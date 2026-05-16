// SPDX-License-Identifier: Apache-2.0
//
// qwen.c -- Qwen3.5 / Qwen3-next architecture specifics.
//
// `#include`-d from slm.c as part of the single-TU build, BUT also
// self-sufficient: build standalone with `-DQWEN_TESTS` to get a
// `qwen-test` binary that runs `qwen_self_test` (forward-pass sanity)
// + `chunked_self_test` (chunked vs autoregressive parity). All
// dependencies are pulled in via the `model.c` header-guarded
// `#include` below.
//
// Holds:
//   - Qwen-specific GGUF parsing: slm_load_config (reads qwen35.*
//     KVs) and slm_load_weights (resolves blk.L.ssm_*/attn_* names).
//   - Gated DeltaNet SSM math: slm_forward_ssm / _ssm_batch.
//   - Per-layer attn/ssm dispatcher: slm_forward_step / _forward_batch
//     (Qwen3-next's full_attention_interval-based interleave).
//   - autoregressive_ref + chunked_self_test (the chunked SSM kernel's
//     parity test is Qwen-math-specific).
//   - qwen_self_test (synthetic mini-Qwen forward pass).
//   - ARENA_F32 macro (forward-pass arena helper, file-scoped).
//
// Model-agnostic plumbing (struct definitions, attention block, KV
// cache, sampling primitives, trace/dump helpers, two-level
// lifecycle) lives in model.c, which this file #includes first.

#ifndef QWEN_C
#define QWEN_C

#include "model.c"       // shared structs, attn, KV, rng, lifecycle


#ifdef LLM_CLI
static const char * SLM_GGUF_PATH_DEFAULT =
    "/Users/leo/Library/Containers/io.github.leok7v.QwenHaiku/Data/"
    "Library/Application Support/Qwen/Qwen3.5-0.8B-Q4_K_M.gguf";
#endif

// ---------------------------------------------------------------------------
// Model config (Qwen3.5 / Qwen3-next GGUF KV reader)
// ---------------------------------------------------------------------------

static int32_t slm_load_config(const struct gguf * g, struct slm_config * c) {
    memset(c, 0, sizeof(*c));
    // Detect arch + pick KV prefix.
    const struct gguf_kv * arch_kv = gguf_find_kv(g, "general.architecture");
    const char * prefix = "qwen3";
    if (arch_kv && arch_kv->v.type == GGUF_VT_STR) {
        size_t cc = 0;
        uint64_t n = gr_u64(arch_kv->v.raw, &cc);
        const char * s = (const char *)(arch_kv->v.raw + cc);
        if (n == 6 && memcmp(s, "qwen35", 6) == 0) {
            prefix = "qwen35";
            // qwen35 is a hybrid SSM+Transformer (Mamba-2/DeltaNet
            // style SSM blocks with full attention every
            // full_attention_interval-th layer). See PLAN.md for the
            // architectural notes; SSM block math is reverse-
            // engineered from tensor shapes/names and is marked
            // "PROBABLY" where uncertain.
            fprintf(stderr,
                "llm: qwen35 hybrid (Mamba-2/DeltaNet + attention)."
                " SSM math is best-guess.\n");
        } else if (n == 5 && memcmp(s, "qwen2", 5) == 0) {
            prefix = "qwen2";
        }
    }
    char key[128];
    #define KV_U32(k, dflt) \
        (snprintf(key, sizeof(key), "%s.%s", prefix, k), \
         (int32_t)gguf_kv_u32(g, key, (dflt)))
    #define KV_F32(k, dflt) \
        (snprintf(key, sizeof(key), "%s.%s", prefix, k), \
         gguf_kv_f32(g, key, (dflt)))
    c->hidden_dim   = KV_U32("embedding_length",                0);
    c->n_layers     = KV_U32("block_count",                     0);
    c->n_heads      = KV_U32("attention.head_count",            0);
    c->n_kv_heads   = KV_U32("attention.head_count_kv",   (uint32_t)c->n_heads);
    c->ffn_dim      = KV_U32("feed_forward_length",             0);
    // qwen35's nominal context_length is 262144 but the KV cache is
    // sized linearly to it (k + v, fp16, per kv-head per layer). At
    // the full 262144 that's 12 GB of committed virtual memory — fine
    // on macOS where zero-fill-on-demand keeps RSS low, lethal on iOS
    // where jetsam counts the virtual commit. Cap at 4096 by default;
    // power users can raise via QWEN_MAX_POS env var.
    int32_t cl = KV_U32("context_length", 2048);
    const char * cap_env = getenv("QWEN_MAX_POS");
    int32_t cap = (cap_env != NULL && atoi(cap_env) > 0) ? atoi(cap_env)
                                                         : 4096;
    c->max_position = (cl < cap) ? cl : cap;
    c->rope_theta   = KV_F32("rope.freq_base",            10000.0f);
    c->norm_eps     = KV_F32("attention.layer_norm_rms_epsilon", 1e-5f);
    int32_t kl      = KV_U32("attention.key_length", 0);
    if (kl > 0) {
        c->head_dim = kl;
    } else if (c->n_heads > 0) {
        c->head_dim = c->hidden_dim / c->n_heads;
    }
    c->rope_dim = KV_U32("rope.dimension_count", 0);
    // mrope sections (qwen35 / qwen3-vl interleaved mrope).
    snprintf(key, sizeof(key), "%s.rope.dimension_sections", prefix);
    const struct gguf_kv * sec_kv = gguf_find_kv(g, key);
    for (int32_t i = 0; i < 4; i++) { c->rope_sections[i] = 0; }
    if (sec_kv && sec_kv->v.type == GGUF_VT_ARRAY
               && sec_kv->v.arr_type == GGUF_VT_I32
               && sec_kv->v.arr_n >= 1) {
        size_t cc = 0;
        int32_t n = (int32_t)sec_kv->v.arr_n;
        if (n > 4) { n = 4; }
        for (int32_t i = 0; i < n; i++) {
            c->rope_sections[i] = (int32_t)gr_u32(sec_kv->v.arr_data, &cc);
        }
    }
    // qwen35 hybrid extras (Gated DeltaNet linear attention).
    if (strcmp(prefix, "qwen35") == 0) {
        c->is_hybrid          = 1;
        c->attn_output_gate   = 1;
        c->full_attn_interval = KV_U32("full_attention_interval",  4);
        c->linear_n_heads     = KV_U32("ssm.group_count",         16);
        c->linear_k_head_dim  = KV_U32("ssm.state_size",         128);
        // GGUF's "ssm.state_size" = head_k_dim; head_v_dim is the
        // same in this model. Total inner V_dim = n_heads*head_v_dim.
        c->linear_v_head_dim  = c->linear_k_head_dim;
        c->linear_conv_kernel = KV_U32("ssm.conv_kernel",          4);
    } else {
        c->is_hybrid          = 0;
        c->attn_output_gate   = 0;
        c->full_attn_interval = 1;
    }
    #undef KV_U32
    #undef KV_F32
    const struct gguf_kv * tk = gguf_find_kv(g, "tokenizer.ggml.tokens");
    if (tk && tk->v.type == GGUF_VT_ARRAY) {
        c->vocab_size = (int32_t)tk->v.arr_n;
    }
    c->bos_id = (int32_t)gguf_kv_u32(g, "tokenizer.ggml.bos_token_id", 0);
    c->eos_id = (int32_t)gguf_kv_u32(g, "tokenizer.ggml.eos_token_id", 0);
    // eot_id is filled in later (after tokenizer_load), when the tokenizer
    // vocab map is available. Initialize to -1 so the stop check
    // ignores it on models without an `<|im_end|>` token.
    c->eot_id = -1;
    return   c->hidden_dim > 0
          && c->n_layers   > 0
          && c->n_heads    > 0
          && c->ffn_dim    > 0
          && c->vocab_size > 0 ? 0 : -1;
}

// ---------------------------------------------------------------------------
// Tokenizer (byte-level BPE) is pulled in from model.c (after struct
// slm_config is in scope there), so qwen.c can drop the include here.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Model weights - pointers into the mmap'd GGUF, plus type tags.
// ---------------------------------------------------------------------------


static int32_t slm_load_weights(const struct gguf * g,
                                const struct slm_config * cfg,
                                struct slm_weights * w) {
    memset(w, 0, sizeof(*w));
    int32_t r = 0;
    r |= slm_resolve_tensor(g, "token_embd.weight",   &w->tok_embd,    1);
    r |= slm_resolve_tensor(g, "output_norm.weight",  &w->output_norm, 1);
    // output.weight is optional; absent => tied embeddings.
    slm_resolve_tensor(g, "output.weight", &w->output, 0);
    if (w->output.data == NULL) { w->output = w->tok_embd; }
    w->layers = (struct slm_layer_w *)oom(
        calloc((size_t)cfg->n_layers, sizeof(struct slm_layer_w)));
    char nm[64];
    for (int32_t L = 0; L < cfg->n_layers; L++) {
        struct slm_layer_w * Lw = &w->layers[L];
        // Layer-type detection: in qwen35 the SSM layers carry an
        // ssm_a tensor; the attention layers don't. (We could also
        // use full_attn_interval modulo, but probing the GGUF is
        // the safer signal.)
        snprintf(nm, sizeof(nm), "blk.%d.ssm_a", L);
        struct slm_tensor_ref probe = {0};
        Lw->is_ssm = (cfg->is_hybrid
                      && slm_resolve_tensor(g, nm, &probe, 0) == 0
                      && probe.data != NULL) ? 1 : 0;

        // Shared norms + FFN. qwen35 calls the post-mixer norm
        // "post_attention_norm.weight"; classic Qwen uses
        // "ffn_norm.weight". Try both.
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", L);
        r |= slm_resolve_tensor(g, nm, &Lw->attn_norm, 1);
        snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", L);
        if (slm_resolve_tensor(g, nm, &Lw->ffn_norm, 0) != 0
            || Lw->ffn_norm.data == NULL) {
            snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", L);
            r |= slm_resolve_tensor(g, nm, &Lw->ffn_norm, 1);
        }
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", L);
        r |= slm_resolve_tensor(g, nm, &Lw->ffn_gate, 1);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight",   L);
        r |= slm_resolve_tensor(g, nm, &Lw->ffn_up,   1);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", L);
        r |= slm_resolve_tensor(g, nm, &Lw->ffn_down, 1);

        if (Lw->is_ssm) {
            snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight",    L);
            r |= slm_resolve_tensor(g, nm, &Lw->attn_qkv,    1);
            snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight",   L);
            r |= slm_resolve_tensor(g, nm, &Lw->attn_gate,   1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_a",              L);
            r |= slm_resolve_tensor(g, nm, &Lw->ssm_a,       1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight",   L);
            r |= slm_resolve_tensor(g, nm, &Lw->ssm_alpha,   1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight",    L);
            r |= slm_resolve_tensor(g, nm, &Lw->ssm_beta,    1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight",  L);
            r |= slm_resolve_tensor(g, nm, &Lw->ssm_conv1d,  1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_dt.bias",        L);
            r |= slm_resolve_tensor(g, nm, &Lw->ssm_dt_bias, 1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight",    L);
            r |= slm_resolve_tensor(g, nm, &Lw->ssm_norm,    1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight",     L);
            r |= slm_resolve_tensor(g, nm, &Lw->ssm_out,     1);
        } else {
            snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight",      L);
            r |= slm_resolve_tensor(g, nm, &Lw->attn_q, 1);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight",      L);
            r |= slm_resolve_tensor(g, nm, &Lw->attn_k, 1);
            snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight",      L);
            r |= slm_resolve_tensor(g, nm, &Lw->attn_v, 1);
            snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", L);
            slm_resolve_tensor(g, nm, &Lw->attn_q_norm, 0);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", L);
            slm_resolve_tensor(g, nm, &Lw->attn_k_norm, 0);
            snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", L);
            r |= slm_resolve_tensor(g, nm, &Lw->attn_out, 1);
        }
    }
    return r;
}



// ---------------------------------------------------------------------------
// SSM block - Gated DeltaNet (qwen35 / Qwen3-Next linear attention)
//
// Ported from transformers/models/qwen3_next/modeling_qwen3_next.py
// (Qwen3NextGatedDeltaNet.forward + torch_recurrent_gated_delta_rule).
// We use the recurrent (token-wise) kernel for both prefill and decode
// - slower but algorithmically simple. The chunked variant is a
// later optimisation.
//
// Symbol map between Python / GGUF / this C:
//
//   Python                                     GGUF tensor             local
//   ---------------------------------------    --------------------    ------
//   in_proj_qkvz first three chunks (Q,K,V)    attn_qkv.weight         qkv
//   in_proj_qkvz fourth chunk      (Z gate)    attn_gate.weight        z
//   in_proj_ba split 'b'                       ssm_beta.weight         b
//   in_proj_ba split 'a'                       ssm_alpha.weight        a
//   conv1d.weight                              ssm_conv1d.weight       conv_w
//   A_log                                      ssm_a                   a_log
//   dt_bias                                    ssm_dt.bias             dt_bias
//   norm (RMSNormGated) weight                 ssm_norm.weight         norm_w
//   out_proj                                   ssm_out.weight          out_w
//
// Math (single token, our use case):
//   1. h_norm = rms_norm(h, attn_norm)
//   2. qkv = attn_qkv · h_norm                              # (6144,)
//      z   = attn_gate · h_norm                             # (2048,)
//      b   = ssm_beta  · h_norm                             # (16,)
//      a   = ssm_alpha · h_norm                             # (16,)
//   3. conv state shift; conv_in = qkv (current step at last slot)
//   4. mixed_qkv = silu(causal_conv1d(qkv with kernel=4))   # (6144,)
//   5. Split mixed_qkv into Q (2048), K (2048), V (2048).
//   6. Reshape to per-head: Q[h, k_d] (16,128), K[h, k_d] (16,128),
//      V[h, v_d] (16,128).
//   7. L2-norm Q[h, :] and K[h, :] along inner dim (use_qk_l2norm
//      in the kernel call).
//   8. beta = sigmoid(b)                                    # (16,)
//      g    = -exp(A_log) · softplus(a + dt_bias)           # (16,)
//      scale = 1 / sqrt(k_head_dim) ; Q *= scale
//   9. Recurrence per head h (state[h] is (k_d, v_d) = (128,128)):
//        g_h   = exp(g[h])
//        state[h, :, :] *= g_h
//        kv_mem[h, v_d] = Σ_k state[h, k, v_d] · K[h, k]
//        delta[h, v_d] = (V[h, v_d] - kv_mem[h, v_d]) · beta[h]
//        state[h, k, v_d] += K[h, k] · delta[h, v_d]
//        out[h, v_d]      = Σ_k state[h, k, v_d] · Q[h, k]
//  10. core_attn_out flat = (16 * 128) = 2048. Apply gated RMSNorm
//      per-head with z as the silu-gate: for each head h,
//         out[h, :] = silu(z[h, :]) * (norm_w * rmsnorm(out[h, :]))
//      Note ssm_norm is (head_v_dim=128,), same per head.
//  11. result = ssm_out · core_attn_out                    # (1024,)
//
// Reference: NVlabs/GatedDeltaNet (ICLR'25) + qwen3_next paper.
// ---------------------------------------------------------------------------

// Forward-pass arena helper. Captures the local `a` (struct arena *)
// from the enclosing function — every forward_* here declares
// `struct arena * a = c->arena;` as its first line, so the macro
// just lifts the (float*)+sizeof(float) noise out of the call site.
// Stays scoped to qwen.c via the file-internal `#define`.
#define ARENA_F32(n) ((float *)arena_alloc(a, (size_t)(n) * sizeof(float)))

static struct tensor * slm_forward_ssm(struct slm_ctx * c,
                                       int32_t L,
                                       struct tensor * h) {
    struct arena * a = c->arena;
    struct slm_layer_w * Lw = &c->model->W.layers[L];
    int32_t n_heads  = c->model->cfg.linear_n_heads;     // 16
    int32_t k_hd     = c->model->cfg.linear_k_head_dim;  // 128
    int32_t v_hd     = c->model->cfg.linear_v_head_dim;  // 128
    int32_t K_dim    = n_heads * k_hd;            // 2048
    int32_t V_dim    = n_heads * v_hd;            // 2048
    int32_t kK       = c->model->cfg.linear_conv_kernel; //    4
    int32_t conv_dim = 2 * K_dim + V_dim;         // 6144
    assert(Lw->ssm_conv1d.shape[1] == conv_dim);
    // 1. RMSNorm.
    struct tensor attn_norm_w = weights_as_f32_view(&Lw->attn_norm, a);
    struct tensor * h_norm =
        tensor_rms_norm(h, &attn_norm_w, c->model->cfg.norm_eps);
    DUMP("attn_norm", h_norm->data, c->model->cfg.hidden_dim);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_norm-%d", (int)L);
        slm_trace_row(nm, "RMS_NORM", h_norm->data, c->model->cfg.hidden_dim);
    }
    // 2. In-projections.
    struct tensor * qkv_pre = matmul_dispatch(&Lw->attn_qkv,  h_norm);
    DUMP("attn_qkv ", qkv_pre->data, conv_dim);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_qkv-%d", (int)L);
        slm_trace_row(nm, "MUL_MAT", qkv_pre->data, conv_dim);
    }
    struct tensor * z = matmul_dispatch(&Lw->attn_gate, h_norm);
    // GGUF naming ambiguity: ssm_alpha vs ssm_beta - neither the
    // Python code nor the file name documents which is which. The
    // Python reference takes `mixed_ba` and splits into (b, a) in
    // that order, where 'b' goes through sigmoid (-> beta in the
    // recurrence) and 'a' is added to dt_bias before softplus
    // (-> g_log). Per llama.cpp's qwen3_next GGUF converter, the
    // FIRST half of mixed_ba (i.e. 'b') is named ssm_beta, and the
    // SECOND half ('a') is named ssm_alpha. So:
    //   ssm_beta  -> b -> sigmoid -> beta
    //   ssm_alpha -> a -> softplus -> g_log
    struct tensor * b_t = matmul_dispatch(&Lw->ssm_beta,  h_norm);
    struct tensor * a_t = matmul_dispatch(&Lw->ssm_alpha, h_norm);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "alpha-%d", (int)L);
        slm_trace_row(nm, "MUL_MAT", a_t->data, n_heads);
        snprintf(nm, sizeof(nm), "beta-%d", (int)L);
        slm_trace_row(nm, "MUL_MAT", b_t->data, n_heads);
    }
    // shapes: qkv_pre(conv_dim,1), z(V_dim,1), b_t(n_heads,1), a_t(n_heads,1).
    // 3. Shift conv state ring; write qkv_pre into the new slot.
    {
        int32_t head    = c->ssm.conv_head[L];
        size_t  lane_off = ((size_t)L * kK + head) * conv_dim;
        float * lane    = c->ssm.conv_state + lane_off;
        for (int32_t i = 0; i < conv_dim; i++) { lane[i] = qkv_pre->data[i]; }
        c->ssm.conv_head[L] = (head + 1) % kK;
    }
    // 4. Causal conv1d step: for each channel ch, sum_k w[k, ch] *
    //    buffer_slot(k, ch), then SiLU. GGUF storage convention:
    //    ssm_conv1d.weight has shape [4, 6144] = [kernel_size,
    //    conv_dim]. shape[0] is the inner (fastest-changing) dim,
    //    so memory layout is conv_w[ch * kK + k]: per channel, kK
    //    contiguous kernel coefficients. Verified against the
    //    eval-callback dump. Position k=kK-1 = current step; k=0 =
    //    oldest in window.
    int32_t head_after = c->ssm.conv_head[L];
    const float * conv_w = (const float *)Lw->ssm_conv1d.data;
    struct tensor * mixed = tensor_new_2d(a, conv_dim, 1);
    size_t conv_lane_base = (size_t)L * kK * conv_dim;
    // Conv pass: write raw conv output into mixed->data so we can
    // trace it before the SiLU activation. Splitting conv from silu
    // does not change the conv kernel's bit output (acc lives in a
    // register; we only write the final value).
    for (int32_t ch = 0; ch < conv_dim; ch++) {
        float acc = 0.0f;
        const float * wch = conv_w + (size_t)ch * kK;
        for (int32_t k = 0; k < kK; k++) {
            int32_t slot = (head_after + k) % kK;
            float   v    = c->ssm.conv_state[conv_lane_base +
                                             (size_t)slot * conv_dim + ch];
            acc += v * wch[k];
        }
        mixed->data[ch] = acc;
    }
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "conv_raw-%d", (int)L);
        slm_trace_row(nm, "CONV1D", mixed->data, conv_dim);
    }
    // SiLU using ggml's NEON polynomial-approximation expf; bit-
    // identical to ggml's GGML_OP_SILU output. The libm scalar form
    // x / (1 + expf(-x)) differs by 1-2 ULPs and compounds through
    // 24 layers to visible drift, so we use ggml's vectorized variant
    // exclusively for parity.
    simd_silu_f32(conv_dim, mixed->data, mixed->data);
    // 5. Split mixed into Q, K, V. Per the Python reference's
    //    `mixed_qkv = torch.cat((query, key, value), dim=-1)` after
    //    reshape, the FLAT layout in memory is block-concatenated
    //    (head-major within each block):
    //      mixed[0 .. K_dim-1]              = Q  (all heads of Q)
    //      mixed[K_dim .. 2*K_dim-1]        = K
    //      mixed[2*K_dim .. 2*K_dim+V_dim-1] = V
    //    Confirmed by the llama-eval-callback dump: v_conv/k_conv/q_conv
    //    are all VIEWS into conv_output_silu of width 2048 each.
    const float * Q_flat = mixed->data + 0;
    const float * K_flat = mixed->data + K_dim;
    const float * V_flat = mixed->data + 2 * K_dim;
    assert(K_dim == 2048 && V_dim == 2048);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "Q_raw-%d", (int)L);
        slm_trace_row(nm, "VIEW", Q_flat, K_dim);
        snprintf(nm, sizeof(nm), "K_raw-%d", (int)L);
        slm_trace_row(nm, "VIEW", K_flat, K_dim);
        snprintf(nm, sizeof(nm), "V_raw-%d", (int)L);
        slm_trace_row(nm, "VIEW", V_flat, V_dim);
    }
    // 6/7. L2-normalise Q and K per-head along head dim. Then scale
    //      Q by 1/sqrt(k_hd) (the attention temperature).
    //
    // Accumulator shape matches ggml_compute_forward_l2_norm_f32:
    //   sum += (double)(q*q)        -- square in fp32, cast UP to fp64
    //   scale = 1/fmaxf(sqrtf(sum), eps)
    // Doing the multiply in fp32 first (then promoting to fp64 for
    // accumulation) is what ggml does, and gives bit-identical sum;
    // `(double)q * (double)q` keeps more precision but produces a
    // different bit result that compounds. eps applied via fmaxf
    // (not inside sqrt) also matters - same reason. The qwen3-next
    // l2_norm eps is 1e-6f per the GGUF default.
    const float l2_eps = 1e-6f;
    float scale = 1.0f / sqrtf((float)k_hd);
    float Q_norm[2048];
    float K_norm[2048];
    for (int32_t h2 = 0; h2 < n_heads; h2++) {
        double qss = 0.0, kss = 0.0;
        for (int32_t i = 0; i < k_hd; i++) {
            float q = Q_flat[h2 * k_hd + i];
            float k = K_flat[h2 * k_hd + i];
            qss += (double)(q * q);
            kss += (double)(k * k);
        }
        float qrs = 1.0f / fmaxf(sqrtf((float)qss), l2_eps);
        float krs = 1.0f / fmaxf(sqrtf((float)kss), l2_eps);
        for (int32_t i = 0; i < k_hd; i++) {
            Q_norm[h2 * k_hd + i] = Q_flat[h2 * k_hd + i] * qrs * scale;
            K_norm[h2 * k_hd + i] = K_flat[h2 * k_hd + i] * krs;
        }
    }
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "Q_l2norm-%d", (int)L);
        slm_trace_row(nm, "L2_NORM_SCALE", Q_norm, K_dim);
        snprintf(nm, sizeof(nm), "K_l2norm-%d", (int)L);
        slm_trace_row(nm, "L2_NORM", K_norm, K_dim);
    }
    // 8. Per-head beta and g.
    //    NOTE: ssm_a in the GGUF is already pre-computed as the
    //    negative-exp-of-A_log scalar (the converter folds the
    //    -exp() into the stored tensor). Verified via
    //    llama-eval-callback dump: g_in = MUL(softplus(α+dt_bias),
    //    ssm_a) - a plain multiply, no extra exp. So we use
    //    `a_log_w[h]` (badly named here - it's actually neg-exp-A)
    //    directly without exponentiating.
    const float * ssm_a_w   = (const float *)Lw->ssm_a.data;
    const float * dt_bias_w = (const float *)Lw->ssm_dt_bias.data;
    float beta[64], g_t[64], g_log_dbg[64], a_softplus_dbg[64];
    assert(n_heads <= 64);
    for (int32_t h2 = 0; h2 < n_heads; h2++) {
        float bb = b_t->data[h2];
        beta[h2] = 1.0f / (1.0f + expf(-bb));   // sigmoid(b)
        float aa = a_t->data[h2] + dt_bias_w[h2];
        float softplus_a = aa > 20.0f ? aa : logf(1.0f + expf(aa));
        a_softplus_dbg[h2] = softplus_a;
        float g_log = ssm_a_w[h2] * softplus_a; // already neg-exp
        g_log_dbg[h2] = g_log;
        g_t[h2] = expf(g_log);
    }
    {
        // Trace names match the equivalent llama.cpp ggml node names
        // for layer 0 of qwen3-next; see tools/qwen-haiku trace map.
        char nm[48];
        snprintf(nm, sizeof(nm), "beta_in-%d", (int)L);
        slm_trace_row(nm, "SIGMOID", beta, n_heads);
        snprintf(nm, sizeof(nm), "a_softplus-%d", (int)L);
        slm_trace_row(nm, "SOFTPLUS", a_softplus_dbg, n_heads);
        // llama's `g_in-0` is the PRE-EXP g_log value (ssm_a * softplus_a),
        // not the post-exp g. Trace g_log here so byte-equality is direct;
        // g_t = expf(g_log) is the local intermediate we feed into the
        // recurrent state update below.
        snprintf(nm, sizeof(nm), "g_in-%d", (int)L);
        slm_trace_row(nm, "MUL", g_log_dbg, n_heads);
    }
    // 9. Recurrent state update per head.
    //    state[L, h, k, v] flat as ssm_state[((L*n_heads + h)*k_hd + k)*v_hd + v]
    //
    // 9.2 and 9.5 are dot reductions whose bit-output must match
    // ggml's qwen3_next chunked path: that path materialises
    // state * K (or state * Q) as a fp32 tensor via ggml_mul,
    // transposes+conts to make k contiguous, then sums each k-row
    // with `ggml_vec_sum_f32`. On Apple builds, ggml_vec_sum_f32
    // dispatches to Accelerate's `vDSP_sve`. We mirror this exactly:
    // form a contiguous k-inner products buffer per v, then call
    // vDSP_sve. Without Accelerate the fallback is fp64 accumulation
    // which matches ggml's #else branch (also fp64).
    size_t  state_off  = (size_t)L * n_heads * k_hd * v_hd;
    float * state_base = c->ssm.ssm_state + state_off;
    float out_flat[2048];   // n_heads * v_hd = 2048
    for (int32_t h2 = 0; h2 < n_heads; h2++) {
        float gh = g_t[h2];
        float bh = beta[h2];
        float * st = state_base + (size_t)h2 * k_hd * v_hd;
        const float * Q_h = Q_norm + h2 * k_hd;
        const float * K_h = K_norm + h2 * k_hd;
        const float * V_h = V_flat + h2 * v_hd;
        // 9.1 state *= g
        for (int32_t kv = 0; kv < k_hd * v_hd; kv++) { st[kv] *= gh; }
        // 9.2 kv_mem[v] = Σ_k state[k, v] * K[k]
        float kv_mem[128];
#ifdef LLM_USE_ACCELERATE
        float prod[128];  // k-contiguous products for vDSP_sve
#endif
        assert(v_hd <= 128);
        assert(k_hd <= 128);
        for (int32_t v = 0; v < v_hd; v++) {
#ifdef LLM_USE_ACCELERATE
            vDSP_vmul(st + v, v_hd, K_h, 1, prod, 1, (vDSP_Length)k_hd);
            float r;
            vDSP_sve(prod, 1, &r, (vDSP_Length)k_hd);
            kv_mem[v] = r;
#else
            double acc = 0.0;
            for (int32_t k = 0; k < k_hd; k++) {
                acc += (double)(st[k * v_hd + v] * K_h[k]);
            }
            kv_mem[v] = (float)acc;
#endif
        }
        // 9.3 delta[v] = (V[v] - kv_mem[v]) * beta
        float delta[128];
        for (int32_t v = 0; v < v_hd; v++) {
            delta[v] = (V_h[v] - kv_mem[v]) * bh;
        }
        // 9.4 state[k, v] += K[k] * delta[v]
        for (int32_t k = 0; k < k_hd; k++) {
            float kk = K_h[k];
            float * row = st + (size_t)k * v_hd;
            for (int32_t v = 0; v < v_hd; v++) {
                row[v] += kk * delta[v];
            }
        }
        // 9.5 out[h, v] = Σ_k state[k, v] * Q[k]
        for (int32_t v = 0; v < v_hd; v++) {
#ifdef LLM_USE_ACCELERATE
            vDSP_vmul(st + v, v_hd, Q_h, 1, prod, 1, (vDSP_Length)k_hd);
            float r;
            vDSP_sve(prod, 1, &r, (vDSP_Length)k_hd);
            out_flat[h2 * v_hd + v] = r;
#else
            double acc = 0.0;
            for (int32_t k = 0; k < k_hd; k++) {
                acc += (double)(st[k * v_hd + v] * Q_h[k]);
            }
            out_flat[h2 * v_hd + v] = (float)acc;
#endif
        }
    }
    {
        // Recurrent-state output - matches llama's attn_output-0
        // shape (V_dim contiguous, head-major). Trace before any
        // gating / norm so we can pinpoint divergence in the
        // recurrent update separately from the gated rmsnorm.
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_output-%d", (int)L);
        slm_trace_row(nm, "RECURRENT", out_flat, V_dim);
    }
    // 10. Gated RMSNorm with z (per head, v_hd wide):
    //       y[h, :] = silu(z[h, :]) * (norm_w * rmsnorm(out[h, :]))
    //     Implementation per Qwen3NextRMSNormGated: variance over
    //     v_hd, rsqrt, multiply by norm_w, then multiply by silu(z).
    // Accumulator pattern + SiLU formulation match ggml's RMS_NORM
    // and ggml_v_silu (NEON polynomial expf). Without these the
    // step-10 gated output drifts by 1-2 ULPs/element vs llama.cpp.
    const float * norm_w = (const float *)Lw->ssm_norm.data;
    struct tensor * y_norm = tensor_new_2d(a, V_dim, 1);
    float z_silu[2048];
    simd_silu_f32(V_dim, z_silu, z->data);
    for (int32_t h2 = 0; h2 < n_heads; h2++) {
        double ssq = 0.0;
        float * yh = out_flat + h2 * v_hd;
        for (int32_t v = 0; v < v_hd; v++) {
            ssq += (double)(yh[v] * yh[v]);
        }
        float mean = (float)(ssq / (double)v_hd);
        float rs = 1.0f / sqrtf(mean + c->model->cfg.norm_eps);
        const float * sg = z_silu + h2 * v_hd;
        float * yo = y_norm->data + h2 * v_hd;
        // ggml's RMS_NORM/MUL/MUL pipeline stores each intermediate
        // to memory, so each fp32 multiply is a separate operation.
        // Split into 3 distinct write-back stages to prevent clang
        // from fusing into a single FMA (which would change bits).
        for (int32_t v = 0; v < v_hd; v++) {
            yo[v] = yh[v] * rs;            // RMS_NORM scale
        }
        for (int32_t v = 0; v < v_hd; v++) {
            yo[v] = yo[v] * norm_w[v];     // MUL weight
        }
        for (int32_t v = 0; v < v_hd; v++) {
            yo[v] = yo[v] * sg[v];         // MUL silu(z)
        }
    }
    DUMP("conv_silu", mixed->data, conv_dim);
    DUMP("y_norm   ", y_norm->data, V_dim);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "conv_silu-%d", (int)L);
        slm_trace_row(nm, "SILU", mixed->data, conv_dim);
        snprintf(nm, sizeof(nm), "y_norm-%d", (int)L);
        slm_trace_row(nm, "RMS_NORM_GATED", y_norm->data, V_dim);
    }
    // 11. Output projection V_dim -> hidden.
    struct tensor * out_t = matmul_dispatch(&Lw->ssm_out, y_norm);
    DUMP("ssm_out  ", out_t->data, c->model->cfg.hidden_dim);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "ssm_out-%d", (int)L);
        slm_trace_row(nm, "MUL_MAT", out_t->data, c->model->cfg.hidden_dim);
    }
    return out_t;
}

// Multi-token SSM block for batched prefill. Processes `n` tokens
// in one call using the chunked Gated DeltaNet kernel
// (chunked_ssm_step_f32). Returns a [hidden_dim, n] tensor of
// post-output-projection vectors, ready for the residual add.
//
// Any n >= 1 is accepted: the internal multi-chunk loop segments
// the input into successive chunked_ssm_step_f32 calls of size
// min(remaining, CHUNK_SIZE), and the per-head state carries in
// place across chunks. State is updated by the kernel itself, not
// by this wrapper.
//
// `h` is [hidden_dim, n] - the input residual stream for these n
// tokens. The SSM layer reads h as-is; the embedding lookup is
// done by the caller (slm_forward_batch).
static struct tensor * slm_forward_ssm_batch(struct slm_ctx * c,
                                              int32_t L, int32_t n,
                                              struct tensor * h) {
    struct arena * a = c->arena;
    struct slm_layer_w * Lw = &c->model->W.layers[L];
    int32_t n_heads  = c->model->cfg.linear_n_heads;
    int32_t k_hd     = c->model->cfg.linear_k_head_dim;
    int32_t v_hd     = c->model->cfg.linear_v_head_dim;
    int32_t K_dim    = n_heads * k_hd;
    int32_t V_dim    = n_heads * v_hd;
    int32_t kK       = c->model->cfg.linear_conv_kernel;
    int32_t conv_dim = 2 * K_dim + V_dim;
    assert(n > 0);
    // 1. attn_norm per token (rms_norm handles n via x->ne[1]).
    struct tensor attn_norm_w = weights_as_f32_view(&Lw->attn_norm, a);
    struct tensor * h_norm =
        tensor_rms_norm(h, &attn_norm_w, c->model->cfg.norm_eps);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_norm-%d", (int)L);
        slm_trace_batch(nm, "RMS_NORM",
                       h_norm->data, c->model->cfg.hidden_dim, n);
    }
    // 2. In-projections - matmul_dispatch already iterates the n axis.
    struct tensor * qkv_pre = matmul_dispatch(&Lw->attn_qkv,  h_norm);  // [conv_dim, n]
    struct tensor * z       = matmul_dispatch(&Lw->attn_gate, h_norm);  // [V_dim, n]
    struct tensor * b_t     = matmul_dispatch(&Lw->ssm_beta,  h_norm);  // [n_heads, n]
    struct tensor * a_t     = matmul_dispatch(&Lw->ssm_alpha, h_norm);  // [n_heads, n]
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_qkv-%d", (int)L);
        slm_trace_batch(nm, "MUL_MAT", qkv_pre->data, conv_dim, n);
        snprintf(nm, sizeof(nm), "alpha-%d", (int)L);
        slm_trace_batch(nm, "MUL_MAT", a_t->data, n_heads, n);
        snprintf(nm, sizeof(nm), "beta-%d", (int)L);
        slm_trace_batch(nm, "MUL_MAT", b_t->data, n_heads, n);
    }
    // 3. Conv1d step per token (sequential, ring buffer state).
    struct tensor * mixed = tensor_new_2d(a, conv_dim, n);
    const float * conv_w = (const float *)Lw->ssm_conv1d.data;
    size_t conv_lane_base = (size_t)L * kK * conv_dim;
    for (int32_t t = 0; t < n; t++) {
        int32_t head = c->ssm.conv_head[L];
        size_t  lane_off = conv_lane_base + (size_t)head * conv_dim;
        float * lane = c->ssm.conv_state + lane_off;
        for (int32_t i = 0; i < conv_dim; i++) {
            lane[i] = qkv_pre->data[t * conv_dim + i];
        }
        c->ssm.conv_head[L] = (head + 1) % kK;
        int32_t head_after = c->ssm.conv_head[L];
        for (int32_t ch = 0; ch < conv_dim; ch++) {
            float acc = 0.0f;
            const float * wch = conv_w + (size_t)ch * kK;
            for (int32_t k = 0; k < kK; k++) {
                int32_t slot = (head_after + k) % kK;
                float v_l = c->ssm.conv_state[
                    conv_lane_base + (size_t)slot * conv_dim + ch];
                acc += v_l * wch[k];
            }
            mixed->data[t * conv_dim + ch] = acc;
        }
    }
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "conv_raw-%d", (int)L);
        slm_trace_batch(nm, "CONV1D", mixed->data, conv_dim, n);
    }
    // 4. SiLU on all tokens at once.
    simd_silu_f32(n * conv_dim, mixed->data, mixed->data);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "conv_silu-%d", (int)L);
        slm_trace_batch(nm, "SILU", mixed->data, conv_dim, n);
    }
    // 5. Split Q, K, V (per-token contiguous within conv_dim).
    float * Q_all = ARENA_F32((size_t)n * K_dim);
    float * K_all = ARENA_F32((size_t)n * K_dim);
    float * V_all = ARENA_F32((size_t)n * V_dim);
    for (int32_t t = 0; t < n; t++) {
        const float * row = mixed->data + (size_t)t * conv_dim;
        memcpy(Q_all + (size_t)t * K_dim, row,         (size_t)K_dim * sizeof(float));
        memcpy(K_all + (size_t)t * K_dim, row + K_dim, (size_t)K_dim * sizeof(float));
        memcpy(V_all + (size_t)t * V_dim, row + 2 * K_dim, (size_t)V_dim * sizeof(float));
    }
    // 6/7. L2-norm Q, K per head, scale Q by 1/sqrt(k_hd).
    const float l2_eps = 1e-6f;
    float scale = 1.0f / sqrtf((float)k_hd);
    for (int32_t t = 0; t < n; t++) {
        for (int32_t h2 = 0; h2 < n_heads; h2++) {
            double qss = 0.0, kss = 0.0;
            for (int32_t i = 0; i < k_hd; i++) {
                float qv = Q_all[t * K_dim + h2 * k_hd + i];
                float kv = K_all[t * K_dim + h2 * k_hd + i];
                qss += (double)(qv * qv);
                kss += (double)(kv * kv);
            }
            float qrs = 1.0f / fmaxf(sqrtf((float)qss), l2_eps);
            float krs = 1.0f / fmaxf(sqrtf((float)kss), l2_eps);
            for (int32_t i = 0; i < k_hd; i++) {
                Q_all[t * K_dim + h2 * k_hd + i] *= qrs * scale;
                K_all[t * K_dim + h2 * k_hd + i] *= krs;
            }
        }
    }
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "Q_l2norm-%d", (int)L);
        slm_trace_batch(nm, "L2_NORM_SCALE", Q_all, K_dim, n);
        snprintf(nm, sizeof(nm), "K_l2norm-%d", (int)L);
        slm_trace_batch(nm, "L2_NORM", K_all, K_dim, n);
    }
    // 8. beta, g_log per token per head.
    const float * ssm_a_w   = (const float *)Lw->ssm_a.data;
    const float * dt_bias_w = (const float *)Lw->ssm_dt_bias.data;
    float * beta_all  = ARENA_F32((size_t)n * n_heads);
    float * g_log_all = ARENA_F32((size_t)n * n_heads);
    for (int32_t t = 0; t < n; t++) {
        for (int32_t h2 = 0; h2 < n_heads; h2++) {
            float bb = b_t->data[t * n_heads + h2];
            beta_all[t * n_heads + h2] = 1.0f / (1.0f + expf(-bb));
            float aa = a_t->data[t * n_heads + h2] + dt_bias_w[h2];
            float softplus_a = aa > 20.0f ? aa : logf(1.0f + expf(aa));
            g_log_all[t * n_heads + h2] = ssm_a_w[h2] * softplus_a;
        }
    }
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "beta_in-%d", (int)L);
        slm_trace_batch(nm, "SIGMOID", beta_all, n_heads, n);
        snprintf(nm, sizeof(nm), "g_in-%d", (int)L);
        slm_trace_batch(nm, "MUL", g_log_all, n_heads, n);
    }
    // 9. Chunked SSM per head. Per-chunk buffers (sized at CHUNK_SIZE
    //    so the same scratch handles every chunk; the kernel uses the
    //    actual chunk_n as its leading dim).
    // TODO: const size_t n = CHUNK_SIZE * sizeof(float) maybe?
    // TODO: #define slm_alloc() to fight this ugliness: ARENA_F32(CHUNK_SIZE * k_hd); ?
    float * q_chunk    = ARENA_F32(CHUNK_SIZE * k_hd);
    float * k_chunk    = ARENA_F32(CHUNK_SIZE * k_hd);
    float * v_chunk    = ARENA_F32(CHUNK_SIZE * v_hd);
    float * g_log_h    = ARENA_F32(CHUNK_SIZE);
    float * beta_h     = ARENA_F32(CHUNK_SIZE);
    float * out_chunk  = ARENA_F32(CHUNK_SIZE * v_hd);
    // Chunked scratch — reused per head.
    float * sc_gcs        = ARENA_F32(CHUNK_SIZE);
    float * sc_gexp       = ARENA_F32(CHUNK_SIZE);
    float * sc_decay_mask = ARENA_F32(CHUNK_SIZE * CHUNK_SIZE);
    float * sc_k_beta     = ARENA_F32(CHUNK_SIZE * k_hd);
    float * sc_v_beta     = ARENA_F32(CHUNK_SIZE * v_hd);
    float * sc_kk_dot     = ARENA_F32(CHUNK_SIZE * CHUNK_SIZE);
    float * sc_lhs        = ARENA_F32(CHUNK_SIZE * CHUNK_SIZE);
    float * sc_attn       = ARENA_F32(CHUNK_SIZE * CHUNK_SIZE);
    float * sc_v_eff      = ARENA_F32(CHUNK_SIZE * v_hd);
    float * sc_kbeta_gexp = ARENA_F32(CHUNK_SIZE * k_hd);
    float * sc_k_cumdecay = ARENA_F32(CHUNK_SIZE * k_hd);
    float * sc_attn_kq    = ARENA_F32(CHUNK_SIZE * CHUNK_SIZE);
    float * sc_q_g_exp    = ARENA_F32(CHUNK_SIZE * k_hd);
    float * sc_attn_inter = ARENA_F32(CHUNK_SIZE * v_hd);
    float * sc_v_prime    = ARENA_F32(CHUNK_SIZE * v_hd);
    float * sc_v_new      = ARENA_F32(CHUNK_SIZE * v_hd);
    float * sc_v_attn     = ARENA_F32(CHUNK_SIZE * v_hd);
    float * sc_key_gdiff  = ARENA_F32(CHUNK_SIZE * k_hd);
    float * sc_kgd_vnew   = ARENA_F32(k_hd * v_hd);
    size_t state_off  = (size_t)L * n_heads * k_hd * v_hd;
    float * state_base = c->ssm.ssm_state + state_off;
    float * out_flat = ARENA_F32((size_t)n * V_dim);
    // Multi-chunk loop: process min(remaining, CHUNK_SIZE) tokens
    // per call. State carries in place across chunks because
    // chunked_ssm_step_f32 updates it. The tail chunk is whatever
    // length is left (no zero-padding — the kernel takes its own
    // chunk_n parameter).
    int32_t n_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    for (int32_t ck = 0; ck < n_chunks; ck++) {
        int32_t chunk_start = ck * CHUNK_SIZE;
        int32_t chunk_n     = n - chunk_start;
        if (chunk_n > CHUNK_SIZE) { chunk_n = CHUNK_SIZE; }
        // Pack each head's chunk_n tokens; no zero-padding needed
        // because the kernel processes exactly chunk_n tokens this
        // call (state still carries across chunks).
        for (int32_t h2 = 0; h2 < n_heads; h2++) {
            for (int32_t t = 0; t < chunk_n; t++) {
                int32_t tg = chunk_start + t;
                memcpy(q_chunk + (size_t)t * k_hd,
                       Q_all   + (size_t)tg * K_dim + (size_t)h2 * k_hd,
                       (size_t)k_hd * sizeof(float));
                memcpy(k_chunk + (size_t)t * k_hd,
                       K_all   + (size_t)tg * K_dim + (size_t)h2 * k_hd,
                       (size_t)k_hd * sizeof(float));
                memcpy(v_chunk + (size_t)t * v_hd,
                       V_all   + (size_t)tg * V_dim + (size_t)h2 * v_hd,
                       (size_t)v_hd * sizeof(float));
                g_log_h[t] = g_log_all[tg * n_heads + h2];
                beta_h [t] = beta_all [tg * n_heads + h2];
            }
            float * state_h = state_base + (size_t)h2 * k_hd * v_hd;
            chunked_ssm_step_f32(chunk_n, k_hd, v_hd,
                                 q_chunk, k_chunk, v_chunk, g_log_h, beta_h,
                                 state_h, out_chunk,
                                 sc_gcs, sc_gexp, sc_decay_mask,
                                 sc_k_beta, sc_v_beta, sc_kk_dot, sc_lhs,
                                 sc_attn, sc_v_eff, sc_kbeta_gexp, sc_k_cumdecay,
                                 sc_attn_kq, sc_q_g_exp, sc_attn_inter,
                                 sc_v_prime, sc_v_new, sc_v_attn,
                                 sc_key_gdiff, sc_kgd_vnew);
            for (int32_t t = 0; t < chunk_n; t++) {
                int32_t tg = chunk_start + t;
                memcpy(out_flat + (size_t)tg * V_dim + (size_t)h2 * v_hd,
                       out_chunk + (size_t)t * v_hd,
                       (size_t)v_hd * sizeof(float));
            }
        }
    }
    // 10. Gated RMSNorm + z-SiLU per token. Each token's y_norm
    //     depends only on its own out_flat row + z row.
    const float * norm_w = (const float *)Lw->ssm_norm.data;
    struct tensor * y_norm = tensor_new_2d(a, V_dim, n);
    float z_silu_buf[2048];
    for (int32_t t = 0; t < n; t++) {
        simd_silu_f32(V_dim, z_silu_buf, z->data + (size_t)t * V_dim);
        for (int32_t h2 = 0; h2 < n_heads; h2++) {
            double ssq = 0.0;
            float * yh = out_flat + (size_t)t * V_dim + (size_t)h2 * v_hd;
            for (int32_t v = 0; v < v_hd; v++) {
                ssq += (double)(yh[v] * yh[v]);
            }
            float mean = (float)(ssq / (double)v_hd);
            float rs   = 1.0f / sqrtf(mean + c->model->cfg.norm_eps);
            const float * sg = z_silu_buf + (size_t)h2 * v_hd;
            float * yo = y_norm->data + (size_t)t * V_dim + (size_t)h2 * v_hd;
            for (int32_t v = 0; v < v_hd; v++) {
                yo[v] = yh[v] * rs;
            }
            for (int32_t v = 0; v < v_hd; v++) {
                yo[v] = yo[v] * norm_w[v];
            }
            for (int32_t v = 0; v < v_hd; v++) {
                yo[v] = yo[v] * sg[v];
            }
        }
    }
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "y_norm-%d", (int)L);
        slm_trace_batch(nm, "GATED_RMSNORM", y_norm->data, V_dim, n);
    }
    // 11. Output projection V_dim -> hidden_dim, per token.
    struct tensor * out_t = matmul_dispatch(&Lw->ssm_out, y_norm);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "ssm_out-%d", (int)L);
        slm_trace_batch(nm, "MUL_MAT",
                       out_t->data, c->model->cfg.hidden_dim, n);
    }
    return out_t;
}

// Attention block for one transformer layer (non-SSM). Returns the
// post-output-projection tensor; the caller is responsible for the
// residual add. Mirrors slm_forward_ssm's contract.

// Multi-token attention block for batched prefill. Processes `n`
// tokens against the existing KV cache (rows 0..pos_start-1) plus
// the n new rows written by this call (rows pos_start..pos_start+n-1).
//
// Mirrors slm_forward_attn's contract: input is the (hidden_dim, n)
// residual stream, output is the (hidden_dim, n) post-projection
// tensor; caller does the residual add. State writes:
//   - KV cache rows pos_start..pos_start+n-1 (k_rope/v, cast to fp16)
//
// Causal masking is handled by tensor_attention, which uses
// k_offset = pos_start so each query t attends only to keys
// 0..(pos_start + t).

static struct tensor * slm_forward_step(struct slm_ctx * c,
                                        int32_t tok_id, int32_t pos) {
    struct arena * a = c->arena;
    arena_reset(a);

    // 1. Embedding lookup. tok_embd in qwen35 GGUF is Q6_K-quantised
    //    (not fp32), so we can't view it as a plain fp32 row table.
    //    Dequantise the single row we need by routing through
    //    matmul_dispatch with a one-hot activation.
    int32_t hidden_dim = c->model->cfg.hidden_dim;
    struct tensor * h;
    if (c->model->W.tok_embd.type == GGUF_TT_F32) {
        struct tensor * ids = tensor_new_1d(a, 1);
        ids->data[0] = (float)tok_id;
        struct tensor tev = weights_as_f32_view(&c->model->W.tok_embd, a);
        h = tensor_get_rows(&tev, ids);
    } else {
        // tok_embd shape in GGUF is (hidden_dim, vocab_size). Rather
        // than build a one-hot and run a 200M-weight matmul through
        // matmul_dispatch just to extract one row, dequant the single
        // block-aligned chunk of size hidden_dim that lives at
        // tok_id*hidden_dim and copy into h directly. Q6_K is the
        // only quant we hit for tok_embd in Qwen3.5-0.8B-Q4_K_M;
        // other quants would need a corresponding dequant call here.
        h = tensor_new_2d(a, hidden_dim, 1);
        assert(hidden_dim % QK_K == 0);
        int32_t blocks_per_row = hidden_dim / QK_K;
        const q6k_block * row_blocks =
            (const q6k_block *)c->model->W.tok_embd.data
            + (size_t)tok_id * blocks_per_row;
        for (int32_t b = 0; b < blocks_per_row; b++) {
            q6k_dequant_block(row_blocks + b, h->data + b * QK_K);
        }
    }
    // h shape: (hidden_dim, 1)
    slm_trace_row("inp_embd", "GET_ROWS", h->data, c->model->cfg.hidden_dim);

    static int8_t use_ssm_batch = -1;
    if (use_ssm_batch < 0) {
        use_ssm_batch = (getenv("LLM_USE_SSM_BATCH") != NULL) ? 1 : 0;
    }
    for (int32_t L = 0; L < c->model->cfg.n_layers; L++) {
        struct slm_layer_w * Lw = &c->model->W.layers[L];
        struct tensor * mix;
        if (Lw->is_ssm) {
            // Single-token forward: route to the chunked-SSM batched
            // wrapper under LLM_USE_SSM_BATCH=1. For n=1 the chunked
            // kernel is mathematically equivalent to the recurrent
            // autoregressive path; this gate exists so we can run
            // `--single "Hello"` with both paths and diff the traces.
            mix = use_ssm_batch
                ? slm_forward_ssm_batch(c, L, 1, h)
                : slm_forward_ssm(c, L, h);
        } else {
            mix = slm_forward_attn(c, L, pos, h);
        }
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "mix-%d", (int)L);
            slm_trace_row(nm, Lw->is_ssm ? "SSM" : "ATTN",
                         mix->data, c->model->cfg.hidden_dim);
        }
        h = tensor_add(h, mix);

        // FFN: SwiGLU (shared by attention and SSM layers).
        DUMP("[F]residual", h->data, c->model->cfg.hidden_dim);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "mix_residual-%d", (int)L);
            slm_trace_row(nm, "ADD", h->data, c->model->cfg.hidden_dim);
        }
        struct tensor ffn_norm_w =
            weights_as_f32_view(&Lw->ffn_norm, a);
        struct tensor * h_ffn_norm =
            tensor_rms_norm(h, &ffn_norm_w, c->model->cfg.norm_eps);
        DUMP("[F]post_atn", h_ffn_norm->data, c->model->cfg.hidden_dim);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "ffn_norm-%d", (int)L);
            slm_trace_row(nm, "RMS_NORM",
                         h_ffn_norm->data, c->model->cfg.hidden_dim);
        }
        struct tensor * gate     = matmul_dispatch(&Lw->ffn_gate, h_ffn_norm);
        struct tensor * up       = matmul_dispatch(&Lw->ffn_up,   h_ffn_norm);
        // FFN SwiGLU uses NEON-poly SiLU (bit-exact with ggml's
        // GGML_OP_SILU). Scalar libm would drift 1-2 ULP per element.
        struct tensor * gate_act = tensor_new_nd(a, gate->ndim, gate->ne);
        simd_silu_f32((int)tensor_nelements(gate),
                          gate_act->data, gate->data);
        struct tensor * ffn_in   = tensor_mul(gate_act, up);
        struct tensor * ffn_out  = matmul_dispatch(&Lw->ffn_down, ffn_in);
        DUMP("[F]ffn_out ", ffn_out->data, c->model->cfg.hidden_dim);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "ffn_out-%d", (int)L);
            slm_trace_row(nm, "MUL_MAT",
                         ffn_out->data, c->model->cfg.hidden_dim);
        }
        h = tensor_add(h, ffn_out);
        DUMP("[F]post_ffn", h->data, c->model->cfg.hidden_dim);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "l_out-%d", (int)L);
            slm_trace_row(nm, "ADD", h->data, c->model->cfg.hidden_dim);
        }
    }

    // 12. Final RMSNorm + lm_head.
    struct tensor output_norm_w =
        weights_as_f32_view(&c->model->W.output_norm, a);
    struct tensor * h_final =
        tensor_rms_norm(h, &output_norm_w, c->model->cfg.norm_eps);
    slm_trace_row("result_norm", "RMS_NORM",
                 h_final->data, c->model->cfg.hidden_dim);
    struct tensor * logits = matmul_dispatch(&c->model->W.output, h_final);
    slm_trace_row("result_output", "MUL_MAT",
                 logits->data, (int64_t)tensor_nelements(logits));
    return logits;
}

// ---------------------------------------------------------------------------
// Batched (multi-token) forward for prefill.
//
// Processes `n` tokens at positions [pos_start, pos_start + n) in a
// single call, returning the logits for the LAST token. Used for
// prefill of multi-token prompts where the SSM layers benefit from
// the chunked Gated DeltaNet kernel (~10x throughput vs n calls to
// the autoregressive `slm_forward_step` for long prompts).
//
// Per-layer routing:
//   - SSM layer: slm_forward_ssm_batch (chunked, all n tokens in one
//                call; internal multi-chunk loop for n > CHUNK_SIZE).
//   - Attention layer: slm_forward_attn_batch (batched-queries-vs-
//                full-KV-cache kernel; causal mask via k_offset =
//                pos_start so query t attends only to keys
//                0..pos_start+t).
//   - FFN: shape [hidden, n] flows through tensor_rms_norm,
//          matmul_dispatch, tensor_mul, simd_silu_f32 untouched -
//          all four operate over the `n` axis natively.
//
// Caller invariants:
//   - Caller is responsible for advancing `c->pos` (mirrors
//     slm_forward_step semantics; we just compute, no state pointer).
//   - SSM state (`c->ssm.ssm_state`, `c->ssm.conv_state`,
//     `c->ssm.conv_head`) and KV cache rows for
//     pos_start..pos_start+n-1 are written by the two batch helpers.
//
// Returns logits for the LAST token (shape [vocab_size, 1]). Earlier
// tokens' logits are discarded (we only need them for prefill
// state-building; the first sample-able token is pos_start + n - 1).
__attribute__((unused))
static struct tensor * slm_forward_batch(struct slm_ctx * c,
                                         const int32_t * tok_ids,
                                         int32_t n,
                                         int32_t pos_start) {
    assert(n > 0);
    struct arena * a = c->arena;
    arena_reset(a);
    int32_t hidden_dim = c->model->cfg.hidden_dim;
    // 1. Embedding lookup for n tokens. Same Q6_K / F32 split as the
    //    single-token path; we just stack n rows.
    struct tensor * h = tensor_new_2d(a, hidden_dim, n);
    for (int32_t t = 0; t < n; t++) {
        int32_t tok_id = tok_ids[t];
        float * dst = h->data + (size_t)t * hidden_dim;
        if (c->model->W.tok_embd.type == GGUF_TT_F32) {
            const float * src = (const float *)c->model->W.tok_embd.data
                              + (size_t)tok_id * hidden_dim;
            memcpy(dst, src, (size_t)hidden_dim * sizeof(float));
        } else {
            // Q6_K dequant per row, same as slm_forward_step.
            assert(hidden_dim % QK_K == 0);
            int32_t blocks_per_row = hidden_dim / QK_K;
            const q6k_block * row_blocks =
                (const q6k_block *)c->model->W.tok_embd.data
                + (size_t)tok_id * blocks_per_row;
            for (int32_t b = 0; b < blocks_per_row; b++) {
                q6k_dequant_block(row_blocks + b, dst + b * QK_K);
            }
        }
    }
    slm_trace_batch("inp_embd", "GET_ROWS", h->data, hidden_dim, n);
    // 2. Per-layer.
    for (int32_t L = 0; L < c->model->cfg.n_layers; L++) {
        struct slm_layer_w * Lw = &c->model->W.layers[L];
        struct tensor * mix;
        if (Lw->is_ssm) {
            // SSM: process all n tokens via chunked kernel. Internal
            // multi-chunk loop handles n > CHUNK_SIZE.
            mix = slm_forward_ssm_batch(c, L, n, h);
        } else {
            // Attention: batched-queries kernel. Processes all n
            // tokens against the KV cache in one pass; tensor_attention
            // applies the causal mask via k_offset = pos_start so
            // query t attends only to keys 0..(pos_start+t).
            mix = slm_forward_attn_batch(c, L, pos_start, n, h);
        }
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "mix-%d", (int)L);
            slm_trace_batch(nm, Lw->is_ssm ? "SSM" : "ATTN",
                           mix->data, hidden_dim, n);
        }
        // h = h + mix (per-token elementwise add).
        h = tensor_add(h, mix);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "mix_residual-%d", (int)L);
            slm_trace_batch(nm, "ADD", h->data, hidden_dim, n);
        }
        // FFN: rms_norm + SwiGLU + matmul, all n-axis-aware.
        struct tensor ffn_norm_w = weights_as_f32_view(&Lw->ffn_norm, a);
        struct tensor * h_ffn_norm =
            tensor_rms_norm(h, &ffn_norm_w, c->model->cfg.norm_eps);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "ffn_norm-%d", (int)L);
            slm_trace_batch(nm, "RMS_NORM", h_ffn_norm->data, hidden_dim, n);
        }
        struct tensor * gate     = matmul_dispatch(&Lw->ffn_gate, h_ffn_norm);
        struct tensor * up       = matmul_dispatch(&Lw->ffn_up,   h_ffn_norm);
        struct tensor * gate_act = tensor_new_2d(a, gate->ne[0], gate->ne[1]);
        simd_silu_f32((int)tensor_nelements(gate),
                          gate_act->data, gate->data);
        struct tensor * ffn_in   = tensor_mul(gate_act, up);
        struct tensor * ffn_out  = matmul_dispatch(&Lw->ffn_down, ffn_in);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "ffn_out-%d", (int)L);
            slm_trace_batch(nm, "MUL_MAT", ffn_out->data, hidden_dim, n);
        }
        h = tensor_add(h, ffn_out);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "l_out-%d", (int)L);
            slm_trace_batch(nm, "ADD", h->data, hidden_dim, n);
        }
    }
    // 3. Final RMS-norm + lm_head on the LAST token only.
    //    Earlier tokens' logits aren't needed for prefill.
    struct tensor * h_last = tensor_new_2d(a, hidden_dim, 1);
    memcpy(h_last->data,
           h->data + (size_t)(n - 1) * hidden_dim,
           (size_t)hidden_dim * sizeof(float));
    struct tensor output_norm_w =
        weights_as_f32_view(&c->model->W.output_norm, a);
    struct tensor * h_final =
        tensor_rms_norm(h_last, &output_norm_w, c->model->cfg.norm_eps);
    slm_trace_row("result_norm", "RMS_NORM", h_final->data, hidden_dim);
    struct tensor * logits = matmul_dispatch(&c->model->W.output, h_final);
    slm_trace_row("result_output", "MUL_MAT",
                 logits->data, c->model->cfg.vocab_size);
    return logits;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------


// xoroshiro128** PRNG (Blackman/Vigna, public-domain reference).
// Cheap, seedable, much better statistical quality than libc rand().
// Returns a uniform u64; the sampler divides into [0, 1) below.

// ---------------------------------------------------------------------------
// --chunked-test: lives here (not in chunked.c) because the
// reference implementation `autoregressive_ref` IS Qwen3-next math
// (the gated DeltaNet SSM recurrence the chunked kernel must
// match). chunked.c stays a pure SSM kernel; this test asserts
// "chunked-vs-autoregressive agree for Qwen-shaped inputs", which
// is a qwen.c concern.


// ---------------------------------------------------------------------------
// --chunked-test: run the chunked SSM kernel on a 1-real-token chunk
// (padded with 63 zero-tokens) and compare to the autoregressive
// math direct evaluation. Per the degeneracy proof above, they MUST
// agree (mathematically identical operations applied to the same
// fp32 inputs). A 1-2 ULP discrepancy on a handful of elements is
// acceptable - that is intra-kernel reduction order drift. A larger
// discrepancy means our chunked port is wrong.
//
// Lives here so the test sits next to the code it exercises;
// `#include`-d into slm.c as part of the single-TU build, and
// invoked from slm.c's CLI dispatch via `--chunked-test`.

#define CHUNKED_TEST_KHD 128
#define CHUNKED_TEST_VHD 128
// N=8 validates the chunked recurrence against an autoregressive
// reference implementation. Both should agree to within fp32
// accumulation noise (sub-1e-5 relative). The chunked path uses
// a slightly different reduction order, hence the small tolerance.
#define CHUNKED_TEST_NTOK 8

static void autoregressive_ref(int n, int k_hd, int v_hd,
                               const float * Q,       // [n, k_hd]
                               const float * K,       // [n, k_hd]
                               const float * V,       // [n, v_hd]
                               const float * g_log,   // [n]
                               const float * beta,    // [n]
                               float * state,         // [k_hd, v_hd], zeroed by caller
                               float * out) {         // [n, v_hd]
    for (int t = 0; t < n; t++) {
        float g = expf(g_log[t]);
        float b = beta[t];
        // state *= g
        for (int d = 0; d < k_hd; d++) {
            for (int e = 0; e < v_hd; e++) {
                state[d * v_hd + e] *= g;
            }
        }
        // kv_mem[v] = sum_k state[k, v] * K[t, k]
        float kv_mem[CHUNKED_TEST_VHD];
        for (int e = 0; e < v_hd; e++) {
            float s = 0.0f;
            for (int d = 0; d < k_hd; d++) {
                s += state[d * v_hd + e] * K[t * k_hd + d];
            }
            kv_mem[e] = s;
        }
        // delta = (V - kv_mem) * beta
        float delta[CHUNKED_TEST_VHD];
        for (int e = 0; e < v_hd; e++) {
            delta[e] = (V[t * v_hd + e] - kv_mem[e]) * b;
        }
        // state[k, v] += K[t, k] * delta[v]
        for (int d = 0; d < k_hd; d++) {
            float kd = K[t * k_hd + d];
            for (int e = 0; e < v_hd; e++) {
                state[d * v_hd + e] += kd * delta[e];
            }
        }
        // out[t, v] = sum_k state[k, v] * Q[t, k]
        for (int e = 0; e < v_hd; e++) {
            float s = 0.0f;
            for (int d = 0; d < k_hd; d++) {
                s += state[d * v_hd + e] * Q[t * k_hd + d];
            }
            out[t * v_hd + e] = s;
        }
    }
}

__attribute__((unused))
static int32_t chunked_self_test(void) {
    enum { k_hd = CHUNKED_TEST_KHD, v_hd = CHUNKED_TEST_VHD,
           N   = CHUNKED_TEST_NTOK };
    // Deterministic multi-token inputs (4 distinct tokens).
    float Q[N * k_hd], K[N * k_hd], V[N * v_hd];
    float g_log[N], beta_arr_in[N];
    for (int t = 0; t < N; t++) {
        for (int i = 0; i < k_hd; i++) {
            Q[t * k_hd + i] = sinf((float)((t + 1) * (i + 1)) * 0.0173f);
            K[t * k_hd + i] = cosf((float)((t + 1) * (i + 1)) * 0.0211f);
        }
        for (int i = 0; i < v_hd; i++) {
            V[t * v_hd + i] = sinf((float)((t + 1) * (i + 1)) * 0.0149f) * 0.5f;
        }
        g_log[t]       = -0.25f - 0.05f * (float)t;
        beta_arr_in[t] = 0.55f + 0.03f * (float)t;
    }
    // Autoregressive reference (N sequential steps).
    static float auto_state[k_hd * v_hd];
    static float auto_out  [N * v_hd];
    memset(auto_state, 0, sizeof(auto_state));
    memset(auto_out,   0, sizeof(auto_out));
    autoregressive_ref(N, k_hd, v_hd, Q, K, V, g_log, beta_arr_in,
                       auto_state, auto_out);
    // Chunked path: N real tokens padded to CHUNK_SIZE.
    static float q_pad   [CHUNK_SIZE * k_hd];
    static float k_pad   [CHUNK_SIZE * k_hd];
    static float v_pad   [CHUNK_SIZE * v_hd];
    static float g_log_p [CHUNK_SIZE];
    static float beta_p  [CHUNK_SIZE];
    static float state   [k_hd * v_hd];
    static float out     [CHUNK_SIZE * v_hd];
    memset(q_pad,   0, sizeof(q_pad));
    memset(k_pad,   0, sizeof(k_pad));
    memset(v_pad,   0, sizeof(v_pad));
    memset(g_log_p, 0, sizeof(g_log_p));
    memset(beta_p,  0, sizeof(beta_p));
    memset(state,   0, sizeof(state));
    memset(out,     0, sizeof(out));
    for (int t = 0; t < N; t++) {
        for (int i = 0; i < k_hd; i++) {
            q_pad[t * k_hd + i] = Q[t * k_hd + i];
            k_pad[t * k_hd + i] = K[t * k_hd + i];
        }
        for (int i = 0; i < v_hd; i++) {
            v_pad[t * v_hd + i] = V[t * v_hd + i];
        }
        g_log_p[t] = g_log[t];
        beta_p [t] = beta_arr_in[t];
    }
    // Scratch (heap; one-shot test, no perf concern).
    float * sc_gcs        = (float *)oom(calloc(CHUNK_SIZE, sizeof(float)));
    float * sc_gexp       = (float *)oom(calloc(CHUNK_SIZE, sizeof(float)));
    float * sc_decay_mask = (float *)oom(calloc((size_t)CHUNK_SIZE * CHUNK_SIZE, sizeof(float)));
    float * sc_k_beta     = (float *)oom(calloc((size_t)CHUNK_SIZE * k_hd,       sizeof(float)));
    float * sc_v_beta     = (float *)oom(calloc((size_t)CHUNK_SIZE * v_hd,       sizeof(float)));
    float * sc_kk_dot     = (float *)oom(calloc((size_t)CHUNK_SIZE * CHUNK_SIZE, sizeof(float)));
    float * sc_lhs        = (float *)oom(calloc((size_t)CHUNK_SIZE * CHUNK_SIZE, sizeof(float)));
    float * sc_attn       = (float *)oom(calloc((size_t)CHUNK_SIZE * CHUNK_SIZE, sizeof(float)));
    float * sc_v_eff      = (float *)oom(calloc((size_t)CHUNK_SIZE * v_hd,       sizeof(float)));
    float * sc_kbeta_gexp = (float *)oom(calloc((size_t)CHUNK_SIZE * k_hd,       sizeof(float)));
    float * sc_k_cumdecay = (float *)oom(calloc((size_t)CHUNK_SIZE * k_hd,       sizeof(float)));
    float * sc_attn_kq    = (float *)oom(calloc((size_t)CHUNK_SIZE * CHUNK_SIZE, sizeof(float)));
    float * sc_q_g_exp    = (float *)oom(calloc((size_t)CHUNK_SIZE * k_hd,       sizeof(float)));
    float * sc_attn_inter = (float *)oom(calloc((size_t)CHUNK_SIZE * v_hd,       sizeof(float)));
    float * sc_v_prime    = (float *)oom(calloc((size_t)CHUNK_SIZE * v_hd,       sizeof(float)));
    float * sc_v_new      = (float *)oom(calloc((size_t)CHUNK_SIZE * v_hd,       sizeof(float)));
    float * sc_v_attn     = (float *)oom(calloc((size_t)CHUNK_SIZE * v_hd,       sizeof(float)));
    float * sc_key_gdiff  = (float *)oom(calloc((size_t)CHUNK_SIZE * k_hd,       sizeof(float)));
    float * sc_kgd_vnew   = (float *)oom(calloc((size_t)k_hd * v_hd,             sizeof(float)));
    // Run the kernel with the actual N (dynamic chunk size — exercises
    // the same path that slm_forward_ssm_batch takes for partial tail
    // chunks).
    chunked_ssm_step_f32(N, k_hd, v_hd,
                         q_pad, k_pad, v_pad, g_log_p, beta_p,
                         state, out,
                         sc_gcs, sc_gexp, sc_decay_mask,
                         sc_k_beta, sc_v_beta, sc_kk_dot, sc_lhs,
                         sc_attn, sc_v_eff, sc_kbeta_gexp, sc_k_cumdecay,
                         sc_attn_kq, sc_q_g_exp, sc_attn_inter,
                         sc_v_prime, sc_v_new, sc_v_attn,
                         sc_key_gdiff, sc_kgd_vnew);
    // Compare per-token outputs.
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int   max_t = -1, max_e = -1;
    for (int t = 0; t < N; t++) {
        for (int e = 0; e < v_hd; e++) {
            float ae = auto_out[t * v_hd + e];
            float ce = out     [t * v_hd + e];
            float d  = fabsf(ae - ce);
            float r  = (fabsf(ae) > 1e-6f) ? d / fabsf(ae) : d;
            if (d > max_abs_diff) {
                max_abs_diff = d;
                max_rel_diff = r;
                max_t = t;
                max_e = e;
            }
        }
    }
    // Compare final states.
    float max_state_diff = 0.0f;
    int   max_state_d = -1, max_state_e = -1;
    for (int d = 0; d < k_hd; d++) {
        for (int e = 0; e < v_hd; e++) {
            float ae = auto_state[d * v_hd + e];
            float ce = state     [d * v_hd + e];
            float diff = fabsf(ae - ce);
            if (diff > max_state_diff) {
                max_state_diff = diff;
                max_state_d = d;
                max_state_e = e;
            }
        }
    }
    for (int t = 0; t < N; t++) {
        printf("chunked-test t=%d: auto[0..3] = %.7g %.7g %.7g %.7g\n",
               t,
               auto_out[t * v_hd + 0], auto_out[t * v_hd + 1],
               auto_out[t * v_hd + 2], auto_out[t * v_hd + 3]);
        printf("chunked-test t=%d: chunk[0..3] = %.7g %.7g %.7g %.7g\n",
               t,
               out[t * v_hd + 0], out[t * v_hd + 1],
               out[t * v_hd + 2], out[t * v_hd + 3]);
    }
    printf("chunked-test: max |Δ_out| = %.4g at t=%d e=%d (rel %.4g)\n",
           max_abs_diff, max_t, max_e, max_rel_diff);
    printf("chunked-test: max |Δ_state| = %.4g at d=%d e=%d\n",
           max_state_diff, max_state_d, max_state_e);
    // Free scratch.
    free(sc_gcs); free(sc_gexp); free(sc_decay_mask);
    free(sc_k_beta); free(sc_v_beta); free(sc_kk_dot); free(sc_lhs);
    free(sc_attn); free(sc_v_eff); free(sc_kbeta_gexp);
    free(sc_k_cumdecay); free(sc_attn_kq); free(sc_q_g_exp);
    free(sc_attn_inter); free(sc_v_prime); free(sc_v_new);
    free(sc_v_attn); free(sc_key_gdiff); free(sc_kgd_vnew);
    // Pass if rel <= 1e-5 (a few ULPs at fp32 precision).
    return (max_rel_diff <= 1e-5f) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// --self-test: build a tiny synthetic Qwen3.5 model with deterministic
// weights and run one forward step. Validates the data flow (config →
// weights → KV cache → forward step → logits) without needing the
// GGUF file present. CLI-only — library callers don't need it.
//
// Lives here next to the forward pass it exercises; `#include`-d into
// slm.c as part of the single-TU build, and invoked from slm.c's CLI
// dispatch via `--self-test`.
__attribute__((unused))
static int32_t qwen_self_test(void) {
    // Tiny config: 2 layers, hidden=64, heads=4, head_dim=16, ffn=128,
    // vocab=256. Just enough to exercise every kernel.
    struct slm_config cfg = {
        .n_layers = 2, .n_heads = 4, .n_kv_heads = 2,
        .head_dim = 16, .hidden_dim = 64,
        .ffn_dim = 128, .vocab_size = 256,
        .max_position = 64,
        .rope_theta = 10000.0f, .norm_eps = 1e-5f,
        .bos_id = 0, .eos_id = 1,
    };
    struct arena * a = arena_new(8 * 1024 * 1024);
    // Deterministic rng for synth weights.
    uint32_t s = 1u;
    #define RNDF() ({ s = s * 1103515245u + 12345u; \
                      ((float)((s >> 16) & 0x7fff) / 16383.5f - 1.0f) * 0.05f; })
    // Allocate synthetic fp32 weights (no Q4_K to keep --self-test simple).
    int32_t hidden = cfg.hidden_dim;
    int32_t hd     = cfg.head_dim;
    int32_t nkvh   = cfg.n_kv_heads;
    int32_t kvh    = nkvh * hd;
    int32_t ffn    = cfg.ffn_dim;
    // token_embd: (hidden, vocab)
    float * tok_embd = (float *)arena_alloc(
        a, (size_t)hidden * cfg.vocab_size * sizeof(float));
    for (int32_t i = 0; i < hidden * cfg.vocab_size; i++) tok_embd[i] = RNDF();
    // output_norm: (hidden,)
    float * out_norm = ARENA_F32(hidden);
    for (int32_t i = 0; i < hidden; i++) out_norm[i] = 1.0f + RNDF();
    // Per-layer weights:
    struct slm_layer_w * layers = (struct slm_layer_w *)oom(
        calloc(cfg.n_layers, sizeof(struct slm_layer_w)));
    #define ALLOC_F32(dst, n0, n1) do {                                 \
        size_t _bytes = (size_t)(n0) * (n1) * sizeof(float);           \
        float * _p = (float *)arena_alloc(a, _bytes);                   \
        for (size_t _i = 0; _i < (size_t)(n0)*(n1); _i++) _p[_i] = RNDF(); \
        (dst).data    = _p;                                             \
        (dst).type    = GGUF_TT_F32;                                    \
        (dst).n_dims  = ((n1)==1 ? 1 : 2);                              \
        (dst).shape[0] = (n0);                                          \
        (dst).shape[1] = (n1);                                          \
        (dst).shape[2] = 1;                                             \
        (dst).shape[3] = 1;                                             \
    } while (0)

    for (int32_t L = 0; L < cfg.n_layers; L++) {
        ALLOC_F32(layers[L].attn_norm,  hidden, 1);
        ALLOC_F32(layers[L].attn_q,     hidden, hidden);
        ALLOC_F32(layers[L].attn_k,     hidden, kvh);
        ALLOC_F32(layers[L].attn_v,     hidden, kvh);
        ALLOC_F32(layers[L].attn_out,   hidden, hidden);
        ALLOC_F32(layers[L].ffn_norm,   hidden, 1);
        ALLOC_F32(layers[L].ffn_gate,   hidden, ffn);
        ALLOC_F32(layers[L].ffn_up,     hidden, ffn);
        ALLOC_F32(layers[L].ffn_down,   ffn,    hidden);
    }
    struct slm_weights W = {0};
    W.tok_embd.data = tok_embd; W.tok_embd.type = GGUF_TT_F32;
    W.tok_embd.n_dims = 2; W.tok_embd.shape[0] = hidden;
    W.tok_embd.shape[1] = cfg.vocab_size;
    W.tok_embd.shape[2] = 1; W.tok_embd.shape[3] = 1;
    W.output_norm.data = out_norm; W.output_norm.type = GGUF_TT_F32;
    W.output_norm.n_dims = 1; W.output_norm.shape[0] = hidden;
    W.output_norm.shape[1] = 1;
    W.output_norm.shape[2] = 1; W.output_norm.shape[3] = 1;
    W.output = W.tok_embd;  // tied
    W.layers = layers;
    // Synthetic model + ctx (no GGUF, no mmap — config and weights
    // are entirely in-memory for this test).
    struct slm_model m = {0};
    m.cfg = cfg;
    m.W   = W;
    struct slm_ctx c = {0};
    c.model = &m;
    c.arena = a;
    kv_init(&c.kv, cfg.n_layers, cfg.n_kv_heads,
            cfg.head_dim, cfg.max_position);
    // Run forward on 3 dummy tokens.
    int32_t prompt[] = {7, 13, 42};
    printf("self-test: forward pass for %zu tokens...\n",
           sizeof(prompt) / sizeof(prompt[0]));
    int32_t ok = 1;
    for (size_t i = 0; i < sizeof(prompt) / sizeof(prompt[0]); i++) {
        struct tensor * logits = slm_forward_step(&c, prompt[i], (int32_t)i);
        int64_t n = tensor_nelements(logits);
        if (n != cfg.vocab_size) {
            fprintf(stderr, "self-test: logits size mismatch: %lld != %d\n",
                    (long long)n, cfg.vocab_size);
            ok = 0;
        }
        // Check no NaN/Inf.
        int32_t bad = 0;
        for (int64_t k = 0; k < n; k++) {
            float v = logits->data[k];
            if (!(v == v) || v > 1e30f || v < -1e30f) { bad++; }
        }
        if (bad > 0) {
            fprintf(stderr, "self-test: %d non-finite logits at pos %zu\n",
                    (int)bad, i);
            ok = 0;
        } else {
            int32_t am = sample_argmax(logits);
            printf("  pos=%zu tok=%d argmax=%d logit=%.4f\n",
                   i, prompt[i], (int)am, logits->data[am]);
        }
    }

    kv_free(&c.kv);
    free(layers);
    arena_free(a);
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Standalone test harness. Enabled with -DQWEN_TESTS at compile time
// (see `qwen-test` target in llm/Makefile). When qwen.c is included
// from slm.c instead, QWEN_TESTS is undefined and this block emits
// no code, so slm.c's own main() drives the CLI.
//
// The standalone build needs sampler.c too — model.c's lifecycle
// references slm_ctrl_defaults() (declared in slm.h, defined in
// sampler.c). Pull it in here under the QWEN_TESTS guard so the
// non-standalone build still gets exactly one copy via slm.c.
// ---------------------------------------------------------------------------
#ifdef QWEN_TESTS
#include "sampler.c"
int main(int argc, char ** argv) {
    int rc = 0;
    bool want_chunked = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--chunked-test") == 0) { want_chunked = true; }
    }
    if (want_chunked) {
        rc = chunked_self_test();
        fprintf(stderr, "chunked-test: %s\n", rc == 0 ? "PASS" : "FAIL");
    } else {
        rc = qwen_self_test();
        fprintf(stderr, "qwen-self-test: %s\n", rc == 0 ? "PASS" : "FAIL");
    }
    return rc;
}
#endif  // QWEN_TESTS

#endif  // QWEN_C
