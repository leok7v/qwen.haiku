// sampler.c -- llama.cpp-shaped sampler chain.
//
// `#include`-d from llm.c AFTER qwen.c (depends on `struct rng` +
// `rng_uniform` + `sample_argmax`, all defined in qwen.c). Holds
// the `llm_sampler_defaults()` factory plus the sample_with chain:
//
//     repetition penalty -> temperature softmax (top-K) ->
//     top-P (nucleus) -> min-P -> roulette-wheel pick.
//
// Order matches im.ai's `Sampler.swift` (see comment block at
// im.ai/src/model/Sampler.swift:19-32 for the rationale). One
// subtle difference: im.ai normalizes the softmax over the FULL
// vocabulary before top-K filtering; we normalize over the top-K
// survivors (which represent ~99% of probability mass for typical
// Qwen logits, so the top-P cutoff shifts by <1% in practice).
//
// Greedy fast path: when sampler->temperature <= 0, skip the entire
// chain and return argmax over the (penalty-adjusted) logits.

struct llm_sampler llm_sampler_defaults(void) {
    struct llm_sampler s;
    s.temperature        = 0.7f;
    s.top_k              = 40;
    s.top_p              = 0.9f;
    s.min_p              = 0.05f;
    s.repetition_penalty = 1.25f;
    s.repetition_window  = 64;
    s.tools              = true;   // agent dispatch on by default
    s.think              = false;  // reasoning off (gen prompt skips
                                   // the <think> block)
    s.debug              = true;   // surface tool_call /
                                   // tool_response chunks to UI
    return s;
}

// Apply repetition penalty in-place to `logits`: any token id that
// appears in the recent-history window is scaled by /penalty
// (positive logits become less likely) or *penalty (negative logits
// become MORE negative). Standard llama.cpp convention.
static void apply_rep_penalty(struct tensor * logits,
                              const int32_t * history, int32_t hist_n,
                              float penalty, int32_t window) {
    if (penalty == 1.0f || hist_n == 0) { return; }
    int32_t start = (window > 0 && window < hist_n) ? hist_n - window : 0;
    int64_t vlen  = tensor_nelements(logits);
    for (int32_t i = start; i < hist_n; i++) {
        int32_t t = history[i];
        if (t >= 0 && (int64_t)t < vlen) {
            float lv = logits->data[t];
            logits->data[t] = (lv > 0.0f) ? (lv / penalty) : (lv * penalty);
        }
    }
}

// Top-k filter into parallel arrays (idx, val) of length filled.
// Linear scan; k is capped to LLM_SAMPLE_TOPK_MAX so the working set
// fits in a stack buffer.
#define LLM_SAMPLE_TOPK_MAX 256

static int32_t topk_collect(const struct tensor * logits, int32_t k,
                            int32_t * idx, float * val) {
    int64_t n      = tensor_nelements(logits);
    if (k <= 0 || k > LLM_SAMPLE_TOPK_MAX) { k = LLM_SAMPLE_TOPK_MAX; }
    int32_t filled = 0;
    for (int64_t i = 0; i < n; i++) {
        float lv = logits->data[i];
        if (filled < k) {
            idx[filled] = (int32_t)i;
            val[filled] = lv;
            filled++;
        } else {
            int32_t worst = 0;
            for (int32_t j = 1; j < k; j++) {
                if (val[j] < val[worst]) { worst = j; }
            }
            if (lv > val[worst]) {
                idx[worst] = (int32_t)i;
                val[worst] = lv;
            }
        }
    }
    return filled;
}

// Softmax over `filled` candidates with temperature; writes the
// normalized probability into `val` (replacing logits).
static void topk_softmax(float * val, int32_t filled, float temperature) {
    float m = val[0];
    for (int32_t j = 1; j < filled; j++) {
        if (val[j] > m) { m = val[j]; }
    }
    float sum = 0.0f;
    for (int32_t j = 0; j < filled; j++) {
        val[j] = expf((val[j] - m) / temperature);
        sum   += val[j];
    }
    if (sum > 0.0f) {
        for (int32_t j = 0; j < filled; j++) { val[j] /= sum; }
    }
}

// Sort (idx, val) pairs by val descending using insertion sort
// (filled <= 256 in practice; cheaper than qsort overhead).
static void topk_sort_desc(int32_t * idx, float * val, int32_t filled) {
    for (int32_t i = 1; i < filled; i++) {
        float   v = val[i];
        int32_t k = idx[i];
        int32_t j = i - 1;
        while (j >= 0 && val[j] < v) {
            val[j + 1] = val[j];
            idx[j + 1] = idx[j];
            j--;
        }
        val[j + 1] = v;
        idx[j + 1] = k;
    }
}

static int32_t sample_with(struct tensor * logits,
                           const struct llm_sampler * sp,
                           struct rng * rng,
                           const int32_t * history, int32_t hist_n) {
    apply_rep_penalty(logits, history, hist_n,
                      sp->repetition_penalty, sp->repetition_window);
    if (sp->temperature <= 0.0f) {
        return sample_argmax(logits);
    }
    int32_t idx[LLM_SAMPLE_TOPK_MAX];
    float   val[LLM_SAMPLE_TOPK_MAX];
    int32_t k = sp->top_k > 0 ? sp->top_k : LLM_SAMPLE_TOPK_MAX;
    if (k > LLM_SAMPLE_TOPK_MAX) { k = LLM_SAMPLE_TOPK_MAX; }
    int32_t filled = topk_collect(logits, k, idx, val);
    topk_softmax(val, filled, sp->temperature);
    topk_sort_desc(idx, val, filled);
    // Top-p (nucleus): keep the smallest prefix whose cumulative
    // probability >= top_p. Effective only when 0 < top_p < 1.
    int32_t cutoff = filled;
    if (sp->top_p > 0.0f && sp->top_p < 1.0f) {
        float acc = 0.0f;
        for (int32_t j = 0; j < filled; j++) {
            acc += val[j];
            if (acc >= sp->top_p) { cutoff = j + 1; j = filled; }
        }
    }
    // Min-p: drop tokens whose probability < min_p * top_prob.
    if (sp->min_p > 0.0f) {
        float thresh = sp->min_p * val[0];
        int32_t j2   = 1;
        while (j2 < cutoff && val[j2] >= thresh) { j2++; }
        cutoff = j2;
    }
    // Re-normalize and roulette-wheel sample from the surviving set.
    float sum = 0.0f;
    for (int32_t j = 0; j < cutoff; j++) { sum += val[j]; }
    float u = rng_uniform(rng) * sum;
    float c = 0.0f;
    int32_t picked = 0;
    for (int32_t j = 0; j < cutoff; j++) {
        c += val[j];
        if (u <= c) { picked = j; j = cutoff; }
    }
    return idx[picked];
}
