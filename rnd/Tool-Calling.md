# Architectural Plan: Local Tool-Calling Optimization for Qwen3.5 0.8B

This document consolidates the complete architectural strategy, design criteria, formatting protocols, and production-ready C implementation for executing reliable tool-calling and web research pipelines using the ultra-low-overhead Qwen3.5 0.8B Q4 GGUF model.

---

## 1. Practical Feasibility & Validation

Practical deployment reports confirm that Qwen3.5 0.8B (operating under Q4_K_M or Q8_0 quantizations) is highly viable for local edge orchestration, such as offline home automation routing and local tool execution loops. 

### Key Characteristics
- Sub-half-second time-to-first-token latency on commodity edge hardware or dev boards.
- Zero-shot deterministic routing works reliably when the action space is restricted.
- Adding multi-shot or few-shot examples inside the system prompt frequently confuses the 0.8B model, causing processing loops or repetitive syntax generation. It functions best with strict, minimal zero-shot instructions.
- Reasoning/Thinking mode must be explicitly deactivated. When active, the model degrades into token-wasting loops or leaks thought tokens into the structural parameter definitions.

---

## 2. Tool Definition & Messaging Formats

The model natively processes tool availability when injected into the system prompt using an XML-wrapped JSON layout. It is also capable of falling back to standard OpenAI-style schema strings.

### System Prompt Tool Injection Layout
<|im_start|>system
# Tools
You may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "websearch", "description": "Search DuckDuckGo.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>

### Model Generation Target Output (OpenAI JSON Fallback Shape)
When executing tool requests, the model produces raw text enclosed in structural tags:
<tool_call>
{"name": "websearch", "arguments": {"query": "bitcoin price today"}}
</tool_call>

### Context Tool Response Ingestion
To pass execution payloads back into the history context, wrap the data inside a user message variant using a <tool_response> block:
<|im_start|>user
<tool_response>
{"status": "success", "data": "Bitcoin price is..."}
</tool_response><|im_end|>

---

## 3. Inference Parameters & Sampler Tuning

Deterministic syntax completion is mandatory for preserving the integrity of structural parameters. Deviating from these thresholds leads to malformed syntax errors:

- Temperature: Set strictly to 0.0 or a maximum of 0.1.
- Top_P: Keep at 1.0 (when temperature is 0.0). If temperature is 0.1, clamp Top_P to 0.8 or 0.9 to purge fringe malformed structural tokens.
- Repetition Penalty: Clamp between 1.0 and 1.05. Avoid thresholds > 1.1, as structural formatting requires repetitive punctuation characters (quotes, braces, XML tags). High penalties degrade syntax compliance.
- Disabling Thinking Mode: Pass the following configuration override argument to the llama.cpp backend or your internal chat formatter to disable thinking completely:
  --chat-template-kwargs '{"enable_thinking":false}'

---

## 4. Decomposed Multi-Context Pipeline Architecture

Rather than executing a complex multi-turn loop inside a single context, this architecture decomposes the processing into separate, single-purpose, sequential contexts sharing the same read-only underlying model weights concurrently.

### Functional Sequence
1. Context 1 (The Router): Zero-shot context running with tool definition enabled. Parses the user question and issues a clean, structured <tool_call> for websearch.
2. App Dispatch 1: Takes the generated query, runs the underlying DuckDuckGo scrape tool, and returns raw search hits.
3. Context 2 (The Evaluator): Spawns a passive context (tools and thinking disabled). Receives the raw query and search list, then prints out exactly a single clean URL target string.
4. App Dispatch 2: Fetches and distills the page body text content into plain text via curl and a custom whitespace/tag scraper.
5. Context 3 (The Synthesis Engine): Spawns a passive context. Receives the user question and the distilled plain text dump to formulate the definitive answer.

### Advantages
- Prevents context window inflation by dropping intermediate scraping fragments once their sub-tasks finish.
- Isolates schema complexity, minimizing structural failure rates.
- Caches initial system prompts safely, accelerating inference speed across subsequent sub-tasks.

---

## 5. Concrete C Implementation (agent_decomposed_run)

The complete execution engine implementation adheres strictly to the structural protocol: linear processing, single function exit, high density body, explicit initialization of local variables, space padding around pointer declarations, and complete avoidance of early returns.

// Specialized system prompts for the isolated zero-shot contexts
static const char * PROMPT_ROUTER = 
    "You are a routing agent. Your only job is to call the websearch tool "
    "to find information answering the user's request. Output a clean "
    "<tool_call> block.";

static const char * PROMPT_EVALUATOR = 
    "You are a link selector. Given the following user request and search "
    "results, output exactly the single best absolute URL string to fetch. "
    "Do not output markdown, reasoning, or tags. Only output the raw URL.";

static const char * PROMPT_SYNTHESIZER = 
    "You are a research synthesis engine. Answer the user's original query "
    "using only the provided distilled text content from the sourced webpage.";

// Decomposed multi-context execution loop
char * agent_decomposed_run(struct slm_model * model, 
                            const char * question, 
                            const struct slm_sampler * sp, 
                            uint64_t seed) {
    char * final_answer = NULL;
    struct slm_ctrl ctrl_router = slm_ctrl_defaults();
    struct slm_ctrl ctrl_passive = slm_ctrl_defaults();
    
    struct slm_ctx * ctx_router = NULL;
    struct slm_ctx * ctx_eval = NULL;
    struct slm_ctx * ctx_synth = NULL;
    
    struct chars out_router = {0};
    struct chars out_eval = {0};
    struct chars out_synth = {0};
    
    int32_t * ids = NULL;
    int n_ids = 0;
    
    struct agent_call calls[2];
    int n_calls = 0;
    struct tool_result search_res = {0};
    struct tool_result fetch_res = {0};
    struct tool_result distill_res = {0};
    
    struct ts_buf eval_input = {0};
    struct ts_buf synth_input = {0};
    
    // Configure context behaviors
    ctrl_router.tools = true;
    ctrl_router.think = false;
    ctrl_passive.tools = false;
    ctrl_passive.think = false;
    
    ids = (int32_t *)calloc(16384, sizeof(int32_t));
    
    if (ids != NULL && model != NULL && question != NULL) {
        // ====================================================================
        // PHASE 1: Routing & Initial Web Search
        // ====================================================================
        ctx_router = slm_ctx_create(model, PROMPT_ROUTER, &ctrl_router);
        if (ctx_router != NULL) {
            char * f_prompt = jinja_apply_delta(question, NULL, 0);
            if (f_prompt != NULL) {
                n_ids = tokenizer_encode(&ctx_router->model->tok, f_prompt, ids, 16384);
                slm_generate(ctx_router, ids, n_ids, 128, 0, sp, seed, agent_capture_cb, &out_router);
                chars_put(&out_router, "", 0);
                free(f_prompt);
                
                n_calls = agent_parse_tool_calls(out_router.data, calls, 2);
                if (n_calls > 0) {
                    agent_dispatch(&calls[0], &search_res);
                }
                agent_free_calls(calls, n_calls);
            }
        }
        
        // ====================================================================
        // PHASE 2: Isolated Link Evaluation
        // ====================================================================
        if (search_res.ok && search_res.body != NULL) {
            ctx_eval = slm_ctx_create(model, PROMPT_EVALUATOR, &ctrl_passive);
            if (ctx_eval != NULL) {
                ts_printf(&eval_input, "User Query: %s\n\nSearch Results:\n%s", question, search_res.body);
                char * e_prompt = jinja_apply_delta(eval_input.data, NULL, 0);
                if (e_prompt != NULL) {
                    n_ids = tokenizer_encode(&ctx_eval->model->tok, e_prompt, ids, 16384);
                    slm_generate(ctx_eval, ids, n_ids, 64, 0, sp, seed, agent_capture_cb, &out_eval);
                    chars_put(&out_eval, "", 0);
                    free(e_prompt);
                }
            }
        }
        
        // ====================================================================
        // PHASE 3: Page Extraction & Final Synthesis
        // ====================================================================
        if (out_eval.data != NULL && out_eval.count > 0) {
            // Clean up possible stray white spaces from the extracted URL string
            struct ts_buf clean_url = {0};
            struct ts_buf raw_url_buf = {0};
            ts_put(&raw_url_buf, out_eval.data, out_eval.count);
            tools_collapse_ws(&raw_url_buf, &clean_url);
            
            if (clean_url.data != NULL) {
                tools_fetch(clean_url.data, 15, &fetch_res);
                if (fetch_res.ok && fetch_res.body != NULL) {
                    tools_distill(fetch_res.body, strlen(fetch_res.body), &distill_res);
                }
            }
            free(raw_url_buf.data);
            free(clean_url.data);
        }
        
        if (distill_res.ok && distill_res.body != NULL) {
            ctx_synth = slm_ctx_create(model, PROMPT_SYNTHESIZER, &ctrl_passive);
            if (ctx_synth != NULL) {
                ts_printf(&synth_input, "Original Query: %s\n\nSourced Content:\n%s", question, distill_res.body);
                char * s_prompt = jinja_apply_delta(synth_input.data, NULL, 0);
                if (s_prompt != NULL) {
                    n_ids = tokenizer_encode(&ctx_synth->model->tok, s_prompt, ids, 16384);
                    slm_generate(ctx_synth, ids, n_ids, 512, 0, sp, seed, agent_capture_cb, &out_synth);
                    chars_put(&out_synth, "", 0);
                    free(s_prompt);
                    
                    final_answer = out_synth.data;
                    out_synth.data = NULL; // Take ownership
                }
            }
        }
    }
    
    // Fallback if pipeline broke prematurely
    if (final_answer == NULL) {
        final_answer = strdup("(agent: synthesis pipeline failed to converge)");
    }
    
    // Linear resource teardown
    slm_ctx_destroy(ctx_router);
    slm_ctx_destroy(ctx_eval);
    slm_ctx_destroy(ctx_synth);
    
    chars_free(&out_router);
    chars_free(&out_eval);
    chars_free(&out_synth);
    
    tools_result_free(&search_res);
    tools_result_free(&fetch_res);
    tools_result_free(&distill_res);
    
    free(eval_input.data);
    free(synth_input.data);
    free(ids);
    
    return final_answer;
}
