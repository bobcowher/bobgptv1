# Serving bobgpt to OpenWebUI

Goal: chat with a trained checkpoint in OpenWebUI by writing a from-scratch,
OpenAI-compatible HTTP server around `LanguageModel.generate()`. Speed is not a
goal; understanding the contract is.

## Why not llama.cpp (for now)

llama.cpp is not a general runtime. It has a hand-written inference graph per
supported architecture, and its converter only maps HF-format tensors onto one
of those. bobgpt *is* GPT-2 (same math, different names), so the GGUF path
works — but a novel architecture would not. See "The GGUF path" below if we
come back to it.

OpenWebUI doesn't need llama.cpp. It needs anything that speaks the OpenAI API.

## The contract: OpenAI API (de facto, no standards body)

References:
- Chat Completions: https://platform.openai.com/docs/api-reference/chat/create
- Streaming chunks: https://platform.openai.com/docs/api-reference/chat-streaming
- List Models: https://platform.openai.com/docs/api-reference/models/list
- OpenAPI spec (exact field types): https://github.com/openai/openai-openapi
- Server-Sent Events (streaming transport): https://html.spec.whatwg.org/multipage/server-sent-events.html
- OpenWebUI connections: https://docs.openwebui.com

### Minimum surface OpenWebUI needs

Checked against `openapi.json` in github.com/openai/openai-openapi (spec
version 2.3.0, 2026-09-27). "Required" below means required by that schema;
send every required field -- stricter clients (the official `openai` SDK)
validate these shapes. Nullable-but-required fields are sent as `null`.

#### `GET /v1/models` -- populates the model dropdown

```json
{"object": "list",
 "data": [{"id": "bobgpt", "object": "model", "created": 1790000000, "owned_by": "you"}]}
```

- Required: `id`, `object`, `created`, `owned_by`. `created` is a Unix
  timestamp (int, seconds). Any fixed value works, e.g. the checkpoint's mtime.
- `id` is what clients send back as `model` in completion requests.

#### `POST /v1/chat/completions` -- request


```json
{"model": "bobgpt",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "stream": true, "temperature": 0.8, "max_tokens": 100}
```

- Required: `model`, `messages`. Everything else is optional.
- `messages[].content` is **either a string or an array of content parts**
  (`[{"type": "text", "text": "..."}, ...]`). Handle both: join the `text`
  parts, ignore image/audio parts.
- Honor: `stream` (default false), `temperature` (0-2, default 1), `top_p`
  (0-1, default 1), and a token limit. `max_tokens` is marked deprecated in
  favor of `max_completion_tokens`; clients send either, so accept both.
- `stop`: a string or an array of up to 4 strings. The returned text must
  **not** include the stop sequence.
- `stream_options: {"include_usage": true}` -- see streaming below.
- OpenWebUI sends extra fields. Ignore unknown fields rather than rejecting
  the request (Pydantic's default).

#### Non-streaming response

```json
{"id": "chatcmpl-<unique>",
 "object": "chat.completion",
 "created": 1790000000,
 "model": "bobgpt",
 "choices": [{"index": 0,
              "message": {"role": "assistant", "content": "generated text", "refusal": null},
              "logprobs": null,
              "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59}}
```

- Required: `id`, `object`, `created`, `model`, `choices`; each choice needs
  `index`, `message`, `logprobs`, `finish_reason`; `message` needs `role`,
  `content`, **and `refusal`** (null). `usage` is optional but send it.
- `created` here is *now* (time of the request), unlike the model's.
- `finish_reason`: `"stop"` if generation hit the end marker / EOS / a stop
  string, `"length"` if it hit `max_tokens`.
- `usage` counts tokens (tiktoken), not characters.

#### Streaming response (OpenWebUI's default)

`Content-Type: text/event-stream`. Each event is `data: <json>\n\n`. Every
chunk shares the same `id` and `created`:

```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1790000000,"model":"bobgpt","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1790000000,"model":"bobgpt","choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1790000000,"model":"bobgpt","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}

data: [DONE]
```

- Required per chunk: `id`, `object`, `created`, `model`, `choices`; each
  choice needs `index`, `delta`, `finish_reason` (null until the last chunk).
  Every field inside `delta` is optional, and `logprobs` is optional here.
- Convention (what OpenAI sends): first chunk carries `delta.role`; middle
  chunks carry only `delta.content`; the last choice chunk has an empty `delta`
  and the `finish_reason`; then the literal `data: [DONE]`.
- If the request has `stream_options.include_usage: true`, send one more chunk
  before `[DONE]` with `"choices": []` and the `usage` object (other chunks
  then carry `"usage": null`).
- Token-by-token decoding can split a multi-byte UTF-8 character across two
  tokens. Buffer until the bytes decode cleanly before emitting a chunk.

#### Errors

Non-2xx with `{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": null}}`.
All four of `message`, `type`, `param`, `code` are required (`param`/`code`
may be null). Note FastAPI's own validation errors (422 `{"detail": ...}`)
don't match this shape unless you add an exception handler.

## The part no standard covers: chat template

The API hands over a list of `{role, content}` messages; the model eats a flat
string. How to flatten them is our call (Llama, ChatML, etc. each invented
their own). **Ours is already fixed by training**: `chat_template.py` is the
format `python_qa` is rendered in (via `scripts/build_mix.py`), so the server
must use the same one -- import `ROLE_HEADERS` / `END_MARKER` from it rather
than retyping the strings. A training conversation looks like:

```
### System
You are a patient Python tutor. ...

### Question
What does `None` mean in Python?

### Answer
`None` is Python's single value for "no value" ...

### End<|endoftext|>
```

So to prompt: render the conversation's turns the same way, then end with
`### Answer\n` and let the model write the answer. (`render()` itself
appends `### End`, which is right for training data but not for a prompt.)
Stop when the model emits `### End`, `<|endoftext|>` (token 50256 --
`generate()` takes `eos_id`), or starts a new `### Question`.

Open questions:
- Context is only 256 tokens. What gets truncated when the conversation is
  longer -- oldest turns first? Keep the system turn?
- `max_tokens` + prompt must fit in 256.
- Only multi-turn *format* is ours to define: the training Q&A is all
  single-turn (system, question, answer). Multi-turn chats will work
  mechanically but the model has never seen one.
- The Q&A is a tiny share of pretraining (run 21: ~0.06% of tokens), so
  expect it to drift into web/docs text often until there's more Q&A data.

## Build order

1. `/v1/models` + non-streaming `/v1/chat/completions`. Test with `curl`.
2. Streaming. `generate()` already produces one token per loop iteration —
   turn it into a generator that yields tokens instead of returning at the end.
   This is the step that actually changes the generation code.
3. Point OpenWebUI at `http://<host>:<port>/v1`.
4. Rough edges: `max_tokens`, stop strings, context trimming.

## Streaming: implementation spec

The wire format is above ("Streaming response"). This section is how to get
there. Acceptance test: the `== POST /v1/chat/completions (streaming)` block
of `scripts/check_api.sh` (8 checks, incl. `include_usage`), plus watching
text appear word by word in Open WebUI.

### Shape of the solution

Three layers, each testable on its own:

1. **Token generator** (model side). A generator version of
   `LanguageModel.generate()`: same loop, but `yield` each new token id
   instead of collecting them. It stops on its own at EOS or
   `max_new_tokens`. It knows nothing about text, stop strings, or HTTP.
   Test it in a REPL: `list(gen(...))` should equal what `generate()` returns
   for the same input at `temperature=0`.
2. **Text stream** (decode + stop logic). Consumes token ids, yields text
   pieces that are safe to show, and finally reports *why* it ended
   (`"stop"` or `"length"`). All the hard parts live here (traps below).
   Pure Python: you can unit-test it with a fake list of token ids, no model.
3. **SSE framing** (HTTP side). Wraps layer 2 in the chunk JSON, prefixes
   `data: `, ends each event with `\n\n`, appends `data: [DONE]\n\n`. Return
   it with FastAPI's `StreamingResponse(..., media_type="text/event-stream")`.

The payoff of layering: make the **non-streaming path consume the same text
stream** and join the pieces. Then stop handling, trimming and
`finish_reason` exist once, and both paths can't drift apart. Today
`get_chat_completion` does its trimming after the fact on the whole string;
that approach can't work when text has already been sent.

### Traps (each one is why layer 2 exists)

- **You can't un-send text.** Non-streaming finds `### End` and slices it
  off. Streaming may already have sent `###` before `End` is generated. Fix:
  hold back any tail of the pending text that could still become a stop
  string. Emit only what is definitely not the start of one. Example, stop
  string `"### End"`, pending text `"use a tuple.\n\n##"`: emit
  `"use a tuple."`, hold `"\n\n##"`. Next token `"#"`, still a possible
  prefix, keep holding. Next token `" End"`, full match: drop the held text,
  finish with `"stop"`. If the next token had been `"#"`+`" Note"`, the hold
  is released and emitted.
  Stop strings: `END_MARKER`, `"\n### Question"`, plus the request's `stop`
  (string or list). Hint: for each stop string, check the suffixes of
  pending text against its prefixes; the longest match is what you hold.
- **Whitespace trimming.** Non-streaming `.strip()`s at the end. Streaming
  equivalent: drop leading whitespace until the first non-space character,
  and hold trailing whitespace (don't emit `"\n\n"` until something
  non-space follows). Otherwise every answer ends in a blank line, because
  the training data has `"\n\n### End"`.
- **Split UTF-8.** One token is bytes, not always a whole character
  (emoji, accented letters, `—`). `tokenizer.decode([tok])` on half a
  character gives `"�"` (�). Fix: accumulate bytes with
  `tokenizer.decode_single_token_bytes(tok)` and only turn them into text
  when `bytes.decode("utf-8")` succeeds. A `try/except UnicodeDecodeError`
  around it is fine.
- **finish_reason.** `"stop"` if a stop string or EOS ended it, `"length"`
  if the generator ran out at `max_tokens`. Layer 2 knows which; layer 1
  should make it knowable (EOS vs. running out).
- **Same `id` and `created` on every chunk.** Generate them once per
  request, before the generator starts. Also: `id` must be unique per
  request (`"chatcmpl-" + uuid4().hex`); the hard-coded `"5"` will make
  Open WebUI merge or overwrite messages eventually.
- **Usage counts.** Prompt tokens are known up front; completion tokens =
  number of tokens the model generated (including held/discarded ones, since
  they cost compute), not characters emitted.
- **Return type.** With `stream: true` you return a `StreamingResponse`,
  not the Pydantic model. That's fine with `response_model=` on the route:
  FastAPI passes any returned `Response` through without validating it, so
  the model still checks the non-streaming branch. The route itself must not
  contain `yield` (it would become a generator and its JSON `return` would be
  lost). Branch early: `if req.stream: return StreamingResponse(gen())`, with
  the yields in a separate (or nested) function. FastAPI doc:
  https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
- **Sync generator is fine.** A plain `def` generator gets iterated in
  FastAPI's threadpool, so a slow model doesn't freeze the server. Don't
  make it `async def` unless the body actually awaits something; a blocking
  `model(...)` call inside `async def` blocks the event loop.
- **`torch.no_grad()` across `yield`.** A `with torch.no_grad():` around the
  loop inside a generator holds while the generator is suspended. That's what
  you want, but it's per-thread state, and the threadpool may resume the
  generator on a different thread. Safest: put the `no_grad` (or
  `@torch.inference_mode()`) around each forward call, not around the yield.
- **Client hangs up.** Open WebUI's stop button closes the connection.
  Starlette stops iterating your generator, so generation stops. Nothing to
  build, but worth watching happen once (a print in the loop).

### Build order

1. Layer 1. Check it matches `generate()` at temperature 0.
2. Layer 2 with a fake token list. Cases to cover: `### End` split across
   tokens, a near-miss (`"### Ending"` if you want to be thorough), an emoji
   split across tokens, trailing `"\n\n"` before `### End`, and running out
   at max_tokens with text still held (emit it, `"length"`).
3. Layer 3, `curl -N` to watch it arrive (`-N` turns off curl's buffering):
   `curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"bobgpt","messages":[{"role":"user","content":"What is a dict?"}],"stream":true}'`
4. Switch non-streaming onto layer 2. `check_api.sh` should still pass
   its non-streaming block.
5. `include_usage`, then Open WebUI.

### Out of date above

The model is now 1024-token context (`GPT_CONFIG_124M`), not 256. The
"Open questions" bullets about 256 still apply at 1024, just less often.

## The GGUF path (parked)

If we ever want llama.cpp speed/quantization:

1. Export `model.pth` → HF `GPT2LMHeadModel` folder. `load_gpt_weights` in
   `languagemodel.py` is the map in reverse. Traps:
   - Linear weights need `.T` (HF GPT-2 uses `Conv1D`).
   - Q/K/V are fused into `c_attn` — concatenate ours.
   - `qkv_bias: False` → write zero biases for `c_attn`.
   - Our `out_head` is untied from `tok_emb` → `tie_word_embeddings=False`,
     and verify `lm_head.weight` survives GGUF conversion (check the
     `GPT2Model` class in llama.cpp's `convert_hf_to_gguf.py`).
   - `n_positions` = 256.
   - Tokenizer: tiktoken `gpt2` == HF `GPT2TokenizerFast.from_pretrained("gpt2")`.
2. Verify: HF logits vs our logits on the same input, within ~1e-4. A missed
   transpose fails silently.
3. `python convert_hf_to_gguf.py <hf_dir> --outtype f16`
4. `llama-server -m bobgpt.gguf -c 256` → OpenAI API on :8080.

References: `transformers/models/gpt2/modeling_gpt2.py` (the de facto HF
format), `configuration_gpt2.py`, llama.cpp `gguf-py/gguf/tensor_mapping.py`,
GGUF spec at https://github.com/ggml-org/ggml/blob/master/docs/gguf.md.

Architecture note: if bobgpt moves off GPT-2 (RMSNorm, RoPE, SwiGLU), the
export target becomes HF Llama format — llama.cpp's best-supported path.
