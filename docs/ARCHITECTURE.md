# Modernizing the architecture

Plan written 2026-10-07, for work on a branch (`arch-modern`). Robert implements;
Claude explains, reviews and points at sources.

bobgpt is still the GPT-2 design from Raschka's book: learned position embeddings,
LayerNorm, a 4x GELU MLP, biased linear layers, and an output head separate from the
token embedding. Much of what small models gained since GPT-2 came from changes to that
block and to the optimizer, not only from size and data. None of it can be added to
run 31's weights: a new architecture means a new pretrain (run 31 took 43 hours), so
each change is tested small first.

## What a current model does: Laguna XS 2.1

Read from the model file on lab (`/data/models/Laguna-XS-2.1-Q4_K_M.gguf`, with the
`gguf` package), since there was no official model card to find. A 33B-total, ~3B-active
mixture-of-experts coder from Poolside.

| | Laguna XS 2.1 | bobgpt (run 31) |
|---|---|---|
| Size | 40 layers, width 2048, vocab 100,352 | 24 layers, width 1024, vocab 50,257 |
| Norm | RMSNorm, pre-norm (eps 1e-6) | LayerNorm, pre-norm |
| Q/K | RMSNorm per head on Q and K (`attn_q_norm`, `attn_k_norm`) | none |
| Attention output | per-head sigmoid gate (`attn_gate`, width x heads) | none |
| Heads | grouped-query: 48 or 64 query heads, 8 key/value heads, head dim 128 | 16 heads, full K/V, head dim 64 |
| Positions | RoPE: base 500k on 64 of 128 dims for full layers, base 10k on all dims for sliding layers; YaRN 32x from 8k to 256k | learned table (1024 x 1024) |
| Attention span | sliding window 512, one full-attention layer every four (head counts repeat 48, 64, 64, 64) | full |
| Feed-forward | SwiGLU (gate, up, down); layer 0 dense (8192), the rest MoE: 256 experts of 512, top 8, one shared expert, bias-based load balancing (`exp_probs_b`) | GELU MLP, hidden 4096 |
| Embeddings | input and output separate | separate |

## What carries over to a 406M model on one 3090

Take:
- **RoPE** in place of the learned position table. It's what every current model uses,
  and later it allows a longer context without retraining the table.
- **RMSNorm** in place of LayerNorm. Same quality, less work.
- **SwiGLU** feed-forward with hidden ~ 8/3 x width (2730, round to a multiple of 64),
  so the parameter count stays about the same. Reliably a little lower loss.
- **No biases** in the linear layers.
- **QK-norm**: RMSNorm on each head's queries and keys before the dot product. Stops
  attention logits from blowing up, which is what lets you raise the learning rate.
  Qwen3, OLMo 2 and Gemma 3 use it.
- **Gated attention** (Laguna's `attn_gate`): a per-head sigmoid of a linear map of the
  block input, multiplied into each head's output. About 16k parameters per layer at
  our size. The Qwen team's gated-attention paper reports it removes attention sinks
  and loss spikes.
- **Tied embeddings**: one matrix for input embedding and output head. The head is 51M
  of our 406M parameters (13%); small models (SmolLM, Qwen3-0.6B, Llama 3.2 1B) tie it
  and spend the budget on layers.

Maybe:
- **Grouped-query attention**. It saves KV-cache memory at long context; at 1024 tokens
  it buys little. Cheap to add if we move to longer context.
- **Deeper, thinner** at the same parameter count (MobileLLM: depth beats width below 1B).
  Changes the speed too, so measure tokens/s alongside loss.

Skip:
- **Mixture of experts**. Training memory scales with total parameters (~16 bytes each
  with AdamW), and 24 GB is already the ceiling.
- **Sliding-window attention and YaRN**. They pay off at long context, not at 1024.

Training changes, separate from the architecture:
- **Muon** optimizer for the 2D weight matrices (AdamW stays on embeddings and norms).
  modded-nanogpt and Kimi's Moonlight report 1.3-2x better token efficiency.
- **Warmup-stable-decay** learning rate schedule in place of cosine, and a higher peak
  learning rate once QK-norm is in.

## How to test without a 43-hour run per idea

1. Branch `arch-modern` off develop. Each change is a switch in the config dict, so
   GPT-2 checkpoints (runs 20-35) still load with the switches off. `config_from_state_dict`
   needs to learn the new switches from the state dict, the way it reads `qkv_bias`.
2. Small runs: 124M, ~300M tokens of the current pretrain mix, same seed and data order.
   That's about 1.5 hours each on the 3090.
   - a. baseline (today's model)
   - b. RoPE + RMSNorm + SwiGLU + no bias + tied embeddings
   - c. b + QK-norm + gated attention
   - d. c + Muon (optimizer, not architecture: compare to c only)
3. Compare at equal tokens: val loss curve, `eval_frozen.py` (books, python_docs), tokens/s
   and peak memory. A change that wins on loss but costs 30% speed may lose at equal hours.
4. The winner becomes the next 406M pretrain, then post-training, then the ranking
   eval (docs/EVAL.md section 5) against run 34/35.

## Order to build it in

Each step is small enough to test alone (shapes, a loss that falls on a tiny batch):
RMSNorm, then SwiGLU, then RoPE (the one with real math; test that rotating q and k by
the same position leaves q.k unchanged relative to their offset), then tied embeddings,
then QK-norm, then the attention gate.

## Sources

- Laguna XS 2.1 architecture: the GGUF metadata and tensor shapes above.
- RoPE: Su et al., "RoFormer" (2021). SwiGLU: Shazeer, "GLU Variants Improve Transformer"
  (2020). RMSNorm: Zhang and Sennrich (2019).
- QK-norm: used in OLMo 2, Qwen3 and Gemma 3 reports.
- Gated attention: Qwen team, "Gated Attention for Large Language Models" (2025).
- MobileLLM: Liu et al. (2024). Muon: Keller Jordan (2024); "Muon is Scalable" (Moonshot, 2025).
- modded-nanogpt (Keller Jordan): a running record of tricks that win at GPT-2 small scale.
