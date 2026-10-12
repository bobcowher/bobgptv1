import torch
import torch.nn as nn

# RMSNorm's default eps is the input dtype's machine epsilon: 1.2e-7 in fp32, 0.0078 in bf16.
# 1e-6 is what Llama, Qwen and OLMo use.
NORM_EPS = 1e-6

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_length = cfg["context_length"]  # no pos_emb to read it from
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        torch.nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg)
                  for _ in range(cfg["n_layers"])]
                )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=NORM_EPS)
        self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias=False
                )

        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg["ff_hidden"]
        self.gate_proj = nn.Linear(cfg['emb_dim'], hidden , bias=False)
        self.upscale_proj = nn.Linear(cfg['emb_dim'], hidden, bias=False)
        self.downscale_proj = nn.Linear(hidden, cfg['emb_dim'], bias=False)

    def forward(self, x):
        gate = nn.functional.silu(self.gate_proj(x))  # how much of each unit to let through
        value = self.upscale_proj(x)                       # what each unit carries
        return self.downscale_proj(gate * value)           # element-wise product, back to emb_dim


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"]
                )

        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=NORM_EPS)
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=NORM_EPS)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
        

def apply_rope(x, cos, sin):
    """Rotate each (x[i], x[i + half]) pair of the last dim by the table's angle.

    x: (b, heads, tokens, head_dim); cos/sin: (tokens, head_dim).
    For a pair (a, b): a' = a cos - b sin, b' = a sin + b cos.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert (d_out % num_heads == 0), \
                "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)
        # QK-norm: RMSNorm each head's query and key, so attention logits can't
        # grow without bound as the weights do. Only the learned scale sets them.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=NORM_EPS)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=NORM_EPS)
        # Gated attention (Qwen, 2025): each head's output is scaled by a sigmoid
        # of the token's input, so a head can switch itself off for that token
        # instead of attending somewhere harmless. Zero init: every gate starts at 0.5.
        self.gate = nn.Linear(d_in, num_heads, bias=False)
        nn.init.zeros_(self.gate.weight)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length),
                           diagonal=1)
                )

        # RoPE table: the angle each dim pair turns at each position, fast pairs
        # first, down to slow ones. Split-halves pairing (dim i with dim i + head_dim/2),
        # so both halves share an angle. Built in fp32, rebuilt rather than saved.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        positions = torch.arange(context_length).float()
        angles = positions[:, None] * inv_freq[None, :]   # (context_length, head_dim/2)
        angles = torch.cat([angles, angles], dim=-1)      # (context_length, head_dim)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Normalize before RoPE, as Qwen3 and OLMo 2 do: the rotation keeps the norm.
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        # Permute to (b, num_heads, num_tokens, head_dim) for multi-head attention
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # Rotate queries and keys by position, so q.k depends on their distance.
        # Values carry content and stay unrotated.
        cos = self.cos[:num_tokens].to(queries.dtype)
        sin = self.sin[:num_tokens].to(queries.dtype)
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Fused causal attention: same math as scores -> causal mask -> softmax(/sqrt(d))
        # -> dropout -> @ values, but never materializes the (tokens x tokens) score
        # matrix, so memory grows linearly with context instead of quadratically.
        # self.mask is no longer read here; it stays registered so older checkpoints load,
        # and config_from_state_dict reads context_length from its size.
        context_vec = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
                )

        # Transpose back to (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.permute(0, 2, 1, 3)

        # (b, num_tokens, num_heads) gates, one per head, broadcast over head_dim.
        context_vec = context_vec * torch.sigmoid(self.gate(x)).unsqueeze(-1)

        context_vec = context_vec.contiguous().view(
                b, num_tokens, self.d_out
                )
        context_vec = self.out_proj(context_vec)
        return context_vec
