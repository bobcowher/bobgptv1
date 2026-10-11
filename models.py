import torch
import torch.nn as nn

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        torch.nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        torch.nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.01)

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg)
                  for _ in range(cfg["n_layers"])]
                )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias=False
                )

        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device)
                )
        x = tok_embeds + pos_embeds
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
        self.norm1 = nn.RMSNorm(cfg["emb_dim"])
        self.norm2 = nn.RMSNorm(cfg['emb_dim'])
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
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length),
                           diagonal=1)
                )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Permute to (b, num_heads, num_tokens, head_dim) for multi-head attention
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # Fused causal attention: same math as scores -> causal mask -> softmax(/sqrt(d))
        # -> dropout -> @ values, but never materializes the (tokens x tokens) score
        # matrix, so memory grows linearly with context instead of quadratically.
        # self.mask is no longer read; it stays registered so older checkpoints load.
        context_vec = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
                )

        # Transpose back to (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.permute(0, 2, 1, 3)

        context_vec = context_vec.contiguous().view(
                b, num_tokens, self.d_out
                )
        context_vec = self.out_proj(context_vec)
        return context_vec
