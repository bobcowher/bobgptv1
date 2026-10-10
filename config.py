
GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "ff_hidden": 2048,  # SwiGLU: ~8/3 x emb_dim, a multiple of 64
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
        }

GPT_CONFIG_406M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "ff_hidden": 2752,  # SwiGLU: ~8/3 x emb_dim, a multiple of 64
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": False
        }

GPT_CONFIG_838M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1280,
        "ff_hidden": 3392,  # SwiGLU: ~8/3 x emb_dim, a multiple of 64
        "n_heads": 20,
        "n_layers": 36,
        "drop_rate": 0.1,
        "qkv_bias": False
        }



def config_from_state_dict(state):
    """The config a checkpoint was trained with, read from its weight shapes.

    Context length is pos_emb's rows; width and depth come from the weights.
    n_heads isn't stored anywhere, but every config here uses 64-dim heads.
    ff_hidden is the SwiGLU hidden width, the gate projection's rows.
    """
    emb_dim = state["tok_emb.weight"].shape[1]
    return {
        "vocab_size": state["tok_emb.weight"].shape[0],
        "context_length": state["pos_emb.weight"].shape[0],
        "emb_dim": emb_dim,
        "n_heads": emb_dim // 64,
        "ff_hidden": state["trf_blocks.0.ff.gate_proj.weight"].shape[0],
        "n_layers": len({k.split(".")[1] for k in state if k.startswith("trf_blocks.")}),
        "drop_rate": 0.0,
        "qkv_bias": "trf_blocks.0.att.W_query.bias" in state,
    }
