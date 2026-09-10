from dataset import *
from config import GPT_CONFIG_124M 
from languagemodel import LanguageModel
# from gpt_download import download_and_load_gpt2

model_configs = {
"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
"gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
"gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
"gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"

NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

gpt = LanguageModel(gpt_config=NEW_CONFIG)
gpt.model.eval()
gpt.load_the_model("checkpoints/gpt.pth")

gpt.generate_and_print_sample(start_context="Hello. How's your day going? ")
