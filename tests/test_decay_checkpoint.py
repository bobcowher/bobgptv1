import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from languagemodel import LanguageModel

CFG = {"vocab_size": 50257, "context_length": 16, "emb_dim": 64, "ff_hidden": 128,
       "n_heads": 2, "n_layers": 1, "drop_rate": 0.0, "qkv_bias": False}


def test_wsd_saves_weights_at_decay_start(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # TensorBoard writes runs/ here
    batches = [(torch.randint(0, 100, (2, 16)), torch.randint(0, 100, (2, 16)))] * 10
    model = LanguageModel(CFG, train_loader=batches, val_loader=batches[:1],
                          checkpoint_path=str(tmp_path / "model.pth"))
    model.train(num_epochs=1, eval_freq=100, eval_iter=1, start_context="Hi",
                warmup_steps=1, max_steps=10, lr_schedule="wsd",
                decay_checkpoint_path=str(tmp_path / "decay_start.pth"))
    decay_start = torch.load(tmp_path / "decay_start.pth")
    final = torch.load(tmp_path / "model.pth")
    # Saved after step 8 of 10 (decay is the last 20%), so the decay steps moved the weights since.
    assert not torch.equal(decay_start["tok_emb.weight"], final["tok_emb.weight"])
