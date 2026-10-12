import copy

import torch

from config import GPT_CONFIG_124M
from models import GPTModel
from optim import MuonAdamW, make_optimizer, orthogonalize

TINY = {**GPT_CONFIG_124M, "context_length": 32, "emb_dim": 128, "ff_hidden": 320,
        "n_heads": 2, "n_layers": 2, "drop_rate": 0.0}


def test_muon_gets_block_matrices_only():
    model = GPTModel(TINY)
    adam_group, muon_group = make_optimizer(model, "muon", lr=6e-4).param_groups
    # Per block: W_query, W_key, W_value, out_proj, gate, up, down.
    assert len(muon_group["params"]) == 7 * TINY["n_layers"]
    assert all(p.ndim == 2 for p in muon_group["params"])
    assert any(p is model.tok_emb.weight for p in adam_group["params"])
    n_all = sum(p.numel() for p in model.parameters())
    n_split = sum(p.numel() for g in (adam_group, muon_group) for p in g["params"])
    assert n_split == n_all


def test_orthogonalize_flattens_singular_values():
    for shape in [(128, 128), (320, 128), (128, 320)]:
        s = torch.linalg.svdvals(orthogonalize(torch.randn(shape)).float())
        # The bulk lands in about [0.7, 1.2]. A random square matrix's smallest
        # singular values start near 0 and five steps don't lift them all the way.
        assert s.max() < 1.3 and s.quantile(0.05) > 0.6, (shape, s.quantile(0.05), s.max())


def test_adam_half_matches_torch_adamw():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(10))
    b = torch.nn.Parameter(a.detach().clone())
    ours = MuonAdamW([{"params": [a], "muon": False, "lr": 1e-2, "weight_decay": 0.1}])
    ref = torch.optim.AdamW([b], lr=1e-2, weight_decay=0.1)
    for _ in range(5):
        g = torch.randn(10)
        a.grad, b.grad = g.clone(), g.clone()
        ours.step()
        ref.step()
    assert torch.allclose(a, b, atol=1e-6)


def test_muon_trains_tiny_model():
    torch.manual_seed(0)
    model = GPTModel(TINY)
    x = torch.randint(0, 1000, (8, 32))
    losses = {}
    for name in ["adamw", "muon"]:
        m = copy.deepcopy(model)
        opt = make_optimizer(m, name, lr=1e-3)
        for _ in range(30):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(m(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
            loss.backward()
            opt.step()
        losses[name] = loss.item()
    print(losses)
    assert losses["muon"] < 9.0 and losses["adamw"] < 9.0  # both well below ln(50257) = 10.8
