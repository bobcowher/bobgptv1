from collections import defaultdict

import torch


@torch.compile
def orthogonalize(G, steps=5):
    """Approximately replace each matrix's singular values with 1 (Newton-Schulz, in bf16).

    G is (rows, cols) or a stack (n, rows, cols) of same-shape matrices, done in
    one batched call. The quintic's coefficients trade exactness for speed:
    singular values land roughly in [0.7, 1.2], which works as well as the exact
    U V^T in practice. From Keller Jordan's Muon (github.com/KellerJordan/Muon).
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)  # spectral <= Frobenius <= 1
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if tall else X


class MuonAdamW(torch.optim.Optimizer):
    """Muon for the 2D weights inside the blocks, AdamW for everything else.

    One optimizer with two kinds of param group (marked by "muon"), so the
    LambdaLR schedule scales both learning rates by the same factor.
    Muon: momentum, then orthogonalize the update, so every direction in the
    matrix moves by about the same amount. Embeddings, the tied head, norms and
    biases aren't hidden-layer matrices and stay on AdamW, which is torch's own
    (fused on the GPU) fed this group's scheduled lr each step.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            if group["muon"]:
                group.setdefault("momentum", 0.95)
            else:
                group.setdefault("betas", (0.9, 0.999))
                group.setdefault("eps", 1e-8)
        super().__init__(param_groups, defaults={})
        self.adamw = {}
        for i, group in enumerate(self.param_groups):
            if not group["muon"]:
                self.adamw[i] = torch.optim.AdamW(
                    group["params"], lr=group["lr"], betas=group["betas"], eps=group["eps"],
                    weight_decay=group["weight_decay"], fused=group["params"][0].is_cuda)

    @torch.no_grad()
    def step(self, closure=None):
        for i, group in enumerate(self.param_groups):
            if not group["muon"]:
                self.adamw[i].param_groups[0]["lr"] = group["lr"]
                self.adamw[i].step()
                continue

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            for p in params:
                if not self.state[p]:
                    self.state[p]["momentum"] = torch.zeros_like(p)
            grads = [p.grad for p in params]
            bufs = [self.state[p]["momentum"] for p in params]
            torch._foreach_lerp_(bufs, grads, 1 - group["momentum"])
            updates = torch._foreach_lerp(grads, bufs, group["momentum"])  # Nesterov

            # Same-shape matrices (all the W_query's, all the up projections, ...)
            # are orthogonalized together: 3 batched calls a step instead of 84.
            by_shape = defaultdict(list)
            for p, u in zip(params, updates):
                by_shape[p.shape].append((p, u))
            lr, wd = group["lr"], group["weight_decay"]
            for (rows, cols), pairs in by_shape.items():
                ps = [p for p, _ in pairs]
                ortho = orthogonalize(torch.stack([u for _, u in pairs])).to(ps[0].dtype)
                # Wide and tall matrices get the same per-element step size.
                scale = max(1, rows / cols) ** 0.5
                if wd:
                    torch._foreach_mul_(ps, 1 - lr * wd)
                torch._foreach_add_(ps, list(ortho.unbind(0)), alpha=-lr * scale)


def make_optimizer(model, name, lr, weight_decay=0.1, muon_lr=0.01):
    """AdamW on everything, or Muon on the blocks' matrices plus AdamW on the rest."""
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                 fused=next(model.parameters()).is_cuda)
    if name != "muon":
        raise ValueError(f"unknown optimizer {name!r}")
    # The attention gate maps to one number per head, closer to an output head
    # than a hidden matrix, so it stays on AdamW.
    muon = [p for name, p in model.trf_blocks.named_parameters()
            if p.ndim == 2 and not name.endswith("att.gate.weight")]
    muon_ids = {id(p) for p in muon}
    adam = [p for p in model.parameters() if id(p) not in muon_ids]
    # AdamW group first: the training loop logs param_groups[0]'s LR, which keeps
    # the logged schedule comparable with the all-AdamW runs.
    # Muon gets no weight decay: Keller Jordan's default at this scale. Decoupled
    # decay at lr 0.01 would be 17x stronger per step than AdamW's at 6e-4.
    return MuonAdamW([
        {"params": adam, "muon": False, "lr": lr, "weight_decay": weight_decay},
        {"params": muon, "muon": True, "lr": muon_lr, "weight_decay": 0.0},
    ])
