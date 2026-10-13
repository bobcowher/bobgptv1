import torch


@torch.compile
def orthogonalize(G, steps=5):
    """Approximately replace G's singular values with 1 (Newton-Schulz, in bf16).

    The quintic's coefficients trade exactness for speed: singular values land
    roughly in [0.7, 1.2], which works as well as the exact U V^T in practice.
    From Keller Jordan's Muon (github.com/KellerJordan/Muon).
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.mT
    X = X / (X.norm() + 1e-7)  # spectral norm <= Frobenius norm <= 1
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
    biases aren't hidden-layer matrices and stay on AdamW.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            if group["muon"]:
                group.setdefault("momentum", 0.95)
            else:
                group.setdefault("betas", (0.9, 0.999))
                group.setdefault("eps", 1e-8)
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["muon"]:
                    if not state:
                        state["momentum"] = torch.zeros_like(p)
                    buf = state["momentum"]
                    buf.lerp_(p.grad, 1 - group["momentum"])
                    update = p.grad.lerp(buf, group["momentum"])  # Nesterov
                    update = orthogonalize(update)
                    # Wide and tall matrices get the same per-element step size.
                    scale = max(1, p.size(0) / p.size(1)) ** 0.5
                    p.mul_(1 - lr * wd)
                    p.add_(update.to(p.dtype), alpha=-lr * scale)
                else:
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    state["exp_avg"].lerp_(p.grad, 1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                    m_hat = state["exp_avg"] / (1 - beta1 ** state["step"])
                    v_hat = state["exp_avg_sq"] / (1 - beta2 ** state["step"])
                    p.mul_(1 - lr * wd)
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-lr)


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
