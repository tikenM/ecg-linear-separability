"""
Black-box baseline for the XAI-alignment experiment (fix #4).

A compact 1D-CNN trained on the raw (band-passed) beat. We then extract
attribution maps (gradient saliency and Integrated Gradients) over the raw
waveform. xai_alignment.py aggregates these to the shared frequency-band space
and compares them against the linear model's coefficient-derived band
importance.

torch is imported lazily so the rest of the package works without it.
"""
from __future__ import annotations

import numpy as np


def _torch():
    import torch
    return torch


class ECGCNN:
    """Thin wrapper around a small 1D-CNN. Kept framework-agnostic on the
    outside so callers stay clean."""

    def __init__(self, n_leads: int, seg_len: int, n_classes: int,
                 lr: float = 1e-3, epochs: int = 30, batch: int = 256,
                 device: str | None = None, seed: int = 13):
        self.n_leads = n_leads
        self.seg_len = seg_len
        self.n_classes = n_classes
        self.lr, self.epochs, self.batch, self.seed = lr, epochs, batch, seed
        self._device = device
        self.net = None
        self.classes_ = None

    # -- architecture -------------------------------------------------------- #
    def _build(self):
        torch = _torch()
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self, n_leads, n_classes):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Conv1d(n_leads, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.head = nn.Linear(64, n_classes)

            def forward(self, x):
                z = self.body(x).squeeze(-1)
                return self.head(z)

        return Net(self.n_leads, self.n_classes)

    @property
    def device(self):
        torch = _torch()
        if self._device:
            return self._device
        return "cuda" if torch.cuda.is_available() else "cpu"

    # -- fit / predict ------------------------------------------------------- #
    def fit(self, X, y, class_weight=None):
        torch = _torch()
        torch.manual_seed(self.seed)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.tensor(y_idx, dtype=torch.long)
        self.net = self._build().to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        if class_weight is None:
            counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(float)
            w = torch.tensor((counts.sum() / (len(counts) * (counts + 1e-6))),
                             dtype=torch.float32, device=self.device)
        else:
            w = torch.tensor(class_weight, dtype=torch.float32, device=self.device)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        n = len(Xt)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                xb = Xt[idx].to(self.device)
                yb = yt[idx].to(self.device)
                opt.zero_grad()
                loss = lossf(self.net(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        torch = _torch()
        self.net.eval()
        with torch.no_grad():
            xb = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)
            logits = self.net(xb)
            idx = logits.argmax(1).cpu().numpy()
        return self.classes_[idx]

    # -- attribution --------------------------------------------------------- #
    def saliency(self, X) -> np.ndarray:
        """|d logit_pred / d input|, shape (n, n_leads, seg_len)."""
        torch = _torch()
        self.net.eval()
        xb = torch.tensor(np.asarray(X), dtype=torch.float32,
                          device=self.device, requires_grad=True)
        logits = self.net(xb)
        top = logits.max(1).values.sum()
        top.backward()
        return xb.grad.abs().detach().cpu().numpy()

    def integrated_gradients(self, X, steps: int = 32, baseline=None) -> np.ndarray:
        """Integrated Gradients attribution, shape (n, n_leads, seg_len)."""
        torch = _torch()
        self.net.eval()
        X = np.asarray(X, dtype=np.float32)
        xb = torch.tensor(X, device=self.device)
        base = torch.zeros_like(xb) if baseline is None else torch.tensor(
            baseline, dtype=torch.float32, device=self.device)
        total = torch.zeros_like(xb)
        target = None
        for a in np.linspace(1.0 / steps, 1.0, steps):
            pt = (base + a * (xb - base)).clone().requires_grad_(True)
            logits = self.net(pt)
            if target is None:
                target = logits.argmax(1)
            sel = logits.gather(1, target[:, None]).sum()
            grad = torch.autograd.grad(sel, pt)[0]
            total = total + grad
        ig = (xb - base) * total / steps
        return ig.abs().detach().cpu().numpy()

    # -- cost accounting (fix #2 / #4) -------------------------------------- #
    def cost(self) -> dict:
        torch = _torch()
        n_params = sum(p.numel() for p in self.net.parameters())
        # rough MAC count for the conv stack at this seg_len
        size_kb = n_params * 4 / 1024.0
        return {"params": int(n_params), "approx_size_kb": float(size_kb)}
