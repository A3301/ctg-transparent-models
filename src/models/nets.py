"""Neural network architectures. All take x (B, C, T) and return a dict with 'logit' (B,)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- CNN / TCN backbone
class ResBlock(nn.Module):
    def __init__(self, cin, cout, k, stride, drop):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2)
        self.b2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(drop)
        self.skip = nn.Conv1d(cin, cout, 1, stride) if (cin != cout or stride != 1) else nn.Identity()

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.drop(h)
        h = self.b2(self.c2(h))
        return F.relu(h + self.skip(x))


class CNNEncoder(nn.Module):
    """1-D residual CNN; returns pooled embedding and the temporal feature map."""

    def __init__(self, in_ch=4, width=32, n_stages=4, kernel=7, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, kernel, 2, kernel // 2), nn.BatchNorm1d(width), nn.ReLU())
        layers, c = [], width
        for s in range(n_stages):
            co = width * (2 ** min(s, 2))
            layers.append(ResBlock(c, co, kernel, 2, drop))
            c = co
        self.blocks = nn.Sequential(*layers)
        self.out_dim = 2 * c

    def forward(self, x):
        h = self.blocks(self.stem(x))
        emb = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return emb, h


class CNN(nn.Module):
    def __init__(self, in_ch=4, width=32, n_stages=4, kernel=7, drop=0.2, **_):
        super().__init__()
        self.enc = CNNEncoder(in_ch, width, n_stages, kernel, drop)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.out_dim, 1))

    def forward(self, x, **_):
        emb, _ = self.enc(x)
        return {"logit": self.head(emb).squeeze(-1)}


# ----------------------------------------------------------------------------- GRU
class GRUNet(nn.Module):
    def __init__(self, in_ch=4, hidden=64, layers=1, pool=4, drop=0.2, **_):
        super().__init__()
        self.pool = pool
        self.proj = nn.Conv1d(in_ch, hidden, 5, 1, 2)
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True,
                          dropout=drop if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(4 * hidden, 1))

    def forward(self, x, **_):
        x = F.avg_pool1d(x, self.pool) if self.pool > 1 else x
        h = F.relu(self.proj(x)).transpose(1, 2)
        o, _ = self.gru(h)
        emb = torch.cat([o.mean(1), o.amax(1)], 1)
        return {"logit": self.head(emb).squeeze(-1)}


# ----------------------------------------------------------------------------- Patch Transformer
class PatchTransformer(nn.Module):
    def __init__(self, in_ch=4, d=64, layers=2, heads=4, patch=30, drop=0.2, max_len=3600, **_):
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(in_ch * patch, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, max_len // patch + 2, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, x, **_):
        B, C, T = x.shape
        n = T // self.patch
        p = x[:, :, :n * self.patch].reshape(B, C, n, self.patch).permute(0, 2, 1, 3).reshape(B, n, C * self.patch)
        h = torch.cat([self.cls.expand(B, -1, -1), self.embed(p)], 1) + self.pos[:, :n + 1]
        h = self.tf(h)
        return {"logit": self.head(h[:, 0]).squeeze(-1)}


# ----------------------------------------------------------------------------- MCNN (Petrozziello 2019)
class _Branch12(nn.Module):
    """5 x (conv k, f filters, BN, ReLU, maxpool 2) -> dropout -> FC(out)  (as described by Mendis 2024)."""

    def __init__(self, length, k=9, f=40, out=10, drop=0.3):
        super().__init__()
        layers, c, L = [], 1, length
        for _ in range(5):
            layers += [nn.Conv1d(c, f, k, 1, k // 2), nn.BatchNorm1d(f), nn.ReLU(), nn.MaxPool1d(2)]
            c, L = f, L // 2
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(f * L, out))

    def forward(self, x):
        return self.fc(self.conv(x))


class MCNN(nn.Module):
    """Inputs at 0.25 Hz: x[:,0]=FHR, x[:,2]=UC (zeros if poor quality); q = 10-dim quality vector."""

    def __init__(self, length=900, k=9, f=40, n_q=10, drop=0.3, **_):
        super().__init__()
        self.fhr = _Branch12(length, k, f, n_q, drop)
        self.uc = _Branch12(length, k, f, n_q, drop)
        self.q = nn.Linear(n_q, n_q)
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(2 * n_q, n_q), nn.ReLU(), nn.Dropout(drop),
                                nn.Linear(n_q, n_q), nn.ReLU(), nn.Dropout(drop), nn.Linear(n_q, 1))

    def forward(self, x, q=None, **_):
        a = self.fhr(x[:, 0:1]) * self.q(q)
        b = self.uc(x[:, 2:3])
        return {"logit": self.fc(torch.cat([a, b], 1)).squeeze(-1)}


# ----------------------------------------------------------------------------- FHR-LINet (Mendis 2024)
class FHRLINet(nn.Module):
    """BN -> 3 parallel conv (k=5,15,25; 160 filters, ReLU) -> concat -> maxpool2 -> conv128 k7 ->
    conv128 k9 -> GAP -> FC64 -> dropout -> logit. FHR only, input-length invariant."""

    def __init__(self, drop=0.5, **_):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)
        self.b = nn.ModuleList([nn.Conv1d(1, 160, k, 1, k // 2) for k in (5, 15, 25)])
        self.c1 = nn.Conv1d(480, 128, 7, 1, 3)
        self.c2 = nn.Conv1d(128, 128, 9, 1, 4)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x, **_):
        h = self.bn(x[:, 0:1])
        h = torch.cat([F.relu(c(h)) for c in self.b], 1)
        h = F.max_pool1d(h, 2)
        h = F.relu(self.c2(F.relu(self.c1(h))))
        return {"logit": self.fc(h.mean(-1)).squeeze(-1)}


REGISTRY = {"cnn": CNN, "gru": GRUNet, "transformer": PatchTransformer, "mcnn": MCNN, "fhrlinet": FHRLINet}
