"""Training utilities shared by all neural models.

* class-weighted BCE (pos_weight = n_neg / n_pos of the training data)
* early stopping on validation loss (max_epochs, patience)
* post-hoc calibration on the validation split:
    prior correction  logit' = logit - log(pos_weight)   (undoes the class-weight shift)
    temperature scaling  p = sigmoid(logit' / T)
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F


def _to(t, device):
    return None if t is None else torch.as_tensor(t, device=device)


def predict_logits(model, X, extra=None, device="cuda", bs=128):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = _to(X[i:i + bs], device)
            kw = {k: _to(v[i:i + bs], device) for k, v in (extra or {}).items()}
            o = model(xb, **kw)
            outs.append({k: v.float().cpu() for k, v in o.items() if torch.is_tensor(v)})
    return {k: torch.cat([o[k] for o in outs]).numpy() for k in outs[0]}


def fit_temperature(logits, y):
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / logT.exp(), t)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(logT.exp().item())
    return float(np.clip(T, 0.05, 20.0))


def train_model(model, Xtr, ytr, Xva, yva, cfg, extra_tr=None, extra_va=None, aux_tr=None, aux_va=None,
                seed=0, device="cuda", max_epochs=60, patience=10, crop=None):
    """Train with early stopping. `aux_*` = dict of auxiliary targets passed to model.loss (NeSy).
    `crop` = (length, stride) -> training samples are expanded into fixed crops (FHR-LINet)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    n_pos = max(float(ytr.sum()), 1.0)
    pos_weight = (len(ytr) - n_pos) / n_pos
    if crop is not None:
        L, S = crop
        starts = list(range(0, Xtr.shape[-1] - L + 1, S))
        Xtr = np.concatenate([Xtr[..., s:s + L] for s in starts])
        ytr = np.concatenate([ytr] * len(starts))
        if aux_tr is not None:
            aux_tr = {k: np.concatenate([v] * len(starts)) for k, v in aux_tr.items()}
    Xt, yt = _to(Xtr, device), _to(ytr.astype(np.float32), device)
    Xv, yv = _to(Xva, device), _to(yva.astype(np.float32), device)
    et = {k: _to(v, device) for k, v in (extra_tr or {}).items()}
    ev = {k: _to(v, device) for k, v in (extra_va or {}).items()}
    at = {k: _to(v, device) for k, v in (aux_tr or {}).items()}
    av = {k: _to(v, device) for k, v in (aux_va or {}).items()}
    pw = torch.tensor(pos_weight, device=device)
    params = model.param_groups(cfg) if hasattr(model, "param_groups") else model.parameters()  # NeSy-v2: rule group
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 1e-3))
    bs = cfg.get("bs", 32)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def loss_fn(out, y, aux):
        if hasattr(model, "loss"):
            return model.loss(out, y, aux, pw)
        return F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pw)

    best, best_state, bad = math.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g).to(device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            out = model(Xt[idx], **{k: v[idx] for k, v in et.items()})
            loss = loss_fn(out, yt[idx], {k: v[idx] for k, v in at.items()})
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, len(Xv), 128):
                out = model(Xv[i:i + 128], **{k: v[i:i + 128] for k, v in ev.items()})
                vl += float(loss_fn(out, yv[i:i + 128], {k: v[i:i + 128] for k, v in av.items()})) * len(out["logit"])
            vl /= len(Xv)
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.pos_weight_ = pos_weight
    model.epochs_ = ep + 1
    return model


def calibrated_probs(model, X, extra, Xva, yva, extra_va, device="cuda"):
    """Prior-correct + temperature-scale on validation; return (test_probs, val_probs, T, out_test)."""
    shift = math.log(model.pos_weight_)
    ov = predict_logits(model, Xva, extra_va, device)
    ot = predict_logits(model, X, extra, device)
    lv, lt = ov["logit"] - shift, ot["logit"] - shift
    T = fit_temperature(lv, yva)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    return sig(lt / T), sig(lv / T), T, ot
