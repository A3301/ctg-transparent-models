"""NeSy-v2, the end-to-end model NeSy-E2E (docs/analiz_plani_v2.md §4.1).

Changes w.r.t. the weak-label NeSyFIGO of the benchmark (src/models/nesy.py), everything else unchanged:
  * frame head at 4 s resolution (first residual block output + upsampled deepest map) giving
    ACC / DEC / UC event probabilities and the E/L/V/P deceleration type;
  * expert supervision with the quality mask (§3.2, asymmetric: positives always, negatives only
    where q = 1), window concepts from expert events (§3.4), weak labels only for concepts with no
    expert reference (baseline, variability, long reduced / increased variability);
  * fixed loss weights 1 / 1 / 0.1 (not tuned);
  * separate optimizer group for the FIGO rule parameters: lr x10, no weight decay.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.nesy import BINARY, NeSyFIGO

EXPERT_BITS = ["accel", "rep_any", "rep_late", "rep_variable", "early", "prolonged", "very_prolonged",
               "rep_late_long"]
EXPERT_IDX = [BINARY.index(b) for b in EXPERT_BITS]
WEAK_BITS = ["reduced_var_long", "increased_var_long"]
WEAK_IDX = [BINARY.index(b) for b in WEAK_BITS]
W_FRAME, W_WIN_EXPERT, W_WIN_WEAK = 1.0, 1.0, 0.1
RULE_LR_MULT = 10.0
N_BINS = 900


class NeSyV2(NeSyFIGO):
    def __init__(self, in_ch=4, width=32, n_stages=4, kernel=7, drop=0.2, n_bins=N_BINS, **_):
        super().__init__(in_ch, width, n_stages, kernel, drop, lam=0.0, use_logic=True)
        self.n_bins = n_bins
        c1 = self.enc.blocks[0].c2.out_channels
        cdeep = self.enc.out_dim // 2
        self.frame = nn.Sequential(nn.Conv1d(c1 + cdeep, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                                   nn.Dropout(drop), nn.Conv1d(64, 3 + 4, 1))

    def encode(self, x):
        h0 = self.enc.stem(x)
        h1 = self.enc.blocks[0](h0)
        h = h1
        for blk in list(self.enc.blocks)[1:]:
            h = blk(h)
        emb = torch.cat([h.mean(-1), h.amax(-1)], 1)
        up = F.interpolate(h, size=h1.shape[-1], mode="linear", align_corners=False)
        fr = self.frame(torch.cat([h1, up], 1))
        fr = F.interpolate(fr, size=self.n_bins, mode="linear", align_corners=False)
        return emb, fr

    def forward(self, x, **_):
        emb, fr = self.encode(x)
        emb = self.drop(emb)
        lb = F.log_softmax(self.h_b(emb), 1)
        lv = F.log_softmax(self.h_v(emb), 1)
        lbits = self.h_bits(emb)
        logit, cls = self.logic(lb, lv, lbits)
        return {"logit": logit, "figo_probs": cls, "c_b": lb.exp(), "c_v": lv.exp(),
                "c_bits": torch.sigmoid(lbits), "lb": lb, "lv": lv, "lbits": lbits,
                "f_logit": fr[:, :3], "f_type": fr[:, 3:],
                "f_prob": torch.sigmoid(fr[:, :3]), "f_tprob": torch.softmax(fr[:, 3:], 1)}

    # ------------------------------------------------------------------ loss
    def loss(self, out, y, aux, pos_weight):
        main = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pos_weight)
        if not aux:
            return main
        # frame: aux["frame"] (B,3,T) {0,1}; aux["fmask"] (B,3,T) weights; aux["ftype"] (B,T) -1..3
        fl = F.binary_cross_entropy_with_logits(out["f_logit"], aux["frame"].float(), reduction="none")
        w = aux["fmask"].float()
        terms = []
        for k in range(3):
            s = w[:, k].sum()
            if s > 0:
                terms.append((fl[:, k] * w[:, k]).sum() / s)
        ft = aux["ftype"].long()
        if (ft >= 0).any():
            terms.append(F.cross_entropy(out["f_type"], ft.clamp(min=0), reduction="none")[ft >= 0].mean())
        l_frame = torch.stack(terms).mean() if terms else main * 0
        # window concepts with expert reference (-1 = unknown)
        we = aux["wexp"].float()
        m = we >= 0
        l_wexp = (F.binary_cross_entropy_with_logits(out["lbits"][:, EXPERT_IDX], we.clamp(min=0), reduction="none")[m].mean()
                  if m.any() else main * 0)
        # weak labels only where no expert reference exists
        wt = [F.nll_loss(out["lb"], aux["c_baseline"], ignore_index=-1),
              F.nll_loss(out["lv"], aux["c_variability"], ignore_index=-1)]
        for j, name in zip(WEAK_IDX, WEAK_BITS):
            t = aux["c_" + name].float()
            mm = t >= 0
            if mm.any():
                wt.append(F.binary_cross_entropy_with_logits(out["lbits"][mm, j], t[mm]))
        wt = [t for t in wt if torch.isfinite(t)]
        l_weak = torch.stack(wt).mean() if wt else main * 0
        return main + W_FRAME * l_frame + W_WIN_EXPERT * l_wexp + W_WIN_WEAK * l_weak

    def param_groups(self, cfg):
        rule = [p for n, p in self.named_parameters() if n.startswith("logic.")]
        rest = [p for n, p in self.named_parameters() if not n.startswith("logic.")]
        lr, wd = cfg.get("lr", 1e-3), cfg.get("wd", 1e-3)
        return [{"params": rest, "lr": lr, "weight_decay": wd},
                {"params": rule, "lr": lr * RULE_LR_MULT, "weight_decay": 0.0}]
