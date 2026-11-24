"""Plan B stage 1: expert-supervised event detector (docs/analiz_plani_v3.md §4.1).

Same encoder and frame head as NeSy-v2, trained with the frame loss only (masked BCE for ACC / DEC /
UC + deceleration-type CE, the v2 quality mask). No outcome loss, no window concepts: the outcome
logit computed by the inherited forward pass is never trained and never used.
Input has a 5th channel, FHR minus the rule-based baseline curve (bpm / 25), because FIGO event
definitions are baseline-relative.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.nesy_v2 import NeSyV2


def frame_loss(out, aux):
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
    return torch.stack(terms).mean() if terms else out["f_logit"].sum() * 0


class DetNet(NeSyV2):
    def loss(self, out, y, aux, pos_weight):
        if not aux:      # no expert targets in this batch split: nothing to learn
            return out["f_logit"].sum() * 0
        return frame_loss(out, aux)

    def param_groups(self, cfg):
        return self.parameters()
