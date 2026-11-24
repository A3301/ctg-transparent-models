"""Neuro-symbolic FIGO model: neural concept detectors + exact probabilistic-logic layer.

Program (ProbLog semantics; text in FIGO_PROGRAM, used for numerical verification):
  neural annotated disjunctions  baseline in {sevbrady(<100), brady(100-110), normal, tachy}
                                 variability in {reduced, normal, increased, sinusoidal}
  neural facts                   accel, rep_any, rep_late, rep_variable, early, prolonged,
                                 very_prolonged, reduced_var_long, increased_var_long, rep_late_long
  FIGO 2015                      pathological :- baseline(sevbrady) ; reduced_var_long ;
                                    increased_var_long ; variability(sinusoidal) ; rep_late_long ;
                                    very_prolonged.
                                 normal :- baseline(normal), variability(normal), \\+rep_any,
                                    \\+prolonged, \\+pathological.
                                 suspicious :- \\+pathological, \\+normal.
  acidemia (learnable weights)   w_i :: acid :- body_i.   (noisy-OR, incl. a leak rule)
Inference: exact weighted model counting by enumerating 4*4*2^10 = 16384 worlds (log space).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.nets import CNNEncoder

BASELINE = ["sevbrady", "brady", "normal", "tachy"]
VARIAB = ["reduced", "normal", "increased", "sinusoidal"]
BINARY = ["accel", "rep_any", "rep_late", "rep_variable", "early", "prolonged", "very_prolonged",
          "reduced_var_long", "increased_var_long", "rep_late_long"]
CONCEPT_KEYS = {"baseline": "c_baseline", "variability": "c_variability", "accel": "c_accel",
                "rep_any": "c_rep_any", "rep_late": "c_rep_late", "rep_variable": "c_rep_variable",
                "early": "c_early", "prolonged": "c_prolonged", "very_prolonged": "c_very_prolonged",
                "reduced_var_long": "c_reduced_var_long", "increased_var_long": "c_increased_var_long",
                "rep_late_long": "c_rep_late_long"}
CLIN_FACTORS = ["gest_ge41", "mat_age_ge35", "nulliparous", "diabetes", "hypertension", "preeclampsia",
                "liq_praecox", "pyrexia", "induced"]   # pre-labour clinical priors (flagged experiment only)
RULES = ["leak", "pathological", "suspicious", "rep_late", "rep_variable", "prolonged",
         "tachycardia", "no_accel", "reduced_variability", "rep_any"]


# Atoms used in the FIGO class definitions are enumerated; atoms that appear in exactly one
# acidemia rule and nowhere else (accel, rep_late, rep_variable) are marginalised in closed form:
# a noisy-OR rule w::acid :- f with independent fact f contributes the factor (1 - w*P(f)).
# 'early' appears in no rule (concept only). This is exact, and 16x cheaper than full enumeration.
ENUM_BITS = ["rep_any", "prolonged", "very_prolonged", "reduced_var_long", "increased_var_long", "rep_late_long"]
INDEP_RULES = {"rep_late": ("rep_late", False), "rep_variable": ("rep_variable", False),
               "no_accel": ("accel", True)}          # rule -> (atom, negated)


def _world_table():
    worlds = list(itertools.product(range(4), range(4), *[(0, 1)] * len(ENUM_BITS)))
    W = np.array(worlds, dtype=np.int64)
    b, v = W[:, 0], W[:, 1]
    bit = {n: W[:, 2 + i] for i, n in enumerate(ENUM_BITS)}
    path = ((b == 0) | (bit["reduced_var_long"] == 1) | (bit["increased_var_long"] == 1) | (v == 3)
            | (bit["rep_late_long"] == 1) | (bit["very_prolonged"] == 1)).astype(np.float32)
    normal = ((b == 2) & (v == 1) & (bit["rep_any"] == 0) & (bit["prolonged"] == 0) & (path == 0)).astype(np.float32)
    susp = ((path == 0) & (normal == 0)).astype(np.float32)
    bodies = {"leak": np.ones(len(W), np.float32), "pathological": path, "suspicious": susp,
              "prolonged": bit["prolonged"], "tachycardia": (b == 3), "reduced_variability": (v == 0),
              "rep_any": bit["rep_any"]}
    enum_rules = [r for r in RULES if r not in INDEP_RULES]
    R = np.stack([np.asarray(bodies[r], np.float32) for r in enum_rules])
    Ob = np.eye(4, dtype=np.float32)[b].T
    Ov = np.eye(4, dtype=np.float32)[v].T
    Bits = W[:, 2:].T.astype(np.float32)
    cls = np.stack([normal, susp, path])
    return Ob, Ov, Bits, R, cls, enum_rules


class FIGOLogic(nn.Module):
    """Exact differentiable WMC over the FIGO program (verified against ProbLog)."""

    def __init__(self, learn_rules=True, init_w=0.1, init_leak=0.05):
        super().__init__()
        Ob, Ov, Bits, R, cls, enum_rules = _world_table()
        for n, a in dict(Ob=Ob, Ov=Ov, Bits=Bits, R=R, cls=cls).items():
            self.register_buffer(n, torch.tensor(a))
        self.enum_idx = [RULES.index(r) for r in enum_rules]
        self.indep = [(RULES.index(r), BINARY.index(atom), neg) for r, (atom, neg) in INDEP_RULES.items()]
        self.bit_idx = [BINARY.index(n) for n in ENUM_BITS]
        init = torch.full((len(RULES),), float(np.log(init_w / (1 - init_w))))
        init[0] = float(np.log(init_leak / (1 - init_leak)))
        self.rule_logit = nn.Parameter(init, requires_grad=learn_rules)

    def rule_weights(self):
        return torch.sigmoid(self.rule_logit)

    def forward(self, logp_b, logp_v, logit_bits, extra_lnot=None):
        """logp_b/logp_v: (B,4) log-probs; logit_bits: (B,10) over BINARY. Returns logit(P(acid)), class probs.
        extra_lnot: optional (B,) sum of log(1 - w_k * P(body_k)) for additional independent rules."""
        w = self.rule_weights().clamp(max=1 - 1e-6)
        le = logit_bits[:, self.bit_idx]
        lq, lnq = F.logsigmoid(le), F.logsigmoid(-le)
        logPw = logp_b @ self.Ob + logp_v @ self.Ov + lq @ self.Bits + lnq @ (1 - self.Bits)   # (B, nw)
        lnot = torch.log1p(-w[self.enum_idx]) @ self.R                                        # (nw,)
        l1 = torch.logsumexp(logPw + lnot, dim=1)                                             # enumerated part
        for ri, ai, neg in self.indep:
            pf = torch.sigmoid(-logit_bits[:, ai]) if neg else torch.sigmoid(logit_bits[:, ai])
            l1 = l1 + torch.log1p(-w[ri] * pf)
        if extra_lnot is not None:
            l1 = l1 + extra_lnot
        l1 = l1.clamp(max=-1e-7)                                                              # log(1 - P(acid))
        logit = torch.log(-torch.expm1(l1)) - l1
        return logit, logPw.exp() @ self.cls.T


class NeSyFIGO(nn.Module):
    def __init__(self, in_ch=4, width=32, n_stages=4, kernel=7, drop=0.2, lam=0.1,
                 use_logic=True, rules_only=False, n_clin=0, **_):
        super().__init__()
        self.n_clin = n_clin
        if n_clin:
            self.clin_logit = nn.Parameter(torch.full((n_clin,), float(np.log(0.05 / 0.95))))
        self.lam, self.use_logic, self.rules_only = lam, use_logic, rules_only
        if not rules_only:
            self.enc = CNNEncoder(in_ch, width, n_stages, kernel, drop)
            D = self.enc.out_dim
            self.drop = nn.Dropout(drop)
            self.h_b, self.h_v, self.h_bits = nn.Linear(D, 4), nn.Linear(D, 4), nn.Linear(D, len(BINARY))
        if use_logic:
            self.logic = FIGOLogic()
        else:
            self.lin = nn.Linear(4 + 4 + len(BINARY), 1)

    def concepts(self, x, cvec=None):
        if self.rules_only:  # weak labels (one-hot / uniform when undefined) instead of a network
            eps = 1e-4
            pb, pv, bits = cvec[:, :4], cvec[:, 4:8], cvec[:, 8:]
            return (torch.log(pb.clamp(min=eps)), torch.log(pv.clamp(min=eps)),
                    torch.logit(bits.clamp(eps, 1 - eps)))
        emb, _ = self.enc(x)
        emb = self.drop(emb)
        return F.log_softmax(self.h_b(emb), 1), F.log_softmax(self.h_v(emb), 1), self.h_bits(emb)

    def forward(self, x, cvec=None, clin=None, **_):
        lb, lv, lbits = self.concepts(x, cvec)
        out = {"c_b": lb.exp(), "c_v": lv.exp(), "c_bits": torch.sigmoid(lbits),
               "lb": lb, "lv": lv, "lbits": lbits}
        if self.use_logic:
            extra = None
            if self.n_clin and clin is not None:   # w_k :: acid :- risk_factor_k  (deterministic facts)
                wk = torch.sigmoid(self.clin_logit).clamp(max=1 - 1e-6)
                extra = (clin * torch.log1p(-wk)).sum(1)
            logit, cls = self.logic(lb, lv, lbits, extra)
            out.update(logit=logit, figo_probs=cls)
        else:
            out["logit"] = self.lin(torch.cat([lb.exp(), lv.exp(), torch.sigmoid(lbits)], 1)).squeeze(-1)
        return out

    def loss(self, out, y, aux, pos_weight):
        main = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pos_weight)
        if self.lam <= 0 or self.rules_only or not aux:
            return main
        terms = [F.nll_loss(out["lb"], aux["c_baseline"], ignore_index=-1),
                 F.nll_loss(out["lv"], aux["c_variability"], ignore_index=-1)]
        for i, n in enumerate(BINARY):
            t = aux[CONCEPT_KEYS[n]].float()
            m = t >= 0
            if m.any():
                terms.append(F.binary_cross_entropy_with_logits(out["lbits"][m, i], t[m]))
        terms = [t for t in terms if torch.isfinite(t)]
        return main + self.lam * torch.stack(terms).mean()


def build_nesy(name, cfg, in_ch=4):
    kw = {k: v for k, v in cfg.items() if k in ("width", "n_stages", "kernel", "drop", "lam")}
    if name == "nesy":
        return NeSyFIGO(in_ch, **kw)
    if name == "nesy_nologic":
        return NeSyFIGO(in_ch, use_logic=False, **kw)
    if name == "nesy_nocsup":
        kw["lam"] = 0.0
        return NeSyFIGO(in_ch, **kw)
    if name == "nesy_clin":
        return NeSyFIGO(in_ch, n_clin=len(CLIN_FACTORS), **kw)
    if name == "nesy_rulesonly":
        return NeSyFIGO(in_ch, rules_only=True, lam=0.0)
    raise ValueError(name)


def concept_vector(table) -> np.ndarray:
    """Weak labels -> (N, 18) [baseline one-hot(4), variability one-hot(4), 10 binaries]; undefined -> uniform/0.5."""
    def onehot(v, k):
        o = np.full((len(v), k), 1.0 / k, np.float32)
        ok = np.isfinite(v)
        o[ok] = np.eye(k, dtype=np.float32)[v[ok].astype(int)]
        return o
    b = onehot(table["c_baseline"].values.astype(float), 4)
    v = onehot(table["c_variability"].values.astype(float), 4)
    bits = np.stack([np.nan_to_num(table[CONCEPT_KEYS[n]].values.astype(float), nan=0.5) for n in BINARY], 1)
    return np.concatenate([b, v, bits.astype(np.float32)], 1)


FIGO_PROGRAM = r"""
% neural predicates are replaced by probabilistic facts with the network's probabilities
{facts}
pathological :- baseline(sevbrady).
pathological :- reduced_var_long.
pathological :- increased_var_long.
pathological :- variability(sinusoidal).
pathological :- rep_late_long.
pathological :- very_prolonged.
normal :- baseline(normal), variability(normal), \+rep_any, \+prolonged, \+pathological.
suspicious :- \+pathological, \+normal.
{rules}
query(acid). query(normal). query(suspicious). query(pathological).
"""
RULE_BODIES = {"leak": "true", "pathological": "pathological", "suspicious": "suspicious",
               "rep_late": "rep_late", "rep_variable": "rep_variable", "prolonged": "prolonged",
               "tachycardia": "baseline(tachy)", "no_accel": "\\+accel",
               "reduced_variability": "variability(reduced)", "rep_any": "rep_any"}


def problog_text(pb, pv, pbits, w) -> str:
    facts = ["; ".join(f"{p:.10f}::baseline({n})" for p, n in zip(pb, BASELINE)) + ".",
             "; ".join(f"{p:.10f}::variability({n})" for p, n in zip(pv, VARIAB)) + "."]
    facts += [f"{p:.10f}::{n}." for p, n in zip(pbits, BINARY)]
    rules = [f"{wi:.10f}::acid :- {RULE_BODIES[r]}." for wi, r in zip(w, RULES)]
    return FIGO_PROGRAM.format(facts="\n".join(facts), rules="\n".join(rules))
