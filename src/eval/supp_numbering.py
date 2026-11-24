"""Supplementary sections are cited in the manuscript by key, e.g. "Supplementary S[prereg]".

Numbers are assigned in order of first citation in the main text (from the Introduction on), so the supplement
is always cited in sequence; sections that the text never cites keep their default order at the end.
Used by src/eval/make_supplements.py (section titles) and scripts/build_submission.py (citations).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEYS = {
    "paper1": ["prereg", "figo", "eventfull", "iou", "perrepeat", "window", "dectype", "ruleweights", "intervention", "fhrma",
               "refaudit", "hyper", "controlled"],
    "paper2": ["prereg", "figo", "vardefs", "finalmodel", "robust", "calib", "netbenefit", "shift", "ablation", "subgroup", "tripod", "review"],
}
KEYS["combined"] = KEYS["paper1"] + [k for k in KEYS["paper2"] if k not in KEYS["paper1"]]   # all sections of this paper
MANUSCRIPT = {"paper1": ROOT / "manuscript/paper1_concept_fidelity.md",
              "paper2": ROOT / "manuscript/paper2_transparent_models.md",
              "combined": ROOT / "manuscript/combined/combined_paper.md"}
TOKEN = re.compile(r"S\[([a-z0-9]+)\]")


def numbers(paper: str) -> dict[str, int]:
    text = MANUSCRIPT[paper].read_text(encoding="utf-8")
    start = text.find("## 1. Introduction")
    body = text[start:] if start >= 0 else text
    keys = KEYS[paper]
    unknown = set(TOKEN.findall(body)) - set(keys)
    assert not unknown, f"unknown supplement keys in {paper}: {unknown}"

    def first(k):
        i = body.find(f"S[{k}]")
        return (i if i >= 0 else 10 ** 9, keys.index(k))

    return {k: n + 1 for n, k in enumerate(sorted(keys, key=first))}


def resolve(text: str, paper: str) -> str:
    num = numbers(paper)
    return TOKEN.sub(lambda m: f"S{num[m.group(1)]}", text)
