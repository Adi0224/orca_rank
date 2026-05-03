"""Extract #### answer and GSM8K-style exact match."""

from __future__ import annotations

import re


def normalize_number(s: str) -> str:
    s = s.strip().replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return ""
    v = float(m.group(0))
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return str(v)


def gold_from_answer_field(answer_full: str) -> str:
    tail = answer_full.split("####")[-1].strip()
    return normalize_number(tail)


def extract_from_generation(text: str) -> str:
    if "####" in text:
        return gold_from_answer_field(text)
    return normalize_number(text.split("\n")[-1])


def gsm8k_exact_batch(generated: list[str], gold_answer_fields: list[str]) -> tuple[float, int, int]:
    ok = 0
    total = len(generated)
    for gen, ga in zip(generated, gold_answer_fields, strict=False):
        g = extract_from_generation(gen)
        gv = gold_from_answer_field(ga)
        ok += int(g == gv != "")
    em = ok / total if total else 0.0
    return float(em), ok, total
