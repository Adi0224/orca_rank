#!/usr/bin/env python3
"""Merge runs/*/metrics.json into metrics_grid.csv."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def find_metrics_under(base: Path) -> tuple[dict, Path] | None:
    cand = base / "metrics.json"
    if cand.is_file():
        return json.loads(cand.read_text()), cand
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Flatten metrics.json from run directories.")
    p.add_argument("--runs-root", type=Path, default=Path("runs"), help="Parent of individual run dirs")
    p.add_argument("--out", type=Path, default=Path("metrics_grid.csv"))
    args = p.parse_args()

    rows: list[dict[str, object]] = []
    if not args.runs_root.is_dir():
        print("Missing runs-root:", args.runs_root)
        return

    for d in sorted(args.runs_root.iterdir()):
        if not d.is_dir():
            continue
        parsed = find_metrics_under(d)
        if parsed is None:
            continue
        m, cand_path = parsed
        cfg = m.get("config", {})
        rows.append({
            "path": str(cand_path),
            "method": cfg.get("method"),
            "lora_r": cfg.get("lora_r"),
            "seed": cfg.get("seed"),
            "val_em": m.get("val_exact_match_mean"),
            "stage_a_wall_s": m.get("wall_seconds_stage_a"),
            "stage_b_wall_s": m.get("wall_seconds_stage_b"),
        })

    if not rows:
        print("No metrics.json found under subdirs of", args.runs_root)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Wrote", args.out.as_posix(), "rows=", len(rows))


if __name__ == "__main__":
    main()
