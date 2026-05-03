#!/usr/bin/env python3
"""
Scan Condor return tarballs (`orca_<Cluster>_<Proc>.tar.gz`), read embedded
`runs/*/metrics.json` without fully unpacking; write CSV, TAKEAWAYS.txt, PNG figures.

  python scripts/analyze_chtc_tarballs.py \\
    --input ~/Desktop/639/orca_rank/lora_runs \\
    --output-dir ./chtc_summaries
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tarfile
from pathlib import Path
from typing import Any

_ROW_NUM_RE = re.compile(r"_r(\d+)_")


def discover_tarballs(inputs: list[Path], *, recurse: bool) -> list[Path]:
    paths: list[Path] = []
    pat = re.compile(r"^orca_\d+_\d+\.tar\.gz$", re.IGNORECASE)
    for root in inputs:
        if root.is_file() and tarfile.is_tarfile(root):
            if pat.match(root.name):
                paths.append(root)
            continue
        if not root.is_dir():
            continue
        iterator = root.rglob("orca_*.tar.gz") if recurse else root.glob("orca_*.tar.gz")
        for p in sorted(iterator):
            if pat.match(p.name):
                paths.append(p)
    return sorted(set(paths), key=lambda x: x.as_posix())


def read_metrics_from_tar(tf: tarfile.TarFile) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for m in tf.getmembers():
        if not m.isfile():
            continue
        name = m.name.lstrip("./")
        if "/runs/" not in name or not name.endswith("metrics.json"):
            continue
        if name.endswith("metrics.partial.json"):
            continue
        fobj = tf.extractfile(m)
        if fobj is None:
            continue
        raw = fobj.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        out.append((name, data))
    return out


def tarball_cluster_id(path: Path) -> str | None:
    mm = re.search(r"orca_(\d+)_(\d+)", path.name)
    if not mm:
        return None
    return f"{mm.group(1)}.{mm.group(2)}"


def flatten_run_row(tarball: Path, member: str, mobj: dict[str, Any]) -> dict[str, Any]:
    cfg = mobj.get("config") or {}
    tail = str(cfg.get("output_dir") or "")
    tag = ""
    if "runs/" in tail:
        tag = tail.split("runs/", 1)[-1].strip("/")
    else:
        parts = member.split("/")
        for i, p in enumerate(parts):
            if p == "runs" and i + 1 < len(parts):
                tag = parts[i + 1]
                break
    rr = cfg.get("lora_r")
    row: dict[str, Any] = {
        "tarball": tarball.name,
        "cluster_job": tarball_cluster_id(tarball) or "",
        "metrics_path_in_tar": member,
        "job_tag": tag,
        "method": cfg.get("method"),
        "seed": cfg.get("seed"),
        "lora_r": rr,
        "hard_train_samples": cfg.get("hard_train_samples"),
        "val_samples": cfg.get("val_samples"),
        "easy_pool_samples": cfg.get("easy_pool_samples"),
        "stage_b_epochs": cfg.get("stage_b_epochs"),
        "embedder_epochs": cfg.get("embedder_epochs"),
        "max_stage_b_steps": cfg.get("max_stage_b_steps"),
        "torch_version": cfg.get("torch_version"),
        "cuda_device": cfg.get("cuda_device"),
        "wall_seconds_stage_a": mobj.get("wall_seconds_stage_a"),
        "wall_seconds_stage_b": mobj.get("wall_seconds_stage_b"),
        "epochs_ran_stage_b": mobj.get("epochs_ran_stage_b"),
        "global_steps_stage_b": mobj.get("global_steps_stage_b"),
        "train_loss_epochs_mean_tail": mobj.get("train_loss_epochs_mean_tail"),
        "val_exact_match_mean": mobj.get("val_exact_match_mean"),
        "dump_val_predictions": cfg.get("dump_val_predictions"),
    }
    rr2 = row.get("lora_r")
    if (rr2 is None or (isinstance(rr2, float) and math.isnan(rr2))) and isinstance(tag, str):
        gm = _ROW_NUM_RE.search(tag)
        if gm:
            try:
                row["lora_r"] = int(gm.group(1))
            except ValueError:
                pass
    return row


def load_all_rows(tarballs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tb in tarballs:
        try:
            with tarfile.open(tb, mode="r:*") as tf:
                for member, mobj in read_metrics_from_tar(tf):
                    rows.append(flatten_run_row(tb, member, mobj))
        except (OSError, tarfile.ReadError) as e:
            print("WARN skipping tarball:", tb, e)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt_cell(r.get(k)) for k in keys})


def _fmt_cell(v: Any) -> Any:
    if isinstance(v, (list, dict)):
        return json.dumps(v, default=str)
    return v


def build_takeaways(df: Any, pd_mod: Any) -> str:
    lines: list[str] = []
    lines.append("CHTC tarball summary (embedded metrics.json)\n")
    lines.append(f"runs parsed: {len(df)}")
    if df.empty:
        return "\n".join(lines)

    em_col = "val_exact_match_mean"
    if em_col not in df.columns:
        lines.append("(missing val_exact_match_mean)")
        return "\n".join(lines)

    sub = df[df[em_col].notna()].copy()
    if sub.empty:
        lines.append("No finite val_exact_match_mean.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Mean validation EM ± std across seeds (grouped method × LoRA rank):")
    cols = ["method", "lora_r"]
    if all(c in sub.columns for c in cols):
        grp = (
            sub.groupby(cols)[em_col].agg(["mean", "std", "count"]).reset_index().sort_values(cols)
        )
        for _, r in grp.iterrows():
            lr = int(r["lora_r"]) if pd_mod.notna(r["lora_r"]) else None
            st = f"{r['mean']:.6f}"
            if r["count"] > 1 and pd_mod.notna(r["std"]):
                st += f" ± {float(r['std']):.6f}"
            st += f"  (seeds aggregated: {int(r['count'])})"
            lines.append(f"  {r['method']}  rank={lr} → {st}")
    lines.append("")

    idx = sub[em_col].idxmax()
    best = df.loc[idx]
    lines.append("Best single run (max val_exact_match_mean):")
    lines.append(
        f"  em={best.get(em_col)}  method={best.get('method')}  rank={best.get('lora_r')}  "
        f"seed={best.get('seed')}  file={best.get('tarball')}"
    )
    lines.append("")

    if "epochs_ran_stage_b" in df.columns:
        er = df["epochs_ran_stage_b"].dropna()
        if len(er):
            lines.append(f"epochs_ran_stage_b range: [{er.min()}, {er.max()}]")
    lines.append("")
    lines.append(
        "Note: val_exact_match_mean is computed on GSM8K train-derived validation rows saved in metrics "
        "(not HF split=test unless you evaluate test separately)."
    )
    return "\n".join(lines)


def plot_figs(df: Any, out_dir: Path, pd_mod: Any) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

    em = "val_exact_match_mean"
    if df.empty or em not in df.columns:
        return paths
    dd = df[(df[em].notna()) & (df["method"].notna() if "method" in df.columns else False)].copy()
    if "method" not in df.columns or dd.empty:
        return paths

    try:
        import seaborn as sns

        if "lora_r" in dd.columns and dd["lora_r"].notna().any():
            g = (
                dd.groupby(["method", "lora_r"])[em]
                .mean()
                .reset_index()
                .rename(columns={em: "mean_em"})
            )
            ranks = sorted([x for x in g["lora_r"].unique() if pd_mod.notna(x)], key=float)
            plt.figure(figsize=(9, 5))
            sns.barplot(data=g, x="lora_r", y="mean_em", hue="method", order=ranks)
            plt.ylabel("mean val_exact_match_mean (within group)")
            plt.xlabel("LoRA rank")
            plt.legend(loc="upper right")
            plt.tight_layout()
            p = out_dir / "fig_val_em_method_by_rank.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            paths.append(p)

            pivot = dd.pivot_table(index="method", columns="lora_r", values=em, aggfunc="mean")
            plt.figure(figsize=(max(6, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.8)))
            vmax = float(pivot.max(skipna=True))
            vmin = float(pivot.min(skipna=True))
            if math.isnan(vmax):
                vmax = 0.0
            if math.isnan(vmin):
                vmin = 0.0
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".4f",
                cmap="viridis",
                vmin=min(0.0, vmin),
                vmax=max(vmax * 1.2 + 1e-9, 0.001),
            )
            plt.title("Mean val_exact_match_mean (agg. seeds)")
            plt.ylabel("method")
            plt.xlabel("LoRA rank")
            plt.tight_layout()
            p2 = out_dir / "fig_val_em_heatmap.png"
            plt.savefig(p2, dpi=150, bbox_inches="tight")
            plt.close()
            paths.append(p2)
        else:
            plt.figure(figsize=(6, 4))
            m = dd.groupby("method")[em].mean().sort_values()
            m.plot.barh(color="steelblue")
            plt.xlabel("mean val_exact_match_mean")
            plt.tight_layout()
            p3 = out_dir / "fig_val_em_by_method.png"
            plt.savefig(p3, dpi=150)
            plt.close()
            paths.append(p3)
    except Exception as e:
        print("WARN plot (seaborn):", e)

    wc = "wall_seconds_stage_b"
    if wc in df.columns:
        wt = pd_mod.to_numeric(df[wc], errors="coerce").dropna()
        if len(wt) > 1:
            plt.figure(figsize=(6, 4))
            plt.hist(wt, bins=min(12, len(wt)), color="steelblue", edgecolor="white")
            plt.xlabel("wall_seconds_stage_b")
            plt.ylabel("count runs")
            plt.tight_layout()
            p = out_dir / "fig_wall_stage_b_seconds.png"
            plt.savefig(p, dpi=150)
            plt.close()
            paths.append(p)

    try:
        if "seed" in df.columns and "method" in df.columns and len(df["seed"].unique()) > 1:
            pivot_s = dd.pivot_table(index="seed", columns="method", values=em, aggfunc="mean")
            pivot_s.plot(marker="o", figsize=(7, 4))
            plt.ylabel("mean val_exact_match_mean")
            plt.xlabel("seed")
            plt.legend(title="method")
            plt.tight_layout()
            p = out_dir / "fig_val_em_by_seed.png"
            plt.savefig(p, dpi=150)
            plt.close()
            paths.append(p)
    except Exception as e:
        print("WARN plot seed sweep:", e)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Orca CHTC tarballs + plots.")
    ap.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="Dirs with orca_*.tar.gz and/or tarball paths.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chtc_analysis_out"),
        help="Output folder for CSV, TAKEAWAYS.txt, figures/*.png",
    )
    ap.add_argument(
        "--no-recurse",
        action="store_true",
        help="Only top-level tarball names in dirs (no rglob).",
    )
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas required: pip install pandas") from None

    tbs = discover_tarballs(list(args.input), recurse=not args.no_recurse)
    if not tbs:
        print("No matching orca_*_*_*.tar.gz under:", args.input)
        raise SystemExit(1)

    rows = load_all_rows(tbs)
    if not rows:
        print("No metrics.json embedded in tarballs.")
        raise SystemExit(2)

    df = pd.DataFrame(rows)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "runs_summary.csv"
    write_csv(rows, csv_path)
    takeaway_path = out / "TAKEAWAYS.txt"
    takeaway_path.write_text(build_takeaways(df, pd) + "\n", encoding="utf-8")
    fig_dir = out / "figures"
    plotted = plot_figs(df, fig_dir, pd)

    print(takeaway_path.read_text(encoding="utf-8"))
    print(f"Wrote {csv_path} rows={len(df)} figures={len(plotted)} in {fig_dir}")


if __name__ == "__main__":
    # Allow multiprocessing loaders on macOS + MPL
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
