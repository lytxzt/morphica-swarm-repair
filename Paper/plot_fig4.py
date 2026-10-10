#!/usr/bin/env python3
"""Generate the UGV transfer figure from frozen benchmark artifacts.

The source of truth is the frozen 1,200-case CSV under Logs/ugv.  The visual
panel uses the matched mixed-terrain replay case only as an illustration; all
quantitative panels use the complete final matrix.

Usage:
    python3 Paper/plot_fig4.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "Logs/ugv/manuscript_matrix/metrics/ugv_summary.csv"
DEFAULT_TRAJECTORY = (
    ROOT / "Logs/ugv/render_demo/"
    "case_mixed_random_MORPHICA_N100_D50_r50_cr120_w1.2_seed7.json"
)
DEFAULT_TERRAIN = ROOT / "Logs/ugv/render_demo/terrain/mixed_N100_s10013"
DEFAULT_OUTPUT = ROOT / "Paper/figs/fig4"

METHODS = ["MORPHICA", "CEN", "DEMD", "CR-MGC", "ML-DAGL", "HERO"]
METHOD_COLORS = {
    "MORPHICA": "#007C6C",
    "CEN": "#666666",
    "DEMD": "#AA4499",
    "CR-MGC": "#CC6677",
    "ML-DAGL": "#3366AA",
    "HERO": "#BBBBBB",
}
TERRAIN_COLORS = {
    "ridge": "#8C564B",
    "rolling": "#D55E00",
    "roughland": "#009E73",
    "mixed": "#0072B2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.pad_inches": 0.02,
    })


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def paired_seed_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap the mean while retaining each selected seed's paired D rows."""
    rng = np.random.default_rng(seed)
    groups = []
    for terrain, terrain_frame in frame.groupby("terrain", sort=True):
        pivot = terrain_frame.pivot(index="seed_index", columns="D", values=value_column)
        if pivot.isna().any().any() or set(pivot.columns) != {30, 50}:
            raise ValueError(f"Incomplete paired-D rows for terrain {terrain!r}")
        groups.append(pivot.sort_index().to_numpy(dtype=float))

    total = sum(values.size for values in groups)
    samples = np.zeros(n_bootstrap, dtype=float)
    for values in groups:
        n_seed = values.shape[0]
        draw = rng.integers(0, n_seed, size=(n_bootstrap, n_seed))
        samples += values[draw].sum(axis=(1, 2))
    samples /= total
    return float(frame[value_column].mean()), *map(float, np.quantile(samples, [0.025, 0.975]))


def method_statistics(matrix: pd.DataFrame, n_bootstrap: int) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for index, method in enumerate(METHODS):
        subset = matrix.loc[matrix["algorithm"] == method].copy()
        success_mean, success_low, success_high = paired_seed_bootstrap(
            subset, "success", n_bootstrap, 20260723 + index)
        subset["rmst_time"] = subset["T_conn"].fillna(subset["horizon"])
        rmst_mean, rmst_low, rmst_high = paired_seed_bootstrap(
            subset, "rmst_time", n_bootstrap, 20260823 + index)
        successful = subset.loc[subset["success"] == 1, "T_conn"]
        stats[method] = {
            "n": int(len(subset)),
            "successes": int(subset["success"].sum()),
            "success_rate": success_mean,
            "success_ci_low": success_low,
            "success_ci_high": success_high,
            "rmst": rmst_mean,
            "rmst_ci_low": rmst_low,
            "rmst_ci_high": rmst_high,
            "successful_t_conn_mean": float(successful.mean()),
        }
    return stats


def plot_replay(ax: plt.Axes, trajectory_path: Path, terrain_dir: Path) -> dict[str, int]:
    data = json.loads(trajectory_path.read_text())
    frames = data["frames"]
    height = np.load(terrain_dir / "heightmap.npy")
    obstacle = np.load(terrain_dir / "obstacle_mask.npy")
    terrain_meta = json.loads((terrain_dir / "metadata.json").read_text())
    bounds = np.asarray(terrain_meta["bounds"], dtype=float)
    extent = (bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1])

    image = ax.imshow(height, extent=extent, origin="lower", cmap="YlGn",
                      interpolation="bilinear", alpha=0.95)
    ax.contour(np.linspace(extent[0], extent[1], height.shape[1]),
               np.linspace(extent[2], extent[3], height.shape[0]), height,
               levels=7, colors="#425240", linewidths=0.25, alpha=0.45)
    if obstacle.any():
        masked = np.ma.masked_where(~obstacle, obstacle)
        ax.imshow(masked, extent=extent, origin="lower", cmap="Greys",
                  interpolation="nearest", alpha=0.65)

    initial, final = frames[0], frames[-1]
    remain = np.asarray(initial["remain_list"], dtype=int)
    initial_positions = np.asarray(initial["positions"], dtype=float)
    final_positions = np.asarray(final["positions"], dtype=float)
    initial_components = np.asarray(initial["component_id"], dtype=int)
    palette = plt.get_cmap("tab10")
    for index in remain:
        color = palette(initial_components[index] % 10)
        ax.plot([initial_positions[index, 0], final_positions[index, 0]],
                [initial_positions[index, 1], final_positions[index, 1]],
                color=color, linewidth=0.35, alpha=0.22, zorder=2)
    ax.scatter(initial_positions[remain, 0], initial_positions[remain, 1],
               c=[palette(initial_components[index] % 10) for index in remain],
               s=7, linewidths=0.25, edgecolors="white", zorder=3)
    ax.scatter(final_positions[remain, 0], final_positions[remain, 1],
               c="#007C6C", marker="^", s=11, linewidths=0.25,
               edgecolors="white", zorder=4)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    k_initial = len(set(initial_components[index] for index in remain))
    final_components = np.asarray(final["component_id"], dtype=int)
    k_final = len(set(final_components[index] for index in remain))
    ax.text(0.02, 0.98, f"mixed terrain, $K$: {k_initial} $\\rightarrow$ {k_final}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6,
            bbox={"boxstyle": "square,pad=0.18", "facecolor": "white", "alpha": 0.82,
                  "edgecolor": "none"})
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#666666",
               markeredgecolor="white", markersize=4, label="initial survivors"),
        Line2D([], [], marker="^", linestyle="", markerfacecolor="#007C6C",
               markeredgecolor="white", markersize=4, label=f"connected, step {final['step']}"),
    ], loc="lower right", frameon=False, handletextpad=0.25, borderpad=0.15)
    return {"initial_k": int(k_initial), "final_k": int(k_final),
            "final_step": int(final["step"]), "frames": len(frames)}


def plot_success(ax: plt.Axes, stats: dict[str, dict[str, float]]) -> None:
    values = np.array([stats[method]["success_rate"] * 100 for method in METHODS])
    lower = np.array([stats[method]["success_ci_low"] * 100 for method in METHODS])
    upper = np.array([stats[method]["success_ci_high"] * 100 for method in METHODS])
    x = np.arange(len(METHODS))
    ax.bar(x, values, color=[METHOD_COLORS[method] for method in METHODS], width=0.72,
           linewidth=0)
    ax.errorbar(x, values, yerr=np.vstack([values - lower, upper - values]),
                fmt="none", color="#222222", linewidth=0.7, capsize=1.8, capthick=0.7)
    ax.set_xticks(x, METHODS, rotation=35, ha="right")
    ax.set_ylim(0, 108)
    ax.set_ylabel("success rate (%)")
    ax.set_title("Matched terrain recovery", loc="left", fontweight="bold")
    for xi, value in zip(x, values):
        ax.text(xi, value + 2.1, f"{value:.0f}", ha="center", va="bottom", fontsize=5.8)
    despine(ax)


def plot_rmst(ax: plt.Axes, stats: dict[str, dict[str, float]]) -> None:
    values = np.array([stats[method]["rmst"] for method in METHODS])
    lower = np.array([stats[method]["rmst_ci_low"] for method in METHODS])
    upper = np.array([stats[method]["rmst_ci_high"] for method in METHODS])
    x = np.arange(len(METHODS))
    ax.bar(x, values, color=[METHOD_COLORS[method] for method in METHODS], width=0.72,
           linewidth=0)
    ax.errorbar(x, values, yerr=np.vstack([values - lower, upper - values]),
                fmt="none", color="#222222", linewidth=0.7, capsize=1.8, capthick=0.7)
    ax.set_xticks(x, METHODS, rotation=35, ha="right")
    ax.set_ylim(0, 540)
    ax.set_ylabel("RMST to connectivity (steps)")
    ax.set_title("Failures right-censored at 600", loc="left", fontweight="bold")
    for xi, value in zip(x, values):
        ax.text(xi, value + 13, f"{value:.0f}", ha="center", va="bottom", fontsize=5.8)
    despine(ax)


def plot_paired_cen(ax: plt.Axes, matrix: pd.DataFrame) -> dict[str, int]:
    pivot = matrix.pivot(index=["terrain", "D", "seed_index"], columns="algorithm",
                         values=["success", "T_conn"])
    both = pivot.loc[(pivot[("success", "MORPHICA")] == 1) &
                     (pivot[("success", "CEN")] == 1)].copy()
    x = both[("T_conn", "MORPHICA")].to_numpy(dtype=float)
    y = both[("T_conn", "CEN")].to_numpy(dtype=float)
    terrain = both.index.get_level_values("terrain")
    for name in TERRAIN_COLORS:
        mask = terrain == name
        ax.scatter(x[mask], y[mask], s=12, color=TERRAIN_COLORS[name], alpha=0.72,
                   linewidths=0, label=name)
    lower, upper = 1.0, max(float(x.max()), float(y.max())) * 1.16
    ax.plot([lower, upper], [lower, upper], color="#333333", linewidth=0.8,
            linestyle="--", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("MORPHICA $T_{\\mathrm{conn}}$ (steps)")
    ax.set_ylabel("CEN $T_{\\mathrm{conn}}$ (steps)")
    ax.set_title("Paired successes: MORPHICA faster", loc="left", fontweight="bold")
    ax.legend(title="terrain", frameon=False, loc="upper left", title_fontsize=5.8,
              handletextpad=0.25, borderpad=0.15)
    despine(ax)
    return {
        "joint_successes": int(len(both)),
        "morphica_faster": int((x < y).sum()),
        "morphica_only_success": int(((pivot[("success", "MORPHICA")] == 1) &
                                       (pivot[("success", "CEN")] == 0)).sum()),
        "cen_only_success": int(((pivot[("success", "MORPHICA")] == 0) &
                                  (pivot[("success", "CEN")] == 1)).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--terrain-dir", type=Path, default=DEFAULT_TERRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        parser.error("--n-bootstrap must be at least 100")

    matrix = pd.read_csv(args.matrix)
    expected_rows = len(METHODS) * 4 * 2 * 25
    if len(matrix) != expected_rows:
        raise ValueError(f"Expected {expected_rows} final-matrix rows, found {len(matrix)}")
    if set(matrix["algorithm"]) != set(METHODS):
        raise ValueError("Final matrix does not contain the expected six algorithms")
    if matrix["invalid_initial_graph"].astype(bool).any():
        raise ValueError("Frozen matrix contains invalid initial scenarios")

    set_style()
    stats = method_statistics(matrix, args.n_bootstrap)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 6.0), dpi=300, layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.06, 1.0), height_ratios=(1.0, 1.0))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    for label, axis in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        axis.text(-0.13, 1.05, label, transform=axis.transAxes, fontsize=9,
                  fontweight="bold", va="bottom", ha="left")

    replay = plot_replay(ax_a, args.trajectory, args.terrain_dir)
    plot_success(ax_b, stats)
    plot_rmst(ax_c, stats)
    paired = plot_paired_cen(ax_d, matrix)

    stem = output_dir / "Fig4_ugv_transfer"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "panel": "Fig. 4",
        "matrix": str(args.matrix.relative_to(ROOT)),
        "matrix_sha256": sha256(args.matrix),
        "trajectory": str(args.trajectory.relative_to(ROOT)),
        "trajectory_sha256": sha256(args.trajectory),
        "terrain_dir": str(args.terrain_dir.relative_to(ROOT)),
        "bootstrap": {
            "n_resamples": args.n_bootstrap,
            "unit": "seed, retaining paired D=30/D=50 rows within terrain",
            "confidence_interval": "percentile 95%",
        },
        "method_statistics": stats,
        "replay": replay,
        "paired_morphica_cen": paired,
    }
    stats_path = stem.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {stem.with_suffix('.pdf')}")
    print(f"Wrote {stem.with_suffix('.png')}")
    print(f"Wrote {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
