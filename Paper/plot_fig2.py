#!/usr/bin/env python3
"""Draw individual Fig. 2 panels for the MORPHICA manuscript.

Use one command per panel, then compose the paper figure in LaTeX:

  python Paper/plot_fig2.py --panel a
  python Paper/plot_fig2.py --panel b

Existing sweep logs provide panels b-e. Panel a requires per-step positions,
which are not stored in the sweep text files, so this script reruns one
deterministic MORPHICA case and caches the trajectory under Paper/figs.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "Logs" / "algorithm_sweep"
DATA_ROOT = SWEEP_ROOT / "N1000"
FIG_DIR = ROOT / "Paper" / "figs"
PANEL_DIR = FIG_DIR / "fig2"
CACHE_DIR = FIG_DIR / "fig2_cache"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_STEP = 1799
CR = 120.0

ALGORITHMS = [
    {
        "label": "MORPHICA",
        "dir": "MORPHICA",
        "prefix": "MORPHICA",
        "color": "#007C6C",
        "priority": 10,
    },
    {
        "label": "ML-DAGL",
        "dir": "ML-DAGL",
        "prefix": "ML-DAGL",
        "color": "#3366AA",
        "priority": 9,
    },
    {
        "label": "DEMD",
        "dir": "DEMD",
        "prefix": "DEMD",
        "color": "#AA4499",
        "priority": 8,
    },
    {
        "label": "CEN",
        "dir": "CEN",
        "prefix": "CEN",
        "color": "#666666",
        "priority": 7,
    },
    {
        "label": "CR-MGC",
        "dir": "CR-MGC",
        "prefix": "CR-MGC",
        "color": "#CC6677",
        "priority": 6,
    },
    {
        "label": "GDR-TS",
        "dir": "GDR_TS",
        "prefix": "GDR_TS",
        "color": "#DDCC77",
        "priority": 5,
    },
    {
        "label": "Hybrid-MADDPG",
        "dir": "Hybrid_MADDPG_APF",
        "prefix": "Hybrid_MADDPG_APF",
        "color": "#44AA99",
        "priority": 4,
    },
    {
        "label": "HERO",
        "dir": "HERO",
        "prefix": "HERO",
        "color": "#BBBBBB",
        "priority": 3,
    },
    {
        "label": "SIDR",
        "dir": "SIDR",
        "prefix": "SIDR",
        "color": "#999999",
        "priority": 2,
    },
]

ALGO_BY_LABEL = {item["label"]: item for item in ALGORITHMS}

TRAJ_MAPPING = {
    "MORPHICA": ("E4_MORPHICA", "MORPHICA"),
    "ML-DAGL": ("E4_ML-DAGL", "ML-DAGL"),
    "DEMD": ("E4_DEMD", "DEMD"),
    "CEN": ("E4_CEN", "CEN"),
    "CR-MGC": ("E4_CR-MGC", "CR-MGC"),
}

PANEL_STEMS = {
    "a": "Fig2a_spatial_sequence",
    "b": "Fig2b_scale_violin",
    "c": "Fig2c_damage_sweep",
    "d": "Fig2d_subnet_dynamics",
    "e": "Fig2e_degree_footprint",
}

METRIC_PANEL_FIGSIZE = (3.6, 2.55)
METRIC_PANEL_ADJUST = {
    "left": 0.20,
    "right": 0.97,
    "bottom": 0.17,
    "top": 0.88,
}
PANEL_LEGEND_FONTSIZE = 5.3


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.0,
        }
    )


def make_metric_panel() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=METRIC_PANEL_FIGSIZE, dpi=300)
    fig.subplots_adjust(**METRIC_PANEL_ADJUST)
    return fig, ax


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def result_path(algo: dict, destroyed: int, num_agents: int = 1000) -> Path:
    return SWEEP_ROOT / f"N{num_agents}" / algo["dir"] / f"{algo['prefix']}_d{destroyed}_50cases.txt"


def read_list_section(path: Path, name: str) -> list:
    text = path.read_text()
    match = re.search(rf"{re.escape(name)}:\s*\n\s*(\[[^\]]*\])", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find section {name!r} in {path}")
    raw = match.group(1)
    if "..." in raw:
        raise ValueError(f"Section {name!r} in {path} is truncated with ellipsis")
    return ast.literal_eval(raw)


def load_connect_steps(
    destroyed: int,
    labels: list[str] | None = None,
    num_agents: int = 1000,
) -> dict[str, np.ndarray]:
    out = {}
    wanted = set(labels) if labels else None
    for algo in ALGORITHMS:
        if wanted and algo["label"] not in wanted:
            continue
        path = result_path(algo, destroyed, num_agents=num_agents)
        if not path.exists():
            continue
        out[algo["label"]] = np.asarray(read_list_section(path, "connect_step"), dtype=float)
    return out


def load_degrees(destroyed: int, labels: list[str], num_agents: int = 1000) -> dict[str, np.ndarray]:
    out = {}
    for label in labels:
        algo = ALGO_BY_LABEL[label]
        path = result_path(algo, destroyed, num_agents=num_agents)
        if not path.exists():
            continue
        out[label] = np.asarray(read_list_section(path, "degree_distribution"), dtype=float)
    return out


def max_step_for(num_agents: int) -> float:
    # Fixed recording cutoffs, independent of which methods hit the horizon.
    return float({100: 599, 200: 799, 500: 1279, 1000: 1799, 2000: 2559}[num_agents])


def ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.sort(np.asarray(values, dtype=float))
    xs = np.unique(vals)
    ys = np.array([(vals > x).mean() for x in xs], dtype=float)
    return xs, ys


def import_simulation_modules(num_agents: int, destroyed: int, ablation: str):
    """Import the repo simulation stack with Configurations parsed for N=1000."""
    old_argv = sys.argv[:]
    sys.argv = [
        old_argv[0],
        "--num-agents",
        str(num_agents),
        "--destroyed",
        str(destroyed),
        "--algorithm",
        "MORPHICA",
        "--ablation",
        ablation,
    ]
    try:
        import Configurations as C
        from Environment import Environment
        from Swarm import Swarm
    finally:
        sys.argv = old_argv
    return C, Environment, Swarm


def pairwise_adjacency(positions: np.ndarray, cr: float = CR) -> np.ndarray:
    coords = np.asarray(positions, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    adj = d2 <= cr * cr
    np.fill_diagonal(adj, False)
    return adj


def component_labels(positions: np.ndarray, cr: float = CR) -> tuple[np.ndarray, int]:
    adj = pairwise_adjacency(positions, cr)
    n = adj.shape[0]
    labels = np.full(n, -1, dtype=int)
    comp = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        labels[start] = comp
        stack = [start]
        while stack:
            node = stack.pop()
            nbrs = np.where(adj[node] & (labels < 0))[0]
            labels[nbrs] = comp
            stack.extend(nbrs.tolist())
        comp += 1
    return labels, comp


def edge_segments(positions: np.ndarray, cr: float = CR, max_edges: int = 1400) -> np.ndarray:
    xy = np.asarray(positions[:, :2], dtype=float)
    adj = pairwise_adjacency(positions, cr)
    ii, jj = np.where(np.triu(adj, k=1))
    if len(ii) > max_edges:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(ii), size=max_edges, replace=False)
        ii = ii[keep]
        jj = jj[keep]
    return np.stack([xy[ii], xy[jj]], axis=1) if len(ii) else np.empty((0, 2, 2))


def run_representative_case(
    cache_path: Path,
    num_agents: int = 1000,
    destroyed: int = 500,
    case_index: int = 0,
    ablation: str = "full",
    force: bool = False,
) -> dict:
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    C, Environment, Swarm = import_simulation_modules(num_agents, destroyed, ablation)

    env = Environment()
    swarm = Swarm(algorithm_mode=14, use_pretrained=False)
    env_positions = env.reset()
    swarm.reset()

    seed_value = list(C.config_random_seed)[case_index] + 1718
    np.random.seed(seed_value)
    random.seed(seed_value)

    _, destroy_list = env.stochastic_destroy(mode=2, num_of_destroyed=destroyed)
    swarm.destroy_happens(deepcopy(destroy_list), deepcopy(env_positions))

    remain_list = np.asarray(swarm.remain_list, dtype=int)
    destroy_positions = np.asarray(C.config_initial_swarm_positions[destroy_list], dtype=float)

    positions_by_step = []
    subnet_by_step = []
    steps = []
    max_step = int(0.8 * C.config_width)

    for step in range(max_step + 1):
        swarm.make_remain_positions()
        labels, n_components = component_labels(swarm.remain_positions, C.config_communication_range)
        positions_by_step.append(np.asarray(swarm.remain_positions, dtype=np.float32))
        subnet_by_step.append(n_components)
        steps.append(step)
        if n_components == 1 and step > 0:
            break
        actions, _ = swarm.take_actions()
        next_positions = swarm.true_positions + actions * C.config_constant_speed
        swarm.update_true_positions(next_positions)

    positions_by_step = np.asarray(positions_by_step, dtype=np.float32)
    subnet_by_step = np.asarray(subnet_by_step, dtype=int)
    steps = np.asarray(steps, dtype=int)
    connect_step = int(steps[-1])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        positions_by_step=positions_by_step,
        subnet_by_step=subnet_by_step,
        steps=steps,
        remain_list=remain_list,
        destroy_list=np.asarray(destroy_list, dtype=int),
        destroy_positions=destroy_positions.astype(np.float32),
        connect_step=np.asarray(connect_step),
        seed_value=np.asarray(seed_value),
        cr=np.asarray(C.config_communication_range, dtype=float),
        space_range=np.asarray(C.config_space_range, dtype=float),
    )
    data = np.load(cache_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def select_frames(traj: dict) -> list[int]:
    connect_step = int(np.asarray(traj["connect_step"]).item())
    desired = [0, int(round(connect_step * 0.33)), int(round(connect_step * 0.66)), connect_step]
    steps = np.asarray(traj["steps"], dtype=int)
    selected = []
    for step in desired:
        idx = int(np.argmin(np.abs(steps - step)))
        if idx not in selected:
            selected.append(idx)
    while len(selected) < 4:
        selected.append(len(steps) - 1)
    return selected[:4]


def plot_representative_sequence(fig, axes, traj: dict) -> None:
    frames = select_frames(traj)
    positions_by_step = traj["positions_by_step"]
    subnet_by_step = np.asarray(traj["subnet_by_step"], dtype=int)
    steps = np.asarray(traj["steps"], dtype=int)
    space_range = np.asarray(traj["space_range"], dtype=float)
    cr = float(np.asarray(traj["cr"]).item())
    center = space_range[:2] * 0.5
    palette = plt.get_cmap("tab20")

    for i, (ax, frame_idx) in enumerate(zip(axes, frames)):
        positions = positions_by_step[frame_idx]
        labels, _ = component_labels(positions, cr)
        colors = np.array([palette((label % 20) / 19.0) for label in labels])
        if subnet_by_step[frame_idx] == 1:
            colors[:, :3] = mcolors.to_rgb(ALGO_BY_LABEL["MORPHICA"]["color"])
            colors[:, 3] = 0.88
        else:
            colors[:, 3] = 0.78

        segs = edge_segments(positions, cr, max_edges=1300)
        if len(segs):
            ax.add_collection(LineCollection(segs, colors="#C7CDD1", linewidths=0.18, alpha=0.42))

        if i == 0:
            destroyed = np.asarray(traj["destroy_positions"])
            ax.scatter(
                destroyed[:, 0],
                destroyed[:, 1],
                marker="x",
                s=3.0,
                linewidths=0.25,
                c="#D95F5F",
                alpha=0.18,
                rasterized=True,
            )

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=5.2,
            c=colors,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            center[0],
            center[1],
            marker="*",
            s=18,
            c="#1A1A1A",
            linewidths=0.2,
            zorder=5,
        )

        ax.set_xlim(0, space_range[0])
        ax.set_ylim(0, space_range[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#D9DEE2")
            spine.set_linewidth(0.45)
        ax.set_title(
            f"t={int(steps[frame_idx])}   K={int(subnet_by_step[frame_idx])}",
            pad=2.0,
        )

    axes[0].text(
        0.02,
        0.04,
        "removed nodes",
        transform=axes[0].transAxes,
        color="#B44B4B",
        fontsize=5.5,
        ha="left",
        va="bottom",
    )


def add_sequence_arrows(fig, axes) -> None:
    for left, right in zip(axes[:-1], axes[1:]):
        left_box = left.get_position()
        right_box = right.get_position()
        y = 0.5 * (left_box.y0 + left_box.y1)
        start = (left_box.x1 + 0.018, y)
        end = (right_box.x0 - 0.018, y)
        arrow = FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=6,
            lw=0.55,
            color="#6C747A",
            alpha=0.72,
        )
        fig.add_artist(arrow)


def plot_recovery_violin(ax, steps_by_algo: dict[str, np.ndarray]) -> None:
    labels = [algo["label"] for algo in ALGORITHMS if algo["label"] in steps_by_algo]
    labels = sorted(labels, key=lambda x: ALGO_BY_LABEL[x]["priority"])
    y_positions = np.arange(len(labels))
    rng = np.random.default_rng(7)

    values_for_violin = []
    for label in labels:
        values = np.asarray(steps_by_algo[label], dtype=float)
        if np.ptp(values) < 1e-9:
            values = values + rng.normal(0.0, 0.6, size=len(values))
        values_for_violin.append(np.clip(values, 1, None))

    parts = ax.violinplot(
        values_for_violin,
        positions=y_positions,
        vert=False,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, label in zip(parts["bodies"], labels):
        color = ALGO_BY_LABEL[label]["color"]
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.22 if label != "MORPHICA" else 0.48)

    for y, label in zip(y_positions, labels):
        values = np.asarray(steps_by_algo[label], dtype=float)
        color = ALGO_BY_LABEL[label]["color"]
        jitter = rng.uniform(-0.25, 0.25, size=len(values))
        ax.scatter(
            np.clip(values, 1, None),
            y + jitter,
            s=4.5,
            c=color,
            alpha=0.36 if label != "MORPHICA" else 0.62,
            linewidths=0,
            rasterized=True,
        )
        q25, med, q75 = np.percentile(values, [25, 50, 75])
        ax.plot([q25, q75], [y, y], color=color, lw=1.6, solid_capstyle="round")
        ax.scatter([med], [y], s=8, c=color, edgecolors="white", linewidths=0.35, zorder=5)

    ax.axvline(MAX_STEP, color="#222222", lw=0.45, ls="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlim(55, 2050)
    ax.set_xticks([60, 100, 200, 500, 1000, 1800])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("steps to connectivity")
    ax.set_title("50% node loss, 50 cases")
    ax.grid(axis="x", color="#E5E8EA", lw=0.45)
    despine(ax)


def available_damage_values(num_agents: int, label: str = "MORPHICA") -> list[int]:
    algo = ALGO_BY_LABEL[label]
    root = SWEEP_ROOT / f"N{num_agents}" / algo["dir"]
    if not root.exists():
        return []
    values = []
    pattern = re.compile(rf"{re.escape(algo['prefix'])}_d(\d+)_50cases\.txt$")
    for path in root.glob(f"{algo['prefix']}_d*_50cases.txt"):
        match = pattern.match(path.name)
        if match:
            values.append(int(match.group(1)))
    return sorted(values)


def sweep_stats(num_agents: int, labels: list[str], damage_values: list[int] | None = None) -> dict[str, list[dict]]:
    if damage_values is None:
        damage_values = available_damage_values(num_agents)
    out = {label: [] for label in labels}
    for destroyed in damage_values:
        for label in labels:
            values_by_label = load_connect_steps(destroyed, labels=[label], num_agents=num_agents)
            if label not in values_by_label:
                continue
            values = values_by_label[label]
            out[label].append(
                {
                    "N": num_agents,
                    "destroyed": destroyed,
                    "damage_ratio": destroyed / num_agents,
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                    "mean": float(np.mean(values)),
                    "p90": float(np.percentile(values, 90)),
                    "success_rate": float(np.mean(values < MAX_STEP)),
                    "n_cases": int(len(values)),
                }
            )
    return out


def plot_damage_sweep(ax) -> dict:
    focus_labels = ["MORPHICA", "ML-DAGL"]
    highlighted_baselines = ["CEN", "DEMD", "CR-MGC"]
    envelope_labels = ["GDR-TS", "Hybrid-MADDPG", "HERO", "SIDR"]
    all_labels = focus_labels + highlighted_baselines + envelope_labels
    stats = sweep_stats(1000, all_labels)
    horizon = max_step_for(1000)

    damage_axis = np.array([row["damage_ratio"] * 100 for row in stats["MORPHICA"]])
    external_by_damage = []
    for damage in damage_axis:
        medians = []
        for label in envelope_labels:
            for row in stats.get(label, []):
                if np.isclose(row["damage_ratio"] * 100, damage):
                    medians.append(row["median"] / horizon)
                    break
        external_by_damage.append(medians)
    ext_low = np.array([np.min(values) if values else np.nan for values in external_by_damage])
    ext_high = np.array([np.max(values) if values else np.nan for values in external_by_damage])
    valid = ~np.isnan(ext_low)
    ax.fill_between(
        damage_axis[valid],
        ext_low[valid],
        ext_high[valid],
        color="#D3D7DA",
        alpha=0.42,
        lw=0,
        zorder=1,
        label="other external range",
    )
    ax.plot(
        damage_axis[valid],
        ext_low[valid],
        color="#777F85",
        lw=0.9,
        ls=(0, (2.5, 1.6)),
        zorder=2,
        label="best other external",
    )

    for label in highlighted_baselines:
        rows = stats.get(label, [])
        if not rows:
            continue
        x = np.array([row["damage_ratio"] * 100 for row in rows])
        med = np.array([row["median"] for row in rows]) / horizon
        color = ALGO_BY_LABEL[label]["color"]
        ax.plot(
            x,
            med,
            color=color,
            lw=0.92,
            alpha=0.78,
            marker="o",
            ms=1.8,
            zorder=2.5,
            label=label,
        )

    for label in focus_labels:
        rows = stats.get(label, [])
        if not rows:
            continue
        x = np.array([row["damage_ratio"] * 100 for row in rows])
        med = np.array([row["median"] for row in rows]) / horizon
        q25 = np.array([row["q25"] for row in rows]) / horizon
        q75 = np.array([row["q75"] for row in rows]) / horizon
        color = ALGO_BY_LABEL[label]["color"]
        ax.fill_between(
            x,
            q25,
            q75,
            color=color,
            alpha=0.09 if label != "MORPHICA" else 0.17,
            lw=0,
            zorder=2,
        )
        ax.plot(
            x,
            med,
            color=color,
            lw=1.85 if label == "MORPHICA" else 1.25,
            marker="o",
            ms=2.8 if label == "MORPHICA" else 2.35,
            zorder=3,
            label=label,
        )
        ax.text(
            96.5,
            med[-1],
            label,
            color=color,
            fontsize=6.2,
            ha="left",
            va="center",
            clip_on=False,
        )

    morphica_rows = stats["MORPHICA"]
    mldagl_rows = stats["ML-DAGL"]
    mldagl_by_damage = {round(row["damage_ratio"] * 100): row["median"] for row in mldagl_rows}
    for target_damage in [50, 75]:
        row = min(morphica_rows, key=lambda item: abs(item["damage_ratio"] * 100 - target_damage))
        damage = round(row["damage_ratio"] * 100)
        baseline = mldagl_by_damage.get(damage)
        if baseline is None:
            continue
        gain = 100.0 * (baseline - row["median"]) / baseline
        ax.text(
            damage + 1.2,
            row["median"] / horizon + 0.035,
            f"{gain:.0f}% faster",
            fontsize=5.7,
            color=ALGO_BY_LABEL["MORPHICA"]["color"],
            ha="left",
            va="bottom",
        )

    ax.axhline(1.0, color="#222222", lw=0.5, ls="--", alpha=0.58)
    ax.text(5.5, 1.012, "horizon", fontsize=5.5, color="#5E6469", ha="left", va="bottom")
    handles, labels = ax.get_legend_handles_labels()
    order = [
        labels.index(label)
        for label in [
            "MORPHICA",
            "ML-DAGL",
            "CEN",
            "DEMD",
            "CR-MGC",
            "other external range",
            "best other external",
        ]
        if label in labels
    ]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.92),
        frameon=False,
        ncol=2,
        handlelength=1.25,
        columnspacing=0.72,
        borderaxespad=0.0,
        fontsize=PANEL_LEGEND_FONTSIZE,
    )
    ax.set_xlim(2, 106)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([5, 25, 50, 75, 95])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel("node loss (%)")
    ax.set_ylabel("$T_{conn}/h$")
    ax.set_title("Damage-varying recovery at N = 1000")
    ax.grid(axis="y", color="#E5E8EA", lw=0.45, which="major")
    despine(ax)
    return {
        "normalization": horizon,
        "highlighted_baselines": highlighted_baselines,
        "other_external_envelope": envelope_labels,
        "damage_sweep": stats,
    }


def scale_damage_matrix(
    num_agents_values: list[int],
    damage_ratios: list[float],
    baseline_labels: list[str],
) -> tuple[np.ndarray, list[list[str]], dict]:
    matrix = np.full((len(num_agents_values), len(damage_ratios)), np.nan, dtype=float)
    winners = [["" for _ in damage_ratios] for _ in num_agents_values]
    raw = {}
    for i, num_agents in enumerate(num_agents_values):
        raw[num_agents] = {}
        for j, ratio in enumerate(damage_ratios):
            destroyed = int(round(num_agents * ratio))
            morphica_steps = load_connect_steps(destroyed, labels=["MORPHICA"], num_agents=num_agents).get("MORPHICA")
            if morphica_steps is None:
                continue
            morphica_median = float(np.median(morphica_steps))
            best_label = None
            best_median = None
            for label in baseline_labels:
                baseline_steps = load_connect_steps(destroyed, labels=[label], num_agents=num_agents).get(label)
                if baseline_steps is None:
                    continue
                median = float(np.median(baseline_steps))
                if best_median is None or median < best_median:
                    best_median = median
                    best_label = label
            if best_median is None or best_median <= 0:
                continue
            improvement = 100.0 * (best_median - morphica_median) / best_median
            matrix[i, j] = improvement
            winners[i][j] = best_label or ""
            raw[num_agents][f"{int(ratio * 100)}%"] = {
                "destroyed": destroyed,
                "morphica_median": morphica_median,
                "best_external": best_label,
                "best_external_median": best_median,
                "relative_reduction_percent": float(improvement),
            }
    return matrix, winners, raw


def plot_scale_damage_map(ax) -> dict:
    num_agents_values = [200, 500, 1000, 2000]
    labels = ["MORPHICA", "ML-DAGL", "DEMD", "CEN", "CR-MGC"]
    offsets = np.linspace(-0.28, 0.28, len(labels))
    width = 0.12
    stats = {"damage_ratio": 0.50, "normalization": {}, "series": {}}

    for group_idx, num_agents in enumerate(num_agents_values):
        destroyed = int(0.5 * num_agents)
        horizon = max_step_for(num_agents)
        stats["normalization"][num_agents] = horizon
        steps_by_label = load_connect_steps(destroyed, labels=labels, num_agents=num_agents)
        for label_idx, label in enumerate(labels):
            values = steps_by_label.get(label)
            if values is None:
                continue
            normalized = np.asarray(values, dtype=float) / horizon
            xpos = group_idx + offsets[label_idx]
            parts = ax.violinplot(
                [normalized],
                positions=[xpos],
                widths=width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            color = ALGO_BY_LABEL[label]["color"]
            body = parts["bodies"][0]
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.42 if label == "MORPHICA" else 0.24)
            q25, median, q75 = np.percentile(normalized, [25, 50, 75])
            ax.plot([xpos, xpos], [q25, q75], color=color, lw=1.15, solid_capstyle="round", zorder=4)
            ax.scatter([xpos], [median], s=11 if label == "MORPHICA" else 8, color=color, edgecolors="white", linewidths=0.35, zorder=5)
            stats["series"].setdefault(label, {})[num_agents] = {
                "destroyed": destroyed,
                "horizon": horizon,
                "median_normalized": float(median),
                "q25_normalized": float(q25),
                "q75_normalized": float(q75),
                "median_steps": float(np.median(values)),
            }

    for label in labels:
        ax.scatter([], [], color=ALGO_BY_LABEL[label]["color"], s=14, label=label)

    ax.axhline(1.0, color="#222222", lw=0.5, ls="--", alpha=0.58)
    ax.text(3.45, 1.012, "horizon", fontsize=5.5, color="#5E6469", ha="right", va="bottom")
    ax.set_xlim(-0.55, len(num_agents_values) - 0.45)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(np.arange(len(num_agents_values)))
    ax.set_xticklabels([f"N={n}" for n in num_agents_values])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("$T_{conn}/h$")
    ax.set_xlabel("swarm size at 50% node loss")
    ax.set_title("Scale-varying recovery distribution")
    ax.grid(axis="y", color="#E5E8EA", lw=0.45)
    ax.legend(
        loc="upper left",
        frameon=False,
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
        fontsize=PANEL_LEGEND_FONTSIZE,
    )
    despine(ax)
    return stats


def plot_tail_ccdf(ax, steps_by_algo: dict[str, np.ndarray]) -> None:
    labels = ["MORPHICA", "ML-DAGL", "DEMD", "CEN", "CR-MGC", "GDR-TS"]
    labels = [label for label in labels if label in steps_by_algo]
    for label in labels:
        values = np.asarray(steps_by_algo[label], dtype=float)
        xs, ys = ccdf(values)
        ax.step(
            np.clip(xs, 1, None),
            ys,
            where="post",
            color=ALGO_BY_LABEL[label]["color"],
            lw=1.35 if label == "MORPHICA" else 0.9,
            alpha=0.96 if label == "MORPHICA" else 0.74,
            label=label,
        )
    ax.axvline(MAX_STEP, color="#222222", lw=0.45, ls="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlim(120, 2050)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([150, 300, 600, 1000, 1800])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("threshold step")
    ax.set_ylabel("Pr($T_{conn}$ > threshold)")
    ax.set_title("75% node loss tail risk")
    ax.grid(axis="both", color="#E5E8EA", lw=0.45)
    ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.4, fontsize=PANEL_LEGEND_FONTSIZE)
    despine(ax)


def traj_path(label: str, destroyed: int = 500, num_cases: int = 20, suffix: str = "_traj.csv") -> Path:
    dirname, prefix = TRAJ_MAPPING[label]
    return DATA_ROOT / dirname / f"{prefix}_d{destroyed}_{num_cases}cases{suffix}"


def read_traj_csv(label: str, destroyed: int = 500, num_cases: int = 20) -> pd.DataFrame:
    path = traj_path(label, destroyed, num_cases, "_traj.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def load_traj_connect_steps(label: str, destroyed: int = 500, num_cases: int = 20) -> np.ndarray:
    dirname, prefix = TRAJ_MAPPING[label]
    path = DATA_ROOT / dirname / f"{prefix}_d{destroyed}_{num_cases}cases_traj.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    txt_path = traj_path(label, destroyed, num_cases, ".txt")
    return np.asarray(read_list_section(txt_path, "connect_step"), dtype=float)


def padded_case_curves(df: pd.DataFrame, max_step: int = 1800) -> np.ndarray:
    cases = sorted(df["case"].dropna().unique())
    curves = []
    for case in cases:
        rows = df[df["case"] == case].sort_values("step")
        curve = np.full(max_step + 1, np.nan, dtype=float)
        for _, row in rows.iterrows():
            step = int(row["step"])
            if 0 <= step <= max_step and not pd.isna(row["num_subnets"]):
                curve[step] = float(row["num_subnets"])
        valid = np.where(~np.isnan(curve))[0]
        if len(valid) == 0:
            continue
        first = valid[0]
        curve[:first] = curve[first]
        last_value = curve[first]
        for idx in range(first, max_step + 1):
            if not np.isnan(curve[idx]):
                last_value = curve[idx]
            else:
                curve[idx] = last_value
        curves.append(curve)
    return np.asarray(curves, dtype=float)


def normalized_k_curves(k_curves: np.ndarray) -> np.ndarray:
    k0 = np.maximum(k_curves[:, [0]], 1.0)
    return k_curves / k0


def cumulative_k_drop_curves(k_curves: np.ndarray) -> np.ndarray:
    denom = np.maximum(k_curves[:, [0]] - 1.0, 1.0)
    drops = np.maximum(-(np.diff(k_curves, axis=1)), 0.0)
    return np.concatenate([np.zeros((len(k_curves), 1)), np.cumsum(drops, axis=1)], axis=1) / denom


def process_metrics_from_k(k_curves: np.ndarray, connect_steps: np.ndarray, horizon_step: int = MAX_STEP) -> dict[str, float]:
    rows = []
    for idx, curve in enumerate(k_curves):
        k0 = max(float(curve[0]), 1.0)
        denom = max(k0 - 1.0, 1.0)
        conn = float(connect_steps[idx]) if idx < len(connect_steps) else float(horizon_step)
        diffs = np.diff(curve)
        drops = np.maximum(-diffs, 0.0)
        rises = np.maximum(diffs, 0.0)
        first_drops = np.where(drops > 0)[0]
        first_merge = float(first_drops[0] + 1) if len(first_drops) else float(horizon_step)
        rows.append(
            {
                "initial_k": k0,
                "normalized_fragmentation_exposure": float(np.sum(curve - 1.0) / ((horizon_step + 1) * denom)),
                "first_effective_merge": first_merge,
                "first_merge_to_connectivity": float(max(conn - first_merge, 0.0)),
                "effective_merge_lower_bound": float(np.sum(drops)),
                "normalized_effective_merge_lower_bound": float(np.sum(drops) / denom),
                "merge_churn": float(np.sum(rises)),
                "re_split_count": float(np.sum(rises > 0)),
            }
        )
    stats = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        stats[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
        }
    return stats


def trajectory_process_summary(
    labels: list[str],
    destroyed_values: list[int],
    num_cases: int = 20,
    horizon_step: int = MAX_STEP,
) -> dict:
    summary = {}
    for destroyed in destroyed_values:
        summary[str(destroyed)] = {}
        for label in labels:
            curves = padded_case_curves(read_traj_csv(label, destroyed, num_cases), max_step=horizon_step)
            connect = load_traj_connect_steps(label, destroyed, num_cases)
            summary[str(destroyed)][label] = process_metrics_from_k(curves, connect, horizon_step)
    return summary


def plot_subnet_dynamics(
    ax,
    merge_ax=None,
    destroyed: int = 500,
    num_cases: int = 20,
    max_step: int = 900,
) -> dict:
    labels = ["MORPHICA", "ML-DAGL", "DEMD", "CEN", "CR-MGC"]
    x = np.arange(0, max_step + 1)
    merge_axes = merge_ax if merge_ax is not None else ax.twinx()
    stats = {}
    for label in labels:
        curves = padded_case_curves(read_traj_csv(label, destroyed, num_cases), max_step=max_step)
        if len(curves) == 0:
            continue
        norm_curves = normalized_k_curves(curves)
        median = np.nanmedian(norm_curves, axis=0)
        lo = np.nanpercentile(norm_curves, 25, axis=0)
        hi = np.nanpercentile(norm_curves, 75, axis=0)
        color = ALGO_BY_LABEL[label]["color"]
        ax.plot(
            x,
            median,
            color=color,
            lw=1.45 if label == "MORPHICA" else 0.95,
            label=label,
        )
        ax.fill_between(x, lo, hi, color=color, alpha=0.09 if label != "MORPHICA" else 0.16, lw=0)
        if label in {"MORPHICA", "ML-DAGL"}:
            merge_curves = cumulative_k_drop_curves(curves)
            merge_median = np.nanmedian(merge_curves, axis=0)
            merge_axes.plot(
                x,
                merge_median,
                color=color,
                lw=1.05 if label == "MORPHICA" else 0.8,
                ls=(0, (2.4, 1.4)),
                alpha=0.82,
            )
        full_curves = padded_case_curves(read_traj_csv(label, destroyed, num_cases), max_step=MAX_STEP)
        stats[label] = process_metrics_from_k(
            full_curves,
            load_traj_connect_steps(label, destroyed, num_cases),
            horizon_step=MAX_STEP,
        )
    ax.set_xlim(0, max_step)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 150, 300, 450, 600, 750, 900])
    ax.set_xlabel("" if merge_ax is not None else "step")
    ax.set_ylabel("$K(t)/K(0)$")
    ax.set_title("Topological repair dynamics")
    ax.grid(axis="both", color="#E5E8EA", lw=0.45)
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.1, fontsize=4.8, columnspacing=0.55)
    merge_axes.set_xlim(0, max_step)
    merge_axes.set_ylim(0, 1.55)
    merge_axes.set_xticks([0, 150, 300, 450, 600, 750, 900])
    if merge_ax is not None:
        ax.tick_params(labelbottom=False)
        merge_axes.set_xlabel("step")
        merge_axes.set_ylabel("$B_K/(K_0-1)$")
        merge_axes.grid(axis="both", color="#E5E8EA", lw=0.45)
        merge_axes.text(
            0.02,
            0.80,
            "dashed: K-drop lower bound",
            transform=merge_axes.transAxes,
            fontsize=5.0,
            color="#5E6469",
            ha="left",
            va="top",
        )
        despine(merge_axes)
    else:
        merge_axes.set_ylabel("cumulative $K$ drops")
        merge_axes.spines["top"].set_visible(False)
        merge_axes.tick_params(axis="y", width=0.6, length=2.5, labelsize=5.5)
    despine(ax)
    return stats


def plot_degree_ccdf(ax, degrees_by_algo: dict[str, np.ndarray]) -> None:
    labels = ["MORPHICA", "ML-DAGL", "DEMD", "CR-MGC", "CEN"]
    labels = [label for label in labels if label in degrees_by_algo]
    for label in labels:
        degrees = np.asarray(degrees_by_algo[label], dtype=float)
        max_k = int(max(2, degrees.max()))
        ks = np.arange(1, max_k + 1)
        ys = np.array([(degrees >= k).mean() for k in ks], dtype=float)
        ax.plot(
            ks,
            np.clip(ys, 1e-5, None),
            color=ALGO_BY_LABEL[label]["color"],
            lw=1.35 if label == "MORPHICA" else 0.9,
            alpha=0.96 if label == "MORPHICA" else 0.75,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 520)
    ax.set_ylim(8e-5, 1.08)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 500])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("final node degree $k$")
    ax.set_ylabel("Pr($K \\geq k$)")
    ax.set_title("Sparse repaired topology")
    ax.grid(axis="both", color="#E5E8EA", lw=0.45, which="major")
    ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.4)
    despine(ax)


def plot_degree_footprint(ax, degrees_by_algo: dict[str, np.ndarray]) -> dict:
    labels = ["MORPHICA", "ML-DAGL", "DEMD", "CR-MGC", "CEN"]
    labels = [label for label in labels if label in degrees_by_algo]
    y_positions = np.arange(len(labels))[::-1]
    stats = {}
    for y, label in zip(y_positions, labels):
        degrees = np.asarray(degrees_by_algo[label], dtype=float)
        q10, median, q90, q95 = np.percentile(degrees, [10, 50, 90, 95])
        color = ALGO_BY_LABEL[label]["color"]
        ax.plot([q10, q90], [y, y], color=color, lw=3.4, alpha=0.18, solid_capstyle="round")
        ax.plot([median, q95], [y, y], color=color, lw=1.5, alpha=0.85, solid_capstyle="round")
        ax.scatter([median], [y], s=20, color=color, edgecolors="white", linewidths=0.45, zorder=4)
        ax.scatter([q95], [y], s=12, marker="|", color=color, linewidths=1.0, zorder=4)
        stats[label] = {
            "mean": float(np.mean(degrees)),
            "median": float(median),
            "q10": float(q10),
            "q90": float(q90),
            "p95": float(q95),
            "max": float(np.max(degrees)),
        }
    ax.set_xscale("log")
    ax.set_xlim(1.5, 520)
    ax.set_xticks([2, 5, 10, 20, 50, 100, 500])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("final node degree")
    ax.set_title("Topology footprint after repair")
    ax.grid(axis="x", color="#E5E8EA", lw=0.45, which="major")
    despine(ax)
    ax.text(
        0.02,
        0.05,
        "dot: median   tick: p95",
        transform=ax.transAxes,
        fontsize=5.8,
        color="#5E6469",
        ha="left",
        va="bottom",
    )
    return stats


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def panel_stats(values_by_label: dict[str, np.ndarray]) -> dict:
    return {
        label: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
        }
        for label, values in values_by_label.items()
    }


def representative_cache_path(args) -> Path:
    return CACHE_DIR / (
        f"representative_MORPHICA_N{args.num_agents}_d{args.rep_destroyed}_case{args.rep_case}.npz"
    )


def load_representative_trajectory(args) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return run_representative_case(
        cache_path=representative_cache_path(args),
        num_agents=args.num_agents,
        destroyed=args.rep_destroyed,
        case_index=args.rep_case,
        ablation=args.ablation,
        force=args.force_representative,
    )


def build_panel(args) -> tuple[plt.Figure, dict]:
    set_style()
    panel = args.panel

    if panel == "a":
        traj = load_representative_trajectory(args)
        fig = plt.figure(figsize=(7.2, 2.02), dpi=300)
        gs = GridSpec(1, 4, figure=fig, wspace=0.54)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
        plot_representative_sequence(fig, axes, traj)
        fig.subplots_adjust(left=0.035, right=0.995, bottom=0.08, top=0.82)
        add_sequence_arrows(fig, axes)
        panel_label(axes[0], "a", x=-0.16, y=1.08)
        stats = {
            "panel": "a",
            "representative_cache": str(representative_cache_path(args)),
            "representative_connect_step": int(np.asarray(traj["connect_step"]).item()),
        }
        return fig, stats

    if panel == "b":
        fig, ax = make_metric_panel()
        stats = plot_scale_damage_map(ax)
        panel_label(ax, "b")
        return fig, {"panel": "b", **stats}

    if panel == "c":
        fig, ax = make_metric_panel()
        stats = plot_damage_sweep(ax)
        panel_label(ax, "c")
        return fig, {"panel": "c", "N": 1000, **stats}

    if panel == "d":
        fig = plt.figure(figsize=METRIC_PANEL_FIGSIZE, dpi=300)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[2.7, 1.0], hspace=0.08)
        ax = fig.add_subplot(gs[0])
        merge_ax = fig.add_subplot(gs[1], sharex=ax)
        fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.86)
        process_stats = plot_subnet_dynamics(ax, merge_ax=merge_ax)
        panel_label(ax, "d")
        return fig, {
            "panel": "d",
            "d": 500,
            "trajectory_cases_per_algorithm": 20,
            "max_step_shown": 900,
            "process_metrics_from_K": {
                "500": process_stats,
                "750": trajectory_process_summary(
                    ["MORPHICA", "ML-DAGL", "DEMD", "CEN", "CR-MGC"],
                    [750],
                    num_cases=20,
                )["750"],
            },
        }

    if panel == "e":
        degrees_50 = load_degrees(500, ["MORPHICA", "ML-DAGL", "DEMD", "CR-MGC", "CEN"])
        fig, ax = make_metric_panel()
        stats = plot_degree_footprint(ax, degrees_50)
        panel_label(ax, "e")
        return fig, {
            "panel": "e",
            "d": 500,
            "degree_distribution": stats,
        }

    raise ValueError(f"Unsupported panel: {panel}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=sorted(PANEL_STEMS), required=True)
    parser.add_argument("--num-agents", type=int, default=1000)
    parser.add_argument("--rep-destroyed", type=int, default=500)
    parser.add_argument("--rep-case", type=int, default=0)
    parser.add_argument("--ablation", type=str, default="full")
    parser.add_argument("--force-representative", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PANEL_DIR,
        help="Directory for individual panel outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, stats = build_panel(args)
    output_stem = args.output_dir / PANEL_STEMS[args.panel]
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    stats_path = output_stem.with_suffix(".stats.json")

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=450)
    plt.close(fig)

    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {stats_path}")
    if "representative_connect_step" in stats:
        print(f"Representative connect_step={stats['representative_connect_step']}")


if __name__ == "__main__":
    main()
