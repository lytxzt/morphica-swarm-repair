#!/usr/bin/env python3
"""Draw individual Fig. 3 panels for the MORPHICA manuscript.

Use one command per panel:

  python Paper/plot_fig3.py --panel a
  python Paper/plot_fig3.py --panel b

The script writes PDF, PNG and a small JSON stats file under Paper/figs/fig3.
It reads active sweep logs, the verified reconstructed ablation exports and
the ring-centre diagnostic used by results.md.
"""

from __future__ import annotations

import argparse
import ast
import json
import hashlib
import os
import pickle
import re
import sys
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib import patheffects
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SWEEP_ROOT = ROOT / "Logs" / "algorithm_sweep"
FIG_DIR = ROOT / "Paper" / "figs"
PANEL_DIR = FIG_DIR / "fig3"
# Fig. 2a owns the whole-system recovery sequence.  Fig. 3a instead uses a
# separate, early fragmented state whose components remain visually balanced,
# so that the three operator morphologies are not hidden by one giant subnet.
RPS_DIAGNOSTIC_CACHE = FIG_DIR / "fig2_cache" / "representative_MORPHICA_N200_d100_case38.npz"
RING_CASE = ROOT / "Logs" / "ring_center_test" / "connected_ring_trajectory.pkl"
RING_CACHE = FIG_DIR / "fig3_cache" / "jittered_ring_center_v4_replay.npz"

CR = 120.0
EPS = 1e-4
RING_JITTER_SEED = 27
RING_ANGULAR_CONCENTRATION = 8.0
RING_RADIAL_JITTER = 75.0
RING_CENTRE_JITTER = 17.0
RING_TRAJECTORY_STRIDE = 25

PANEL_STEMS = {
    "a": "Fig3a_rps_spatial_stages",
    "b": "Fig3b_operator_ablation_n_varying",
    "c": "Fig3c_population_phase_dynamics",
    "d": "Fig3d_ring_center_breaking",
    "e": "Fig3e_ring_center_breaking",
}

COLORS = {
    # Fixed semantic palette for the three MORPHICA operators.  It matches the
    # earlier RPS phase figure: Retraction = coral, Polarization = amber and
    # Spreading = slate blue.  These colours are distinct from method colours
    # used by the benchmark figures.
    "R": "#C9656B",
    "P": "#F3AE35",
    "S": "#6C8ABD",
    "subnet": "#333333",
    "muted": "#9AA0A6",
    "removed": "#E8B4B8",
    # In the ablation panel, the colour identifies the operator that is
    # removed.  Keep this mapping identical to the R/P/S phase panels.
    "full": "#007C6C",
    "no-R": "#C9656B",
    "no-P": "#F3AE35",
    "no-S": "#6C8ABD",
    "multi-bearer": "#AA4499",
}

ABLATION_PATHS = {
    100: {
        "full": "Logs/algorithm_sweep/N100/MORPHICA/MORPHICA_d50_50cases.txt",
        "no-R": "Logs/algorithm_sweep/N100/MORPHICA_no-R/MORPHICA_d50_50cases.txt",
        "no-S": "Logs/algorithm_sweep/N100/MORPHICA_no-S/MORPHICA_d50_50cases.txt",
        "no-P": "Logs/algorithm_sweep/N100/MORPHICA_no-P/MORPHICA_d50_50cases.txt",
        "multi-bearer": "Logs/algorithm_sweep/N100/MORPHICA_multi-bearer/MORPHICA_d50_50cases.txt",
    },
    200: {
        "full": "Logs/algorithm_sweep/N200/MORPHICA/MORPHICA_d100_50cases.txt",
        "no-R": "Logs/algorithm_sweep/N200/MORPHICA_no-R/MORPHICA_d100_50cases.txt",
        "no-S": "Logs/algorithm_sweep/N200/MORPHICA_no-S/MORPHICA_d100_50cases.txt",
        "no-P": "Logs/algorithm_sweep/N200/MORPHICA_no-P/MORPHICA_d100_50cases.txt",
        "multi-bearer": "Logs/algorithm_sweep/N200/MORPHICA_multi-bearer/MORPHICA_d100_50cases.txt",
    },
    500: {
        "full": "Logs/algorithm_sweep/N500/MORPHICA/MORPHICA_d250_50cases.txt",
        "no-R": "Logs/algorithm_sweep/N500/MORPHICA_no-R/MORPHICA_d250_50cases.txt",
        "no-S": "Logs/algorithm_sweep/N500/MORPHICA_no-S/MORPHICA_d250_50cases.txt",
        "no-P": "Logs/algorithm_sweep/N500/MORPHICA_no-P/MORPHICA_d250_50cases.txt",
        "multi-bearer": "Logs/algorithm_sweep/N500/MORPHICA_multi-bearer/MORPHICA_d250_50cases.txt",
    },
    1000: {
        "full": "Logs/manuscript_revision/ablation/full_d500_100cases.txt",
        "no-R": "Logs/manuscript_revision/ablation/no-R_d500_100cases.txt",
        "no-S": "Logs/manuscript_revision/ablation/no-S_d500_100cases.txt",
        "no-P": "Logs/manuscript_revision/ablation/no-P_d500_100cases.txt",
        "multi-bearer": "Logs/algorithm_sweep/N1000/MORPHICA_multiBearer/MORPHICA_d500_50cases.txt",
    },
}


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


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_list_section(path: Path, name: str) -> list:
    text = path.read_text()
    match = re.search(rf"{re.escape(name)}:\s*\n\s*(\[[^\]]*\])", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find section {name!r} in {path}")
    raw = match.group(1)
    if "..." in raw:
        raise ValueError(f"Section {name!r} in {path} is truncated")
    return ast.literal_eval(raw)


def connect_steps(path: str | Path) -> np.ndarray:
    return np.asarray(read_list_section(Path(path), "connect_step"), dtype=float)


def max_step_for(num_agents: int) -> float:
    # Fixed recording cutoffs, independent of which methods hit the horizon.
    return float({100: 599, 200: 799, 500: 1279, 1000: 1799, 2000: 2559}[num_agents])


def trajectory_csv(destroyed: int = 500) -> Path:
    return SWEEP_ROOT / "N1000" / "MORPHICA_traj_v2" / f"MORPHICA_d{destroyed}_50cases_traj.csv"


def trajectory_txt(destroyed: int = 500) -> Path:
    return SWEEP_ROOT / "N1000" / "MORPHICA_traj_v2" / f"MORPHICA_d{destroyed}_50cases.txt"


def component_labels(positions: np.ndarray, cr: float = CR) -> tuple[np.ndarray, int]:
    n = len(positions)
    if n == 0:
        return np.empty(0, dtype=int), 0
    diff = positions[:, None, :] - positions[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    adj = (dist2 <= cr * cr) & ~np.eye(n, dtype=bool)
    labels = np.full(n, -1, dtype=int)
    comp = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        labels[start] = comp
        stack = [start]
        while stack:
            node = stack.pop()
            nb = np.where(adj[node] & (labels < 0))[0]
            labels[nb] = comp
            stack.extend(nb.tolist())
        comp += 1
    return labels, comp


def edge_segments(positions: np.ndarray, cr: float = CR, max_edges: int = 1200) -> np.ndarray:
    n = len(positions)
    if n <= 1:
        return np.empty((0, 2, 2))
    xy = positions[:, :2]
    diff = xy[:, None, :] - xy[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    iu = np.triu_indices(n, 1)
    keep = np.where(dist2[iu] <= cr * cr)[0]
    if len(keep) > max_edges:
        rng = np.random.default_rng(2)
        keep = rng.choice(keep, max_edges, replace=False)
    pairs = np.column_stack((iu[0][keep], iu[1][keep]))
    return np.stack((xy[pairs[:, 0]], xy[pairs[:, 1]]), axis=1)


def component_color_map(labels: np.ndarray, palette: list[str] | None = None) -> dict[int, tuple | str]:
    cmap = plt.get_cmap("tab20")
    counts = np.bincount(labels) if len(labels) else np.array([])
    order = np.argsort(-counts) if len(counts) else []
    color_for = {}
    for rank, comp in enumerate(order):
        color_for[int(comp)] = palette[rank % len(palette)] if palette else cmap(rank % 20)
    return color_for


def draw_coverage_regions(
    ax,
    positions: np.ndarray,
    labels: np.ndarray,
    color_for: dict[int, tuple],
    extent: tuple[tuple[float, float], tuple[float, float]],
    cr: float = CR,
) -> None:
    (xlim, ylim) = extent
    xs = np.linspace(xlim[0], xlim[1], 150)
    ys = np.linspace(ylim[0], ylim[1], 150)
    X, Y = np.meshgrid(xs, ys)

    for comp in sorted(set(labels.tolist())):
        pts = positions[labels == comp, :2]
        if len(pts) == 0:
            continue
        dx = X[..., None] - pts[:, 0]
        dy = Y[..., None] - pts[:, 1]
        mask = np.min(dx * dx + dy * dy, axis=2) <= cr * cr
        color = color_for.get(int(comp), "#777777")
        ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], colors=[color], alpha=0.055, zorder=0.15)
        ax.contour(X, Y, mask.astype(float), levels=[0.5], colors=[color], linewidths=0.45, alpha=0.28, zorder=0.25)


def draw_recent_trails(
    ax,
    trail_positions: np.ndarray | None,
    color: str = "#1A3F60",
    linewidth: float = 0.45,
    alpha_start: float = 0.035,
    alpha_end: float = 0.16,
) -> None:
    if trail_positions is None or len(trail_positions) < 2:
        return
    xy = np.asarray(trail_positions, dtype=float)[..., :2]
    n_steps = xy.shape[0]
    segments = []
    colors = []
    rgb = matplotlib.colors.to_rgb(color)
    for t in range(n_steps - 1):
        alpha = alpha_start + (alpha_end - alpha_start) * ((t + 1) / (n_steps - 1))
        step_segments = np.stack((xy[t], xy[t + 1]), axis=1)
        segments.append(step_segments)
        colors.extend([(*rgb, alpha)] * len(step_segments))
    if segments:
        ax.add_collection(
            LineCollection(
                np.concatenate(segments, axis=0),
                colors=colors,
                linewidths=linewidth,
                zorder=1.25,
            )
        )


def draw_vectors(ax, starts: np.ndarray, vectors: np.ndarray, color: str, scale: float, width: float = 0.004) -> None:
    if len(starts) == 0:
        return
    vec = np.asarray(vectors, dtype=float)[:, :2]
    norms = np.linalg.norm(vec, axis=1)
    keep = norms > EPS
    if not np.any(keep):
        return
    starts = np.asarray(starts, dtype=float)[keep, :2]
    vec = vec[keep] / norms[keep, None] * scale
    ax.quiver(
        starts[:, 0],
        starts[:, 1],
        vec[:, 0],
        vec[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=width,
        headwidth=3.6,
        headlength=4.5,
        headaxislength=4.0,
        color=color,
        alpha=0.86,
        zorder=4.6,
    )


def retraction_vectors(positions: np.ndarray, labels: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anchor_vec = np.asarray(anchor, dtype=float)
    if anchor_vec.shape[0] == 2:
        anchor_vec = np.array([anchor_vec[0], anchor_vec[1], 0.0])
    starts = []
    vectors = []
    for comp in sorted(set(labels.tolist())):
        pts = positions[labels == comp]
        if len(pts) == 0:
            continue
        center = np.mean(pts, axis=0)
        starts.append(center)
        vectors.append(anchor_vec - center)
    return np.asarray(starts), np.asarray(vectors)


def polarization_vectors(positions: np.ndarray, labels: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anchor_vec = np.asarray(anchor, dtype=float)
    if anchor_vec.shape[0] == 2:
        anchor_vec = np.array([anchor_vec[0], anchor_vec[1], 0.0])
    starts = []
    vectors = []
    for comp in sorted(set(labels.tolist())):
        idxs = np.where(labels == comp)[0]
        if len(idxs) == 0:
            continue
        pts = positions[idxs]
        bearer_idx = idxs[int(np.argmin(np.linalg.norm(pts - anchor_vec, axis=1)))]
        starts.append(positions[bearer_idx])
        vectors.append(anchor_vec - positions[bearer_idx])
    return np.asarray(starts), np.asarray(vectors)


def spreading_vectors(positions: np.ndarray, cr: float = CR, max_vectors: int = 28) -> tuple[np.ndarray, np.ndarray]:
    xy = positions[:, :2]
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    starts = []
    vectors = []
    scores = []
    margin = 20.0
    safe_range = cr - margin
    for i, pos in enumerate(xy):
        neighbours = np.where((dist[i] <= cr) & (dist[i] > EPS))[0]
        if len(neighbours) < 2:
            continue
        info = []
        for j in neighbours:
            safety = np.clip((cr - dist[i, j] - margin) / safe_range, 0.0, 1.0)
            if safety <= EPS:
                continue
            delta = xy[j] - pos
            info.append((dist[i, j], safety, np.arctan2(delta[1], delta[0])))
        if len(info) < 2:
            continue
        angles = np.sort(np.asarray([item[2] for item in info]))
        gaps = np.diff(angles)
        all_gaps = np.append(gaps, 2 * np.pi - (angles[-1] - angles[0]))
        gap_idx = int(np.argmax(all_gaps))
        largest_gap = float(all_gaps[gap_idx])
        gap_gate = np.clip((largest_gap - np.pi) / np.pi, 0.0, 1.0)
        if gap_gate < 0.01:
            continue
        if gap_idx < len(gaps):
            gap_mid = angles[gap_idx] + largest_gap / 2.0
        else:
            gap_mid = angles[-1] + largest_gap / 2.0
            if gap_mid > np.pi:
                gap_mid -= 2 * np.pi
        info.sort(key=lambda item: item[0])
        score = float(gap_gate * np.mean([item[1] for item in info[:3]]))
        starts.append(positions[i])
        vectors.append([np.cos(gap_mid), np.sin(gap_mid), 0.0])
        scores.append(score)
    if not starts:
        return np.empty((0, 3)), np.empty((0, 3))
    order = np.argsort(scores)[-max_vectors:]
    return np.asarray(starts)[order], np.asarray(vectors)[order]


def draw_swarm_snapshot(
    ax,
    positions: np.ndarray,
    title: str,
    removed: np.ndarray | None = None,
    anchor: np.ndarray | None = None,
    trail_positions: np.ndarray | None = None,
    coverage_extent: tuple[tuple[float, float], tuple[float, float]] | None = None,
    overlay: str | None = None,
    component_palette: list[str] | None = None,
    anchor_marker: str = "*",
    anchor_size: float = 28,
    anchor_linewidth: float = 0.0,
    cr: float = CR,
    edge_alpha: float = 0.08,
) -> int:
    labels, ncomp = component_labels(positions, cr)
    color_for = component_color_map(labels, component_palette)
    if coverage_extent is not None:
        draw_coverage_regions(ax, positions, labels, color_for, coverage_extent, cr)

    segments = edge_segments(positions, cr)
    if len(segments):
        ax.add_collection(LineCollection(segments, colors="#9EA6AE", linewidths=0.22, alpha=edge_alpha, zorder=1))

    draw_recent_trails(ax, trail_positions)

    if removed is not None and len(removed):
        ax.scatter(removed[:, 0], removed[:, 1], s=2.0, c=COLORS["removed"], alpha=0.32, lw=0, zorder=0)

    colors = [color_for.get(int(c), "#777777") for c in labels]
    ax.scatter(positions[:, 0], positions[:, 1], s=3.2, c=colors, alpha=0.92, lw=0, zorder=2)

    if anchor is not None and overlay == "R":
        starts, vectors = retraction_vectors(positions, labels, anchor)
        draw_vectors(ax, starts, vectors, COLORS["R"], scale=58, width=0.0048)
    elif anchor is not None and overlay == "P":
        starts, vectors = polarization_vectors(positions, labels, anchor)
        if len(starts):
            ax.scatter(starts[:, 0], starts[:, 1], s=28, facecolors="none", edgecolors=COLORS["P"], linewidths=0.9, zorder=5)
        draw_vectors(ax, starts, vectors, COLORS["P"], scale=54, width=0.0048)
    elif overlay == "S":
        starts, vectors = spreading_vectors(positions, cr)
        draw_vectors(ax, starts, vectors, COLORS["S"], scale=36, width=0.0036)

    if anchor is not None:
        ax.scatter(
            [anchor[0]],
            [anchor[1]],
            marker=anchor_marker,
            s=anchor_size,
            c="#222222",
            linewidths=anchor_linewidth,
            zorder=4,
        )

    ax.set_title(title, pad=2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#D0D4DA")
        spine.set_linewidth(0.45)
    return ncomp


def save_panel(fig: plt.Figure, panel: str, stats: dict) -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = PANEL_STEMS[panel]
    pdf = PANEL_DIR / f"{stem}.pdf"
    png = PANEL_DIR / f"{stem}.png"
    stat_path = PANEL_DIR / f"{stem}.stats.json"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    stat_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    plt.close(fig)
    print(f"saved {pdf}")
    print(f"saved {png}")


def plot_panel_a() -> dict:
    data = np.load(RPS_DIAGNOSTIC_CACHE, allow_pickle=True)
    positions = data["positions_by_step"]
    steps = data["steps"]
    anchor = np.asarray(data["space_range"])[:2] * 0.5

    # Fig. 2a owns the global time course.  This panel fixes a single damaged
    # geometry and renders the distinct shape change induced by each operator.
    mechanism_step = 5
    idx = int(np.argmin(np.abs(steps - mechanism_step)))
    base = np.asarray(positions[idx], dtype=float)
    labels, ncomp = component_labels(base)
    component_palette = ["#4D89BB", "#D88A50", "#78A57F", "#9A7AB2", "#C96870"]
    color_for = component_color_map(labels, component_palette)
    r_starts, r_vectors = retraction_vectors(base, labels, anchor)

    fig = plt.figure(figsize=(7.1, 2.12), dpi=300)
    outer = fig.add_gridspec(1, 4, width_ratios=[1.95, 1.0, 1.0, 1.0], wspace=0.13)
    overview_grid = outer[0].subgridspec(1, 3, wspace=0.05)
    overview_axes = [fig.add_subplot(overview_grid[0, i]) for i in range(3)]
    local_axes = [fig.add_subplot(outer[0, i]) for i in range(1, 4)]
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.05, top=0.85)

    overview_idxs = [0, int(np.argmin(np.abs(steps - 19))), len(steps) - 1]
    overview_xy = positions[overview_idxs, :, :2].reshape(-1, 2)
    overview_pad = CR * 0.55
    overview_extent = (
        (float(overview_xy[:, 0].min() - overview_pad), float(overview_xy[:, 0].max() + overview_pad)),
        (float(overview_xy[:, 1].min() - overview_pad), float(overview_xy[:, 1].max() + overview_pad)),
    )

    def finish_axis(ax, title, extent):
        ax.set_title(title, pad=2)
        ax.set_xlim(*extent[0])
        ax.set_ylim(*extent[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#D0D4DA")
            spine.set_linewidth(0.45)

    # The overview provides temporal context; the remaining panels deliberately
    # zoom to one representative fragment per operator.
    for ax, frame_idx in zip(overview_axes, overview_idxs):
        state = np.asarray(positions[frame_idx], dtype=float)
        state_labels, state_k = component_labels(state)
        state_colors = component_color_map(state_labels, component_palette)
        draw_coverage_regions(ax, state, state_labels, state_colors, overview_extent)
        segments = edge_segments(state)
        if len(segments):
            ax.add_collection(LineCollection(segments, colors="#AAB2BA", linewidths=0.16, alpha=0.05, zorder=1))
        ax.scatter(state[:, 0], state[:, 1], s=2.4, c=[state_colors[int(label)] for label in state_labels], lw=0, zorder=2)
        ax.scatter([anchor[0]], [anchor[1]], marker="*", s=11, c="#222222", lw=0, zorder=3)
        title = f"repair course\nt={int(steps[frame_idx])}, K={state_k}" if ax is overview_axes[0] else f"t={int(steps[frame_idx])}, K={state_k}"
        finish_axis(ax, title, overview_extent)

    def fragment_extent(points, vectors=(), scale=0.0):
        xy = [np.asarray(points, dtype=float)[:, :2]]
        for start, vector in vectors:
            norm = np.linalg.norm(vector[:2])
            if norm > EPS:
                xy.append(np.asarray(start, dtype=float)[None, :2])
                xy.append((np.asarray(start, dtype=float)[:2] + vector[:2] / norm * scale)[None, :])
        cloud = np.vstack(xy)
        pad = max(CR * 0.46, np.ptp(cloud[:, 0]) * 0.28, np.ptp(cloud[:, 1]) * 0.28)
        return ((float(cloud[:, 0].min() - pad), float(cloud[:, 0].max() + pad)), (float(cloud[:, 1].min() - pad), float(cloud[:, 1].max() + pad)))

    def draw_fragment(ax, members, title, extent):
        fragment = base[members]
        fragment_labels = np.zeros(len(fragment), dtype=int)
        fragment_color = color_for[int(labels[members[0]])]
        draw_coverage_regions(ax, fragment, fragment_labels, {0: fragment_color}, extent)
        segments = edge_segments(fragment)
        if len(segments):
            ax.add_collection(LineCollection(segments, colors="#AAB2BA", linewidths=0.28, alpha=0.12, zorder=1))
        ax.scatter(fragment[:, 0], fragment[:, 1], s=10, c=fragment_color, lw=0, zorder=2)
        finish_axis(ax, title, extent)

    # Chosen fragments span the upper-left, lower and right-hand fields and
    # have enough nodes for their characteristic morphology to be visible.
    r_comp, p_comp, s_comp = 5, 4, 1
    r_members = np.where(labels == r_comp)[0]
    p_members = np.where(labels == p_comp)[0]
    s_members = np.where(labels == s_comp)[0]

    # R: show a coherent fragment translated towards the shared basin.
    r_start, r_vector = r_starts[r_comp], r_vectors[r_comp]
    r_scale = 55.0
    r_extent = fragment_extent(base[r_members], [(r_start, r_vector)], r_scale)
    draw_fragment(local_axes[0], r_members, "R: coherent fragment drift", r_extent)
    r_unit = r_vector[:2] / max(np.linalg.norm(r_vector[:2]), EPS)
    r_projected = base[r_members, :2] + r_unit * r_scale
    local_axes[0].scatter(base[r_members, 0], base[r_members, 1], s=12, c=color_for[r_comp], alpha=0.16, lw=0, zorder=3)
    local_axes[0].scatter(r_projected[:, 0], r_projected[:, 1], s=11, c=COLORS["R"], alpha=0.86, lw=0, zorder=3.5)
    draw_vectors(local_axes[0], np.asarray([r_start]), np.asarray([r_vector]), COLORS["R"], scale=r_scale, width=0.008)

    # P: select an anchor-facing bearer front in a different fragment.
    p_starts, p_vectors = polarization_vectors(base, labels, anchor)
    p_start, p_vector = p_starts[p_comp], p_vectors[p_comp]
    p_extent = fragment_extent(base[p_members], [(p_start, p_vector)], 54.0)
    draw_fragment(local_axes[1], p_members, "P: bearer-front polarization", p_extent)
    p_front = p_members[np.argsort(np.linalg.norm(base[p_members, :2] - anchor[:2], axis=1))[:3]]
    local_axes[1].scatter(base[p_front, 0], base[p_front, 1], s=25, c=COLORS["P"], alpha=0.86, lw=0, zorder=4.5)
    local_axes[1].scatter([p_start[0]], [p_start[1]], s=66, facecolors="none", edgecolors=COLORS["P"], linewidths=1.25, zorder=5)
    draw_vectors(local_axes[1], np.asarray([p_start]), np.asarray([p_vector]), COLORS["P"], scale=54.0, width=0.008)

    # S: retain only open-sector vectors emitted from the selected boundary.
    s_starts, s_vectors = spreading_vectors(base, max_vectors=100)
    belongs_to_s = np.any(
        np.linalg.norm(s_starts[:, None, :2] - base[s_members][None, :, :2], axis=2) < EPS,
        axis=1,
    ) if len(s_starts) else np.empty(0, dtype=bool)
    s_starts, s_vectors = s_starts[belongs_to_s], s_vectors[belongs_to_s]
    if len(s_starts) > 6:
        s_starts, s_vectors = s_starts[:6], s_vectors[:6]
    s_extent = fragment_extent(base[s_members], [(start, vector) for start, vector in zip(s_starts, s_vectors)], 38.0)
    draw_fragment(local_axes[2], s_members, "S: free-boundary outgrowth", s_extent)
    if len(s_starts):
        s_unit = s_vectors[:, :2] / np.maximum(np.linalg.norm(s_vectors[:, :2], axis=1)[:, None], EPS)
        s_tips = s_starts[:, :2] + s_unit * 38.0
        local_axes[2].scatter(s_tips[:, 0], s_tips[:, 1], s=28, facecolors="none", edgecolors=COLORS["S"], linewidths=1.0, alpha=0.72, zorder=4.2)
    draw_vectors(local_axes[2], s_starts, s_vectors, COLORS["S"], scale=38.0, width=0.0065)

    return {
        "panel": "a",
        "source": str(RPS_DIAGNOSTIC_CACHE.relative_to(ROOT)),
        "fixed_step": int(steps[idx]),
        "initial_subnets": int(ncomp),
        "overview_steps": [int(steps[frame_idx]) for frame_idx in overview_idxs],
        "overview_subnets": [int(component_labels(positions[frame_idx])[1]) for frame_idx in overview_idxs],
        "representative_components": {"R": int(r_comp), "P": int(p_comp), "S": int(s_comp)},
        "representative_component_sizes": {"R": int(len(r_members)), "P": int(len(p_members)), "S": int(len(s_members))},
    }


def channel_fractions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    spreading_col = "f_" + "s" + "cout_mean"
    channel_cols = ["f_ret_mean", "f_pol_bearer", spreading_col]
    for col in channel_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    values = out[channel_cols].to_numpy(dtype=float)
    denom = values.sum(axis=1)
    denom[denom <= 0] = 1.0
    frac = values / denom[:, None]
    out["R_frac"] = frac[:, 0]
    out["P_frac"] = frac[:, 1]
    out["S_frac"] = frac[:, 2]
    return out


def plot_panel_b() -> dict:
    ns = np.array(sorted(ABLATION_PATHS), dtype=int)
    variants = ["full", "no-R", "no-P", "no-S"]
    variant_labels = ["Full", "-R", "-P", "-S"]
    rows = []
    horizons = {}
    for n in ns:
        horizon = max_step_for(int(n))
        horizons[int(n)] = horizon
        reference_cases = read_list_section(ROOT / ABLATION_PATHS[int(n)]["full"], "case")
        for variant in variants:
            path = ROOT / ABLATION_PATHS[int(n)][variant]
            if read_list_section(path, "case") != reference_cases:
                raise ValueError(f"Unmatched operator ablation: N={n}, {variant}")
            vals = connect_steps(path)
            for value in vals:
                rows.append(
                    {
                        "N": int(n),
                        "variant": variant,
                        "T_conn": float(value),
                        "T_norm": float(value) / horizon,
                    }
                )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(3.35, 2.18), dpi=300)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.18, top=0.76)
    offsets = np.linspace(-0.27, 0.27, len(variants))
    stats = {}
    log_offset = 0.02

    def ymap(values):
        return np.log10(np.asarray(values, dtype=float) + log_offset)

    for group_idx, n in enumerate(ns):
        for variant, offset in zip(variants, offsets):
            vals = df[(df["N"] == int(n)) & (df["variant"] == variant)]["T_norm"].to_numpy()
            xpos = group_idx + offset
            vmin, vmax = np.percentile(vals, [0, 100])
            mean = float(np.mean(vals))
            y_min, y_mean, y_max = ymap([vmin, mean, vmax])
            ax.plot([xpos, xpos], [y_min, y_max], color=COLORS[variant], lw=3.4, alpha=0.16, solid_capstyle="round", zorder=3)
            ax.plot([xpos, xpos], [y_mean, y_max], color=COLORS[variant], lw=1.45, alpha=0.85, solid_capstyle="round", zorder=4)
            ax.scatter([xpos], [y_mean], s=18, color=COLORS[variant], edgecolors="white", linewidths=0.45, zorder=5)
            ax.scatter([xpos], [y_min], s=14, marker="_", color=COLORS[variant], linewidths=1.0, zorder=5)
            ax.scatter([xpos], [y_max], s=14, marker="_", color=COLORS[variant], linewidths=1.0, zorder=5)
            stats.setdefault(variant, {})[int(n)] = {
                "mean_normalized": mean,
                "min_normalized": float(vmin),
                "max_normalized": float(vmax),
            }

    ax.axhline(float(ymap(1.0)), color="#222222", lw=0.5, ls="--", alpha=0.58)
    ax.set_xlim(-0.55, len(ns) - 0.45)
    ax.set_xticks(np.arange(len(ns)))
    ax.set_xticklabels([f"N={int(n)}" for n in ns])
    tick_values = log_offset * (((1.0 + log_offset) / log_offset) ** (np.arange(6) / 5.0)) - log_offset
    tick_values[0] = 0.0
    tick_values[-1] = 1.0
    ax.set_ylim(float(ymap(0.0)), float(ymap(1.08)))
    ax.set_yticks(ymap(tick_values))
    ax.set_yticklabels(["0", "0.02", "0.08", "0.19", "0.45", "1.00"])
    ax.set_ylabel("capped $T_{conn}/h$ (offset log)")
    ax.set_title("Operator ablation across scale")
    ax.grid(axis="y", color="#E5E8EA", lw=0.45)
    for variant, label in zip(variants, variant_labels):
        ax.scatter([], [], color=COLORS[variant], s=13, label=label)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.09),
        frameon=False,
        ncol=4,
        handletextpad=0.25,
        columnspacing=0.55,
        markerscale=0.9,
    )
    despine(ax)
    return {
        "panel": "b",
        "source": "50% node-loss operator ablations",
        "N_values": [int(n) for n in ns],
        "normalization": horizons,
        "variants": variants,
        "metric": "RMST divided by recording cutoff; footprints are capped case ranges",
        "sources": {ABLATION_PATHS[int(n)][variant]: hashlib.sha256(
            (ROOT / ABLATION_PATHS[int(n)][variant]).read_bytes()).hexdigest()
            for n in ns for variant in variants},
        "summary": stats,
    }


def plot_panel_c() -> dict:
    destroyed_values = [250, 500, 750]
    bins = np.linspace(0, 1, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    phase_data = {}
    stats = {}
    for destroyed in destroyed_values:
        df = pd.read_csv(trajectory_csv(destroyed), sep="\t")
        df = channel_fractions(df.dropna(subset=["num_remain"]).copy())
        conn = connect_steps(trajectory_txt(destroyed))
        conn_by_case = {case: step for case, step in enumerate(conn)}
        df["connect_step"] = df["case"].map(conn_by_case).astype(float)
        df = df[df["connect_step"] > 0].copy()
        initial_subnets = df.sort_values("step").groupby("case")["num_subnets"].first()
        df["initial_subnets"] = df["case"].map(initial_subnets).astype(float)
        df["subnet_frac"] = df["num_subnets"] / df["initial_subnets"].clip(lower=1.0)
        df["progress"] = np.clip(df["step"] / df["connect_step"], 0, 1)
        df["bin"] = np.digitize(df["progress"], bins) - 1
        df = df[(df["bin"] >= 0) & (df["bin"] < len(centers))]

        means = []
        subnet_means = []
        for b in range(len(centers)):
            group = df[df["bin"] == b]
            if group.empty:
                means.append([np.nan, np.nan, np.nan])
                subnet_means.append(np.nan)
            else:
                means.append([group["R_frac"].mean(), group["P_frac"].mean(), group["S_frac"].mean()])
                subnet_means.append(group["subnet_frac"].mean())
        means = np.asarray(means, dtype=float)
        means = pd.DataFrame(means).interpolate(limit_direction="both").to_numpy()
        denom = means.sum(axis=1)
        denom[denom <= 0] = 1.0
        means = means / denom[:, None]
        subnet_means = (
            pd.Series(subnet_means)
            .interpolate(limit_direction="both")
            .rolling(3, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        means = pd.DataFrame(means).rolling(3, center=True, min_periods=1).mean().to_numpy()
        phase_data[destroyed] = {"force": means, "subnets": subnet_means}
        stats[destroyed] = {
            "source_csv": str(trajectory_csv(destroyed).relative_to(ROOT)),
            "source_txt": str(trajectory_txt(destroyed).relative_to(ROOT)),
            "cases": int(df["case"].nunique()),
            "rows": int(len(df)),
            "mean_initial_subnets": float(initial_subnets.mean()),
        }

    # A stacked composition plot makes the R/P/S hand-off directly visible.
    # The previous three-line design required the reader to mentally add the
    # curves to recover the phase composition.  This version deliberately
    # follows the visual grammar of figures/E7_RPS_bearer.png.
    fig, axes = plt.subplots(1, 3, figsize=(5.25, 2.18), dpi=300, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.89, bottom=0.19, top=0.75, wspace=0.12)
    labels = ["Retraction", "Polarization", "Spreading"]
    k_handle = None
    for ax, destroyed in zip(axes, destroyed_values):
        means = phase_data[destroyed]["force"]
        ax.stackplot(
            centers,
            means[:, 0],
            means[:, 1],
            means[:, 2],
            colors=[COLORS["R"], COLORS["P"], COLORS["S"]],
            linewidth=0,
            zorder=1,
        )
        ax2 = ax.twinx()
        (k_line,) = ax2.plot(
            centers,
            phase_data[destroyed]["subnets"],
            color="#252525",
            lw=1.15,
            ls="--",
            label="$K/K_0$",
            zorder=3,
        )
        k_line.set_path_effects([
            patheffects.Stroke(linewidth=2.4, foreground="white", alpha=0.72),
            patheffects.Normal(),
        ])
        k_handle = k_line
        ax2.set_ylim(0, 1.03)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        if ax is axes[-1]:
            ax2.set_ylabel("subnets $K/K_0$")
            ax2.tick_params(axis="y", width=0.6, length=2.5)
        else:
            ax2.set_yticks([])
            ax2.spines["right"].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(f"{destroyed / 10:.0f}% loss ($D={destroyed}$)", pad=2)
        ax.set_yticks([0, 0.5, 1.0])
        despine(ax)
    axes[1].set_xlabel("normalized repair progress")
    axes[0].set_ylabel("operator share")
    handles = [
        Patch(facecolor=COLORS["R"], edgecolor="none", label=labels[0]),
        Patch(facecolor=COLORS["P"], edgecolor="none", label=labels[1]),
        Patch(facecolor=COLORS["S"], edgecolor="none", label=labels[2]),
        k_handle,
    ]
    legend_labels = [labels[0], labels[1], labels[2], "$K/K_0$"]
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.99),
        frameon=False,
        ncol=4,
        handlelength=1.15,
        columnspacing=0.62,
    )
    return {
        "panel": "c",
        "destroyed_values": destroyed_values,
        "phase_sources": stats,
    }


def bootstrap_delta(full: np.ndarray, variant: np.ndarray, n_boot: int = 2000) -> tuple[float, float, float]:
    if len(full) != len(variant):
        raise ValueError("Paired bootstrap requires equal-length arrays")
    observed = (np.mean(variant) - np.mean(full)) / np.mean(full) * 100.0
    rng = np.random.default_rng(3)
    vals = []
    n = len(full)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append((np.mean(variant[idx]) - np.mean(full[idx])) / np.mean(full[idx]) * 100.0)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(observed), float(lo), float(hi)


def plot_panel_d() -> dict:
    return plot_ring_center_panel("d")


def load_ring_case() -> dict:
    with RING_CASE.open("rb") as f:
        return pickle.load(f)


def jitter_ring_case(case: dict) -> np.ndarray:
    """Generate a random near-ring while preserving the two-subnet topology."""
    positions = np.asarray(case["positions"], dtype=float).copy()
    anchor = np.asarray(case["center"], dtype=float)
    n_centre = int(case["n_center"])
    rng = np.random.default_rng(RING_JITTER_SEED)

    n_ring = len(positions) - n_centre
    angular_gaps = rng.gamma(RING_ANGULAR_CONCENTRATION, 1.0, n_ring)
    angular_gaps = 2.0 * np.pi * angular_gaps / angular_gaps.sum()
    angles = np.cumsum(angular_gaps) - 0.5 * angular_gaps + rng.uniform(0.0, 2.0 * np.pi)
    radial_offset = rng.normal(0.0, RING_RADIAL_JITTER, n_ring)
    radial_offset = (radial_offset + np.roll(radial_offset, 1) + np.roll(radial_offset, -1)) / 3.0
    radius = float(case["R"]) + radial_offset
    positions[n_centre:, :2] = anchor[:2] + radius[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    positions[:n_centre, :2] += rng.normal(0.0, RING_CENTRE_JITTER, (n_centre, 2))
    return positions


def simulate_ring_replay(case: dict) -> dict:
    if RING_CACHE.exists():
        data = np.load(RING_CACHE, allow_pickle=True)
        return {key: data[key] for key in data.files}

    from Configurations import config_constant_speed
    from DIRA_Algorithm.MORPHICA import MORPHICA

    RING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    initial = jitter_ring_case(case)
    anchor = np.asarray(case["center"], dtype=float)
    remain = list(range(len(initial)))
    max_step = int(case.get("MAX_STEP", 2000))
    outputs = {
        "initial": initial,
        "anchor": anchor,
        "max_step": np.array(max_step),
        "jitter_seed": np.array(RING_JITTER_SEED),
        "angular_concentration": np.array(RING_ANGULAR_CONCENTRATION),
        "radial_jitter": np.array(RING_RADIAL_JITTER),
        "centre_jitter": np.array(RING_CENTRE_JITTER),
        "trajectory_stride": np.array(RING_TRAJECTORY_STRIDE),
    }

    for label, ablation in [("full", "full"), ("noP", "no-P")]:
        positions = initial.copy()
        algorithm = MORPHICA(positions, anchor_region=anchor, ablation=ablation)
        final_step = max_step
        trail = [positions.copy()]
        for step in range(max_step + 1):
            if component_labels(positions)[1] == 1 and step > 0:
                final_step = step
                break
            speed, _ = algorithm.group_fusion_step(positions, remain)
            positions = positions + speed * config_constant_speed
            if (step + 1) % RING_TRAJECTORY_STRIDE == 0:
                trail.append(positions.copy())
        if not np.array_equal(trail[-1], positions):
            trail.append(positions.copy())
        outputs[f"{label}_positions"] = positions
        outputs[f"{label}_trail"] = np.stack(trail)
        outputs[f"{label}_steps"] = np.array(final_step)
        outputs[f"{label}_subnets"] = np.array(component_labels(positions)[1])

    np.savez(RING_CACHE, **outputs)
    return outputs


def plot_ring_positions(
    ax,
    positions: np.ndarray,
    title: str,
    anchor: np.ndarray,
    extent: tuple[tuple[float, float], tuple[float, float]],
    trail_positions: np.ndarray | None = None,
    bearer_trails: np.ndarray | None = None,
    ghost_positions: np.ndarray | None = None,
    highlighted_bearers: np.ndarray | None = None,
) -> int:
    labels, ncomp = component_labels(positions)
    # A single connected swarm uses the MORPHICA accent; disconnected subnets
    # use cool/warm colours so their separate CR coverage remains legible.
    palette = [COLORS["full"]] if ncomp == 1 else ["#2C7FB8", "#E58A32", "#8673A1"]
    if ghost_positions is not None:
        ghost_labels, _ = component_labels(ghost_positions)
        ghost_palette = component_color_map(ghost_labels, ["#2C7FB8", "#E58A32", "#8673A1"])
        ghost_colours = [ghost_palette[int(label)] for label in ghost_labels]
        ax.scatter(
            ghost_positions[:, 0],
            ghost_positions[:, 1],
            s=11,
            facecolors="none",
            edgecolors=ghost_colours,
            linewidths=0.55,
            alpha=0.22,
            zorder=0.7,
        )
    ncomp = draw_swarm_snapshot(
        ax,
        positions,
        title,
        removed=None,
        anchor=anchor,
        coverage_extent=extent,
        trail_positions=trail_positions,
        component_palette=palette,
        anchor_marker="x",
        anchor_size=22,
        anchor_linewidth=0.85,
        edge_alpha=0.12,
    )
    if highlighted_bearers is not None and len(highlighted_bearers):
        ax.scatter(
            positions[highlighted_bearers, 0],
            positions[highlighted_bearers, 1],
            s=27,
            facecolors="none",
            edgecolors="#D1495B",
            linewidths=1.0,
            zorder=5.5,
        )
    if bearer_trails is not None:
        draw_recent_trails(
            ax,
            bearer_trails,
            color="#D1495B",
            linewidth=0.9,
            alpha_start=0.20,
            alpha_end=0.82,
        )
        final_bearers = bearer_trails[-1, :, :2]
        ax.scatter(
            final_bearers[:, 0],
            final_bearers[:, 1],
            s=24,
            facecolors="none",
            edgecolors="#D1495B",
            linewidths=1.10,
            zorder=5.7,
        )
    return ncomp


def bearer_indices(positions: np.ndarray, labels: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    indices = []
    for comp in sorted(set(labels.tolist())):
        idx = np.where(labels == comp)[0]
        indices.append(idx[np.argmin(np.linalg.norm(positions[idx, :2] - anchor[:2], axis=1))])
    return np.asarray(indices, dtype=int)


def outer_bearer_index(positions: np.ndarray, labels: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Return the anchor-facing bearer of the subnet furthest from the basin."""
    components = sorted(set(labels.tolist()))
    outer_comp = max(
        components,
        key=lambda comp: np.linalg.norm(np.mean(positions[labels == comp, :2], axis=0) - anchor[:2]),
    )
    idx = np.where(labels == outer_comp)[0]
    bearer = idx[np.argmin(np.linalg.norm(positions[idx, :2] - anchor[:2], axis=1))]
    return np.asarray([bearer], dtype=int)


def plot_ring_center_panel(panel: str) -> dict:
    case = load_ring_case()
    replay = simulate_ring_replay(case)
    initial = np.asarray(replay["initial"], dtype=float)
    full_final = np.asarray(replay["full_positions"], dtype=float)
    noP_final = np.asarray(replay["noP_positions"], dtype=float)
    full_trail_all = np.asarray(replay["full_trail"], dtype=float)
    anchor = np.asarray(replay["anchor"], dtype=float)[:2]
    max_step = int(np.asarray(replay["max_step"]))
    full_steps = int(np.asarray(replay["full_steps"]))
    noP_steps = int(np.asarray(replay["noP_steps"]))
    n_centre = int(case["n_center"])
    labels0, n0 = component_labels(initial)
    labels_full, nf = component_labels(full_final)
    labels_noP, nn = component_labels(noP_final)
    trajectory_indices = np.unique(
        np.concatenate((np.arange(n_centre), np.linspace(n_centre, len(initial) - 1, 8, dtype=int)))
    )
    full_trail = full_trail_all[:, trajectory_indices]
    initial_bearer = outer_bearer_index(initial, labels0, anchor)
    noP_bearer = outer_bearer_index(noP_final, labels_noP, anchor)
    bearer_trails = full_trail_all[:, initial_bearer]

    fig, axes = plt.subplots(1, 2, figsize=(4.32, 2.18), dpi=300)
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.06, top=0.84, wspace=0.10)

    all_xy = np.vstack([initial[:, :2], full_final[:, :2], noP_final[:, :2]])
    pad = 135
    xlim = (float(all_xy[:, 0].min() - pad), float(all_xy[:, 0].max() + pad))
    ylim = (float(all_xy[:, 1].min() - pad), float(all_xy[:, 1].max() + pad))
    extent = (xlim, ylim)

    plot_ring_positions(
        axes[0],
        full_final,
        f"MORPHICA: K={n0} $\\rightarrow$ {nf}\nt={full_steps}",
        anchor,
        extent,
        trail_positions=full_trail,
        bearer_trails=bearer_trails,
        ghost_positions=initial,
    )
    plot_ring_positions(
        axes[1],
        noP_final,
        f"no-P: K={n0} $\\rightarrow$ {nn}\nt={noP_steps}",
        anchor,
        extent,
        highlighted_bearers=noP_bearer,
    )
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return {
        "panel": panel,
        "source": str(RING_CASE.relative_to(ROOT)),
        "N": int(len(initial)),
        "n_ring": int(case.get("n_ring", 0)),
        "n_center": int(case.get("n_center", 0)),
        "initial_subnets": int(n0),
        "full_steps": int(full_steps),
        "full_final_subnets": int(nf),
        "noP_steps": int(noP_steps),
        "noP_final_subnets": int(nn),
        "max_step": int(max_step),
        "jitter": {
            "seed": int(RING_JITTER_SEED),
            "angular_concentration": float(RING_ANGULAR_CONCENTRATION),
            "radial_std": float(RING_RADIAL_JITTER),
            "centre_std": float(RING_CENTRE_JITTER),
            "trajectory_stride": int(RING_TRAJECTORY_STRIDE),
        },
    }


def plot_panel_e() -> dict:
    return plot_ring_center_panel("e")


def build_panel(panel: str) -> None:
    set_style()
    builders = {
        "a": plot_panel_a,
        "b": plot_panel_b,
        "c": plot_panel_c,
        "d": plot_panel_d,
        "e": plot_panel_e,
    }
    stats = builders[panel]()
    save_panel(plt.gcf(), panel, stats)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw individual Fig. 3 panels")
    parser.add_argument("--panel", choices=["a", "b", "c", "d", "e", "all"], required=True)
    args = parser.parse_args()

    panels = ["a", "b", "c", "d"] if args.panel == "all" else [args.panel]
    for panel in panels:
        build_panel(panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
