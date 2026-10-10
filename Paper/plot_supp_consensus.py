#!/usr/bin/env python3
"""Supplementary source-free substrate analysis for the MORPHICA manuscript.

This script is a post-processing analysis. It does not modify the controller.
It reconstructs communication graphs from the cached representative Fig. 2
trajectory, infers communication-edge births, and compares distributed
consensus on the fragmented and repaired graph substrates.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "Paper" / "figs" / "fig2_cache" / "representative_MORPHICA_N1000_d500_case0.npz"
OUT_DIR = ROOT / "Paper" / "figs" / "supp"
OUT_STEM = OUT_DIR / "SuppFig_consensus_substrate"


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
            "savefig.pad_inches": 0.02,
        }
    )


def adjacency(positions: np.ndarray, cr: float) -> np.ndarray:
    xyz = np.asarray(positions, dtype=float)
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    adj = dist2 <= cr * cr
    np.fill_diagonal(adj, False)
    return adj


def component_labels_from_adj(adj: np.ndarray) -> tuple[np.ndarray, int]:
    n = adj.shape[0]
    labels = np.full(n, -1, dtype=int)
    comp = 0
    for start in range(n):
        if labels[start] >= 0:
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


def infer_edge_births(positions_by_step: np.ndarray, cr: float) -> dict[str, np.ndarray | int]:
    initial_adj = adjacency(positions_by_step[0], cr)
    initial_labels, initial_k = component_labels_from_adj(initial_adj)
    seen_cross_edges: set[tuple[int, int]] = set()
    births = []
    lower_bound_births = []
    subnet_counts = []
    previous_k = initial_k

    for pos in positions_by_step:
        adj = adjacency(pos, cr)
        _, k = component_labels_from_adj(adj)
        subnet_counts.append(k)

        ii, jj = np.where(np.triu(adj, k=1))
        cross_edges = {
            (int(i), int(j))
            for i, j in zip(ii, jj)
            if initial_labels[i] != initial_labels[j]
        }
        new_births = cross_edges - seen_cross_edges
        births.append(len(new_births))
        lower_bound_births.append(max(previous_k - k, 0))
        seen_cross_edges.update(new_births)
        previous_k = k

    return {
        "initial_k": int(initial_k),
        "subnet_counts": np.asarray(subnet_counts, dtype=int),
        "edge_births": np.asarray(births, dtype=int),
        "lower_bound_births": np.asarray(lower_bound_births, dtype=int),
        "total_edge_births": int(np.sum(births)),
        "total_lower_bound_births": int(np.sum(lower_bound_births)),
    }


def metropolis_weights(adj: np.ndarray) -> np.ndarray:
    deg = adj.sum(axis=1).astype(float)
    w = np.zeros(adj.shape, dtype=float)
    ii, jj = np.where(np.triu(adj, k=1))
    for i, j in zip(ii, jj):
        value = 1.0 / (1.0 + max(deg[i], deg[j]))
        w[i, j] = value
        w[j, i] = value
    row_sum = w.sum(axis=1)
    np.fill_diagonal(w, 1.0 - row_sum)
    return w


def consensus_error_curve(adj: np.ndarray, values: np.ndarray, iterations: int = 200) -> np.ndarray:
    w = metropolis_weights(adj)
    z = values.astype(float).copy()
    target = float(np.mean(values))
    initial_error = float(np.mean((z - target) ** 2))
    if initial_error <= 0:
        initial_error = 1.0
    errors = []
    for _ in range(iterations + 1):
        errors.append(float(np.mean((z - target) ** 2) / initial_error))
        z = w @ z
    return np.asarray(errors, dtype=float)


def main() -> None:
    if not CACHE.exists():
        raise FileNotFoundError(f"Missing trajectory cache: {CACHE}")

    set_style()
    data = np.load(CACHE, allow_pickle=True)
    positions_by_step = np.asarray(data["positions_by_step"], dtype=float)
    steps = np.asarray(data["steps"], dtype=int)
    cr = float(np.asarray(data["cr"]).item())
    connect_step = int(np.asarray(data["connect_step"]).item())

    birth_info = infer_edge_births(positions_by_step, cr)
    subnet_counts = np.asarray(birth_info["subnet_counts"], dtype=int)
    edge_births = np.asarray(birth_info["edge_births"], dtype=int)
    lower_bound_births = np.asarray(birth_info["lower_bound_births"], dtype=int)

    rng = np.random.default_rng(7)
    values = rng.normal(size=positions_by_step.shape[1])
    values = (values - np.mean(values)) / np.std(values)

    adj_initial = adjacency(positions_by_step[0], cr)
    adj_repaired = adjacency(positions_by_step[-1], cr)
    initial_curve = consensus_error_curve(adj_initial, values)
    repaired_curve = consensus_error_curve(adj_repaired, values)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35), dpi=300)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.20, top=0.86, wspace=0.32)

    ax = axes[0]
    ax.plot(steps, subnet_counts, color="#222222", lw=1.2, label="subnet count")
    birth_steps = steps[edge_births > 0]
    birth_values = edge_births[edge_births > 0]
    if len(birth_steps):
        ax.vlines(birth_steps, ymin=0, ymax=birth_values, color="#007C6C", lw=0.8, alpha=0.65, label="edge births")
    ax.set_xlabel("recovery step")
    ax.set_ylabel("count")
    ax.set_title("edge birth restores topology")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    iters = np.arange(len(initial_curve))
    ax.semilogy(iters, initial_curve, color="#CC6677", lw=1.4, label=f"fragmented, K={subnet_counts[0]}")
    ax.semilogy(iters, repaired_curve, color="#007C6C", lw=1.4, label="repaired, K=1")
    ax.set_xlabel("consensus iteration")
    ax.set_ylabel("normalized error")
    ax.set_title("substrate enables consensus")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)

    stats = {
        "source_cache": str(CACHE.relative_to(ROOT)),
        "num_agents": int(positions_by_step.shape[1]),
        "num_steps": int(len(steps)),
        "connect_step": connect_step,
        "initial_subnets": int(subnet_counts[0]),
        "final_subnets": int(subnet_counts[-1]),
        "total_cross_fragment_edge_births": int(birth_info["total_edge_births"]),
        "total_lower_bound_edge_births_from_K": int(birth_info["total_lower_bound_births"]),
        "edge_birth_steps": [int(s) for s in birth_steps.tolist()],
        "edge_birth_counts": [int(v) for v in birth_values.tolist()],
        "lower_bound_birth_steps": [int(s) for s in steps[lower_bound_births > 0].tolist()],
        "initial_consensus_error_at_200": float(initial_curve[-1]),
        "repaired_consensus_error_at_200": float(repaired_curve[-1]),
        "consensus_iterations": int(len(initial_curve) - 1),
    }
    OUT_STEM.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved {OUT_STEM.with_suffix('.pdf')}")
    print(f"Saved {OUT_STEM.with_suffix('.png')}")
    print(f"Saved {OUT_STEM.with_suffix('.stats.json')}")


if __name__ == "__main__":
    main()
