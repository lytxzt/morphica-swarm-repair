#!/usr/bin/env python3
"""Draw the four conceptual panels for manuscript Fig. 1.

The figure is assembled from deterministic Matplotlib primitives so graph
geometry, equations and labels remain editable in the PDF output.

Usage:
  python Paper/plot_fig1.py --panel a
  python Paper/plot_fig1.py --panel all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "Paper" / "figs"
PANEL_DIR = FIG_DIR / "fig1"
BIOLOGICAL_DIR = PANEL_DIR / "biological"

TEXT = "#202428"
MUTED = "#66707A"
EDGE = "#87919A"
LIGHT_EDGE = "#C5CBD1"
R_COLOR = "#C9656B"
P_COLOR = "#F3AE35"
S_COLOR = "#6C8ABD"
EDGE_BIRTH = "#A64A86"
BASIN = "#83ADD3"
FRAGMENT_COLORS = ("#4C78A8", "#59A14F", "#B07AA1")
BIOLOGICAL_ASSETS = {
    "R": BIOLOGICAL_DIR / "retraction.png",
    "P": BIOLOGICAL_DIR / "polarization.png",
    "S": BIOLOGICAL_DIR / "spreading.png",
}

PANEL_SIZES = {
    "a": (3.10, 1.95),
    "b": (3.85, 1.95),
    "c": (4.50, 2.85),
    "d": (2.55, 2.85),
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "savefig.facecolor": "white",
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.0,
        }
    )


def setup_axis(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("auto")
    ax.axis("off")


def panel_heading(ax: plt.Axes, letter: str, title: str, x: float, y: float) -> None:
    ax.text(x, y, letter, fontsize=9, fontweight="bold", color=TEXT, va="top")
    ax.text(x + 0.55, y, title, fontsize=8, fontweight="bold", color=TEXT, va="top")


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = TEXT,
    linewidth: float = 1.0,
    mutation_scale: float = 8,
    connectionstyle: str = "arc3",
    linestyle: str = "-",
    zorder: int = 4,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_node(
    ax: plt.Axes,
    xy: tuple[float, float],
    *,
    color: str,
    radius: float = 0.16,
    edgecolor: str | None = None,
    linewidth: float = 0.75,
    zorder: int = 5,
) -> None:
    ax.scatter(
        [xy[0]],
        [xy[1]],
        s=(radius * 50.0) ** 2,
        facecolor=color,
        edgecolor=edgecolor or TEXT,
        linewidth=linewidth,
        zorder=zorder,
    )


def draw_graph(
    ax: plt.Axes,
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    color: str,
    radius: float = 0.15,
    edge_color: str = EDGE,
    edge_width: float = 0.75,
    node_edge: str | None = None,
) -> None:
    for i, j in edges:
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color=edge_color,
            linewidth=edge_width,
            zorder=2,
        )
    for xy in positions:
        draw_node(ax, tuple(xy), color=color, radius=radius, edgecolor=node_edge)


def draw_domain(ax: plt.Axes, vertices: np.ndarray, color: str, alpha: float = 0.10) -> None:
    ax.add_patch(
        Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=0.65,
            alpha=alpha,
            zorder=0,
        )
    )


def draw_basin(ax: plt.Axes, xy: tuple[float, float], scale: float = 1.0) -> None:
    for radius, alpha in ((0.54, 0.08), (0.34, 0.13), (0.15, 0.25)):
        ax.add_patch(
            Circle(
                xy,
                radius * scale,
                facecolor=BASIN,
                edgecolor="none",
                alpha=alpha,
                zorder=0,
            )
        )
    ax.add_patch(
        Circle(
            xy,
            0.54 * scale,
            facecolor="none",
            edgecolor=BASIN,
            linewidth=0.6,
            linestyle=(0, (2, 2)),
            alpha=0.75,
            zorder=1,
        )
    )


def draw_tissue(
    ax: plt.Axes,
    centre: tuple[float, float],
    *,
    healed: bool,
    scale: float = 1.0,
) -> list[tuple[float, float]]:
    cx, cy = centre
    cells: list[tuple[float, float]] = []
    for row in range(5):
        y = cy + (row - 2) * 0.55 * scale
        for col in range(7):
            x = cx + (col - 3) * 0.53 * scale + (row % 2) * 0.26 * scale
            if not healed and abs(x - cx) < 0.58 * scale:
                continue
            cells.append((x, y))

    width = 3.75 * scale
    height = 2.75 * scale
    ax.add_patch(
        Ellipse(
            (cx, cy),
            width,
            height,
            facecolor="#F8EAF1",
            edgecolor="#D9A8BF",
            linewidth=0.7,
            zorder=0,
        )
    )
    if not healed:
        ax.add_patch(
            Polygon(
                np.asarray(
                    [
                        (cx - 0.38 * scale, cy + 1.42 * scale),
                        (cx + 0.55 * scale, cy + 1.42 * scale),
                        (cx + 0.31 * scale, cy + 0.25 * scale),
                        (cx + 0.47 * scale, cy - 1.42 * scale),
                        (cx - 0.46 * scale, cy - 1.42 * scale),
                        (cx - 0.22 * scale, cy - 0.18 * scale),
                    ]
                ),
                closed=True,
                facecolor="white",
                edgecolor="#D9A8BF",
                linewidth=0.7,
                zorder=1,
            )
        )

    for cell_idx, (x, y) in enumerate(cells):
        is_new = healed and abs(x - cx) < 0.62 * scale
        face = "#ECA4C9" if is_new else "#F6D9E7"
        edge = "#9B4D73" if is_new else "#B56C8E"
        radius = 0.225 * scale
        rng = np.random.default_rng(1000 + cell_idx + (200 if healed else 0))
        angles = np.linspace(0, 2 * np.pi, 9)[:-1]
        radii = radius * (1.0 + rng.uniform(-0.11, 0.11, size=angles.size))
        vertices = np.column_stack((x + radii * np.cos(angles), y + radii * np.sin(angles)))
        ax.add_patch(
            Polygon(vertices, closed=True, facecolor=face, edgecolor=edge, linewidth=0.55, zorder=3)
        )
        ax.add_patch(
            Circle(
                (x + rng.uniform(-0.035, 0.035) * scale, y + rng.uniform(-0.025, 0.025) * scale),
                0.055 * scale,
                facecolor="#8E5273",
                edgecolor="none",
                alpha=0.72,
                zorder=4,
            )
        )
    return cells


def draw_panel_a(ax: plt.Axes) -> None:
    setup_axis(ax, (0, 10), (0, 6.3))
    panel_heading(ax, "a", "Morphogenesis with material sources", 0.05, 6.18)
    ax.text(
        5.0,
        5.25,
        r"$\partial_t\rho+\nabla\!\cdot J=\sigma,\quad \sigma\ne0$",
        ha="center",
        va="center",
        fontsize=8.5,
        color=TEXT,
    )

    draw_tissue(ax, (2.45, 2.70), healed=False, scale=0.78)
    draw_tissue(ax, (7.72, 2.70), healed=True, scale=0.78)
    add_arrow(ax, (4.20, 2.70), (5.75, 2.70), linewidth=1.0, mutation_scale=9)

    add_arrow(ax, (1.85, 3.00), (2.32, 2.92), color="#8E5273", linewidth=0.8)
    add_arrow(ax, (3.18, 2.45), (2.73, 2.55), color="#8E5273", linewidth=0.8)
    ax.text(2.50, 4.12, r"migration flux $J$", ha="center", fontsize=6.7, color=MUTED)
    ax.annotate(
        r"cell source $\sigma$",
        xy=(7.60, 3.00),
        xytext=(8.70, 4.08),
        fontsize=6.7,
        color="#8E5273",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#8E5273", lw=0.65),
    )
    ax.text(2.45, 0.70, "damaged tissue", ha="center", fontsize=6.8, color=MUTED)
    ax.text(7.72, 0.70, "restored continuity", ha="center", fontsize=6.8, color=MUTED)


def fragment_geometries() -> list[tuple[np.ndarray, list[tuple[int, int]], np.ndarray]]:
    frag1 = np.asarray([(1.15, 3.35), (1.75, 4.20), (2.85, 4.15), (3.35, 3.28), (2.45, 2.62), (1.35, 2.55)])
    edge1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4), (2, 4)]
    hull1 = np.asarray([(0.70, 2.25), (1.42, 4.65), (3.05, 4.55), (3.70, 3.25), (2.55, 2.18)])

    frag2 = np.asarray([(5.15, 1.78), (5.85, 2.48), (6.80, 2.18), (7.35, 1.40), (6.20, 0.98)])
    edge2 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4), (1, 3)]
    hull2 = np.asarray([(4.72, 1.62), (5.65, 2.90), (7.05, 2.60), (7.78, 1.28), (6.08, 0.62)])

    frag3 = np.asarray([(9.30, 3.35), (9.95, 4.30), (11.05, 4.52), (12.12, 3.86), (11.88, 2.72), (10.75, 2.52), (9.72, 2.38)])
    edge3 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0), (1, 5), (2, 5), (0, 5)]
    hull3 = np.asarray([(8.85, 3.25), (9.63, 4.72), (11.25, 4.90), (12.55, 4.00), (12.20, 2.35), (10.45, 2.10), (9.25, 2.15)])
    return [(frag1, edge1, hull1), (frag2, edge2, hull2), (frag3, edge3, hull3)]


def draw_panel_b(ax: plt.Axes) -> None:
    setup_axis(ax, (0, 13.2), (0, 6.3))
    panel_heading(ax, "b", "Source-free swarm constraint", 0.05, 6.18)
    ax.text(
        8.15,
        5.62,
        r"$\partial_t\rho+\nabla\!\cdot J=0$",
        ha="center",
        fontsize=8.5,
        color=TEXT,
    )
    ax.text(
        11.25,
        5.62,
        r"$N_{\mathrm{surv}}=\mathrm{constant}$",
        ha="center",
        fontsize=7.2,
        color=MUTED,
    )

    geometries = fragment_geometries()
    for idx, (positions, edges, hull) in enumerate(geometries):
        color = FRAGMENT_COLORS[idx]
        draw_domain(ax, hull, color, alpha=0.09)
        draw_graph(ax, positions, edges, color=color, radius=0.16, node_edge="white")
        representative = positions[1]
        ax.add_patch(
            Circle(
                representative,
                0.86,
                facecolor="none",
                edgecolor=color,
                linewidth=0.6,
                linestyle=(0, (3, 2)),
                alpha=0.62,
                zorder=1,
            )
        )
        add_arrow(
            ax,
            tuple(positions[0]),
            tuple(positions[1]),
            color=color,
            linewidth=0.8,
            linestyle=(0, (2, 2)),
            mutation_scale=7,
        )

    draw_basin(ax, (7.45, 4.23), scale=0.88)
    ax.text(7.45, 5.02, "inherited positional cue", ha="center", fontsize=6.4, color=MUTED)

    ax.plot([3.50, 4.95], [2.92, 2.05], color=R_COLOR, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(4.27, 2.50, r"$\times$", color=R_COLOR, fontsize=11, ha="center", va="center")
    ax.text(4.32, 3.35, "no cross-fragment\nsignalling", color=R_COLOR, fontsize=6.2, ha="center")

    ax.text(1.15, 1.08, "CR-only relays", color=MUTED, fontsize=6.5, ha="left")
    ax.text(12.75, 0.55, "Topology must be regenerated by redistribution", color=TEXT, fontsize=6.8, ha="right")


def draw_morphogenetic_icon(ax: plt.Axes, kind: str, origin: tuple[float, float], color: str) -> None:
    ox, oy = origin
    image = plt.imread(BIOLOGICAL_ASSETS[kind])
    ax.imshow(image, extent=(ox - 1.35, ox + 1.35, oy - 1.03, oy + 1.03), aspect="auto", zorder=1)
    if kind == "R":
        add_arrow(ax, (ox - 0.80, oy + 0.35), (ox - 0.25, oy + 0.08), color=color, linewidth=0.85, mutation_scale=7)
        add_arrow(ax, (ox + 0.82, oy + 0.34), (ox + 0.24, oy + 0.08), color=color, linewidth=0.85, mutation_scale=7)
        add_arrow(ax, (ox, oy - 0.72), (ox, oy - 0.18), color=color, linewidth=0.85, mutation_scale=7)
    elif kind == "P":
        ax.add_patch(Circle((ox + 0.30, oy + 0.22), 0.18, facecolor="none", edgecolor=color, linewidth=1.0, zorder=4))
        add_arrow(ax, (ox + 0.15, oy + 0.22), (ox - 0.48, oy + 0.05), color=color, linewidth=1.0, mutation_scale=8)
    else:
        for start, end in [((-0.68, 0.40), (-1.10, 0.72)), ((0.68, 0.42), (1.12, 0.72)), ((0.54, -0.44), (1.02, -0.76))]:
            add_arrow(
                ax,
                (ox + start[0], oy + start[1]),
                (ox + end[0], oy + end[1]),
                color=color,
                linewidth=0.85,
                mutation_scale=7,
            )


def draw_robot_strategy(ax: plt.Axes, kind: str, origin: tuple[float, float], color: str) -> None:
    ox, oy = origin
    positions = np.asarray([(-0.72, 0.10), (-0.30, 0.68), (0.42, 0.58), (0.78, -0.08), (0.12, -0.55), (-0.56, -0.48)]) + (ox, oy)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4), (2, 4)]
    for i, j in edges:
        ax.plot([positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]], color=EDGE, lw=0.7, zorder=1)
    for idx, xy in enumerate(positions):
        node_color = color if kind == "P" and idx == 0 else "#DCE5EC"
        node_edge = color if kind == "P" and idx == 0 else EDGE
        draw_node(ax, tuple(xy), color=node_color, radius=0.13, edgecolor=node_edge, linewidth=0.55)

    if kind == "R":
        draw_basin(ax, (ox - 1.45, oy), scale=0.45)
        for idx in (1, 2, 4):
            x, y = positions[idx]
            add_arrow(ax, (x, y), (x - 0.78, y), color=color, linewidth=0.9, mutation_scale=7)
    elif kind == "P":
        x, y = positions[0]
        add_arrow(ax, (x, y), (x - 1.02, y + 0.02), color=color, linewidth=1.1, mutation_scale=8)
    else:
        for idx, delta in ((1, (-0.45, 0.42)), (2, (0.35, 0.48)), (3, (0.52, -0.08)), (5, (-0.48, -0.22))):
            x, y = positions[idx]
            add_arrow(ax, (x, y), (x + delta[0], y + delta[1]), color=color, linewidth=0.85, mutation_scale=7)


def draw_operator_column(
    ax: plt.Axes,
    x: float,
    symbol: str,
    name: str,
    morph_label: str,
    movement_label: str,
    color: str,
) -> None:
    ax.add_patch(Circle((x - 0.72, 8.55), 0.25, facecolor=color, edgecolor="white", linewidth=0.7, zorder=5))
    ax.text(x - 0.72, 8.55, symbol, ha="center", va="center", fontsize=7, fontweight="bold", color="white", zorder=6)
    ax.text(x - 0.30, 8.55, name, ha="left", va="center", fontsize=7.2, fontweight="bold", color=TEXT)
    draw_morphogenetic_icon(ax, symbol, (x, 6.66), color)
    ax.text(x, 5.48, morph_label, ha="center", va="top", fontsize=5.8, color=MUTED, linespacing=1.05)
    add_arrow(ax, (x, 4.88), (x, 4.30), color=LIGHT_EDGE, linewidth=0.8, mutation_scale=7)
    draw_robot_strategy(ax, symbol, (x, 3.08), color)
    ax.text(x, 1.94, movement_label, ha="center", va="top", fontsize=5.8, color=TEXT, linespacing=1.05)


def draw_panel_c(ax: plt.Axes) -> None:
    setup_axis(ax, (0, 16), (0, 10))
    panel_heading(ax, "c", "Morphogenetic functions become displacement strategies", 0.05, 9.86)

    draw_operator_column(ax, 3.35, "R", "Retraction", "fragments retract\ntowards the field source", "fragment-wide\nbasinward drift", R_COLOR)
    draw_operator_column(ax, 8.25, "P", "Polarization", "one basin-facing\nfront differentiates", "bearer-only\npolar displacement", P_COLOR)
    draw_operator_column(ax, 13.15, "S", "Spreading", "free boundary grows\ninto open space", "boundary-local\nangular-void expansion", S_COLOR)

    symbols = [(6.35, R_COLOR, "R"), (7.45, P_COLOR, "P"), (8.55, S_COLOR, "S")]
    ax.text(7.45, 0.90, "concurrent combination", ha="center", fontsize=5.7, color=MUTED)
    for idx, (x, color, symbol) in enumerate(symbols):
        ax.add_patch(Circle((x, 0.38), 0.20, facecolor=color, edgecolor="white", linewidth=0.6, zorder=5))
        ax.text(x, 0.38, symbol, ha="center", va="center", fontsize=6, fontweight="bold", color="white", zorder=6)
        if idx < 2:
            ax.text(x + 0.55, 0.38, "+", ha="center", va="center", fontsize=7, color=TEXT)
    add_arrow(ax, (8.90, 0.38), (10.00, 0.38), color=TEXT, linewidth=1.0, mutation_scale=8)
    ax.text(10.18, 0.38, r"movement flux $v_i$", ha="left", va="center", fontsize=6.3, color=TEXT)
def edge_birth_positions() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    left = np.asarray([(1.05, 0.10), (1.40, 0.82), (2.20, 0.74), (2.48, 0.05), (1.72, -0.38)])
    right = np.asarray([(3.50, -0.10), (4.05, 0.52), (4.82, 0.22), (4.48, -0.52), (3.75, -0.62)])
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]
    return left, right, edges, edges


def draw_edge_birth_state(ax: plt.Axes, origin: tuple[float, float], *, fused: bool) -> None:
    ox, oy = origin
    left, right, e1, e2 = edge_birth_positions()
    left = left + (ox, oy)
    right = right + (ox, oy)
    left_hull = np.asarray([(0.62, -0.20), (1.22, 1.12), (2.48, 1.02), (2.82, -0.18), (1.68, -0.76)]) + (ox, oy)
    right_hull = np.asarray([(3.14, -0.28), (3.92, 0.86), (5.16, 0.52), (4.72, -0.90), (3.48, -0.93)]) + (ox, oy)
    if fused:
        fused_hull = np.asarray([(0.62, -0.20), (1.22, 1.12), (2.48, 1.02), (3.92, 0.86), (5.16, 0.52), (4.72, -0.90), (3.48, -0.93), (1.68, -0.76)]) + (ox, oy)
        draw_domain(ax, fused_hull, "#4F9E91", alpha=0.12)
    else:
        draw_domain(ax, left_hull, FRAGMENT_COLORS[0], alpha=0.11)
        draw_domain(ax, right_hull, FRAGMENT_COLORS[2], alpha=0.11)
    draw_graph(ax, left, e1, color=FRAGMENT_COLORS[0], radius=0.14, node_edge="white")
    draw_graph(ax, right, e2, color=FRAGMENT_COLORS[2], radius=0.14, node_edge="white")
    if fused:
        ax.plot([left[3, 0], right[0, 0]], [left[3, 1], right[0, 1]], color=EDGE_BIRTH, lw=2.0, zorder=3)
        add_arrow(
            ax,
            tuple(left[3]),
            tuple(right[0]),
            color=EDGE_BIRTH,
            linewidth=0.9,
            linestyle=(0, (2, 2)),
            mutation_scale=7,
            zorder=4,
        )
    else:
        ax.plot(
            [left[3, 0], right[0, 0]],
            [left[3, 1], right[0, 1]],
            color=LIGHT_EDGE,
            lw=0.7,
            ls=(0, (2, 2)),
            zorder=1,
        )


def draw_panel_d(ax: plt.Axes) -> None:
    setup_axis(ax, (0, 6.2), (0, 10))
    panel_heading(ax, "d", "Edge birth closes the loop", 0.05, 9.86)

    ax.text(0.42, 8.70, r"separate domains, $G(X_t)$", fontsize=6.1, color=MUTED, fontweight="bold")
    draw_edge_birth_state(ax, (0.25, 7.25), fused=False)
    ax.text(3.10, 6.45, "open CR gap", fontsize=5.8, color=MUTED, ha="center", va="top")

    add_arrow(ax, (3.10, 6.20), (3.10, 5.10), color=EDGE_BIRTH, linewidth=1.25, mutation_scale=9)
    ax.text(1.90, 5.70, "RPS movement", color=MUTED, fontsize=5.7, ha="right", va="center")
    ax.text(3.10, 5.00, "communication-edge birth", color=EDGE_BIRTH, fontsize=5.9, ha="center", va="top")

    ax.text(0.42, 4.52, r"fused domain, $G(X_{t+\Delta t})$", fontsize=6.1, color=MUTED, fontweight="bold")
    draw_edge_birth_state(ax, (0.25, 3.05), fused=True)
    add_arrow(
        ax,
        (5.38, 2.62),
        (5.72, 6.72),
        color="#3C746B",
        linewidth=0.85,
        mutation_scale=7,
        connectionstyle="arc3,rad=-0.30",
    )
    ax.text(5.92, 4.72, "signal re-equilibration", rotation=90, fontsize=5.2, color="#3C746B", ha="center", va="center")
    ax.text(5.72, 6.92, r"$z_{t+\Delta t}$", fontsize=6.0, color="#3C746B", ha="center")

    ax.plot([0.70, 5.50], [1.38, 1.38], color=LIGHT_EDGE, lw=0.55)
    ax.text(3.10, 0.94, "graph merge  +  signalling-domain fusion", ha="center", fontsize=6.1, color=TEXT)
    ax.text(3.10, 0.40, "the next signal-mechanics cycle uses the enlarged domain", ha="center", fontsize=5.6, color=MUTED)


PANEL_DRAWERS = {
    "a": draw_panel_a,
    "b": draw_panel_b,
    "c": draw_panel_c,
    "d": draw_panel_d,
}


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    plt.close(fig)


def render_panel(panel: str) -> None:
    fig = plt.figure(figsize=PANEL_SIZES[panel], dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    PANEL_DRAWERS[panel](ax)
    save_figure(fig, PANEL_DIR / f"Fig1{panel}")


def render_combined() -> None:
    fig = plt.figure(figsize=(7.09, 5.12), dpi=300, facecolor="white")
    axes = {
        "a": fig.add_axes([0.015, 0.605, 0.420, 0.380]),
        "b": fig.add_axes([0.450, 0.605, 0.535, 0.380]),
        "c": fig.add_axes([0.015, 0.020, 0.625, 0.550]),
        "d": fig.add_axes([0.660, 0.020, 0.325, 0.550]),
    }
    for panel, ax in axes.items():
        PANEL_DRAWERS[panel](ax)
    save_figure(fig, FIG_DIR / "Fig1_source_free_morphogenesis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=["a", "b", "c", "d", "combined", "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    if args.panel in PANEL_DRAWERS:
        render_panel(args.panel)
        return
    if args.panel == "combined":
        render_combined()
        return
    for panel in PANEL_DRAWERS:
        render_panel(panel)
    render_combined()


if __name__ == "__main__":
    main()
