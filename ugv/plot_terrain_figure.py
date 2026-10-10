#!/usr/bin/env python3
"""Generate 1×4 terrain recovery trajectory figure for the UGV paper.

One panel per terrain (ridge, rolling, roughland, mixed), each showing:
- terrain heightmap background
- destroyed agents (grey ×)
- surviving agent trajectory paths from t=0 to T_conn
- initial (t=0) and final (t=T_conn) positions, coloured by component
- CR edges at final step (thin lines)

Usage:
    python ugv/plot_terrain_figure.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from terrain import TerrainMap

TRAJECTORY_DIR = Path('data/logs/ugv/terrain_trajectories')
OUTPUT_DIR = Path('data/logs/ugv/figures')
TERRAINS = ['ridge', 'rolling', 'roughland', 'mixed']
TERRAIN_LABELS = {'ridge': 'Ridge', 'rolling': 'Rolling',
                  'roughland': 'Roughland', 'mixed': 'Mixed'}
SEED = 7


def load_trajectory(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data['metadata'], data['frames']


def load_terrain(meta, trajectory_path):
    terrain_dir = meta.get('terrain_dir', '')
    td = Path(trajectory_path).parent / terrain_dir if terrain_dir else None
    bounds = np.array([[0.0, float(meta.get('width', 750))],
                       [0.0, float(meta.get('length', 750))]])
    if td is not None and td.exists():
        meta_path = td / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                tmeta = json.load(f)
            bounds = np.array(tmeta['bounds'])
    return TerrainMap(meta['terrain'], bounds,
                      seed=meta.get('terrain_seed', None))


def main():
    N_PANELS = len(TERRAINS)

    fig = plt.figure(figsize=(20, 5.2))
    # Use GridSpec for 1 row × 4 cols with room for colorbar
    gs = plt.GridSpec(1, N_PANELS, figure=fig, wspace=0.06,
                      left=0.03, right=0.93, top=0.88, bottom=0.08)

    cmap_comp = plt.cm.tab10
    traj_alphas = [0.25, 0.40, 0.35, 0.30]  # vary so overlapping paths differ

    for col, terrain_name in enumerate(TERRAINS):
        json_path = TRAJECTORY_DIR / (
            f'case_{terrain_name}_random_MORPHICA_N100_D50_r50_cr120'
            f'_w1.2_seed{SEED}.json')
        if not json_path.exists():
            print(f'Missing: {json_path}', file=sys.stderr)
            continue
        meta, frames = load_trajectory(json_path)
        terrain = load_terrain(meta, json_path)

        N = meta['N']
        T_conn = meta['summary'].get('T_conn', len(frames) - 1)
        K0 = meta['summary'].get('K_initial', '?')
        T_conn = min(T_conn, len(frames) - 1)

        extent = [terrain.bounds[0, 0], terrain.bounds[0, 1],
                  terrain.bounds[1, 0], terrain.bounds[1, 1]]

        ax = fig.add_subplot(gs[0, col])

        # ── terrain heightmap ──
        hm = ax.imshow(terrain._heightmap, origin='lower', extent=extent,
                       cmap='terrain', interpolation='bilinear', alpha=0.82,
                       aspect='equal')

        # ── hard hazards as building outlines (contour, not grid cells) ──
        if terrain._obstacle_mask.any():
            # Build coordinate grids for contour tracing
            xs = np.linspace(terrain.bounds[0, 0], terrain.bounds[0, 1],
                             terrain._obstacle_mask.shape[1] + 1)
            ys = np.linspace(terrain.bounds[1, 0], terrain.bounds[1, 1],
                             terrain._obstacle_mask.shape[0] + 1)
            # pad mask edges with 0 so contour closes at boundary
            padded = np.pad(terrain._obstacle_mask.astype(np.float64), 1,
                            mode='constant', constant_values=0.0)
            xp = np.linspace(terrain.bounds[0, 0] - terrain.resolution,
                             terrain.bounds[0, 1] + terrain.resolution,
                             padded.shape[1])
            yp = np.linspace(terrain.bounds[1, 0] - terrain.resolution,
                             terrain.bounds[1, 1] + terrain.resolution,
                             padded.shape[0])
            ax.contourf(xp, yp, padded, levels=[0.5, 1.0],
                        colors=['#aa3333'], alpha=0.30, antialiased=True)

        # ── trajectory paths (thin lines, t=0 → T_conn) ──
        # Trace one agent per initial component for visual clarity
        init_comp = np.array(frames[0]['component_id'])
        init_remain = set(frames[0]['remain_list'])
        final_comp = np.array(frames[min(T_conn, len(frames)-1)]['component_id'])
        final_remain = set(frames[min(T_conn, len(frames)-1)]['remain_list'])

        # Pick representative agents: one from each initial component
        traced = set()
        for cid in sorted(set(init_comp)):
            candidates = [i for i in range(N)
                          if i in init_remain and init_comp[i] == cid]
            if candidates:
                traced.add(candidates[len(candidates) // 2])

        # Draw paths
        for i in sorted(traced):
            pts = []
            for f in frames[:T_conn + 1]:
                if i in f['remain_list']:
                    pts.append(f['positions'][i][:2])
            if len(pts) < 2:
                continue
            pts = np.array(pts)
            alpha = traj_alphas[i % len(traj_alphas)]
            ax.plot(pts[:, 0], pts[:, 1], color='white', lw=0.5,
                    alpha=alpha, zorder=1, solid_joinstyle='round')

        # ── destroyed agents (t=0) ──
        destroyed = [i for i in range(N) if i not in init_remain]
        if destroyed:
            pos0 = np.array(frames[0]['positions'])
            ax.scatter(pos0[destroyed, 0], pos0[destroyed, 1],
                       c='#444444', s=6, marker='x', alpha=0.35,
                       zorder=2, linewidths=0.4)

        # ── initial positions (small, semitransparent circles) ──
        pos0 = np.array(frames[0]['positions'])
        for cid in sorted(set(init_comp)):
            mask = [(i in init_remain and init_comp[i] == cid) for i in range(N)]
            if not any(mask):
                continue
            ax.scatter(pos0[mask, 0], pos0[mask, 1],
                       c=[cmap_comp(cid % 10)], s=12, marker='o',
                       edgecolors='#333333', linewidths=0.1,
                       alpha=0.45, zorder=3)

        # ── final positions (larger, opaque markers) ──
        fr_end = frames[T_conn]
        pos_end = np.array(fr_end['positions'])
        for cid in sorted(set(final_comp)):
            mask = [(i in final_remain and final_comp[i] == cid)
                    for i in range(N)]
            if not any(mask):
                continue
            ax.scatter(pos_end[mask, 0], pos_end[mask, 1],
                       c=[cmap_comp(cid % 10)], s=22, marker='o',
                       edgecolors='#111111', linewidths=0.3,
                       alpha=0.90, zorder=4)

        # ── CR edges at final step ──
        edges = fr_end.get('edges', [])
        if edges:
            for a, b in edges[:2000]:  # cap at 2000 edges
                pa, pb = pos_end[a, :2], pos_end[b, :2]
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                        color='cyan', lw=0.15, alpha=0.35, zorder=1.5)

        # ── decor ──
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)', fontsize=9, labelpad=2)
        if col == 0:
            ax.set_ylabel('y (m)', fontsize=9)
        else:
            ax.tick_params(axis='y', labelleft=False)

        title = (f'{TERRAIN_LABELS[terrain_name]}\n'
                 f'K(0)={K0}  $T_{{\\mathrm{{conn}}}}$={T_conn}')
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=5)

        # ── compact legend (first panel only) ──
        if col == 0:
            from matplotlib.lines import Line2D
            custom = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markersize=4, markeredgewidth=0,
                       label='initial (t=0)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markersize=6, markeredgewidth=0.3,
                       markeredgecolor='#111111',
                       label=f'final (t={T_conn})'),
                Line2D([0], [0], marker='x', color='#444444',
                       markersize=4, markeredgewidth=0.4,
                       label='destroyed', lw=0),
                Line2D([0], [0], color='white', lw=0.8, alpha=0.6,
                       label='path'),
                Line2D([0], [0], color='cyan', lw=0.3, alpha=0.5,
                       label='CR edge'),
            ]
            ax.legend(handles=custom, fontsize=6.5,
                      loc='lower left', framealpha=0.82,
                      ncol=3, columnspacing=0.8, handletextpad=0.5)

    # ── colorbar for height ──
    cbar_ax = fig.add_axes([0.945, 0.12, 0.008, 0.72])
    cbar = fig.colorbar(hm, cax=cbar_ax)
    cbar.set_label('Height (m)', fontsize=9, labelpad=6)
    cbar.ax.tick_params(labelsize=7)

    # ── global title ──
    fig.suptitle(
        'MORPHICA Source-Free Recovery Across Four Terrains\n'
        f'N=100  D=50  CR=120 m  seed={SEED}  '
        '— trajectories from initial fragmentation to first connectivity',
        fontsize=12.5, fontweight='bold', y=0.985)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt, ext in [('pdf', 'pdf'), ('png', 'png')]:
        p = OUTPUT_DIR / f'Fig4a_terrain_recovery_1x4.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'→ {p}  ({p.stat().st_size // 1024} KB)')
    plt.close(fig)


if __name__ == '__main__':
    main()
