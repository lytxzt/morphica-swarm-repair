#!/usr/bin/env python3
"""Generate diagnostic plots for a single UGV case from saved trajectory."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from terrain import TerrainMap


def load_trajectory(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data['metadata'], data['frames']


def plot_terrain_overview(terrain, meta, save_path):
    """Heightmap + costmap + obstacle mask side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    extent = [terrain.bounds[0, 0], terrain.bounds[0, 1],
              terrain.bounds[1, 0], terrain.bounds[1, 1]]

    # heightmap
    im0 = axes[0].imshow(terrain._heightmap, origin='lower', extent=extent,
                         cmap='terrain', interpolation='bilinear')
    axes[0].set_title('Heightmap (m)', fontsize=12)
    axes[0].set_xlabel('x (m)'); axes[0].set_ylabel('y (m)')
    plt.colorbar(im0, ax=axes[0], shrink=0.78)

    # costmap
    im1 = axes[1].imshow(terrain._costmap, origin='lower', extent=extent,
                         cmap='YlOrRd', interpolation='bilinear',
                         vmin=1.0, vmax=max(terrain._costmap.max(), 2.0))
    axes[1].set_title('Traversal Cost', fontsize=12)
    axes[1].set_xlabel('x (m)'); axes[1].set_ylabel('y (m)')
    plt.colorbar(im1, ax=axes[1], shrink=0.78)

    # slope
    slope = np.arctan(np.sqrt(terrain._grad_x**2 + terrain._grad_y**2)) \
            * 180.0 / np.pi
    im2 = axes[2].imshow(slope, origin='lower', extent=extent,
                         cmap='Reds', interpolation='bilinear', vmin=0, vmax=35)
    axes[2].set_title('Slope (°)', fontsize=12)
    axes[2].set_xlabel('x (m)'); axes[2].set_ylabel('y (m)')
    cbar2 = plt.colorbar(im2, ax=axes[2], shrink=0.78)
    cbar2.ax.axhline(25, color='k', lw=1, ls='--', label='max_slope=25°')

    # overlay obstacles on heightmap
    if terrain._obstacle_mask.any():
        obs_y, obs_x = np.where(terrain._obstacle_mask)
        obs_xy = np.column_stack([
            terrain.bounds[0, 0] + (obs_x + 0.5) * terrain.resolution,
            terrain.bounds[1, 0] + (obs_y + 0.5) * terrain.resolution,
        ])
        axes[0].scatter(obs_xy[:, 0], obs_xy[:, 1], c='red', s=2, alpha=0.6,
                        label='hard hazard')
        axes[0].legend(fontsize=8, loc='upper right')

    name = meta.get('terrain', terrain.name)
    fig.suptitle(f'{name} — N={meta["N"]}  seed={meta["seed_index"]}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  terrain overview → {save_path}')


def plot_trajectory_paths(frames, terrain, meta, save_path):
    """Agent trajectories overlaid on costmap, coloured by component."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    extent = [terrain.bounds[0, 0], terrain.bounds[0, 1],
              terrain.bounds[1, 0], terrain.bounds[1, 1]]

    N = meta['N']
    t_max = len(frames) - 1

    # colour each agent by its final component label
    final = frames[-1]
    comp_labels = np.array(final['component_id'])
    n_comps = max(comp_labels.max() + 1, 2)
    cmap = plt.cm.tab10

    for panel_idx, (ax, label) in enumerate(zip(
        axes,
        [f't=0 (damaged)', f't={t_max // 2}', f't={t_max} (final)']
    )):
        if panel_idx == 0:
            fr = frames[0]
        elif panel_idx == 2:
            fr = frames[-1]
        else:
            fr = frames[t_max // 2]

        # costmap background
        ax.imshow(terrain._costmap, origin='lower', extent=extent,
                  cmap='YlOrRd', interpolation='bilinear',
                  vmin=1.0, vmax=max(terrain._costmap.max(), 2.0), alpha=0.4)

        # hard hazards
        if terrain._obstacle_mask.any():
            obs_y, obs_x = np.where(terrain._obstacle_mask)
            from matplotlib.patches import Rectangle
            for ox, oy in zip(obs_x, obs_y):
                rx = terrain.bounds[0, 0] + ox * terrain.resolution
                ry = terrain.bounds[1, 0] + oy * terrain.resolution
                rect = Rectangle((rx, ry), terrain.resolution,
                                 terrain.resolution, color='red', alpha=0.5,
                                 lw=0)
                ax.add_patch(rect)

        pos = np.array(fr['positions'])
        comp = np.array(fr['component_id'])
        remain = set(fr['remain_list'])

        # destroyed agents
        destroyed = [i for i in range(N) if i not in remain]
        if destroyed:
            ax.scatter(pos[destroyed, 0], pos[destroyed, 1],
                       c='gray', s=8, marker='x', alpha=0.4, label='destroyed')

        # remaining agents coloured by component
        for cid in range(n_comps):
            mask = [(i in remain and comp[i] == cid) for i in range(N)]
            if not any(mask):
                continue
            ax.scatter(pos[mask, 0], pos[mask, 1],
                       c=[cmap(cid % 10)], s=25, edgecolors='black',
                       linewidths=0.3, zorder=5,
                       label=f'comp {cid}' if panel_idx == 0 else None)

        # path traces for a subset
        if panel_idx > 0:
            for i in range(N):
                if i not in remain:
                    continue
                xs = [f['positions'][i][0]
                      for f in frames[:fr['step'] + 1]]
                ys = [f['positions'][i][1]
                      for f in frames[:fr['step'] + 1]]
                ax.plot(xs, ys, color=cmap(comp[i] % 10), lw=0.3,
                        alpha=0.5)

        ax.set_title(label, fontsize=12)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        if panel_idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(f'{meta["algorithm"]}  {meta["terrain"]}  '
                 f'N={meta["N"]}  D={meta["D"]}  '
                 f'T_conn={meta["summary"].get("T_conn", "FAILED")}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  trajectory paths → {save_path}')


def plot_connectivity_over_time(frames, meta, save_path):
    """K(t), component sizes, and edge count over time."""
    steps = [f['step'] for f in frames]
    K = np.array([max(np.array(f['component_id']).max() + 1, 1)
                   for f in frames])
    n_edges = np.array([len(f['edges']) for f in frames])
    n_remain = np.array([len(f['remain_list']) for f in frames])

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # K(t)
    axes[0].plot(steps, K, 'b-', lw=1.5)
    axes[0].axhline(1, color='green', ls='--', lw=1, label='connected')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('K(t)')
    axes[0].set_title('Connected Components')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # edge count
    axes[1].plot(steps, n_edges, 'gray', lw=1.5)
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('|E|')
    axes[1].set_title('CR Edge Count')
    axes[1].grid(True, alpha=0.3)

    # component size distribution (stacked bar at sampled steps)
    sample_steps = np.linspace(0, len(frames) - 1, min(10, len(frames)),
                               dtype=int)
    comp_sizes = []
    for si in sample_steps:
        fr = frames[si]
        remain = set(fr['remain_list'])
        comp = np.array(fr['component_id'])
        sizes = []
        for cid in range(K[si]):
            sizes.append(sum((comp == cid) & np.array(
                [i in remain for i in range(len(comp))])))
        sizes.sort(reverse=True)
        comp_sizes.append(sizes)

    max_sizes = max(len(s) for s in comp_sizes)
    for i, sizes in enumerate(comp_sizes):
        sizes += [0] * (max_sizes - len(sizes))
    comp_sizes = np.array(comp_sizes)
    bottom = np.zeros(len(sample_steps))
    cmap = plt.cm.tab10
    for ci in range(max_sizes):
        axes[2].bar(sample_steps, comp_sizes[:, ci], bottom=bottom,
                    color=cmap(ci % 10), label=f'comp {ci}' if ci < 8 else None,
                    width=max(1, (steps[-1] - steps[0]) / 30))
        bottom += comp_sizes[:, ci]
    axes[2].set_xlabel('Step'); axes[2].set_ylabel('Agents')
    axes[2].set_title('Component Size Distribution')
    if max_sizes <= 8:
        axes[2].legend(fontsize=7, loc='upper right')

    status = 'CONNECTED' if meta['summary']['success'] else 'FAILED'
    fig.suptitle(f'{meta["algorithm"]}  {meta["terrain"]}  '
                 f'N={meta["N"]}  D={meta["D"]}  '
                 f'{status} at T={meta["summary"].get("T_conn", steps[-1])}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  connectivity → {save_path}')


def main():
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        # default: pick the most recent roughland case
        import glob
        candidates = sorted(glob.glob(
            'data/logs/ugv/cases/case_roughland_*MORPHICA*.json'))
        if not candidates:
            print('No trajectory found. Run run_ugv_case.py first.')
            return 1
        json_path = Path(candidates[-1])

    if not json_path.exists():
        print(f'File not found: {json_path}')
        return 1

    meta, frames = load_trajectory(json_path)

    # load terrain
    if 'terrain_dir' in meta:
        # terrain_dir is relative to output_dir; resolve from json_path parent
        terrain_dir = json_path.parent / meta['terrain_dir']
    else:
        terrain_dir = json_path.parent / 'terrain' / json_path.stem
    terrain = TerrainMap.load(terrain_dir)

    # output prefix
    out_dir = json_path.parent.parent / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = json_path.stem

    print(f'Plotting: {prefix}')
    print(f'  steps={len(frames)}  K(0)={meta["summary"].get("num_components_initial")}'
          f'  T_conn={meta["summary"].get("T_conn")}')

    plot_terrain_overview(terrain, meta, out_dir / f'{prefix}_terrain.png')
    plot_trajectory_paths(frames, terrain, meta,
                          out_dir / f'{prefix}_trajectory.png')
    plot_connectivity_over_time(frames, meta,
                                out_dir / f'{prefix}_connectivity.png')
    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
