"""TerrainMap: 2.5D terrain for UGV transfer experiments.

Single class with factory-like init paths per terrain family.
Grid-based height, slope, obstacle and cost queries.
Deterministic from seed; save/load as .npy + metadata.json.
"""

import json
from pathlib import Path

import numpy as np


class TerrainMap:
    """2.5D terrain with height, slope, obstacle, and cost queries."""

    def __init__(self, name, bounds, seed=0, resolution=5.0):
        self.name = name
        self.bounds = np.asarray(bounds)          # [[xmin, xmax], [ymin, ymax]]
        self.resolution = float(resolution)

        xmin, xmax = self.bounds[0]
        ymin, ymax = self.bounds[1]
        self._xsize = int((xmax - xmin) / self.resolution) + 1
        self._ysize = int((ymax - ymin) / self.resolution) + 1

        rng = np.random.default_rng(seed)

        if name == 'flat':
            self._init_flat()
        elif name == 'rolling':
            self._init_rolling(rng)
        elif name == 'ridge':
            self._init_ridge(rng)
        elif name == 'roughland':
            self._init_roughland(rng)
        elif name == 'mixed':
            self._init_mixed(rng)
        elif name == 'maze':
            self._init_maze(rng)
        else:
            raise ValueError(f"Unknown terrain name: {name}")

        self._validate()

    # ------------------------------------------------------------------
    # initialisers
    # ------------------------------------------------------------------
    def _init_flat(self):
        self._heightmap = np.zeros((self._ysize, self._xsize), dtype=np.float64)
        self._obstacle_mask = np.zeros((self._ysize, self._xsize), dtype=bool)
        self._costmap = np.ones((self._ysize, self._xsize), dtype=np.float64)
        self._cache_gradient()

    def _init_rolling(self, rng):
        xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], self._xsize)
        ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], self._ysize)
        X, Y = np.meshgrid(xs, ys)

        span_x = self.bounds[0, 1] - self.bounds[0, 0]
        span_y = self.bounds[1, 1] - self.bounds[1, 0]
        n_waves = rng.integers(3, 8)
        h = np.zeros_like(X)
        for _ in range(n_waves):
            A = rng.uniform(5, 30)
            wx = rng.uniform(0.5, 2.0) * 2 * np.pi / max(span_x, 1.0)
            wy = rng.uniform(0.5, 2.0) * 2 * np.pi / max(span_y, 1.0)
            px = rng.uniform(0, 2 * np.pi)
            py = rng.uniform(0, 2 * np.pi)
            h += A * np.sin(wx * X + px) * np.cos(wy * Y + py)

        self._heightmap = h.astype(np.float64)
        self._obstacle_mask = np.zeros((self._ysize, self._xsize), dtype=bool)
        self._costmap = np.ones((self._ysize, self._xsize), dtype=np.float64)
        self._cache_gradient()

    def _init_ridge(self, rng):
        """Parallel ridges with steep crests, wide traversable valleys."""
        xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], self._xsize)
        ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], self._ysize)
        X, Y = np.meshgrid(xs, ys)
        span_y = self.bounds[1, 1] - self.bounds[1, 0]
        span_x = self.bounds[0, 1] - self.bounds[0, 0]

        for attempt in range(30):
            n_ridges = rng.integers(2, 5)
            h = np.zeros_like(X)
            for k in range(n_ridges):
                ridge_y = self.bounds[1, 0] + span_y * (k + 0.5) / n_ridges
                A = rng.uniform(50, 90)  # need A/σ >= 0.77 for >25° slope
                wy = rng.uniform(1.5, 3.0) * 2 * np.pi / max(span_y, 1.0)
                wx = rng.uniform(0.3, 1.0) * 2 * np.pi / max(span_x, 1.0)
                px = rng.uniform(0, 2 * np.pi)
                h += A * np.exp(-0.5 * ((Y - ridge_y) * wy /
                                        (2 * np.pi)) ** 2) \
                     * (1.0 + 0.3 * np.sin(wx * X + px))

            gy, gx = np.gradient(h, self.resolution)
            slope = np.arctan(np.sqrt(gx**2 + gy**2)) * 180.0 / np.pi
            steep_frac = np.mean(slope > 25.0)
            flat_frac = np.mean(slope < 25.0)
            if steep_frac >= 0.03 and flat_frac >= 0.35:
                break
        else:
            # fallback: one moderate ridge with wide valleys
            h = np.zeros_like(X)
            ridge_y = self.bounds[1, 0] + span_y * 0.5
            h += 50.0 * np.exp(-0.5 * ((Y - ridge_y) * 0.012) ** 2)

        self._heightmap = h.astype(np.float64)
        self._obstacle_mask = np.zeros((self._ysize, self._xsize), dtype=bool)
        self._costmap = np.ones((self._ysize, self._xsize), dtype=np.float64)
        self._cache_gradient()

    def _init_roughland(self, rng):
        """Outdoor surface stress: rolling hills + traversable high-cost patches.

        No hard hazards.  Cost represents brush, loose rock, mud, or rough
        ground — traversable but expensive.  All slopes are constrained below
        max_slope_deg so that cost, not slope, is the primary terrain
        differentiator.
        """
        xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], self._xsize)
        ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], self._ysize)
        X, Y = np.meshgrid(xs, ys)
        span_x = self.bounds[0, 1] - self.bounds[0, 0]
        span_y = self.bounds[1, 1] - self.bounds[1, 0]

        for attempt in range(20):
            n_waves = rng.integers(3, 7)
            h = np.zeros_like(X)
            for _ in range(n_waves):
                A = rng.uniform(5, 25)
                wx = rng.uniform(0.5, 1.5) * 2 * np.pi / max(span_x, 1.0)
                wy = rng.uniform(0.5, 1.5) * 2 * np.pi / max(span_y, 1.0)
                px = rng.uniform(0, 2 * np.pi)
                py = rng.uniform(0, 2 * np.pi)
                h += A * np.sin(wx * X + px) * np.cos(wy * Y + py)

            # check max slope — scale down if too steep
            gy, gx = np.gradient(h, self.resolution)
            max_slope = float(np.arctan(
                np.sqrt(gx**2 + gy**2)).max() * 180.0 / np.pi)
            if max_slope <= 25.0:
                self._heightmap = h.astype(np.float64)
                break
            # scale down to bring max slope within limit (with 1° margin)
            scale = 24.0 / max(max_slope, 1.0)
            h *= scale
            gy2, gx2 = np.gradient(h, self.resolution)
            max_slope2 = float(np.arctan(
                np.sqrt(gx2**2 + gy2**2)).max() * 180.0 / np.pi)
            if max_slope2 <= 25.0:
                self._heightmap = h.astype(np.float64)
                break
        else:
            # fallback: very gentle rolling
            h = np.zeros_like(X)
            for _ in range(2):
                A = rng.uniform(3, 10)
                wx = rng.uniform(0.3, 0.8) * 2 * np.pi / max(span_x, 1.0)
                wy = rng.uniform(0.3, 0.8) * 2 * np.pi / max(span_y, 1.0)
                px = rng.uniform(0, 2 * np.pi)
                py = rng.uniform(0, 2 * np.pi)
                h += A * np.sin(wx * X + px) * np.cos(wy * Y + py)
            self._heightmap = h.astype(np.float64)

        # no hard hazards — cost is the only surface constraint
        self._obstacle_mask = np.zeros((self._ysize, self._xsize), dtype=bool)
        self._costmap = np.ones((self._ysize, self._xsize), dtype=np.float64)

        # traversable high-cost patches (brush, loose rock, mud)
        # use a mix of elliptical patches at varying cost levels
        n_patches = rng.integers(6, 16)
        for _ in range(n_patches):
            pw = rng.uniform(span_x * 0.04, span_x * 0.18)
            ph = rng.uniform(span_y * 0.04, span_y * 0.18)
            px_c = rng.uniform(self.bounds[0, 0] + pw, self.bounds[0, 1] - pw)
            py_c = rng.uniform(self.bounds[1, 0] + ph, self.bounds[1, 1] - ph)
            # cost 2–8 representing different surface types
            patch_cost = rng.uniform(2.0, 8.0)

            ix_min = int((px_c - pw / 2 - self.bounds[0, 0]) / self.resolution)
            ix_max = int((px_c + pw / 2 - self.bounds[0, 0]) / self.resolution) + 1
            iy_min = int((py_c - ph / 2 - self.bounds[1, 0]) / self.resolution)
            iy_max = int((py_c + ph / 2 - self.bounds[1, 0]) / self.resolution) + 1
            ix_min = max(0, ix_min); ix_max = min(self._xsize, ix_max)
            iy_min = max(0, iy_min); iy_max = min(self._ysize, iy_max)
            # elliptical falloff for natural patch edges
            cy = (iy_min + iy_max - 1) / 2.0
            cx = (ix_min + ix_max - 1) / 2.0
            ry = max((iy_max - iy_min) / 2.0, 1.0)
            rx = max((ix_max - ix_min) / 2.0, 1.0)
            iy_arr, ix_arr = np.ogrid[iy_min:iy_max, ix_min:ix_max]
            dist_norm = np.sqrt(((iy_arr - cy) / ry) ** 2
                                + ((ix_arr - cx) / rx) ** 2)
            falloff = np.clip(1.0 - dist_norm, 0.0, 1.0)  # 1 at centre, 0 at edge
            self._costmap[iy_min:iy_max, ix_min:ix_max] = np.maximum(
                self._costmap[iy_min:iy_max, ix_min:ix_max],
                1.0 + (patch_cost - 1.0) * falloff)

        self._cache_gradient()

    def _init_mixed(self, rng):
        """Fig. 4 terrain: rolling hills + sparse hard hazards + cost variation.

        Hard hazards (cliff edge, boulder field, water) are rare and small.
        Cost patches (brush, rough ground) provide traversable variation.
        """
        # rolling base (same as _init_rolling, lower amplitude)
        xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], self._xsize)
        ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], self._ysize)
        X, Y = np.meshgrid(xs, ys)
        span_x = self.bounds[0, 1] - self.bounds[0, 0]
        span_y = self.bounds[1, 1] - self.bounds[1, 0]
        n_waves = rng.integers(3, 6)
        h = np.zeros_like(X)
        for _ in range(n_waves):
            A = rng.uniform(5, 20)
            wx = rng.uniform(0.5, 1.5) * 2 * np.pi / max(span_x, 1.0)
            wy = rng.uniform(0.5, 1.5) * 2 * np.pi / max(span_y, 1.0)
            px = rng.uniform(0, 2 * np.pi)
            py = rng.uniform(0, 2 * np.pi)
            h += A * np.sin(wx * X + px) * np.cos(wy * Y + py)
        self._heightmap = h.astype(np.float64)

        # sparse hard hazards (0–3 small patches: cliff edge, boulder field, water)
        # these are the only binary-blocked cells in outdoor terrain
        self._obstacle_mask = np.zeros((self._ysize, self._xsize), dtype=bool)
        self._costmap = np.ones((self._ysize, self._xsize), dtype=np.float64)
        n_hazards = rng.integers(0, 4)  # often zero — outdoor terrain
        for _ in range(n_hazards):
            ow = rng.uniform(span_x * 0.02, span_x * 0.06)
            oh = rng.uniform(span_y * 0.02, span_y * 0.06)
            ox = rng.uniform(self.bounds[0, 0] + ow, self.bounds[0, 1] - ow)
            oy = rng.uniform(self.bounds[1, 0] + oh, self.bounds[1, 1] - oh)
            ix_min = int((ox - ow / 2 - self.bounds[0, 0]) / self.resolution)
            ix_max = int((ox + ow / 2 - self.bounds[0, 0]) / self.resolution) + 1
            iy_min = int((oy - oh / 2 - self.bounds[1, 0]) / self.resolution)
            iy_max = int((oy + oh / 2 - self.bounds[1, 0]) / self.resolution) + 1
            ix_min = max(0, ix_min); ix_max = min(self._xsize, ix_max)
            iy_min = max(0, iy_min); iy_max = min(self._ysize, iy_max)
            self._obstacle_mask[iy_min:iy_max, ix_min:ix_max] = True

        # traversable cost patches at varying levels
        n_strips = rng.integers(3, 8)
        for _ in range(n_strips):
            sw = rng.uniform(span_x * 0.04, span_x * 0.20)
            sh = rng.uniform(span_y * 0.03, span_y * 0.08)
            sx = rng.uniform(self.bounds[0, 0] + sw, self.bounds[0, 1] - sw)
            sy = rng.uniform(self.bounds[1, 0] + sh, self.bounds[1, 1] - sh)
            patch_cost = rng.uniform(2.0, 6.0)
            ix_min = int((sx - sw / 2 - self.bounds[0, 0]) / self.resolution)
            ix_max = int((sx + sw / 2 - self.bounds[0, 0]) / self.resolution) + 1
            iy_min = int((sy - sh / 2 - self.bounds[1, 0]) / self.resolution)
            iy_max = int((sy + sh / 2 - self.bounds[1, 0]) / self.resolution) + 1
            ix_min = max(0, ix_min); ix_max = min(self._xsize, ix_max)
            iy_min = max(0, iy_min); iy_max = min(self._ysize, iy_max)
            mask = ~self._obstacle_mask[iy_min:iy_max, ix_min:ix_max]
            self._costmap[iy_min:iy_max, ix_min:ix_max][mask] = np.maximum(
                self._costmap[iy_min:iy_max, ix_min:ix_max][mask], patch_cost)

        self._cache_gradient()

    def _init_maze(self, rng):
        """Strong-constraint stress test only — NOT an outdoor terrain.

        Dense rectangular obstacles on flat ground.  Urban/indoor semantics.
        Run separately from the outdoor matrix; never pool with rolling,
        ridge, roughland, or mixed in aggregate tables.
        """
        self._heightmap = np.zeros((self._ysize, self._xsize), dtype=np.float64)
        self._obstacle_mask = np.zeros((self._ysize, self._xsize), dtype=bool)
        self._costmap = np.ones((self._ysize, self._xsize), dtype=np.float64)

        span_x = self.bounds[0, 1] - self.bounds[0, 0]
        span_y = self.bounds[1, 1] - self.bounds[1, 0]
        n_obstacles = rng.integers(5, 16)
        for _ in range(n_obstacles):
            ow = rng.uniform(span_x * 0.03, span_x * 0.15)
            oh = rng.uniform(span_y * 0.03, span_y * 0.15)
            ox = rng.uniform(self.bounds[0, 0] + ow,
                             self.bounds[0, 1] - ow)
            oy = rng.uniform(self.bounds[1, 0] + oh,
                             self.bounds[1, 1] - oh)
            ix_min = int((ox - ow / 2 - self.bounds[0, 0]) / self.resolution)
            ix_max = int((ox + ow / 2 - self.bounds[0, 0]) / self.resolution) + 1
            iy_min = int((oy - oh / 2 - self.bounds[1, 0]) / self.resolution)
            iy_max = int((oy + oh / 2 - self.bounds[1, 0]) / self.resolution) + 1
            ix_min = max(0, ix_min); ix_max = min(self._xsize, ix_max)
            iy_min = max(0, iy_min); iy_max = min(self._ysize, iy_max)
            self._obstacle_mask[iy_min:iy_max, ix_min:ix_max] = True
            self._costmap[iy_min:iy_max, ix_min:ix_max] = 1.0  # same as free; mask blocks movement

        # traversable high-cost strips (rough/mud, not blocked)
        n_strips = rng.integers(2, 6)
        for _ in range(n_strips):
            sw = rng.uniform(span_x * 0.05, span_x * 0.25)
            sh = rng.uniform(span_y * 0.02, span_y * 0.06)
            sx = rng.uniform(self.bounds[0, 0] + sw,
                             self.bounds[0, 1] - sw)
            sy = rng.uniform(self.bounds[1, 0] + sh,
                             self.bounds[1, 1] - sh)
            ix_min = int((sx - sw / 2 - self.bounds[0, 0]) / self.resolution)
            ix_max = int((sx + sw / 2 - self.bounds[0, 0]) / self.resolution) + 1
            iy_min = int((sy - sh / 2 - self.bounds[1, 0]) / self.resolution)
            iy_max = int((sy + sh / 2 - self.bounds[1, 0]) / self.resolution) + 1
            ix_min = max(0, ix_min); ix_max = min(self._xsize, ix_max)
            iy_min = max(0, iy_min); iy_max = min(self._ysize, iy_max)
            # only set cost where no obstacle (traversable)
            mask = ~self._obstacle_mask[iy_min:iy_max, ix_min:ix_max]
            self._costmap[iy_min:iy_max, ix_min:ix_max][mask] = 5.0

        self._cache_gradient()

    # ------------------------------------------------------------------
    # precomputation
    # ------------------------------------------------------------------
    def _cache_gradient(self):
        gy, gx = np.gradient(self._heightmap, self.resolution)
        self._grad_x = gx
        self._grad_y = gy

    # ------------------------------------------------------------------
    # bilinear interpolation helper
    # ------------------------------------------------------------------
    def _bilinear(self, grid, fx, fy):
        """Bilinear interpolation on a 2D grid at fractional indices."""
        x0 = np.clip(np.floor(fx).astype(int), 0, self._xsize - 1)
        x1 = np.clip(x0 + 1, 0, self._xsize - 1)
        y0 = np.clip(np.floor(fy).astype(int), 0, self._ysize - 1)
        y1 = np.clip(y0 + 1, 0, self._ysize - 1)
        wx = fx - x0  # (N,) or scalar
        wy = fy - y0
        # 4-corner interpolation (all element-wise)
        v00 = grid[y0, x0]; v10 = grid[y0, x1]
        v01 = grid[y1, x0]; v11 = grid[y1, x1]
        return (v00 * (1 - wx) * (1 - wy) + v10 * wx * (1 - wy) +
                v01 * (1 - wx) * wy + v11 * wx * wy)

    def _xy_to_frac(self, xy):
        """Float (x, y) → fractional column/row index."""
        xy = np.atleast_2d(np.asarray(xy))
        fx = (xy[:, 0] - self.bounds[0, 0]) / self.resolution
        fy = (xy[:, 1] - self.bounds[1, 0]) / self.resolution
        fx = np.clip(fx, 0.0, float(self._xsize - 1))
        fy = np.clip(fy, 0.0, float(self._ysize - 1))
        return fx, fy

    # ------------------------------------------------------------------
    # public queries
    # ------------------------------------------------------------------
    def height(self, xy):
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        fx, fy = self._xy_to_frac(xy)
        h = self._bilinear(self._heightmap, fx, fy)
        return float(h[0]) if len(h) == 1 else h

    def gradient(self, xy):
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        fx, fy = self._xy_to_frac(xy)
        gx = self._bilinear(self._grad_x, fx, fy)
        gy = self._bilinear(self._grad_y, fx, fy)
        g = np.stack([gx, gy], axis=-1)
        return g[0] if len(g) == 1 else g

    def slope_deg(self, xy):
        g = self.gradient(xy)
        g = np.atleast_2d(g)
        s = np.arctan(np.linalg.norm(g, axis=-1)) * 180.0 / np.pi
        return float(s[0]) if len(s) == 1 else s

    def cost(self, xy):
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        fx, fy = self._xy_to_frac(xy)
        c = self._bilinear(self._costmap, fx, fy)
        return float(c[0]) if len(c) == 1 else c

    def is_obstacle(self, xy):
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        fx, fy = self._xy_to_frac(xy)
        ix = np.clip(np.floor(fx).astype(int), 0, self._xsize - 1)
        iy = np.clip(np.floor(fy).astype(int), 0, self._ysize - 1)
        obs = self._obstacle_mask[iy, ix]
        return bool(obs[0]) if len(obs) == 1 else obs

    def is_inside(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        return (
            (xy[..., 0] >= self.bounds[0, 0])
            & (xy[..., 0] <= self.bounds[0, 1])
            & (xy[..., 1] >= self.bounds[1, 0])
            & (xy[..., 1] <= self.bounds[1, 1])
        )

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def _validate(self):
        """Advisory property checks after generation (warnings only)."""
        slope = np.arctan(np.sqrt(self._grad_x**2 + self._grad_y**2)) \
                * 180.0 / np.pi
        steep_frac = float(np.mean(slope > 25.0)) * 100
        obs_frac = float(np.mean(self._obstacle_mask)) * 100
        cost_range = (float(self._costmap.min()), float(self._costmap.max()))
        z_range = (float(self._heightmap.min()), float(self._heightmap.max()))

        if self.name == 'ridge' and steep_frac < 1.0:
            print(f'  [terrain] WARNING: ridge has only {steep_frac:.1f}% '
                  f'cells slope>25° — slope stress may be weak')
        if self.name == 'maze' and obs_frac < 2.0:
            print(f'  [terrain] WARNING: maze stress test has only '
                  f'{obs_frac:.1f}% obstacle coverage')
        if self.name == 'roughland' and cost_range[1] < 3.0:
            print(f'  [terrain] WARNING: roughland has no significant '
                  f'cost variation (max={cost_range[1]:.1f})')
        if self.name == 'roughland' and steep_frac > 2.0:
            print(f'  [terrain] WARNING: roughland has {steep_frac:.1f}% '
                  f'cells slope>25° — cost may not be the primary constraint')
        if self.name == 'mixed' and z_range[1] - z_range[0] < 1.0:
            print(f'  [terrain] WARNING: mixed terrain is nearly flat '
                  f'(z range={z_range})')
        if self.name == 'mixed' and cost_range[1] <= 1.5:
            print(f'  [terrain] WARNING: mixed terrain has no significant '
                  f'cost variation (max={cost_range[1]:.1f})')

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, dirpath):
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        np.save(dirpath / 'heightmap.npy', self._heightmap)
        np.save(dirpath / 'costmap.npy', self._costmap)
        np.save(dirpath / 'obstacle_mask.npy', self._obstacle_mask)
        meta = {
            'name': self.name,
            'bounds': self.bounds.tolist(),
            'resolution': self.resolution,
        }
        with open(dirpath / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, dirpath):
        dirpath = Path(dirpath)
        with open(dirpath / 'metadata.json') as f:
            meta = json.load(f)
        obj = cls.__new__(cls)
        obj.name = meta['name']
        obj.bounds = np.array(meta['bounds'])
        obj.resolution = meta['resolution']
        obj._xsize = int(
            (obj.bounds[0, 1] - obj.bounds[0, 0]) / obj.resolution
        ) + 1
        obj._ysize = int(
            (obj.bounds[1, 1] - obj.bounds[1, 0]) / obj.resolution
        ) + 1
        obj._heightmap = np.load(dirpath / 'heightmap.npy')
        obj._costmap = np.load(dirpath / 'costmap.npy')
        obj._obstacle_mask = np.load(dirpath / 'obstacle_mask.npy')
        obj._cache_gradient()
        obj._validate()
        return obj
