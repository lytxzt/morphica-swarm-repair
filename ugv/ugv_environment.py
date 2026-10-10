"""UGV unicycle dynamics and terrain-aware environment layer.

Thin wrapper that converts algorithm action vectors (global N×3) into
yaw-limited, slope-limited, obstacle-avoiding displacements on a 2.5-D
terrain.  All methods share the same vehicle layer (plan §2 invariant).
"""

import numpy as np

from Configurations import config_constant_speed, config_communication_range, \
    config_central_point
from ugv.ugv_config import (
    UGV_DT, UGV_V_MAX, UGV_OMEGA_MAX, UGV_AGENT_RADIUS,
    UGV_MAX_SLOPE_DEG, EPS,
)


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════
def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# Footprint safety is a 9-point sampling approximation (centre + 8-point
# circle), not an exact disc–grid intersection test.  At the default 5 m
# resolution with a 0.6 m agent radius, the 1-cell impassable-mask dilation
# (§7.2) compensates by conservatively widening hard-constraint regions.
# Do not claim "full footprint guarantee" in paper text.
_FOOTPRINT_ANGLES = np.linspace(0.0, 2.0 * np.pi, 9)[:-1]
_FOOTPRINT_OFFSETS = np.column_stack([
    np.cos(_FOOTPRINT_ANGLES), np.sin(_FOOTPRINT_ANGLES)])


def _inside_shrunk_bounds(xy, terrain, radius):
    """True if centre xy keeps the full footprint inside terrain bounds."""
    b = terrain.bounds
    return (xy[0] >= b[0, 0] + radius and xy[0] <= b[0, 1] - radius and
            xy[1] >= b[1, 0] + radius and xy[1] <= b[1, 1] - radius)


# ═══════════════════════════════════════════════════════════════════════════
# impassable mask (computed once per terrain + UGV params, stored on terrain)
# ═══════════════════════════════════════════════════════════════════════════


def _build_impassable_mask(terrain, agent_radius, safety_margin, max_slope_deg):
    """Precompute hard-constraint mask: dilated obstacles + steep slope."""
    r_cells = int(np.ceil((agent_radius + safety_margin) / terrain.resolution))
    obs = terrain._obstacle_mask
    if r_cells > 0 and obs.any():
        dilated = obs.copy()
        for _ in range(r_cells):
            prev = dilated.copy()
            dilated[1:, :] |= prev[:-1, :]
            dilated[:-1, :] |= prev[1:, :]
            dilated[:, 1:] |= prev[:, :-1]
            dilated[:, :-1] |= prev[:, 1:]
            dilated[1:, 1:] |= prev[:-1, :-1]
            dilated[1:, :-1] |= prev[:-1, 1:]
            dilated[:-1, 1:] |= prev[1:, :-1]
            dilated[:-1, :-1] |= prev[1:, 1:]
    else:
        dilated = obs.copy()

    slope = np.arctan(np.sqrt(terrain._grad_x**2 + terrain._grad_y**2)) \
            * 180.0 / np.pi
    steep = slope > max_slope_deg

    return dilated | steep


def _get_impassable_mask(terrain, agent_radius, safety_margin, max_slope_deg):
    """Cached impassable mask — stored on terrain to prevent stale-cache bugs."""
    if not hasattr(terrain, '_impassable_cache'):
        terrain._impassable_cache = {}
    key = (agent_radius, safety_margin, max_slope_deg)
    if key not in terrain._impassable_cache:
        terrain._impassable_cache[key] = _build_impassable_mask(
            terrain, agent_radius, safety_margin, max_slope_deg)
    return terrain._impassable_cache[key]


def _impassable_at(mask, xy, terrain):
    """True if (x, y) falls in an impassable cell (nearest-neighbour)."""
    fx = (xy[0] - terrain.bounds[0, 0]) / terrain.resolution
    fy = (xy[1] - terrain.bounds[1, 0]) / terrain.resolution
    ix = int(np.clip(np.floor(fx), 0, terrain._xsize - 1))
    iy = int(np.clip(np.floor(fy), 0, terrain._ysize - 1))
    return bool(mask[iy, ix])


# ═══════════════════════════════════════════════════════════════════════════
# footprint queries
# ═══════════════════════════════════════════════════════════════════════════
def _footprint_max_slope(center_xy, terrain, radius):
    """Max slope over the 8-point circular footprint plus centre."""
    pts = np.vstack([np.atleast_2d(center_xy),
                     np.array(center_xy) + _FOOTPRINT_OFFSETS * radius])
    slopes = terrain.slope_deg(pts)
    return float(np.max(slopes))


def footprint_hits_obstacle(center_xy, terrain, radius):
    """True if any part of the circular footprint intersects an obstacle."""
    if terrain.is_obstacle(center_xy):
        return True
    pts = np.atleast_2d(center_xy) + _FOOTPRINT_OFFSETS * radius
    return bool(np.any(terrain.is_obstacle(pts)))


# ═══════════════════════════════════════════════════════════════════════════
# UGV unicycle integration with shared safety controller (plan §5, §7)
# ═══════════════════════════════════════════════════════════════════════════
def ugv_step(positions, actions, yaw, remain_list, terrain,
             dt=UGV_DT, v_max=UGV_V_MAX, omega_max=UGV_OMEGA_MAX,
             max_slope_deg=UGV_MAX_SLOPE_DEG):
    """One step of unicycle dynamics with shared terrain-aware safety controller.

    Candidate sampling around desired yaw (±90°, N_YAW angles × 2 speeds),
    filtered by impassable mask and footprint slope, scored by alignment /
    progress / cost / slope margin.  The same fixed scoring weights are used
    for every algorithm in a benchmark cell (plan §7).

    Returns (next_positions, next_yaw, diagnostics).
    diagnostics keys: blocked[N], detour_ratio[N], collision[N],
                      raw_action[N×3], executed_action[N×3]
    """
    N = len(positions)
    next_pos = positions.copy()
    next_yaw = yaw.copy()

    blocked = np.zeros(N, dtype=bool)
    detour_ratio = np.ones(N, dtype=np.float64)
    selected_score = np.zeros(N, dtype=np.float64)
    saturated_yaw = np.zeros(N, dtype=bool)
    saturated_speed = np.zeros(N, dtype=bool)
    raw_action = np.zeros((N, 3), dtype=np.float64)
    executed_action = np.zeros((N, 3), dtype=np.float64)

    speed_scale = config_constant_speed / dt
    impassable_mask = _get_impassable_mask(
        terrain, UGV_AGENT_RADIUS, safety_margin=0.3,
        max_slope_deg=max_slope_deg)

    # candidate grid (plan §7.3)
    n_yaw = 7
    dyaw_grid = np.linspace(-np.pi / 2, np.pi / 2, n_yaw)
    speed_fracs = np.array([1.0, 0.5])

    # scoring weights (plan §7.4).  w_clear is zero: the 1-cell dilated
    # impassable mask already enforces a hard clearance margin; no soft
    # distance-to-hazard term is needed on top of that.
    w_dir, w_prog, w_cost, w_slope = 0.55, 0.20, 0.15, 0.10
    cost_max_nominal = 5.0

    for i in remain_list:
        a = actions[i]
        raw_action[i] = a.copy()
        a_horiz_norm = float(np.linalg.norm(a[:2]))

        if a_horiz_norm < EPS:
            # zero desired action → stay in place, not blocked
            next_yaw[i] = yaw[i]
            # positions[i] unchanged
            detour_ratio[i] = 1.0
            blocked[i] = False
            continue

        target_yaw = np.arctan2(a[1], a[0])
        desired_speed = min(v_max, speed_scale * a_horiz_norm)

        # generate candidates
        best_score = -np.inf
        best_cand = None  # (cand_yaw, cand_v, cand_xy)

        for dyaw in dyaw_grid:
            cand_yaw = target_yaw + dyaw
            yaw_delta = np.clip(_wrap_pi(cand_yaw - yaw[i]),
                                -omega_max * dt, omega_max * dt)
            cand_yaw_feas = yaw[i] + yaw_delta

            for sf in speed_fracs:
                cand_v = desired_speed * sf
                cand_xy = np.array([
                    positions[i, 0] + cand_v * np.cos(cand_yaw_feas) * dt,
                    positions[i, 1] + cand_v * np.sin(cand_yaw_feas) * dt,
                ])

                # hard-constraint filter
                if not _inside_shrunk_bounds(cand_xy, terrain, UGV_AGENT_RADIUS):
                    continue
                if _impassable_at(impassable_mask, cand_xy, terrain):
                    continue
                # footprint slope check (plan §7.1: hard constraint)
                fp_slope = _footprint_max_slope(
                    cand_xy, terrain, UGV_AGENT_RADIUS)
                if fp_slope > max_slope_deg:
                    continue

                # score (plan §7.4)
                dir_score = np.cos(_wrap_pi(cand_yaw_feas - target_yaw))
                prog_score = cand_v / v_max
                cost_pen = float(terrain.cost(cand_xy)) / cost_max_nominal
                slope_pen = fp_slope / max_slope_deg

                score = (w_dir * dir_score
                         + w_prog * prog_score
                         - w_cost * cost_pen
                         - w_slope * slope_pen)

                if score > best_score:
                    best_score = score
                    best_cand = (cand_yaw_feas, cand_v, cand_xy)

        if best_cand is not None:
            cand_yaw, cand_v, cand_xy = best_cand
            yaw_delta_chosen = abs(_wrap_pi(cand_yaw - yaw[i]))
            next_yaw[i] = cand_yaw
            next_pos[i, 0] = cand_xy[0]
            next_pos[i, 1] = cand_xy[1]
            z = float(terrain.height(cand_xy))
            next_pos[i, 2] = z
            detour_ratio[i] = cand_v / max(desired_speed, EPS)
            saturated_yaw[i] = (yaw_delta_chosen >= omega_max * dt - EPS)
            saturated_speed[i] = (desired_speed >= v_max - EPS
                                  and cand_v < desired_speed - EPS)
            selected_score[i] = best_score
            executed_action[i] = [np.cos(cand_yaw) * cand_v,
                                  np.sin(cand_yaw) * cand_v, 0.0]
        else:
            # no feasible candidate: rotate toward target but stay in place
            yaw_delta = np.clip(_wrap_pi(target_yaw - yaw[i]),
                                -omega_max * dt, omega_max * dt)
            next_yaw[i] = yaw[i] + yaw_delta
            blocked[i] = True
            detour_ratio[i] = 0.0
            # executed_action stays zero

    # pairwise collision detection — DIAGNOSTIC ONLY (plan §11).
    # Collisions are recorded after positions are written but do not filter
    # candidates or separate agents.  The safety controller is terrain-aware;
    # inter-agent collision avoidance is not part of the current scope.
    # XY distance only (height differences do not relax the collision radius).
    v_collision = np.zeros(N, dtype=bool)
    remain_arr = np.array(remain_list, dtype=int)
    if len(remain_arr) >= 2:
        rp_xy = next_pos[remain_arr, :2]  # XY only
        Gc = np.matmul(rp_xy, rp_xy.T)
        Hc = np.tile(np.diag(Gc), (len(remain_arr), 1))
        Dc = np.sqrt(np.maximum(Hc + Hc.T - 2.0 * Gc, 0.0))
        min_dist = 2.0 * UGV_AGENT_RADIUS
        np.fill_diagonal(Dc, np.inf)
        coll_pairs = np.where(Dc < min_dist)
        for la, lb in zip(coll_pairs[0], coll_pairs[1]):
            if la < lb:
                v_collision[remain_arr[la]] = True
                v_collision[remain_arr[lb]] = True

    diagnostics = {
        'blocked': blocked,
        'detour_ratio': detour_ratio,
        'saturated_yaw': saturated_yaw,
        'saturated_speed': saturated_speed,
        'selected_score': selected_score,
        'collision': v_collision,
        'raw_action': raw_action,
        'executed_action': executed_action,
    }
    return next_pos, next_yaw, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# graph state (CR-only Euclidean — plan §8 invariant)
# ═══════════════════════════════════════════════════════════════════════════
def compute_graph_state(positions, remain_list, communication_range=None):
    """Return (num_components, component_labels, edges).

    component_labels[i] = -1 for destroyed agents.
    edges is list of (i, j) global-index tuples.

    If communication_range is None, uses config_communication_range.
    """
    if communication_range is None:
        communication_range = config_communication_range

    N_total = len(positions)
    remain_arr = np.array(remain_list, dtype=int)
    n = len(remain_arr)

    labels = np.full(N_total, -1, dtype=int)
    edges = []
    if n == 0:
        return 0, labels, edges

    remain_pos = positions[remain_arr]

    # distance / adjacency  (same vectorised form as MORPHICA.check_subnets)
    G = np.matmul(remain_pos, remain_pos.T)
    H_diag = np.tile(np.diag(G), (n, 1))
    dist_sq = H_diag + H_diag.T - 2.0 * G
    D = np.sqrt(np.maximum(dist_sq, 0.0))  # guard against fp rounding below zero
    A = D <= communication_range
    np.fill_diagonal(A, False)

    # edge list
    ui, uj = np.where(np.triu(A))
    for li, lj in zip(ui, uj):
        edges.append((int(remain_arr[li]), int(remain_arr[lj])))

    # BFS component labels
    visited = np.zeros(n, dtype=bool)
    comp_id = 0
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        while stack:
            v = stack.pop()
            labels[remain_arr[v]] = comp_id
            nbrs = np.where(A[v] & ~visited)[0]
            visited[nbrs] = True
            stack.extend(nbrs.tolist())
        comp_id += 1

    return comp_id, labels, edges


# ═══════════════════════════════════════════════════════════════════════════
# initial yaw  (plan §9 step 4: yaw toward formation center)
# ═══════════════════════════════════════════════════════════════════════════
def init_yaw(positions):
    """Initial yaw toward the swarm centre (config_central_point xy)."""
    N = len(positions)
    yaw = np.zeros(N, dtype=np.float64)
    centre_xy = config_central_point[:2]
    for i in range(N):
        d = centre_xy - positions[i, :2]
        if np.linalg.norm(d) > EPS:
            yaw[i] = np.arctan2(d[1], d[0])
    return yaw
