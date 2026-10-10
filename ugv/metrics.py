"""Source-free process metrics from UGV trajectory data (plan §11).

All metrics are computed from the logged CR graph (per-step component labels
and edge sets), without access to algorithm internals.  This keeps the
evaluation fair and reproducible from saved trajectory files alone.
"""

import numpy as np


def _edge_key(i, j):
    """Canonical (min, max) key for an undirected edge."""
    return (i, j) if i < j else (j, i)


def compute_process_metrics(frames, T_conn=None):
    """Compute source-free process metrics from trajectory frames.

    Parameters
    ----------
    frames : list of dict
        Each dict has keys: step, component_id (list), edges (list of pairs),
        remain_list (list of ints).
    T_conn : int or None
        If int, the step at which K(t)=1 was reached. Metrics are computed
        up to T_eval = T_conn.  If None (failed case), metrics are computed
        up to the last frame (T_eval = horizon).

    Returns
    -------
    dict with keys:
        K_initial, K_final
        fragmentation_exposure
        re_split_count
        first_inter_fragment_contact
        first_contact_to_connectivity_latency
        inter_fragment_edge_birth_count
        edge_birth_persistence_mean, edge_birth_persistence_median
        edge_birth_efficiency
        final_edge_count, final_mean_degree
    """
    if not frames:
        return {}

    n_frames = len(frames)
    if T_conn is not None:
        T_eval = T_conn
        # find the frame index for step == T_conn (or last if not found)
        eval_idx = None
        for fi, fr in enumerate(frames):
            if fr['step'] == T_conn:
                eval_idx = fi
                break
        if eval_idx is None:
            eval_idx = n_frames - 1
    else:
        eval_idx = n_frames - 1
        T_eval = frames[-1]['step']

    # ---- K(t) series ----
    # K_all covers t=0..T_eval inclusive; fragmentation_exposure sums
    # t < T_eval only (K_all[:-1]), excluding the terminating frame.
    K_all = np.array([max(int(np.max(fr['component_id'])) + 1, 1)
                       for fr in frames[:eval_idx + 1]], dtype=int)
    K_initial = int(K_all[0])
    K_final = int(K_all[-1])

    # ---- fragmentation_exposure (t < T_eval) ----
    fragmentation_exposure = int(np.sum(K_all[:-1] - 1))

    # ---- re_split_count (t < T_eval - 1) ----
    re_split_count = 0
    K_pre = K_all[:-1]  # t=0..T_eval-1
    if len(K_pre) >= 2:
        re_split_count = int(np.sum(np.maximum(0, K_pre[1:] - K_pre[:-1])))

    # ---- inter-fragment edge tracking ----
    # "First CR-edge birth": the first time a specific pair connects across
    # previously different components.  Tracked in a persistent set so
    # reconnections are not double-counted (plan §11 semantics).
    # Pass 1: scan t=1..T_eval INCLUSIVE for first_contact_step.
    seen_interfragment_edges = set()
    first_contact_step = None
    prev_comp1 = {i: c for i, c in enumerate(frames[0]['component_id'])}
    for fi in range(1, eval_idx + 1):
        fr = frames[fi]
        curr_comp = {i: c for i, c in enumerate(fr['component_id'])}
        curr_edge_set = {_edge_key(e[0], e[1]) for e in fr['edges']}
        remain_set = set(fr['remain_list'])
        for ek in curr_edge_set:
            a, b = ek
            if a not in remain_set or b not in remain_set:
                continue
            prev_ca = prev_comp1.get(a, -1)
            prev_cb = prev_comp1.get(b, -1)
            if prev_ca != prev_cb or prev_ca == -1:
                if ek not in seen_interfragment_edges:
                    seen_interfragment_edges.add(ek)
                    if first_contact_step is None:
                        first_contact_step = fr['step']
        prev_comp1 = curr_comp

    # Pass 2: births for t < T_eval only, with consecutive-survival
    # tracking and right-censoring at T_eval.
    # Only the FIRST continuous survival period is tracked per unique edge
    # (plan: \"for each edge born, number of subsequent consecutive steps\").
    # Reconnections do not create new survival records.
    prev_comp2 = {i: c for i, c in enumerate(frames[0]['component_id'])}
    counted_births = set()       # edge_keys ever counted (never cleared)
    finished_edges = set()       # edge_keys whose survival period ended
    alive_births = {}            # edge_key → birth_step
    edge_survival = {}           # (edge_key, birth_step) → survival steps
    edge_censored = {}           # (edge_key, birth_step) → True if censored
    alive_edges = set()          # (edge_key, birth_step) currently alive

    inter_fragment_birth_count = 0

    for fi in range(1, eval_idx):
        fr = frames[fi]
        step = fr['step']
        curr_comp = {i: c for i, c in enumerate(fr['component_id'])}
        curr_edge_set = {_edge_key(e[0], e[1]) for e in fr['edges']}
        remain_set = set(fr['remain_list'])

        born_this_frame = set()

        for ek in curr_edge_set:
            a, b = ek
            if a not in remain_set or b not in remain_set:
                continue
            prev_ca = prev_comp2.get(a, -1)
            prev_cb = prev_comp2.get(b, -1)
            if prev_ca != prev_cb or prev_ca == -1:
                if ek in seen_interfragment_edges:
                    if ek not in counted_births:
                        counted_births.add(ek)
                        inter_fragment_birth_count += 1
                    # start survival tracking on first appearance only
                    if ek not in alive_births and ek not in finished_edges:
                        alive_births[ek] = step
                        birth_key = (ek, step)
                        alive_edges.add(birth_key)
                        born_this_frame.add(birth_key)
                        edge_survival[birth_key] = 0
                        edge_censored[birth_key] = False

        # update consecutive survival (skip edges born this frame)
        dead = []
        for birth_key in alive_edges:
            if birth_key in born_this_frame:
                continue
            ek, _ = birth_key
            if ek in curr_edge_set:
                edge_survival[birth_key] += 1
            else:
                dead.append(birth_key)
        for birth_key in dead:
            alive_edges.discard(birth_key)
            ek, _ = birth_key
            alive_births.pop(ek, None)
            finished_edges.add(ek)  # prevent re-tracking on reconnection

        prev_comp2 = curr_comp

    # T_eval frame: surviving edges get +1 and are marked right-censored
    edge_birth_censored_count = 0
    if eval_idx > 0 and alive_edges:
        fr_eval = frames[eval_idx]
        eval_edge_set = {_edge_key(e[0], e[1]) for e in fr_eval['edges']}
        for birth_key in list(alive_edges):
            ek, _ = birth_key
            if ek in eval_edge_set:
                edge_survival[birth_key] += 1
                edge_censored[birth_key] = True
                edge_birth_censored_count += 1

    # ---- first_contact_to_connectivity_latency ----
    if first_contact_step is not None and T_conn is not None:
        first_contact_latency = T_conn - first_contact_step
    else:
        first_contact_latency = None

    # ---- edge_birth_persistence ----
    if edge_survival:
        all_survivals = np.array(list(edge_survival.values()), dtype=np.float64)
        # separate exact (uncensored) from right-censored
        exact_mask = np.array([not edge_censored.get(bk, False)
                                for bk in edge_survival], dtype=bool)
        exact_survivals = all_survivals[exact_mask]
        if len(exact_survivals) > 0:
            persistence_mean = float(np.mean(exact_survivals))
            persistence_median = float(np.median(exact_survivals))
        else:
            persistence_mean = None
            persistence_median = None
        # lower-bound mean including censored at observed value
        persistence_lower_bound_mean = float(np.mean(all_survivals))
    else:
        persistence_mean = None
        persistence_median = None
        persistence_lower_bound_mean = None

    # ---- edge_birth_efficiency ----
    edge_birth_efficiency = float(
        inter_fragment_birth_count / max(1, K_initial - 1))

    # ---- terminal degree footprint + LCC ratio ----
    final_fr = frames[eval_idx]
    final_edges = final_fr['edges']
    final_edge_count = len(final_edges)
    degree = {}
    for a, b in final_edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    n_remain = len(final_fr['remain_list'])
    final_mean_degree = (2.0 * final_edge_count / max(n_remain, 1)
                         if n_remain > 0 else 0.0)
    # degree histogram: counts of agents with degree 0,1,2,...
    max_deg = max(degree.values()) if degree else 0
    degree_hist = [sum(1 for i in final_fr['remain_list']
                       if degree.get(i, 0) == d)
                   for d in range(max_deg + 1)]

    # largest connected component ratio at final frame
    final_comp = np.array(final_fr['component_id'])
    remain_set = set(final_fr['remain_list'])
    comp_sizes = {}
    for i, c in enumerate(final_comp):
        if i in remain_set and c >= 0:
            comp_sizes[c] = comp_sizes.get(c, 0) + 1
    max_comp_size = max(comp_sizes.values()) if comp_sizes else 0
    final_lcc_ratio = float(max_comp_size / max(n_remain, 1))

    return dict(
        K_initial=K_initial,
        K_final=K_final,
        T_eval=int(T_eval),
        fragmentation_exposure=fragmentation_exposure,
        re_split_count=re_split_count,
        first_inter_fragment_contact=first_contact_step,
        first_contact_to_connectivity_latency=first_contact_latency,
        inter_fragment_edge_birth_count=inter_fragment_birth_count,
        edge_birth_persistence_mean=persistence_mean,
        edge_birth_persistence_median=persistence_median,
        edge_birth_persistence_lower_bound_mean=persistence_lower_bound_mean,
        edge_birth_censored_count=edge_birth_censored_count,
        edge_birth_efficiency=edge_birth_efficiency,
        final_edge_count=final_edge_count,
        final_mean_degree=final_mean_degree,
        final_degree_histogram=degree_hist,
        final_lcc_ratio=final_lcc_ratio,
    )


def compute_summary_row(metadata, summary, process_metrics):
    """Combine case metadata, solve summary, and process metrics into one flat dict.

    Returns a dict suitable for a CSV row or JSON record.
    """
    row = {}
    # identifiers
    for k in ('algorithm', 'terrain', 'damage_mode',
              'N', 'D', 'requested_D', 'seed_index', 'seed_value', 'terrain_seed'):
        row[k] = metadata.get(k)
    # environment
    for k in ('dt', 'v_max', 'omega_max', 'agent_radius', 'max_slope_deg',
              'communication_range'):
        row[k] = metadata.get(k)
    row['connectivity_model'] = metadata.get('connectivity_model')

    # outcome
    row['success'] = summary.get('success')
    row['T_conn'] = summary.get('T_conn')
    row['invalid_initial_graph'] = summary.get('invalid_initial_graph', False)
    row['n_remain'] = summary.get('n_remain')
    row['horizon'] = summary.get('horizon')

    # per-step diagnostics
    row['blocked_steps'] = summary.get('blocked_steps')
    row['collision_steps'] = summary.get('collision_steps')
    row['saturated_steps'] = summary.get('saturated_steps')
    row['control_saturation_ratio'] = summary.get('control_saturation_ratio')

    # path metrics
    row['path_length_total'] = summary.get('path_length_total')
    row['path_length_mean'] = summary.get('path_length_mean')
    row['terrain_cost_total'] = summary.get('terrain_cost_total')
    row['terrain_cost_mean'] = summary.get('terrain_cost_mean')
    row['excess_terrain_cost_total'] = summary.get('excess_terrain_cost_total')
    row['excess_terrain_cost_mean'] = summary.get('excess_terrain_cost_mean')
    row['mean_cost_per_meter'] = summary.get('mean_cost_per_meter')

    # init quality
    row['init_repositioned_count'] = len(metadata.get('init_repositioned', []))
    row['init_boundary_count'] = metadata.get('init_boundary_count')

    # process metrics
    if process_metrics:
        for k, v in process_metrics.items():
            row[k] = v

    return row
