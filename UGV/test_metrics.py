"""Synthetic tests for process metrics (plan §17 item 9).

Covers the regression-prone edge cases discovered during Phase 2 reviews:
first contact at T_eval, right-censored survival, reconnection not
double-counted, degree histogram, fragmentation exposure off-by-one.

Run:  python UGV/test_metrics.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from UGV.metrics import compute_process_metrics


def _frame(step, component_id, edges, remain_list=None):
    if remain_list is None:
        remain_list = list(range(len(component_id)))
    return dict(step=step, component_id=component_id, edges=edges,
                remain_list=remain_list)


# ── first contact at T_eval ──────────────────────────────────────────────
def test_first_contact_at_T_eval():
    """Edge appearing at T_conn should be detected as first contact."""
    frames = [
        _frame(0, [0, 1], []),
        _frame(1, [0, 0], [(0, 1)]),
    ]
    m = compute_process_metrics(frames, T_conn=1)
    assert m['first_inter_fragment_contact'] == 1
    assert m['first_contact_to_connectivity_latency'] == 0
    # birth at T_eval excluded from birth count
    assert m['inter_fragment_edge_birth_count'] == 0


def test_first_contact_before_T_eval():
    """Edge appearing before T_conn should be detected and counted."""
    frames = [
        _frame(0, [0, 1], []),
        _frame(1, [0, 0], [(0, 1)]),
        _frame(2, [0, 0], [(0, 1)]),
    ]
    m = compute_process_metrics(frames, T_conn=2)
    assert m['first_inter_fragment_contact'] == 1
    assert m['first_contact_to_connectivity_latency'] == 1
    assert m['inter_fragment_edge_birth_count'] == 1


# ── right-censored survival ──────────────────────────────────────────────
def test_censored_edge_survival():
    """Edge born at t=1, alive at T_eval=2 → censored, lower_bound=1."""
    frames = [
        _frame(0, [0, 1], []),
        _frame(1, [0, 0], [(0, 1)]),
        _frame(2, [0, 0], [(0, 1)]),
    ]
    m = compute_process_metrics(frames, T_conn=2)
    assert m['inter_fragment_edge_birth_count'] == 1
    assert m['edge_birth_censored_count'] == 1
    assert m['edge_birth_persistence_mean'] is None  # no exact observations
    assert m['edge_birth_persistence_lower_bound_mean'] == 1.0


def test_exact_survival_when_edge_dies():
    """Edge born at t=1, dies at t=2, T_eval=3 → exact survival=0."""
    frames = [
        _frame(0, [0, 1], []),
        _frame(1, [0, 0], [(0, 1)]),
        _frame(2, [0, 1], []),
        _frame(3, [0, 1], []),
    ]
    m = compute_process_metrics(frames, T_conn=None)  # failed case
    assert m['inter_fragment_edge_birth_count'] == 1
    assert m['edge_birth_censored_count'] == 0
    assert m['edge_birth_persistence_mean'] == 0.0


# ── reconnection not double-counted ──────────────────────────────────────
def test_reconnection_not_double_counted():
    """Same edge born, dies, reconnects → 1 birth, 1 survival sample."""
    frames = [
        _frame(0, [0, 1], []),
        _frame(1, [0, 0], [(0, 1)]),
        _frame(2, [0, 1], []),
        _frame(3, [0, 0], [(0, 1)]),  # reconnect (T_eval)
    ]
    m = compute_process_metrics(frames, T_conn=3)
    assert m['inter_fragment_edge_birth_count'] == 1
    assert m['edge_birth_efficiency'] == 1.0


def test_double_reconnect_still_one_birth():
    """Same edge dies and reconnects twice → still 1 birth."""
    frames = [
        _frame(0, [0, 1], []),
        _frame(1, [0, 0], [(0, 1)]),
        _frame(2, [0, 1], []),
        _frame(3, [0, 0], [(0, 1)]),
        _frame(4, [0, 1], []),
        _frame(5, [0, 0], [(0, 1)]),
    ]
    m = compute_process_metrics(frames, T_conn=5)
    assert m['inter_fragment_edge_birth_count'] == 1


def test_reconnect_only_first_survival_tracked():
    """Reconnection does not add second survival sample."""
    frames = [
        _frame(0, [0, 1], []),
        _frame(1, [0, 0], [(0, 1)]),   # first period starts
        _frame(2, [0, 1], []),          # dies → survival = 0
        _frame(3, [0, 0], [(0, 1)]),   # reconnect, not tracked
        _frame(4, [0, 0], [(0, 1)]),   # T_eval
    ]
    m = compute_process_metrics(frames, T_conn=4)
    # only the first period (0 exact steps) is in edge_survival
    assert m['edge_birth_persistence_mean'] == 0.0
    assert m['edge_birth_censored_count'] == 0


# ── degree histogram ─────────────────────────────────────────────────────
def test_degree_histogram_triangle():
    """Triangle: each agent degree 2 → [0, 0, 3]."""
    frames = [
        _frame(0, [0, 0, 0], [(0, 1), (0, 2), (1, 2)]),
    ]
    m = compute_process_metrics(frames)
    assert m['final_degree_histogram'] == [0, 0, 3]


def test_degree_histogram_chain():
    """Chain 0-1-2 → degrees: 1,2,1 → [0, 2, 1]."""
    frames = [
        _frame(0, [0, 0, 0], [(0, 1), (1, 2)]),
    ]
    m = compute_process_metrics(frames)
    assert m['final_degree_histogram'] == [0, 2, 1]


def test_degree_histogram_isolated():
    """One isolated agent → [1]."""
    frames = [
        _frame(0, [0], [], remain_list=[0]),
    ]
    m = compute_process_metrics(frames)
    assert m['final_degree_histogram'] == [1]


# ── fragmentation exposure off-by-one ────────────────────────────────────
def test_fragmentation_exposure_failed_case():
    """Failed case: K=[3,2], T_eval=1 → exposure = K(0)-1 = 2."""
    frames = [
        _frame(0, [0, 1, 2], []),
        _frame(1, [0, 0, 1], [(0, 1)]),
    ]
    m = compute_process_metrics(frames, T_conn=None)
    assert m['fragmentation_exposure'] == 2  # K(0)-1 only, not K(1)-1


def test_fragmentation_exposure_success():
    """Success: K=[3,2,1], T_eval=2 → exposure = (3-1)+(2-1) = 3."""
    frames = [
        _frame(0, [0, 1, 2], []),
        _frame(1, [0, 0, 1], [(0, 1)]),
        _frame(2, [0, 0, 0], [(0, 1), (0, 2)]),
    ]
    m = compute_process_metrics(frames, T_conn=2)
    assert m['fragmentation_exposure'] == 3


# ── edge cases ───────────────────────────────────────────────────────────
def test_K0_is_1():
    """K(0)=1 → no births, no exposure, no persistence."""
    frames = [
        _frame(0, [0, 0], [(0, 1)]),
    ]
    m = compute_process_metrics(frames)
    assert m['K_initial'] == 1
    assert m['fragmentation_exposure'] == 0
    assert m['inter_fragment_edge_birth_count'] == 0
    assert m['edge_birth_persistence_mean'] is None
    assert m['edge_birth_efficiency'] == 0.0


def test_empty_frames():
    """Empty frame list → empty dict."""
    m = compute_process_metrics([])
    assert m == {}


def test_T_eval_zero():
    """eval_idx=0 → no births, K from single frame."""
    frames = [
        _frame(0, [0, 1, 2], []),
    ]
    m = compute_process_metrics(frames, T_conn=0)
    assert m['K_initial'] == 3
    assert m['fragmentation_exposure'] == 0
    assert m['inter_fragment_edge_birth_count'] == 0


if __name__ == '__main__':
    import inspect
    tests = [(name, fn) for name, fn in inspect.getmembers(
        sys.modules[__name__], inspect.isfunction)
             if name.startswith('test_')]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f'  PASS {name}')
        except AssertionError as e:
            print(f'  FAIL {name}: {e}')
        except Exception as e:
            print(f'  ERROR {name}: {e}')
    print(f'\n{passed}/{len(tests)} passed')
    sys.exit(0 if passed == len(tests) else 1)
