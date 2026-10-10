"""UGV experiment constants  (UGV_experiment_plan.md §5).

Separated from the runner so benchmark and rendering scripts can import
vehicle parameters without depending on the full CLI.
"""

# UGV dynamics defaults
UGV_DT = 0.5                # s per integration step
UGV_V_MAX = 2.0              # m/s physical speed cap
UGV_OMEGA_MAX = 1.2          # rad/s yaw-rate cap
UGV_AGENT_RADIUS = 0.6       # m  collision radius
UGV_MAX_SLOPE_DEG = 25       # degree  slope above this → violation
EPS = 1e-8

# Algorithm name → mode  (mirrors Experiment_Statistic._ALGORITHM_NAME_TO_MODE)
ALGO_MODE = {
    'HERO': 1, 'CEN': 2, 'SIDR': 3, 'CR-MGC': 6, 'DEMD': 7,
    'ML-DAGL': 8, 'GDR-TS': 12, 'Hybrid-MADDPG-APF': 13, 'MORPHICA': 14,
}

# Modes that plan once and cache target positions  (§5 one-shot adapter)
ONE_SHOT_MODES = frozenset({4, 5, 6, 7, 8, 12})

# Arrival threshold for one-shot waypoints  (matching Swarm.py internals)
WAYPOINT_ARRIVE_THRESH = 1.0      # switch to fine positioning
WAYPOINT_STOP_THRESH = 0.01       # considered arrived
