import argparse
import pandas as pd
import numpy as np
import random
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--num-agents', '--num_agents', dest='num_agents', type=int, help='Override number of agents')
parser.add_argument('--dimension', type=int, help='Override swarm dimension')
parser.add_argument('--destroyed', type=int, help='Override number of destructed UAVs')
parser.add_argument('--algorithm', type=str, help='Experiment algorithm name')
parser.add_argument('--seed', type=int, help='Override random seed')
parser.add_argument('--communication-range', '--cr', dest='communication_range', type=int, help='Override communication range (default 120)')
parser.add_argument(
    '--ablation',
    type=str,
    default='full',
    choices=[
        'full',
        'no-S',
        'no-P',
        'no-R',
        'vector-scout',
        'multi-bearer',
        # DIRA_NMI legacy (still accepted but unused for MORPHICA):
        'no-anchor',
        'no-scout',
        'no-align',
        'anchor-only',
        'constant-scouting',
        'no-size',
        'no-neighbour-support',
    ],
    help='Ablation mode (MORPHICA: full, no-S, no-P, no-R, vector-scout, multi-bearer; DIRA_NMI: legacy modes)',
)
# Default weights w_A=0.5, w_S=1.0, w_L=0.0 from iter4 weight scan (best avg steps).
parser.add_argument('--weights', type=str, default='0.5,1.0,0.0', help='DIRA_NMI weights w_A,w_S,w_L')
parser.add_argument(
    '--no-velocity-alignment',
    action='store_true',
    help='Disable local velocity alignment in DIRA_NMI',
)
args, _unknown_args = parser.parse_known_args()


def _parse_dira_nmi_weights(raw_weights):
    try:
        values = [float(item.strip()) for item in raw_weights.split(',')]
    except ValueError:
        parser.error('--weights must be three comma-separated floats, e.g. 1.0,1.0,1.0')

    if len(values) != 3:
        parser.error('--weights must contain exactly three values: w_A,w_S,w_L')

    return values

"""
specify a certain GPU
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# random seed
config_seed = args.seed if args.seed is not None else 1
np.random.seed(config_seed)
random.seed(config_seed + 1)
# torch.manual_seed(1)

# configurations on swarm
config_num_of_agents = 200
config_communication_range = 120
if args.debug:
    config_num_of_agents = 20
if args.num_agents is not None:
    config_num_of_agents = int(args.num_agents)
if args.communication_range is not None:
    config_communication_range = int(args.communication_range)

# configurations on positions
config_dimension = 3
if args.debug:
    config_dimension = 2
if args.dimension is not None:
    config_dimension = int(args.dimension)
config_initial_swarm_positions = pd.read_excel(f"Configurations/swarm_positions_{config_num_of_agents}.xlsx")
config_initial_swarm_positions = config_initial_swarm_positions.values[:, 1:1+config_dimension]
config_initial_swarm_positions = np.array(config_initial_swarm_positions, dtype=np.float64)

# configurations on environment
config_range_list = {"20":320.0, "50":500.0, "100":750.0, "200":1000.0, "500":1600.0, "1000":2250.0, "2000":3200.0}
config_width = config_range_list[f'{config_num_of_agents}']
config_length = config_range_list[f'{config_num_of_agents}']
config_height = 100.0

config_space_range = np.array([config_width, config_length]) if config_dimension == 2 else np.array([config_width, config_length, config_height])
config_central_point = config_space_range * 0.5

config_constant_speed = 1
config_dt = 0.1  # time-step for Hybrid_MADDPG_APF max_time estimation

# configurations on destroy
config_maximum_destroy_num = 50
config_minimum_remain_num = 5

# configurations on meta learning
config_meta_training_epi = 500
# configurations on Graph Convolutional Network
config_K = 1 / 100
config_best_eta = 0.3
config_best_epsilon = 0.99

# configurations on one-off UEDs
config_num_destructed_UAVs = 50  # should be in the range of [1, num_of_UAVs-2]
if args.destroyed is not None:
    config_num_destructed_UAVs = int(args.destroyed)
config_normalize_positions = True

# configurations on DIRA_NMI group-fusion dynamics
config_algorithm_name = args.algorithm
config_dira_nmi_w_A, config_dira_nmi_w_S, config_dira_nmi_w_L = _parse_dira_nmi_weights(args.weights)
config_dira_nmi_ablation = args.ablation
config_morphica_ablation = args.ablation
config_dira_nmi_use_velocity_alignment = not args.no_velocity_alignment

# Non-ideal communication (E9)
parser.add_argument('--packet-loss', type=float, default=0.0,
                    help='Packet loss probability p ∈ [0,1]')
parser.add_argument('--delay', type=int, default=0,
                    help='Communication delay τ (steps of staleness)')
parser.add_argument('--position-noise', type=float, default=0.0,
                    help='Position error σ (meters, Gaussian std)')
_args2, _ = parser.parse_known_args()

_E9_packet_loss = _args2.packet_loss
_E9_delay_steps = _args2.delay
_E9_position_noise = _args2.position_noise

# configurations on training GCN
config_alpha_k = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.9, 0.95, 1, 1.5, 2, 3, 5]
config_gcn_repeat = 100
config_expension_alpha = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
config_d0_alpha = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

config_representation_step = 450

config_random_seed = range(7, 10007)

