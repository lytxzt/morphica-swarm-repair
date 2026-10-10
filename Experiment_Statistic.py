
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import argparse
import os
import json
import sys
import re
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from Environment import Environment
from Swarm import Swarm
from Configurations import *
import Utils

# ==== CLI / demo-mode setup ============================================
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--algorithm', type=str, default=None)
_parser.add_argument('--destroyed-values', type=int, nargs='+', default=None)
_parser.add_argument('--max-cases', type=int, default=None)
_parser.add_argument('--output-dir', type=str, default=None)
_parser.add_argument('--progress-log', type=str, default=None)
_parser.add_argument('--summary-json', type=str, default=None)
_parser.add_argument('--case-offset', type=int, default=0, help='Start from this seed index (for split runs)')
_parser.add_argument('--record-trajectory', action='store_true', help='Record per-step RPS phase dynamics to CSV')
_cli_args, _ = _parser.parse_known_args()

_DEMO_MODE = _cli_args.algorithm is not None

_ALGORITHM_NAME_TO_MODE = {
    "HERO": 1, "CEN": 2, "SIDR": 3, "GCN_2017": 4, "GAT": 5,
    "CR-MGC": 6, "DEMD": 7, "ML-DAGL": 8, "DF-scaled": 9,
    "DIRA": 10, "DIRA_NMI": 11, "GDR_TS": 12, "Hybrid_MADDPG_APF": 13,
    "MORPHICA": 14,
}

if _DEMO_MODE:
    _demo_start_time = time.time()
    _demo_attempt_count = 0
    _demo_elapsed_times = []
    if _cli_args.output_dir and not os.path.exists(_cli_args.output_dir):
        os.makedirs(_cli_args.output_dir)
    if _cli_args.progress_log:
        _d = os.path.dirname(_cli_args.progress_log)
        if _d and not os.path.exists(_d):
            os.makedirs(_d)

max_step = int(0.8 * config_width)
# max_step = 1

use_pretrained = False
draw = False

numpy_speedup = True

if _DEMO_MODE:
    _dnum_iter = _cli_args.destroyed_values if _cli_args.destroyed_values else [100]
    _mode_iter = [_ALGORITHM_NAME_TO_MODE.get(_cli_args.algorithm, 10)]
    _max_case = _cli_args.max_cases if _cli_args.max_cases else 10
else:
    _dnum_iter = range(10, 200, 10)
    _mode_iter = [8]
    _max_case = 50

for dnum in _dnum_iter:
# for dnum in [100]:
    for mode in _mode_iter:
    # for mode in [3]:
        config_algorithm_mode = mode
        algorithm_mode = {1: "HERO",
                          2: "CEN",
                          3: "SIDR",
                          4: "GCN_2017",
                          5: "GAT",
                          6: "CR-MGC",
                          7: "DEMD",
                          8: "ML-DAGL",
                          9: "DF-scaled",
                          10: "DIRA",
                          11: "DIRA_NMI",
                          14: "MORPHICA",
                          15: "MORPHICA",
                          12: "GDR_TS",
                          13: "Hybrid_MADDPG_APF"}

        print("CNS issue Starts...")
        print("------------------------------")
        print(f"Algorithm: {algorithm_mode[config_algorithm_mode]} with {dnum} UAVs destructed")

        # storage
        storage_random_seed = []
        storage_connect_step = []
        storage_connect_positons = []
        storage_node_degree = []
        # 由Roo添加：强化学习指标记录
        storage_rl_rewards = []
        storage_training_time = []

        # change the number of destructed UAVs
        config_num_destructed_UAVs = dnum  # should be in the range of [1, config_num_-2]

        # change the seed to alternate the UED
        # seed = [61, 29, 83, 3, 59, 22, 8, 96, 80, 20, 39, 19, 89, 75, 79, 55, 61, 74, 8, 89, 83, 3, 38, 88, 56, 68, 67, 46, 48, 63, 54, 43, 52, 72, 75, 21, 64, 44, 50, 77, 39, 14, 18, 66, 82, 51, 65, 90, 57, 35, 92, 74, 9, 64, 52, 91, 56, 87, 77, 82, 34, 38, 38, 97, 3, 85, 67, 15, 86, 21, 67, 68, 98, 99, 32, 100, 48, 19, 85, 67, 81, 25, 64, 23, 37, 87, 31, 8, 96, 47, 63, 57, 1, 66, 18, 89, 54, 60, 58, 25]
        seed = config_random_seed

        case = _cli_args.case_offset if _DEMO_MODE else 0
        max_case = _max_case
        if _DEMO_MODE:
            _result_path = os.path.join(_cli_args.output_dir, f"{algorithm_mode[config_algorithm_mode]}_d{dnum}_{max_case}cases.txt")
            _result_dir = os.path.dirname(_result_path)
            if _result_dir and not os.path.exists(_result_dir):
                os.makedirs(_result_dir)
        else:
            _result_path = f'./Logs/damage_3d/d{config_num_destructed_UAVs}/{algorithm_mode[config_algorithm_mode]}_d{config_num_destructed_UAVs}_v2.txt'

        # Resume: load existing results if output file exists
        if _DEMO_MODE and os.path.exists(_result_path):
            with open(_result_path) as f:
                old_text = f.read()
            m = re.search(r'case:\s*\[(.*?)\]', old_text, re.DOTALL)
            if m:
                old_cases = [int(x.strip()) for x in m.group(1).strip('[]').split(',') if x.strip()]
                if old_cases:
                    old_steps = [int(x.strip()) for x in re.search(r'connect_step:\s*\[(.*?)\]', old_text, re.DOTALL).group(1).strip('[]').split(',') if x.strip()]
                    storage_random_seed = old_cases[:]
                    storage_connect_step = old_steps[:]
                    storage_connect_positons = []  # truncated resume — skip positions
                    storage_node_degree = []
                    case = max(old_cases) + 1
                    print(f"Resumed: {len(storage_random_seed)} existing cases, continuing from seed index {case}")

        # for case in range(60):
        while len(storage_random_seed) < max_case and case < len(seed):
            environment = Environment()
            
            swarm = Swarm(algorithm_mode=config_algorithm_mode, use_pretrained=use_pretrained)
            num_cluster_list = []

            environment_positions = environment.reset()
            swarm.reset()

            np.random.seed(seed[case]+1718)
            random.seed(seed[case]+1718)
            # print(seed[case])

            # flag if break the CCN
            break_CCN_flag = True
            # num of connected steps
            num_connected_steps = 0

            destroy_num, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=config_num_destructed_UAVs)
            swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

            initial_remain_positions = []
            destroy_positions = []
            for i in range(config_num_of_agents):
                if i in destroy_list:
                    destroy_positions.append(deepcopy(config_initial_swarm_positions[i]))
                else:
                    initial_remain_positions.append(deepcopy(config_initial_swarm_positions[i]))
            initial_remain_positions = np.array(initial_remain_positions)
            destroy_positions = np.array(destroy_positions)
            A = Utils.make_A_matrix(initial_remain_positions, config_num_of_agents - config_num_destructed_UAVs, config_communication_range)
            num_cluster = environment.check_the_clusters()
            
            # check if the UED break the CCN of the USNET
            if num_cluster == 1:
                # print(f"case {case}: not distructed")
                if _DEMO_MODE:
                    _demo_attempt_count += 1
                    _msg = (f"skipped_attempt dnum={dnum} seed_index={case} "
                            f"valid_case_count={len(storage_random_seed)} "
                            f"reason=initial_connectivity_not_broken")
                    print(_msg)
                    if _cli_args.progress_log:
                        with open(_cli_args.progress_log, 'a') as _pf:
                            _pf.write(_msg + "\n")
                case += 1
                continue

            print("=======================================")

            positions_with_clusters = Utils.split_the_positions_into_clusters(initial_remain_positions, num_cluster, A)

            _case_start_time = time.time()
            if numpy_speedup:
                if _cli_args.record_trajectory:
                    step, final_positions, num, trajectory = swarm.solve(record_trajectory=True)
                else:
                    step, final_positions, num = swarm.solve()
                print(f"case {case}: step {step} ---num of sub-nets {num}")
            else:
                for step in range(max_step):
                    actions, max_time = swarm.take_actions()
                    # actions, max_time, time_consum = swarm.take_actions()
                    # with open(f"./Logs/time/{algorithm_mode[config_algorithm_mode]}.txt", 'a') as f:
                    #     print(time_consum, file=f)

                    environment_next_positions = environment.next_state(deepcopy(actions))
                    swarm.update_true_positions(environment_next_positions)

                    temp_cluster = environment.check_the_clusters()
                    num_cluster_list.append(temp_cluster)
                    # print("---------------------------------------")
                    if temp_cluster == 1:
                        print(f"case {case}: step {step} ---num of sub-nets {environment.check_the_clusters()}")
                    else:
                        num_connected_steps += 1
                        print(f"case {case}: step {step} ---num of sub-nets {environment.check_the_clusters()}", end='\r')
                        if step == max_step-1:
                            print(f"case {case}: step {step} ---num of sub-nets {environment.check_the_clusters()}")

                    if environment.check_the_clusters() == 1:
                        break

                    # update
                    environment.update()
                    environment_positions = deepcopy(environment_next_positions)

                # print("case %d: step %d ---num of clusters %d -- connected" % (case, step, environment.check_the_clusters()))
                
                final_positions = np.take(environment_positions, swarm.remain_list, axis=0)


            final_positions = np.array(final_positions)

            storage_random_seed.append(case)
            storage_connect_step.append(step)
            storage_connect_positons.append(final_positions)

            A = Utils.make_A_matrix(final_positions, len(final_positions), config_communication_range)
            degrees = [int(d) for d in np.sum(A, axis=0)]

            storage_node_degree.extend(deepcopy(degrees))

            print(f"avg: {np.mean(np.array(storage_connect_step))}")

            # Incremental write after each case (survives crashes)
            if _DEMO_MODE:
                with open(_result_path, 'w') as f:
                    print('case:\n', storage_random_seed, file=f)
                    print('\nconnect_step:\n', storage_connect_step, file=f)
                    print('\navg_connect_step:\n', np.mean(np.array(storage_connect_step)), file=f)
                    print('\ndegree_distribution:\n', storage_node_degree, file=f)
                    for i in range(len(storage_random_seed)):
                        print('\n=======================================', file=f)
                        print(f'case {storage_random_seed[i]} -- step {storage_connect_step[i]}', file=f)
                        if i < len(storage_connect_positons):
                            print('final positions:\n', storage_connect_positons[i], file=f)

                # Trajectory CSV
                if _cli_args.record_trajectory and 'trajectory' in dir():
                    traj_path = _result_path.replace('.txt', '_traj.csv')
                    write_header = not os.path.exists(traj_path)
                    TRAJ_COLS = ['step', 'num_subnets', 'num_remain', 'lcc_size', 'f_ret_mean', 'f_pol_mean', 'f_pol_bearer', 'f_scout_mean', 'bearer_count']
                    with open(traj_path, 'a') as tf:
                        if write_header:
                            tf.write('case\t' + '\t'.join(TRAJ_COLS) + '\n')
                        case_idx = storage_random_seed[-1]
                        for row in trajectory:
                            vals = []
                            for c in TRAJ_COLS:
                                if c == 'lcc_size':
                                    sizes = row.get('subnet_sizes', [])
                                    vals.append(str(max(sizes)) if sizes else '')
                                else:
                                    vals.append(str(row.get(c, '')))
                            tf.write(f"{case_idx}\t" + '\t'.join(vals) + '\n')

            _case_elapsed = time.time() - _case_start_time
            if _DEMO_MODE:
                _demo_attempt_count += 1
                _demo_elapsed_times.append(_case_elapsed)
                _running_avg = np.mean(_demo_elapsed_times)
                _est_50 = _running_avg * 50
                _output_path = os.path.abspath(_result_path) if _cli_args.output_dir else "N/A"
                _msg = (f"dnum={dnum} seed_index={case} "
                        f"valid_case_count={len(storage_random_seed)} "
                        f"step={step} subnets={num} "
                        f"case_elapsed_seconds={_case_elapsed:.2f} "
                        f"running_avg_elapsed_seconds={_running_avg:.2f} "
                        f"estimated_seconds_for_50_cases={_est_50:.1f} "
                        f"output_path={_output_path}")
                print(_msg)
                if _cli_args.progress_log:
                    with open(_cli_args.progress_log, 'a') as _pf:
                        _pf.write(_msg + "\n")

            case += 1

            if draw:
                plt.scatter(initial_remain_positions[:,0], initial_remain_positions[:,1], c='black')
                plt.scatter(final_positions[:,0], final_positions[:,1], c='g')
                plt.text(10, 10, f'best time: {max_time}')
                plt.xlim(0, 1000)
                plt.ylim(0, 1000)
                plt.show()

        # with open(f'./Logs/scalable_3d/N{config_num_of_agents}/{algorithm_mode[config_algorithm_mode]}_d{config_num_destructed_UAVs}_v1.txt', 'w') as f:
        with open(_result_path, 'w') as f:
            print('case:\n', storage_random_seed, file=f)
            print('\nconnect_step:\n', storage_connect_step, file=f)
            print('\navg_connect_step:\n', np.mean(np.array(storage_connect_step)), file=f)
            print('\ndegree_distribution:\n', storage_node_degree, file=f)

            # 由Roo添加：强化学习指标输出
            if storage_rl_rewards:
                print('\nrl_rewards:\n', storage_rl_rewards, file=f)
                print('\navg_rl_reward:\n', np.mean(storage_rl_rewards), file=f)
                print('\ntraining_time:\n', storage_training_time, file=f)
                print('\navg_training_time:\n', np.mean(storage_training_time), file=f)

            for i in range(len(storage_random_seed)):
                print('\n=======================================', file=f)
                print(f'case {storage_random_seed[i]} -- step {storage_connect_step[i]}', file=f)
                if i < len(storage_rl_rewards):
                    print(f'RL reward: {storage_rl_rewards[i]}, Training time: {storage_training_time[i]:.2f}s', file=f)
                if i < len(storage_connect_positons):
                    print('final positions:\n', storage_connect_positons[i], file=f)

        # ==== demo summary JSON ========================================
        if _DEMO_MODE and _cli_args.summary_json:
            _total_elapsed = time.time() - _demo_start_time
            _avg_elapsed = np.mean(_demo_elapsed_times) if _demo_elapsed_times else 0.0
            _est_50 = _avg_elapsed * 50
            _est_19x50 = _avg_elapsed * 19 * 50
            _summary = {
                "algorithm": _cli_args.algorithm,
                "algorithm_mode": config_algorithm_mode,
                "destroyed_values": _cli_args.destroyed_values if _cli_args.destroyed_values else [100],
                "max_cases": max_case,
                "valid_cases": len(storage_random_seed),
                "attempts_used": _demo_attempt_count,
                "total_elapsed_seconds": round(_total_elapsed, 2),
                "avg_elapsed_seconds_per_valid_case": round(_avg_elapsed, 2),
                "estimated_seconds_for_50_valid_cases": round(_est_50, 1),
                "estimated_seconds_for_original_19x50_sweep": round(_est_19x50, 1),
                "result_text_path": os.path.abspath(_result_path),
                "progress_log_path": os.path.abspath(_cli_args.progress_log) if _cli_args.progress_log else None,
                "python_executable": sys.executable,
            }
            with open(_cli_args.summary_json, 'w') as _sf:
                json.dump(_summary, _sf, indent=2)
            print(f"summary written to {_cli_args.summary_json}")
