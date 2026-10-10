from copy import deepcopy
import time

import Utils
from Configurations import *
from Environment import Environment

from baselines.Centering import *
from baselines.SIDR import SIDR
from baselines.HERO import HERO

# Keep the original import order (and learned-method seed initialization)
# when the optional environment is installed. CPU methods work without it.
_LEARNED_IMPORT_ERROR = None
try:
    from baselines.CR_MGC import CR_MGC
    from baselines.DEMD import DEMD
    from baselines.GCN_2017 import GCN_2017
    from baselines.GAT import GAT
    from baselines.GDR_TS import GDR_TS
    from baselines.Hybrid_MADDPG_APF import Hybrid_MADDPG_APF
    from baselines.dagl.DAGL_Framework import DAGL
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in {'torch', 'torch_geometric', 'einops'}:
        raise
    _LEARNED_IMPORT_ERROR = exc

from controllers.MORPHICA import MORPHICA


class Swarm:
    def __init__(self, algorithm_mode=1, use_pretrained=False, khop=3):
        self.initial_positions = deepcopy(config_initial_swarm_positions)
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.remain_num = config_num_of_agents
        self.num_of_agents = config_num_of_agents
        self.max_destroy_num = config_maximum_destroy_num

        self.remain_positions = deepcopy(self.initial_positions)
        self.true_positions = deepcopy(self.initial_positions)
        
        self.algorithm_mode = algorithm_mode

        self.if_once_gcn = False
        self.once_destroy_gcn_speed = np.zeros((self.num_of_agents, config_dimension))
        self.max_time = 0

        self.khop = khop
        self._use_pretrained = use_pretrained
        if algorithm_mode not in {1, 2, 3, 14, 15}:
            self._initialize_learned_planners()

        self.hero = HERO(self.initial_positions)

        self.step_info = {}

        # MORPHICA-RPS (morphogenetic repair program)
        self.morphica = MORPHICA(self.initial_positions, ablation=config_morphica_ablation)

        # GDR-TS (Graph-based DRL Topology Self-healing)
        if algorithm_mode not in {1, 2, 3, 14, 15}:
            self._initialize_learned_actors()

        self.if_once_gcn_network = False
        self.once_destroy_gcn_network_speed = np.zeros((self.num_of_agents, config_dimension))

        self.best_final_positions = 0

        self.notice_destroy = False
        self.destination_positions = np.zeros((self.num_of_agents, config_dimension))
        self.inertia_counter = 0
        self.inertia = 100
        self.if_finish = [True for i in range(self.num_of_agents)]

    def _initialize_learned_planners(self):
        if _LEARNED_IMPORT_ERROR is not None:
            raise RuntimeError(
                'This method requires requirements-baselines.txt (Python >=3.9).'
            ) from _LEARNED_IMPORT_ERROR
        self.demd = DEMD()
        self.ddag = DAGL(use_pretrained=self._use_pretrained)
        self.cr_gcm = CR_MGC(use_meta=False)
        self.gcn_2017 = GCN_2017()
        self.gat = GAT()

    def _initialize_learned_actors(self):
        self.gdr_ts_model = GDR_TS()
        self.maddpg_apf_model = Hybrid_MADDPG_APF()

    def destroy_happens(self, destroy_list, environment_positions):
        self.notice_destroy = True
        for destroy_index in destroy_list:
            self.remain_list.remove(destroy_index)
        self.true_positions = deepcopy(environment_positions)
        self.remain_num = len(self.remain_list)
        # self.csds.notice_destroy(deepcopy(destroy_list))

    def update_true_positions(self, environment_positions):
        self.true_positions = deepcopy(environment_positions)

    def reset(self, change_algorithm_mode=False, algorithm_mode=0):
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.remain_num = config_num_of_agents
        self.positions = []
        self.mean_positions = []
        self.target_positions = []
        self.max_time = 0

        if change_algorithm_mode:
            self.algorithm_mode = algorithm_mode
            if algorithm_mode not in {1, 2, 3, 14, 15} and not hasattr(self, 'demd'):
                self._initialize_learned_planners()
                self._initialize_learned_actors()

        self.if_once_gcn = False
        self.once_destroy_gcn_speed = np.zeros((self.num_of_agents, config_dimension))

        self.if_once_gcn_network = False
        self.once_destroy_gcn_network_speed = np.zeros((self.num_of_agents, config_dimension))

    def take_actions(self):
        """
        take actions with global information (GI)
        :return: unit speed vectors
        """
        actions = np.zeros((self.num_of_agents, config_dimension))
        max_time = 0
        self.make_remain_positions()
        flag, num_cluster = Utils.check_if_a_connected_graph(deepcopy(self.remain_positions), len(self.remain_list))

        time_start = time.time()
        if flag:
            return deepcopy(actions), max_time
        else:
            # HERO
            if self.algorithm_mode == 1:
                actions_hero = self.hero.hero(
                    Utils.difference_set([i for i in range(self.num_of_agents)], self.remain_list), self.true_positions)

                #     actions[i] = 0.2 * centering_fly(self.true_positions, self.remain_list, i) + 0.8 * actions_hero[i]
                actions = 0.2 * centering_fly(self.true_positions, self.remain_list) + 0.8 * actions_hero


            # centering
            elif self.algorithm_mode == 2:
                    # actions[i] = centering_fly(self.true_positions, self.remain_list, i)
                actions = centering_fly(self.true_positions, self.remain_list)


            # SIDR
            elif self.algorithm_mode == 3:
                actions = SIDR(self.true_positions, self.remain_list)


            # GCN
            elif self.algorithm_mode == 4:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        if np.linalg.norm(
                                self.true_positions[self.remain_list[i]] - self.best_final_positions[i]) >= 0.55:
                            actions[self.remain_list[i]] = deepcopy(
                                self.once_destroy_gcn_network_speed[self.remain_list[i]])
                        # else:
                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.gcn_2017.gcn(deepcopy(self.true_positions),
                                                                                     deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)


            # GAT
            elif self.algorithm_mode == 5:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        if np.linalg.norm(
                                self.true_positions[self.remain_list[i]] - self.best_final_positions[i]) >= 0.55:
                            actions[self.remain_list[i]] = deepcopy(
                                self.once_destroy_gcn_network_speed[self.remain_list[i]])
                        # else:
                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.gat.gat(deepcopy(self.true_positions),
                                                                                     deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)

            
            # CR-MGC
            elif self.algorithm_mode == 6:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        if np.linalg.norm(self.true_positions[self.remain_list[i]] - self.best_final_positions[i]) >= 0.55:
                            actions[self.remain_list[i]] = deepcopy(self.once_destroy_gcn_network_speed[self.remain_list[i]])

                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.cr_gcm.cr_gcm(deepcopy(self.true_positions),
                                                                                 deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)


            # DEMD
            elif self.algorithm_mode == 7:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        d = np.linalg.norm(self.true_positions[self.remain_list[i]] - self.best_final_positions[i])
                        if d >= 1:
                            actions[self.remain_list[i]] = deepcopy(self.once_destroy_gcn_network_speed[self.remain_list[i]])
                        elif d > 0.0001:
                            actions[self.remain_list[i]] = deepcopy(self.best_final_positions[i] - self.true_positions[self.remain_list[i]])
                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    # actions, max_time, best_final_positions = self.demd.demd(deepcopy(self.true_positions), deepcopy(self.remain_list), self.khop)
                    actions, max_time, best_final_positions = self.demd.demd_adaptive(deepcopy(self.true_positions), deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)
                    
            
            # ML-DAGL
            elif self.algorithm_mode == 8:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        d = np.linalg.norm(self.true_positions[self.remain_list[i]] - self.best_final_positions[i])
                        if d >= 1:
                            actions[self.remain_list[i]] = deepcopy(self.once_destroy_gcn_network_speed[self.remain_list[i]])
                        elif d > 0.0001:
                            actions[self.remain_list[i]] = deepcopy(self.best_final_positions[i] - self.true_positions[self.remain_list[i]])

                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.ddag.dagl(deepcopy(self.true_positions), deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)


            # DF alg
            elif self.algorithm_mode == 14 or self.algorithm_mode == 15:
                if not self.if_once_gcn_network:
                    self.if_once_gcn_network = True
                    self.original_positions = deepcopy(self.true_positions)
                    self.velocities = np.zeros((self.num_of_agents, config_dimension))

                actions, step_info = self.morphica.group_fusion_step(
                    deepcopy(self.true_positions),
                    deepcopy(self.remain_list),
                )
                self.step_info = step_info
                self.velocities = deepcopy(actions)

            # GDR-TS (Graph-based DRL Topology Self-healing)
            elif self.algorithm_mode == 12:
                if self.if_once_gcn_network:
                    # Reuse cached plan: follow pre-computed speed vectors
                    for i in range(len(self.remain_list)):
                        if np.linalg.norm(self.true_positions[self.remain_list[i]] - self.best_final_positions[i]) >= 0.55:
                            actions[self.remain_list[i]] = deepcopy(
                                self.once_destroy_gcn_network_speed[self.remain_list[i]])
                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.gdr_ts_model.gdr_ts(
                        deepcopy(self.true_positions), deepcopy(self.remain_list))

                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)

            # Hybrid MADDPG + APF
            elif self.algorithm_mode == 13:
                # Compute subnets using Utils (equivalent to SubnetGraphBuilder)
                remain_positions = np.array([deepcopy(self.true_positions[i]) for i in self.remain_list])
                A = Utils.make_A_matrix(remain_positions, len(self.remain_list), config_communication_range)
                _, num_clusters = Utils.check_number_of_clusters(
                    Utils.make_D_matrix(A, len(A)) - A, len(A))
                if num_clusters > 0:
                    _, local_subnet_indexes = Utils.split_the_positions_into_clusters_and_indexes(
                        remain_positions, num_clusters, A)
                    # Convert local indices to global indices
                    subnets = [[self.remain_list[idx] for idx in subnet] for subnet in local_subnet_indexes]
                else:
                    subnets = []

                actions, max_time = self.maddpg_apf_model.get_actions(
                    deepcopy(self.true_positions),
                    deepcopy(self.remain_list),
                    subnets,
                )
                self.max_time = deepcopy(max_time)

            else:
                print("No such algorithm")

        time_end = time.time()

        return deepcopy(actions), deepcopy(self.max_time)

    def make_remain_positions(self):
        self.remain_positions = []
        for i in self.remain_list:
            self.remain_positions.append(deepcopy(self.true_positions[i]))
        self.remain_positions = np.array(self.remain_positions)

    def check_if_finish(self, cluster_index):
        flag = True
        for i in range(len(cluster_index)):
            if not self.if_finish[self.remain_list[cluster_index[i]]]:
                flag = False
                break
        return flag
    
    def check_number_of_clusters(self):
        m, n = self.remain_positions.shape
        G = np.matmul(self.remain_positions, self.remain_positions.T)
        H = np.tile(np.diag(G), (m, 1))
        dist_sq = H + H.T - 2 * G
        D_mat = np.sqrt(np.maximum(dist_sq, 0.0))
        A = D_mat <= config_communication_range
        np.fill_diagonal(A, False)
        # BFS component count: O(N²) instead of O(N³) eigendecomposition
        visited = np.zeros(m, dtype=bool)
        num = 0
        for start in range(m):
            if visited[start]:
                continue
            num += 1
            stack = [start]
            visited[start] = True
            while stack:
                v = stack.pop()
                neighbors = np.where(A[v] & ~visited)[0]
                visited[neighbors] = True
                stack.extend(neighbors.tolist())
        return None, num

    def solve(self, record_trajectory=False):
        max_step = int(0.8 * config_width)
        num_of_remain_agents = len(self.remain_list)

        positions = deepcopy(self.remain_positions)
        trajectory = [] if record_trajectory else None

        # t=0 initial state (before first move)
        if record_trajectory:
            self.make_remain_positions()
            _, init_num = self.check_number_of_clusters()
            trajectory.append({'step': 0, 'num_subnets': init_num})

        for step in range(max_step):
            actions, _ = self.take_actions()
            positions = self.true_positions + actions * config_constant_speed

            self.update_true_positions(positions)
            self.make_remain_positions()

            connected_flag, num = self.check_number_of_clusters()

            if record_trajectory:
                row = {'step': step + 1, 'num_subnets': num}
                info = getattr(self, 'step_info', {})
                if isinstance(info, dict) and 'f_ret_mean' in info:
                    row.update(info)
                    row['num_subnets'] = num
                trajectory.append(row)

            if num == 1:
                break

            print(f"step {step} ---num of sub-nets {num}", end='\r')

        if record_trajectory:
            return step, self.remain_positions, num, trajectory
        return step, self.remain_positions, num



if __name__ == "__main__":
    np.random.seed(57)
    random.seed(77)

    environment = Environment()
    swarm = Swarm()

    environment_positions = environment.reset()
    destroy_num, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=100)

    swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

    print(len(swarm.remain_list))

