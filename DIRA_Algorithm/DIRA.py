from copy import deepcopy
from Configurations import *
import numpy as np
import torch
import Utils

alpha = 0.1

FloatTensor = torch.cuda.FloatTensor

stop_exploration = False

class DIRA:
    def __init__(self, initial_positions):
        self.init_positions = deepcopy(initial_positions)
        self.num_of_agents = len(self.init_positions)

        self.num_of_subnets = 0
        self.positions_of_subnets = []
        self.index_of_subnets = []

    def check_subnets(self, node_global_positions, remain_list):
        remain_positions = []
        damage_positions = []

        damage_list = [i for i in range(len(node_global_positions)) if i not in remain_list]
        
        for i in remain_list:
            remain_positions.append(deepcopy(node_global_positions[i]))
        for i in damage_list:
            damage_positions.append(deepcopy(node_global_positions[i]))

        remain_positions = np.array(remain_positions)
        remain_positions_torch = torch.FloatTensor(deepcopy(remain_positions)).type(FloatTensor)
        
        A = Utils.make_A_matrix(remain_positions, len(remain_positions), config_communication_range)
        D = Utils.make_D_matrix(A, len(A))
        L = D - A
        flag, num = Utils.check_number_of_clusters(L, len(L))

        self.num_of_subnets = num
        # print(num)

        self.positions_of_subnets, index_of_subnets = Utils.split_the_positions_into_clusters_and_indexes(remain_positions, num, A)

        self.index_of_subnets = []
        for k in range(self.num_of_subnets):
            index = []
            for i in index_of_subnets[k]:
                index.append(remain_list[i])

            self.index_of_subnets.append(index)
        # print(num, len(self.positions_of_subnets), sum([len(self.index_of_subnets[k]) for k in range(num)]))
        # print(self.index_of_subnets, remain_list)

    def F_cohension(self):
        F_c = np.zeros((self.num_of_agents, config_dimension))
        w_c = np.zeros((self.num_of_agents, config_dimension))
        
        center_of_subnets = [np.mean(subnet, axis=0) for subnet in self.positions_of_subnets]
        # print(center_of_subnets)

        final_positions = deepcopy(config_central_point)

        for k in range(self.num_of_subnets):
            F = final_positions - center_of_subnets[k]
            w = np.min(np.linalg.norm(final_positions - self.positions_of_subnets[k], axis=1)) / (np.linalg.norm(F) + 1e-4)
            
            # print(F, np.linalg.norm(final_positions - self.positions_of_subnets[k], axis=1), w)

            for i in self.index_of_subnets[k]:
                F_c[i], w_c[i] = F, w

        # print(w_c)
        return deepcopy(F_c), deepcopy(w_c)
    
    def F_cohension_v2(self):
        F_c = np.zeros((self.num_of_agents, config_dimension))
        w_c = np.zeros((self.num_of_agents, config_dimension))
        
        center_of_subnets = [np.mean(subnet, axis=0) for subnet in self.positions_of_subnets]
        # print(center_of_subnets)

        final_positions = deepcopy(config_central_point)

        for k in range(self.num_of_subnets):
            # print(F, np.linalg.norm(final_positions - self.positions_of_subnets[k], axis=1), w)
            for i, u in enumerate(self.index_of_subnets[k]):
                F_c[u] = final_positions - self.positions_of_subnets[k][i]
                w_c[u] = np.min(np.linalg.norm(final_positions - self.positions_of_subnets[k], axis=1)) / np.linalg.norm(F_c[u])
                # F_c[i], w_c[i] = F, w

        # print(w_c)
        return deepcopy(F_c), deepcopy(w_c)
    
    def F_explore_v1(self, node_global_positions):
        F_e = np.zeros((self.num_of_agents, config_dimension))
        w_e = np.zeros((self.num_of_agents, config_dimension))

        center_of_subnets = [np.mean(subnet, axis=0) for subnet in self.positions_of_subnets]
        # print(center_of_subnets)

        for k in range(self.num_of_subnets):
            w = (len(self.index_of_subnets[k]) - 1) / self.num_of_agents
            for i in self.index_of_subnets[k]:
                F = node_global_positions[i] - center_of_subnets[k]
                
                F_e[i], w_e[i] = F, w

        # print(F_e, w_e)
        return deepcopy(F_e), deepcopy(w_e)

    def F_align(self, node_global_positions, remain_list):
        F_a = np.zeros((self.num_of_agents, config_dimension))
        w_a = np.zeros((self.num_of_agents, config_dimension))

        for i in remain_list:
            F_j, w_j = [], []
            for j in remain_list:
                if i==j: continue

                d = np.linalg.norm(node_global_positions[i, :] - node_global_positions[j, :])
                if config_communication_range * 0.5 < d <= config_communication_range:
                    F_j.append(node_global_positions[j, :] - node_global_positions[i, :])
                    w_j.append(d/config_communication_range)

            if len(F_j) == 0: F_j, w_j = [0, 0], [0]

            F_j, w_j = np.array(F_j), np.array(w_j)
            # print(F_j, w_j)

            F_a[i] = np.sum([w * F for w, F in zip(w_j, F_j)])
            w_a[i] = np.mean(w_j) - 0.5

        # print(F_a, w_a)
        return deepcopy(F_a), deepcopy(w_a)
    
    def F_align_v2(self, node_global_positions, remain_list):
        F_a = np.zeros((self.num_of_agents, config_dimension))
        w_a = np.zeros((self.num_of_agents, config_dimension))

        for i in remain_list:
            F_j, w_j = [], []
            for j in remain_list:
                if i==j: continue

                d = np.linalg.norm(node_global_positions[i, :] - node_global_positions[j, :])
                e = node_global_positions[j, :] - node_global_positions[i, :]

                if 90 < d <= 120:
                    F_j.append(-e)
                    w_j.append(np.log((120-d)/30))
                elif d <= 70:
                    F_j.append(e)
                    w_j.append(np.log(d/70))

            if len(F_j) == 0: F_j, w_j = [0, 0, 0], [0]

            F_j, w_j = np.array(F_j), np.array(w_j)
            # print(F_j, w_j)

            F_a[i] = np.sum([w * F for w, F in zip(w_j, F_j)], axis=0)
            # print(np.sum([w * F for w, F in zip(w_j, F_j)], axis=0))
            w_a[i] = np.mean(np.abs(w_j))

        # print(F_a, w_a)
        return deepcopy(F_a), deepcopy(w_a)


    def dira(self, node_global_positions, remain_list):
        """
        fly to the center of the swarm is thinks
        :param node_global_positions:
        :return:
        """
        self.check_subnets(node_global_positions, remain_list)

        # print(self.num_of_subnets)

        F_c, w_c = self.F_cohension()
        F_e, w_e = self.F_explore_v1(node_global_positions)
        F_a, w_a = self.F_align_v2(node_global_positions, remain_list)

        # F_c = np.zeros((self.num_of_agents, config_dimension))
        # F_c = np.zeros((self.num_of_agents, config_dimension))
        
        # F_e = np.zeros((self.num_of_agents, config_dimension))
        # w_e = np.zeros((self.num_of_agents, config_dimension))
        
        # F_a = np.zeros((self.num_of_agents, config_dimension))
        # w_a = np.zeros((self.num_of_agents, config_dimension))

        if stop_exploration:
            # limit subnet
            check_list = [i for i in range(self.num_of_subnets)]
            index_of_split_node = []
            for hop in range(20):
                speed = w_c * F_c + w_e * F_e + w_a * F_a
                # print(speed)
                for i in remain_list:
                    if np.linalg.norm(speed[i]) > 1:
                        speed[i] = speed[i] / np.linalg.norm(speed[i])

                check_list_tmp = []

                for k in check_list:
                    if len(self.index_of_subnets[k]) == 1: continue

                    future_positions = deepcopy(self.positions_of_subnets[k])
                    A = Utils.make_A_matrix(future_positions, len(future_positions), config_communication_range)

                    for i in range(len(self.index_of_subnets[k])):
                        future_positions[i] += speed[self.index_of_subnets[k][i]]

                    A_f = Utils.make_A_matrix(future_positions, len(future_positions), config_communication_range)
                    D = Utils.make_D_matrix(A_f, len(A_f))
                    L = D - A_f
                    flag, num_k = Utils.check_number_of_clusters(L, len(L))

                    if num_k > 1:
                        check_list_tmp.append(k)

                        if hop == 0:
                            for i in self.index_of_subnets[k]:
                                # F_e[i], F_a[i] = 0, 0
                                for j in self.index_of_subnets[k]:
                                    if i == j: continue

                                    if A[self.index_of_subnets[k].index(i)][self.index_of_subnets[k].index(j)] == 1 and A_f[self.index_of_subnets[k].index(i)][self.index_of_subnets[k].index(j)] == 0:
                                        index_of_split_node.append(i)
                                        F_e[i], F_a[i] = 0, 0
                                        # F_e[i] = 0

                        else:
                            for i in index_of_split_node:
                                if i not in self.index_of_subnets[k]: continue
                                for j in self.index_of_subnets[k]:
                                    if j in index_of_split_node: continue

                                    if A[self.index_of_subnets[k].index(i)][self.index_of_subnets[k].index(j)] == 1:
                                        index_of_split_node.append(j)
                                        F_e[j], F_a[j] = 0, 0
                                        # F_e[i] = 0

                        # print(self.index_of_subnets[k])
                        # print(hop, sorted(index_of_split_node))

                if len(check_list_tmp) == 0:
                    break

                check_list = deepcopy(check_list_tmp)

        else:
            speed = w_c * F_c + w_e * F_e + w_a * F_a
            for i in remain_list:
                if np.linalg.norm(speed[i]) > 1:
                    speed[i] = speed[i] / np.linalg.norm(speed[i])

        return deepcopy(speed)


def calculate_khop_neighbour(positions, d):
    num_of_agents = len(positions)
    neighbour_i_1hop = []
    for i in range(num_of_agents):
        neighbour = []
        for j in range(num_of_agents):
            if i==j: continue

            if np.linalg.norm(positions[i, :] - positions[j, :]) <= d:
                neighbour.append(j)

        neighbour_i_1hop.append(deepcopy(neighbour))

    all_neighbour = []
    for i in range(num_of_agents):
        cnt = 0
        neighbour_i_all = []
        neighbour_i_multihop = deepcopy(neighbour_i_1hop[i])

        while len(neighbour_i_multihop) > 0:
            neighbour_i_all.append(deepcopy(neighbour_i_multihop))
            neighbour_i_multihop = []

            for j in neighbour_i_all[cnt]:
                for k in neighbour_i_1hop[j]:
                    if k not in sum(neighbour_i_all, []) and k not in neighbour_i_multihop and k != i:
                        neighbour_i_multihop.append(k)

            cnt += 1
            
        all_neighbour.append(deepcopy(neighbour_i_all))

    return deepcopy(all_neighbour)


def check_number_of_clusters_torch(positions, d):
    m, n = positions.shape
    G = torch.mm(positions, positions.T)
    H = torch.tile(torch.diag(G), (m, 1))
    D = torch.sqrt(H + H.T - 2*G)
    # print(m, n, D)
    A = torch.where(D > d, 0, 1.0)
    # print(m, n, A)
    D = torch.diag(torch.sum(A, dim=1))
    # print(m, n, D)
    L = D - A

    e_vals, e_vecs = torch.linalg.eigh(L)
    # print(e_vals.real)
        
    num = torch.sum(torch.where(e_vals.real < 1e-8, 1, 0))
    # print(torch.sum(e_vals), torch.trace(L), torch.sum(A))
    return e_vals.real, num
