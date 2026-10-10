import numpy as np
import torch
from copy import deepcopy

import Utils
from Configurations import *

# Damage Attention Module
def DAM(global_positions, remain_list, khop=1):
    remain_positions = []
    for i in remain_list:
        remain_positions.append(deepcopy(global_positions[i]))
    
    num_remain = len(remain_list)
    
    fixed_positions = deepcopy(remain_positions)
    for i in range(len(global_positions)):
        if i not in remain_list:
            fixed_positions.append(deepcopy(global_positions[i]))

    # virtual central point embedding
    fixed_positions.append(config_central_point)

    num_of_agents = len(fixed_positions)
    
    fixed_positions = np.array(fixed_positions)
    remain_positions = np.array(remain_positions)

    # calculate multi-hop neighborship
    all_neighbor = calculate_khop_neighbour(fixed_positions, config_communication_range)
    # batch_size = max([len(hop) for hop in all_neighbor])-1

    # damage attentive adjacent matrix
    A = make_khop_DA_A_matrix(all_neighbor, fixed_positions, num_remain, khop)

    # proposed bipartite GCN
    D = Utils.make_D_matrix(A, num_of_agents)
    L = D - A
    A_norm = np.linalg.norm(A, ord=np.inf)
    epsilon = 0.5
    A_hat = np.eye(num_of_agents) - epsilon / A_norm * L

    # classic GCN 
    # Ddot = Utils.make_Ddot_matrix(A, num_of_agents)
    # A_hat = np.eye(num_of_agents) - Ddot**(1/2) @ A @ Ddot**(1/2)

    return remain_positions, fixed_positions, A_hat


# Dilated Damage Attention Module (DDAM)
def dilated_DAM(global_positions, remain_list, khop=1, batch_size=10):
    remain_positions = []
    for i in remain_list:
        remain_positions.append(deepcopy(global_positions[i]))
    
    num_remain = len(remain_list)
    
    fixed_positions = deepcopy(remain_positions)
    for i in range(len(global_positions)):
        if i not in remain_list:
            fixed_positions.append(deepcopy(global_positions[i]))

    # virtual central point embedding
    fixed_positions.append(config_central_point)

    num_of_agents = len(fixed_positions)
    
    fixed_positions = np.array(fixed_positions)
    remain_positions = np.array(remain_positions)

    # calculate multi-hop neighborship
    all_neighbor = calculate_khop_neighbour(fixed_positions, config_communication_range)
    # batch_size = max([len(hop) for hop in all_neighbor])-1

    # multi-hop dilated GCN
    A_dilated = []

    for k in range(khop, khop+batch_size):
        # damage attentive adjacent matrix
        A = make_khop_DA_A_matrix(all_neighbor, fixed_positions, num_remain, k)

        # proposed bipartite GCN
        D = Utils.make_D_matrix(A, num_of_agents)
        L = D - A
        A_norm = np.linalg.norm(A, ord=np.inf)
        epsilon = 0.5
        A_dilated_k = np.eye(num_of_agents) - epsilon / A_norm * L

        # classic GCN 
        # Ddot = Utils.make_Ddot_matrix(A, num_of_agents)
        # A_dilated_k = np.eye(num_of_agents) - Ddot**(1/2) @ A @ Ddot**(1/2)

        A_dilated.append(A_dilated_k)

    A_hat = torch.block_diag(torch.FloatTensor(A_dilated[0]))
    for k in range(1, batch_size):
        A_hat = torch.block_diag(A_hat, torch.FloatTensor(A_dilated[k]))

    fixed_positions = np.tile(fixed_positions, (batch_size,1))

    return remain_positions, fixed_positions, A_hat


# khop dilated topology
def calculate_khop_neighbour(positions, d):
    num_of_agents = len(positions)
    # Vectorized 1-hop
    neighbour_i_1hop = []
    for i in range(num_of_agents):
        diffs = positions[i, :] - positions
        dists = np.linalg.norm(diffs, axis=1)
        neighbours = list(np.where((dists > 0) & (dists <= d))[0])
        neighbour_i_1hop.append(neighbours)

    all_neighbour = []
    for i in range(num_of_agents):
        neighbour_i_all = []
        seen = set(neighbour_i_1hop[i])
        seen.add(i)
        frontier = neighbour_i_1hop[i][:]

        while frontier:
            neighbour_i_all.append(frontier[:])
            next_frontier = []
            for j in frontier:
                for k in neighbour_i_1hop[j]:
                    if k not in seen:
                        seen.add(k)
                        next_frontier.append(k)
            frontier = next_frontier

        all_neighbour.append(neighbour_i_all)

    return all_neighbour


# khop dilated damage attention topology
def make_khop_DA_A_matrix(all_neighbor, positions, num_remain, khop=5):
    num_of_agents = len(positions)
    A = np.zeros((num_of_agents, num_of_agents))

    # Precompute hop distances for remain agents
    hop_dist = []
    for i in range(num_remain):
        dist_map = {}
        for hop_idx, hop_nodes in enumerate(all_neighbor[i]):
            for node in hop_nodes:
                if node not in dist_map:
                    dist_map[node] = hop_idx + 1
        hop_dist.append(dist_map)

    for i in range(num_remain):
        hd_i = hop_dist[i]
        for j in range(num_remain, num_of_agents - 1):
            if hd_i.get(j, khop + 1) <= khop:
                A[i, j] = 1
                A[j, i] = 1

    for i in range(num_of_agents - 1):
        A[i, num_of_agents - 1] = 1
        A[num_of_agents-1, i] = 1

    return deepcopy(A)