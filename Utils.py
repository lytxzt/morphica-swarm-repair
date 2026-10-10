import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from operator import itemgetter
from Configurations import *


def make_A_matrix(positions, num_of_agents, d):
    # A = np.zeros((num_of_agents, num_of_agents))
    # for i in range(num_of_agents):
    #     for j in range(i, num_of_agents):
    #         if j == i:
    #             A[i, j] = 0
    #         else:
    #             distance = np.linalg.norm(positions[i, :] - positions[j, :])
    #             if distance <= d:
    #                 A[i, j] = 1
    #                 A[j, i] = 1
    m, n = positions.shape
    G = np.matmul(positions, positions.T)
    H = np.tile(np.diag(G), (m, 1))
    dist_sq = H + H.T - 2 * G
    D = np.sqrt(np.maximum(dist_sq, 0.0))
    A = np.where(D > d, 0, 1.0)
    return deepcopy(A)


def make_D_matrix(A, num_of_agents):
    # D = np.zeros((num_of_agents, num_of_agents))
    # for i in range(num_of_agents):
    #     D[i, i] = np.sum(A[i])
    D = np.diag(np.sum(A, axis=1))
    return deepcopy(D)


def make_Ddot_matrix(A, num_of_agents):
    D = np.zeros((num_of_agents, num_of_agents))
    for i in range(num_of_agents):
        D[i, i] = 1 / np.sum(A[i])
    return deepcopy(D)


def count_clusters_from_A(A):
    """Count connected components via BFS from adjacency matrix. O(N+E)."""
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    n_components = 0
    for start in range(n):
        if visited[start]:
            continue
        n_components += 1
        stack = [start]
        visited[start] = True
        while stack:
            v = stack.pop()
            neighbors = np.where((A[v] > 0) & ~visited)[0]
            visited[neighbors] = True
            stack.extend(neighbors.tolist())
    return n_components == 1, n_components


def check_number_of_clusters(L, num_of_agents):
    """
    Count connected components via BFS on adjacency matrix.
    O(N+E) instead of O(N³) eigendecomposition.
    """
    n = num_of_agents
    # Recover adjacency from Laplacian: L = D - A, and D has row sums on diagonal,
    # so A = -L with diagonal zeroed, then binarize.
    visited = np.zeros(n, dtype=bool)
    n_components = 0
    for start in range(n):
        if visited[start]:
            continue
        n_components += 1
        # BFS from start
        queue = [start]
        visited[start] = True
        while queue:
            v = queue.pop()
            # L[v,:] has -1 for edges, deg on diagonal. Check non-diagonal entries.
            for u in range(n):
                if u != v and not visited[u] and L[v, u] < -0.5:
                    visited[u] = True
                    queue.append(u)
    return n_components == 1, n_components


def calculate_d_max(positions):
    # Vectorized pairwise max distance
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 0)
    return float(np.max(dist))


def find_d_min(positions, num):
    """Binary search on sorted distances to find min d that connects all nodes.
    O(N² log N) — replaces O(N⁴) smallest_d_algorithm."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_mat, 0)
    triu = dist_mat[np.triu_indices(num, k=1)]
    sorted_dists = np.sort(triu)
    lo, hi = 0, len(sorted_dists) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        _, nc = count_clusters_from_A(dist_mat <= sorted_dists[mid])
        if nc == 1:
            hi = mid
        else:
            lo = mid + 1
    return float(sorted_dists[lo])


def check_if_a_connected_graph(remain_positions, remain_num):
    A = make_A_matrix(remain_positions, remain_num, config_communication_range)
    D = make_D_matrix(A, remain_num)
    L = D - A
    connected_flag, num_of_clusters = check_number_of_clusters(L, remain_num)
    return deepcopy(connected_flag), deepcopy(num_of_clusters)


def split_the_positions_into_clusters(positions, num_of_clusters, A):
    positions_with_clusters = []
    remain_list = [i for i in range(len(positions))]
    if num_of_clusters <= 1:
        return None
    else:
        for k in range(num_of_clusters):
            temp_positions = []

            visited = np.zeros(len(remain_list))
            counter = 0
            stack = Stack()
            stack.push(remain_list[0])
            visited[0] = 1
            counter += 1

            while stack.length() != 0:
                current = stack.top_element()
                flag = True
                temp_counter = 0
                for i in remain_list:
                    if A[current, i] == 1 and visited[temp_counter] == 0:
                        visited[temp_counter] = 1
                        counter += 1
                        stack.push(i)
                        flag = False
                        break
                    temp_counter += 1
                if flag:
                    stack.pop()

            visited_node = []
            for j in range(len(remain_list)):
                if visited[j] == 1:
                    visited_node.append(remain_list[j])
            for j in range(counter):
                remain_list.remove(visited_node[j])
                temp_positions.append(deepcopy(positions[visited_node[j]]))
            positions_with_clusters.append(deepcopy(np.array(temp_positions)))
        return deepcopy(positions_with_clusters)


class Stack:
    def __init__(self):
        self.memory = []

    def push(self, num):
        self.memory.append(num)

    def pop(self):
        if len(self.memory) == 0:
            return None
        else:
            temp = self.memory[-1]
            del self.memory[-1]
            return temp

    def length(self):
        return len(self.memory)

    def top_element(self):
        return self.memory[-1]


def split_the_positions_into_clusters_and_indexes(positions, num_of_clusters, A):
    positions_with_clusters = []
    cluster_index = []
    remain_list = [i for i in range(len(positions))]
    if num_of_clusters <= 1:
        positions_with_clusters.append(deepcopy(positions))
        cluster_index.append(remain_list)
        return deepcopy(positions_with_clusters), deepcopy(cluster_index)
    else:
        for k in range(num_of_clusters):
            temp_positions = []
            temp_index = []

            visited = np.zeros(len(remain_list))
            counter = 0
            stack = Stack()
            stack.push(remain_list[0])
            visited[0] = 1
            counter += 1

            while stack.length() != 0:
                current = stack.top_element()
                flag = True
                temp_counter = 0
                for i in remain_list:
                    if A[current, i] == 1 and visited[temp_counter] == 0:
                        visited[temp_counter] = 1
                        counter += 1
                        stack.push(i)
                        flag = False
                        break
                    temp_counter += 1
                if flag:
                    stack.pop()

            visited_node = []
            for j in range(len(remain_list)):
                if visited[j] == 1:
                    visited_node.append(remain_list[j])
                    temp_index.append(remain_list[j])
            for j in range(counter):
                remain_list.remove(visited_node[j])
                temp_positions.append(deepcopy(positions[visited_node[j]]))

            positions_with_clusters.append(deepcopy(np.array(temp_positions)))
            cluster_index.append(deepcopy(temp_index))
        return deepcopy(positions_with_clusters), deepcopy(cluster_index)


def intersection_set(listA, listB):
    intersection_list = [i for i in listA if i in listB]
    return deepcopy(intersection_list)


def difference_set(listA, listB):
    difference_list = [i for i in listA if i not in listB]
    return deepcopy(difference_list)


def union_set(listA, listB):
    union_set = [i for i in listA]
    for i in listB:
        if i not in listA:
            union_set.append(deepcopy(i))
    return deepcopy(union_set)


def smallest_d_algorithm(positions, num, d0):
    A = make_A_matrix(positions, num, d0)
    d_min = deepcopy(d0)
    unsorted_list = []
    dis = []
    sorted_list = []
    for i in range(num - 1):
        for j in range(i + 1, num):
            unsorted_list.append(deepcopy({"start": i,
                                           "end": j,
                                           "distance": np.linalg.norm(positions[i] - positions[j])}))
            dis.append(deepcopy(np.linalg.norm(positions[i] - positions[j])))
    sorted_index = [index for index, value in sorted(enumerate(dis), key=itemgetter(1))]
    for i in range(len(sorted_index)):
        sorted_list.append(unsorted_list[sorted_index[i]])

    # find the threshold
    threshold_for_d0 = 0
    for i in range(len(sorted_index)):
        if sorted_list[i]["distance"] > d_min:
            threshold_for_d0 = deepcopy(i)
            break
    for i in range(threshold_for_d0, len(sorted_index)):
        A[sorted_list[i]["start"], sorted_list[i]["end"]] = 1
        A[sorted_list[i]["end"], sorted_list[i]["start"]] = 1
        D = make_D_matrix(A, num)
        L = D - A
        connected_flag, num_cluster = check_number_of_clusters(L, num)
        if connected_flag:
            d_min = deepcopy(sorted_list[i]["distance"])
            break
    return deepcopy(d_min)

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


def make_khop_A_matrix(all_neighbor, positions, num_remain, khop=5):
    # all_neighbor = calculate_khop_neighbour(positions, d)

    num_of_agents = len(positions)
    A = np.zeros((num_of_agents, num_of_agents))

    for i in range(num_remain):
        for j in range(num_remain, num_of_agents-1):
            hop = 1
            # if np.linalg.norm(positions[j] - central_point) >= embedding_distence : continue
            # dis = np.linalg.norm(positions[j] - central_point) / embedding_distence 
            # dis = np.cos(dis*np.pi/2)
            while(j not in all_neighbor[i][hop-1]):hop+=1
            if hop <= khop:
                A[i, j] = 1
                A[j, i] = 1

    for i in range(num_of_agents-1):
        A[i, num_of_agents-1] = 1
        A[num_of_agents-1, i] = 1

    # print(A)
    return deepcopy(A)


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

    try:
        e_vals, e_vecs = torch.linalg.eigh(L)
    except torch._C._LinAlgError:
        # Regularize ill-conditioned Laplacian
        L_reg = L + 1e-5 * torch.eye(m, device=L.device, dtype=L.dtype)
        try:
            e_vals, e_vecs = torch.linalg.eigh(L_reg)
        except torch._C._LinAlgError:
            # Fallback to numpy eigh (more robust for near-singular matrices)
            L_np = L.cpu().numpy()
            e_vals_np, _ = np.linalg.eigh(L_np)
            e_vals = torch.from_numpy(e_vals_np).to(L.device)

    num = torch.sum(torch.where(e_vals.real < 0.0001, 1, 0))
    return e_vals.real, num
