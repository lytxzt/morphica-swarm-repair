from copy import deepcopy
import numpy as np
from Configurations import *


def centering_fly(node_global_positions, remain_list, index):
    """
    fly to the center of the swarm is thinks
    :param node_global_positions:
    :return:
    """
    remain_positions = []
    self_positions = node_global_positions[index]
    for i in remain_list:
        remain_positions.append(deepcopy(node_global_positions[i]))
    remain_positions = np.array(remain_positions)
    final_positions = np.mean(remain_positions, 0)
    flying_direction = (final_positions - self_positions)/np.linalg.norm(final_positions - self_positions)
    if np.linalg.norm(final_positions - self_positions) < 1:
        flying_direction = np.array([0 for _ in range(2)])
    return deepcopy(flying_direction)


def centering_fly_v2(node_global_positions, remain_list):
    """
    fly to the center of the swarm is thinks
    :param node_global_positions:
    :return:
    """
    remain_positions = []
    # self_positions = node_global_positions[index]
    # for i in remain_list:
    #     remain_positions.append(deepcopy(node_global_positions[i]))
    # remain_positions = np.array(remain_positions)
    remain_positions = np.take(node_global_positions, remain_list)
    final_positions = np.mean(remain_positions, 0)
    
    flying_direction = np.zeros((len(node_global_positions), config_dimension))
    for i in remain_list:
        if np.linalg.norm(final_positions - node_global_positions[i]) >= 1:
            flying_direction[i] = (final_positions - node_global_positions[i])/np.linalg.norm(final_positions - node_global_positions[i])
        
            # flying_direction[i] = np.array([0 for _ in range(config_dimension)])
    return deepcopy(flying_direction)
