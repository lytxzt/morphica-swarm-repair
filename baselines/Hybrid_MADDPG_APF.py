import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import Utils
from Configurations import *

# ==========================================
# 1. MADDPG actor network (fully connected MLP)
# ==========================================
class SimpleMADDPGActor(nn.Module):
    def __init__(self, obs_dim, action_dim=3, hidden_dim=128):
        super(SimpleMADDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        """obs: [batch_size, obs_dim]"""
        return self.net(obs)

# ==========================================
# 2. Hybrid MADDPG + APF entry point
# ==========================================
class Hybrid_MADDPG_APF:
    """
    OPTIMIZED version: vectorized O(n^2) distance computations and batched actor inference.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = config_dimension * 3
        self.actor = SimpleMADDPGActor(obs_dim=obs_dim, action_dim=config_dimension).to(self.device)
        self.actor.eval()

        # APF parameters
        self.k_att = 0.8
        self.k_rep = 15.0
        self.safe_distance = 15.0
        self.max_speed = config_constant_speed

    # ------------------------------------------------------------------
    def get_actions(self, true_positions, remain_list, subnets):
        """
        Fully vectorized action computation.
        All per-agent loops replaced with matrix operations.
        """
        n_remain = len(remain_list)
        if n_remain == 0:
            return np.zeros((config_num_of_agents, config_dimension)), 0.0

        # Extract remain positions as [n_remain, D] matrix
        pos = np.array([true_positions[i] for i in remain_list])  # [n, D]

        # --- Vectorized pairwise distance matrix ---
        # D_ij = ||pos_i - pos_j||
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]      # [n, n, D]
        dist = np.linalg.norm(diff, axis=2)                        # [n, n]

        # --------------------------------------------------------------
        # 1. Batch local observations (replaces per-agent get_local_observation)
        # --------------------------------------------------------------
        nbr_mask = (dist <= config_communication_range) & (np.eye(n_remain) == 0)  # [n, n], exclude self
        nbr_counts = nbr_mask.sum(axis=1, keepdims=True)                            # [n, 1]
        nbr_counts = np.maximum(nbr_counts, 1.0)                                     # avoid /0

        # avg_neighbor_diff[i] = mean(pos_j - pos_i) over neighbors j
        avg_neighbor_diff = (nbr_mask[:, :, np.newaxis] * diff).sum(axis=1) / nbr_counts  # [n, D]

        # obs = [pos, vel(zero), avg_neighbor_diff] — batched
        vel = np.zeros_like(pos)
        obs_np = np.concatenate([pos, vel, avg_neighbor_diff], axis=1)   # [n, 3D]

        # Single batched actor forward pass
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_np).to(self.device)
            v_rl = self.actor(obs_tensor).cpu().numpy() * self.max_speed  # [n, D]

        # --------------------------------------------------------------
        # 2. Vectorized APF computation
        # --------------------------------------------------------------
        f_att = np.zeros((n_remain, config_dimension))
        f_rep = np.zeros((n_remain, config_dimension))

        # --- Attraction: to centroid of other subnets ---
        if len(subnets) > 1 and subnets:
            # Build subnet membership mask: [n_subnets, n_remain]
            subnet_masks = []
            for subnet in subnets:
                mask = np.zeros(n_remain, dtype=bool)
                for idx, agent_id in enumerate(remain_list):
                    if agent_id in subnet:
                        mask[idx] = True
                subnet_masks.append(mask)

            for s_idx, mask_s in enumerate(subnet_masks):
                other_mask = np.zeros(n_remain, dtype=bool)
                for o_idx, mask_o in enumerate(subnet_masks):
                    if o_idx != s_idx:
                        other_mask |= mask_o

                if other_mask.any():
                    center_others = pos[other_mask].mean(axis=0)         # [D]
                    # For each agent in this subnet, compute attraction to center_others
                    idx_in_subnet = np.where(mask_s)[0]
                    if len(idx_in_subnet) > 0:
                        diff_to_center = center_others[np.newaxis, :] - pos[idx_in_subnet]  # [k, D]
                        dist_to_center = np.linalg.norm(diff_to_center, axis=1, keepdims=True)  # [k, 1]
                        dist_to_center = np.maximum(dist_to_center, 1e-8)
                        f_att[idx_in_subnet] += self.k_att * (diff_to_center / dist_to_center)

        # --- Repulsion: vectorized ---
        close_mask = (dist > 0) & (dist < self.safe_distance)  # [n, n]
        # force_mag = k_rep * (1/d - 1/d0) * (1/d^2)
        dist_safe = np.where(close_mask, dist, np.inf)
        inv_d = 1.0 / np.maximum(dist_safe, 1e-8)
        force_mag = self.k_rep * (inv_d - 1.0 / self.safe_distance) * (inv_d ** 2)  # [n, n]
        # direction: unit vector from j to i = (pos_i - pos_j) / d = -diff / d
        unit_dir = -diff / np.maximum(dist[:, :, np.newaxis], 1e-8)               # [n, n, D]
        f_rep = (force_mag[:, :, np.newaxis] * unit_dir).sum(axis=1)              # [n, D]

        # --- Combine APF ---
        v_apf = f_att + f_rep                                                      # [n, D]
        v_apf_mag = np.linalg.norm(v_apf, axis=1, keepdims=True)
        v_apf_mag = np.maximum(v_apf_mag, 1e-8)
        v_apf = np.where(v_apf_mag > self.max_speed,
                         (v_apf / v_apf_mag) * self.max_speed,
                         v_apf)

        # --------------------------------------------------------------
        # 3. Hybrid combination
        # --------------------------------------------------------------
        alpha = 0.5
        v_final = alpha * v_rl + (1 - alpha) * v_apf  # [n, D]

        # Normalize to unit direction
        v_mag = np.linalg.norm(v_final, axis=1, keepdims=True)
        v_mag = np.maximum(v_mag, 1e-8)
        v_dir = v_final / v_mag

        # --------------------------------------------------------------
        # 4. Scatter back to full action array
        # --------------------------------------------------------------
        actions = np.zeros((config_num_of_agents, config_dimension))
        for idx, agent_id in enumerate(remain_list):
            actions[agent_id] = v_dir[idx]

        # Max time estimate
        temp_max_distance = np.max(v_mag) * config_dt
        max_time = temp_max_distance / config_constant_speed if config_constant_speed > 0 else 0

        return actions, max_time
