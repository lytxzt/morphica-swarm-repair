from copy import deepcopy
from torch.optim import Adam
import Utils
from Configurations import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.norm import LayerNorm

# Optimized hyperparameters (speed-focused)
best_hidden_dimension = 128   # was 500 — slimmer model, faster training
best_dropout = 0.1
best_lr = 0.0001
MAX_TRAIN_STEPS = 200          # was 1000 — early convergence suffices for online opt
EARLY_STOP_PATIENCE = 20       # stop if loss doesn't improve for N steps
EARLY_STOP_CONNECTED = 10      # stop if connected for N consecutive steps

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

class GDR_TS:
    """
    GDR-TS Baseline Implementation (Online Optimization Version) — OPTIMIZED
    Based on WCNC 2025: UAV Swarm Network Topology Self-Healing via Graph-Based DRL

    Optimizations:
      - hidden_dim: 500 → 128, layers: 6 → 3 (slimmer GATN)
      - training steps: 1000 → 200 with early stopping
      - AMP mixed precision for faster GPU training
    """
    def __init__(self):
        self.hidden_dimension = best_hidden_dimension
        self.dropout_value = best_dropout
        self.gcn_network = GATN_optimized(
            nfeat=config_dimension, nhid=self.hidden_dimension, nclass=config_dimension,
            dropout=self.dropout_value, if_dropout=True, bias=True
        )
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.gcn_network.cuda()

        self.optimizer = Adam(self.gcn_network.parameters(), lr=best_lr)
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        self.scaler = torch.amp.GradScaler('cuda') if self.use_cuda else None

    def gdr_ts(self, global_positions, remain_list):
        remain_positions = []
        for i in remain_list:
            remain_positions.append(deepcopy(global_positions[i]))
        remain_positions = np.array(remain_positions)
        num_remain = len(remain_list)

        # ==========================================================
        # Step 1: GATE & Critic Module (CMO Operation)
        # ==========================================================
        # Vectorized: compute distance matrix once, reuse for d and A_mat
        diff = remain_positions[:, np.newaxis, :] - remain_positions[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist_mat, 0)

        # d_min = minimum distance to connect all nodes (binary search on sorted distances)
        triu = dist_mat[np.triu_indices(num_remain, k=1)]
        sorted_dists = np.sort(triu)
        lo, hi = 0, len(sorted_dists) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            _, nc = Utils.count_clusters_from_A(dist_mat <= sorted_dists[mid])
            if nc == 1:
                hi = mid
            else:
                lo = mid + 1
        d_min = sorted_dists[lo]
        d_max = float(dist_mat.max())
        d = d_min + (d_max - d_min) * 0.25

        A_mat = (dist_mat <= d).astype(float)
        np.fill_diagonal(A_mat, 0)
        edge_set = list(zip(*np.where(A_mat > 0)))

        A_hat = np.array(edge_set).T

        D_mat = np.diag(np.sum(A_mat, axis=1))
        L_mat = D_mat - A_mat

        max_degree = np.max(np.sum(A_mat, axis=1))
        theta = 1.0 / (max_degree + 1e-5) if max_degree > 0 else 0

        P_hat = deepcopy(remain_positions)
        I = np.eye(num_remain)
        CMO_op = I - theta * L_mat

        q_iterations = 3  # was 5 — fewer iterations, sufficient for initial guess
        for _ in range(q_iterations):
            P_hat = np.matmul(CMO_op, P_hat)

        P_hat_tensor = torch.FloatTensor(P_hat).type(self.FloatTensor)
        remain_positions_tensor = torch.FloatTensor(remain_positions).type(self.FloatTensor)
        A_hat_tensor = torch.LongTensor(A_hat).type(self.LongTensor)

        best_final_positions = 0
        best_loss = 1e12
        connected_streak = 0
        no_improve_steps = 0

        # ==========================================================
        # Step 2: GATN-PT (Position Fine-Tuning Module) — OPTIMIZED
        # ==========================================================
        for train_step in range(MAX_TRAIN_STEPS):
            # AMP mixed precision for faster training
            with torch.amp.autocast('cuda', enabled=self.use_cuda):
                final_positions = self.gcn_network(P_hat_tensor, A_hat_tensor)
                final_positions_scaled = 0.5 * torch.Tensor(config_space_range).type(self.FloatTensor) * final_positions

                # Check connectivity (BFS from A, skip D/L construction)
                final_positions_np = final_positions_scaled.cpu().data.numpy()
                A = Utils.make_A_matrix(final_positions_np, len(final_positions_np), config_communication_range)
                flag, num = Utils.count_clusters_from_A(A)

                mu = 1000.0
                max_moving_distance = torch.max(torch.norm(final_positions_scaled - remain_positions_tensor, dim=1))
                loss = mu * (num - 1) + max_moving_distance

            # Record best
            loss_val = loss.cpu().data.numpy() if self.use_cuda else loss.data.numpy()
            if loss_val < best_loss:
                best_loss = loss_val
                best_final_positions = deepcopy(final_positions_np)
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            # Track connected streak for early stop
            if num == 1:
                connected_streak += 1
            else:
                connected_streak = 0

            # Backward with AMP scaler
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            print("    GDR-TS step %d, loss %.4f, best %.4f, num=%d" % (train_step, loss_val, best_loss, int(num)), end='\r')

            # Early stopping
            if connected_streak >= EARLY_STOP_CONNECTED:
                print("    GDR-TS early stop (connected) at step %d" % train_step)
                break
            if no_improve_steps >= EARLY_STOP_PATIENCE:
                print("    GDR-TS early stop (no improvement) at step %d" % train_step)
                break

        # ==========================================================
        # Step 3: Extract speed vectors
        # ==========================================================
        speed = np.zeros((config_num_of_agents, config_dimension))
        temp_max_distance = 0

        for i in range(num_remain):
            dist = np.linalg.norm(best_final_positions[i] - remain_positions[i])
            if dist > 0:
                speed[remain_list[i]] = (best_final_positions[i] - remain_positions[i]) / dist
            if dist > temp_max_distance:
                temp_max_distance = dist

        max_time = temp_max_distance / config_constant_speed
        return deepcopy(speed), deepcopy(max_time), deepcopy(best_final_positions)


class GATN_optimized(nn.Module):
    """
    Slimmed GATN: 3 layers instead of 6, smaller hidden dimension.
    GAT1 (in→nhid) → Norm → GAT2 (nhid→nhid) → Norm → GAT3 (nhid→nclass)
    """
    def __init__(self, nfeat=3, nhid=128, nclass=3, n_heads=4, dropout=0.5, alpha=0.2, if_dropout=True, bias=True):
        super(GATN_optimized, self).__init__()
        self.gat1 = GATv2Conv(in_channels=nfeat, out_channels=nhid, heads=n_heads, concat=True,
                               negative_slope=alpha, dropout=dropout, bias=bias, add_self_loops=True)
        self.gat2 = GATv2Conv(in_channels=nhid * n_heads, out_channels=nhid, heads=n_heads, concat=True,
                               negative_slope=alpha, dropout=dropout, bias=bias)
        self.gat3 = GATv2Conv(in_channels=nhid * n_heads, out_channels=nclass, heads=n_heads, concat=False,
                               negative_slope=alpha, dropout=0, bias=bias)

        self.norm1 = LayerNorm(nhid * n_heads)
        self.norm2 = LayerNorm(nhid * n_heads)
        self.norm3 = LayerNorm(nclass)

        self.dropout = dropout
        self.training = if_dropout

    def forward(self, x, adj):
        x = self.norm1(self.gat1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm2(self.gat2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm3(self.gat3(x, adj))
        return torch.tanh(x) + 1
