import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
from itertools import permutations
import math
from typing import Tuple

class SlotUtilities:

    def _select_supervised_slots(self, regression, classification):
        scores = torch.sigmoid(classification)
        selected_indices = torch.topk(scores, k=self.num_people, dim=1, largest=True, sorted=True).indices
        gather_regression_indices = selected_indices.unsqueeze(-1).expand(-1, -1, regression.shape[-1])
        selected_regression = torch.gather(regression, dim=1, index=gather_regression_indices)
        selected_classification = torch.gather(classification, dim=1, index=selected_indices)
        return (selected_regression, selected_classification, selected_indices)

    @staticmethod
    def get_linear_positional_encoding(seq_length, d_model):
        pos = torch.arange(seq_length).unsqueeze(1).float()
        scale = torch.linspace(0, 1, d_model).unsqueeze(0)
        encodings = pos * scale
        encodings = (encodings - encodings.mean()) / (encodings.std() + 1e-06)
        return encodings

    @staticmethod
    def reorder_by_sequence_similarity(x):
        """
        Find the globally cheapest sequence of one-to-one slot permutations.

        The first frame fixes the slot labels. Dynamic programming considers all
        permutations in every later frame and minimizes the summed transition
        distance over the complete sequence.
        """
        (b, t, n, d) = x.shape
        if t <= 1 or n <= 1:
            return x
        slot_indices = torch.arange(n, device=x.device)
        slot_permutations = torch.tensor(list(permutations(range(n))), device=x.device, dtype=torch.long)
        num_permutations = slot_permutations.shape[0]
        identity_state = torch.nonzero((slot_permutations == slot_indices).all(dim=1), as_tuple=False).squeeze(1).item()
        with torch.no_grad():
            detached = x.detach()
            path_cost = detached.new_full((b, num_permutations), float('inf'))
            path_cost[:, identity_state] = 0
            backpointers = []
            previous_indices = slot_permutations[:, None, :]
            current_indices = slot_permutations[None, :, :]
            for frame_idx in range(1, t):
                pairwise_distance = torch.cdist(detached[:, frame_idx - 1], detached[:, frame_idx]).square()
                transition_cost = pairwise_distance[:, previous_indices, current_indices].sum(dim=-1)
                total_cost = path_cost.unsqueeze(2) + transition_cost
                (path_cost, best_previous_state) = total_cost.min(dim=1)
                backpointers.append(best_previous_state)
            state_sequence = [path_cost.argmin(dim=1)]
            for best_previous_state in reversed(backpointers):
                previous_state = torch.gather(best_previous_state, 1, state_sequence[-1].unsqueeze(1)).squeeze(1)
                state_sequence.append(previous_state)
            state_sequence.reverse()
            state_sequence = torch.stack(state_sequence, dim=1)
            assignments = slot_permutations[state_sequence]
        return torch.gather(x, 2, assignments.unsqueeze(-1).expand(-1, -1, -1, d))

    @staticmethod
    def reorder_by_nearest_neighbor(x):
        return SlotUtilities.reorder_by_sequence_similarity(x)

class SlotAttention(nn.Module):

    def __init__(self, num_slots: int, dim: int, iters: int=3, eps: float=1e-08):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.scale = dim ** (-0.5)
        self.eps = eps
        self.input_norm = nn.LayerNorm(dim)
        self.slot_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, dim))
        self.xyz_to_slot = nn.Linear(8, dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x, xyz, E_init=None):
        """
        x: (B, N, D)  — input features
        return: slots (B, K, D)
        """
        (B, N, D) = x.shape
        x = self.input_norm(x)
        k = self.to_k(x)
        v = self.to_v(x)
        if E_init is not None:
            with torch.no_grad():
                slots_mu_z = self.xyz_to_slot(E_init)
            sigma = torch.exp(self.slots_log_sigma)
            slots_mu = slots_mu_z + self.slots_mu
            slots = slots_mu
        else:
            sigma = torch.exp(self.slots_log_sigma)
            slots = self.slots_mu + torch.randn(B, self.num_slots, D, device=x.device) * sigma
        for i in range(self.iters):
            slots_normed = self.slot_norm(slots)
            q = self.to_q(slots_normed)
            logits = torch.einsum('bnd,bkd->bnk', k, q) * self.scale
            attn = F.softmax(logits, dim=2)
            attn_out = attn
            attn = attn + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)
            updates = torch.einsum('bnk,bnd->bkd', attn, v)
            slots = self.gru(updates.reshape(-1, D), slots.reshape(-1, D)).view(B, self.num_slots, D)
            slots = slots + self.mlp(self.mlp_norm(slots))
        return (slots, attn_out, attn)

class PointSlotEncoder(nn.Module):
    """
    Differentiable DBSCAN over a batch of point sets.
    
    Input:
        X: tensor of shape (B, N, D)
    Output:
        C: tensor of shape (B, N, N) – soft connectivity matrices
        core_score: tensor of shape (B, N) – soft core‐point scores
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.num_people = self.config.PNT_NUM_PEOPLE
        self.num_points = self.config.PNT_NUM_POINTS
        self.regression_parameter = self.config.REGRESSION_PARAMETER
        self.pnt_emb_dim = 512
        self.raw_eps = nn.Parameter(torch.tensor(0.3, dtype=torch.float32), requires_grad=True)
        self.raw_min_samples = nn.Parameter(torch.tensor(3, dtype=torch.float32), requires_grad=True)
        self.raw_alpha = nn.Parameter(torch.tensor(50, dtype=torch.float32), requires_grad=True)
        self.raw_threshold = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        configured_num_slots = getattr(self.config, 'PNT_NUM_SLOTS', None)
        if configured_num_slots is None:
            configured_num_slots = getattr(self.config, 'PNT_PRED_NUM_SLOTS', 3)
        self.k_max = int(configured_num_slots)
        self.max_hops = 3
        self.tau = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.use_xy_only = True
        self.hop_weights = torch.ones(self.max_hops, dtype=torch.float32)
        self.pointnet = nn.Sequential(nn.Conv1d(8, 64, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128, 384, 1), nn.BatchNorm1d(384))
        self.slot_attention = SlotAttention(num_slots=self.k_max, dim=self.pnt_emb_dim, iters=5)
        self.head_mlp = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(3), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 512))
        self.bn = nn.BatchNorm1d(512)
        config_kpconv = ConfigKPConv()
        config_kpconv.num_kernel_points = 7
        config_kpconv.deform_radius = 6.0
        config_kpconv.KP_extent = 0.8
        config_kpconv.KP_influence = 'gaussian'
        config_kpconv.aggregation_mode = 'sum'
        config_kpconv.in_features_dim = 8
        config_kpconv.modulated = False
        config_kpconv.use_batch_norm = True
        r = 0.5
        in_dim = config_kpconv.in_features_dim
        out_dim_1 = 128
        out_dim_2 = 128
        self.K = config_kpconv.num_kernel_points
        self.kpconv_1 = KPConv(kernel_size=self.K, p_dim=2, in_channels=in_dim, out_channels=out_dim_1, KP_extent=config_kpconv.KP_extent, radius=r, fixed_kernel_points='center', KP_influence=config_kpconv.KP_influence, aggregation_mode=config_kpconv.aggregation_mode, deformable=False, modulated=config_kpconv.modulated)
        self.batchnorm_and_activation_11 = nn.Sequential(nn.BatchNorm1d(out_dim_1, momentum=0.05))
        self.kpconv_2 = KPConv(kernel_size=self.K, p_dim=2, in_channels=out_dim_1, out_channels=out_dim_1, KP_extent=config_kpconv.KP_extent, radius=r, fixed_kernel_points='center', KP_influence=config_kpconv.KP_influence, aggregation_mode=config_kpconv.aggregation_mode, deformable=False, modulated=config_kpconv.modulated)
        self.batchnorm_and_activation_21 = nn.Sequential(nn.BatchNorm1d(out_dim_1, momentum=0.05))

    @staticmethod
    def knn(xyz, q_pts, k):
        dist = torch.cdist(q_pts, xyz)
        (_, idx) = torch.topk(dist, k=k, dim=-1, largest=False)
        return idx

    def forward(self, X: torch.Tensor):
        """
        Args:
            X: (B, N, D) batched input
        Returns:
            C: (B, N, N) soft connectivity matrices
            core_score: (B, N) soft “core” scores
        """
        X = X.squeeze(1)
        (B, N, D) = X.shape
        mask_nonzero = (X.abs().sum(dim=-1, keepdim=True) != 0).float()
        X = torch.cat((X, mask_nonzero), dim=-1)
        if self.use_xy_only:
            xyz = X[:, :, :2]
        else:
            xyz = X[:, :, :3]
        valid_mask = xyz.abs().sum(dim=2) > 0.001
        valid_row = valid_mask.unsqueeze(2).float()
        valid_col = valid_mask.unsqueeze(1).float()
        mask_mat = valid_row * valid_col
        dist = torch.cdist(xyz, xyz, p=2)
        A = torch.sigmoid(-self.raw_alpha * (dist - self.raw_eps))
        c = A.sum(dim=2)
        core_score = torch.sigmoid(self.raw_alpha * (c - self.raw_min_samples))
        core_mat = core_score.unsqueeze(2) * core_score.unsqueeze(1)
        A_core = A * core_mat
        M1 = A_core
        M2 = torch.bmm(M1, A_core)
        M3 = torch.bmm(M2, A_core)
        C = self.hop_weights[0] * M1 + self.hop_weights[1] * M2 + self.hop_weights[2] * M3
        is_empty = X.abs().sum(dim=(1, 2)) < 1e-06
        eye = torch.eye(N, device=X.device).unsqueeze(0).expand(B, -1, -1)
        C = torch.where(is_empty.view(B, 1, 1), eye, C)
        eye = torch.eye(N, device=X.device).unsqueeze(0).expand(B, -1, -1)
        C = C + eye
        connectivity = torch.sigmoid(self.raw_alpha * (C - self.raw_threshold))
        S = connectivity / (connectivity.sum(dim=-1, keepdim=True) + 1e-08)
        E = torch.bmm(S, X[..., :])
        points = X.permute(0, 2, 1).contiguous()
        pnt = self.pointnet(points).permute(0, 2, 1).contiguous()
        xyz = X[:, :, :2]
        k_neighbor = 16
        neighb_inds = self.knn(xyz, xyz, k=k_neighbor)
        x = self.kpconv_1(q_pts=xyz, s_pts=xyz, neighb_inds=neighb_inds, x=X[...])
        x = x.permute(0, 2, 1).contiguous()
        x = self.batchnorm_and_activation_11(x)
        x = x.permute(0, 2, 1).contiguous()
        skip_x1 = x.clone()
        x = self.kpconv_2(q_pts=xyz, s_pts=xyz, neighb_inds=neighb_inds, x=x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.batchnorm_and_activation_21(x)
        x = x.permute(0, 2, 1).contiguous()
        x += skip_x1
        x = nn.LeakyReLU(0.1)(x)
        x = x * valid_row
        pnt = torch.concat((pnt, x), dim=-1)
        pnt = self.bn(pnt.permute(0, 2, 1)).permute(0, 2, 1)
        (slots, attn_out, attn) = self.slot_attention(pnt, xyz, E_init=None)
        agg_p = torch.einsum('bns,bnd->bsd', attn, xyz)
        slots_flat = slots.view(B, -1)
        y = slots_flat
        return y

def radius_gaussian(sq_r, sig, eps=1e-09):
    """
    Batched version of the radius‐Gaussian influence:
      sq_r:   (B, N_q, N_nbr, K)       squared distances
      sig:    scalar or tensor broadcastable to that shape
    Returns
      exp(− sq_r / (2 * sig² + eps)), same shape as sq_r.
    """
    return torch.exp(-sq_r / (2 * sig ** 2 + eps))

class KPConv(nn.Module):

    def __init__(self, kernel_size: int, p_dim: int, in_channels: int, out_channels: int, KP_extent: float, radius: float, fixed_kernel_points: str='center', KP_influence: str='linear', aggregation_mode: str='sum', deformable: bool=False, modulated: bool=False):
        """
        Batched KPConv (absorbs a leading “batch” dimension in forward).
        Arguments:
          kernel_size        = K  (number of kernel points)
          p_dim              = 3  (point‐dimensionality, usually 3)
          in_channels        = C_in
          out_channels       = C_out
          KP_extent          = influence radius of each kernel point
          radius             = radius used to initialize kernel points
          fixed_kernel_points= 'center' | 'verticals' | 'none' (same as before)
          KP_influence       = 'constant' | 'linear' | 'gaussian'
          aggregation_mode   = 'sum' | 'closest'
          deformable         = whether to generate offsets
          modulated          = whether to generate modulations (deformable+modulated)
        """
        super(KPConv, self).__init__()
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None
        self.weights = nn.Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32), requires_grad=True)
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(kernel_size=self.K, p_dim=self.p_dim, in_channels=self.in_channels, out_channels=self.offset_dim, KP_extent=KP_extent, radius=radius, fixed_kernel_points=fixed_kernel_points, KP_influence=KP_influence, aggregation_mode=aggregation_mode, deformable=False, modulated=False)
            self.offset_bias = nn.Parameter(torch.zeros((self.offset_dim,), dtype=torch.float32), requires_grad=True)
        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None
        self.reset_parameters()
        K_pts_numpy = load_kernels(self.radius, self.K, dimension=self.p_dim, fixed=self.fixed_kernel_points)
        self.kernel_points = nn.Parameter(torch.tensor(K_pts_numpy, dtype=torch.float32), requires_grad=False)

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)

    def forward(self, q_pts: torch.Tensor, s_pts: torch.Tensor, neighb_inds: torch.LongTensor, x: torch.Tensor) -> torch.Tensor:
        """
        Batched forward. Returns (B, N_q, C_out).
        Step‐by‐step:
         1) If deformable: predict offsets (+ modulations) via a nested KPConv.
         2) Collect “shadow point” row at distance 1e6 for each batch.
         3) Gather neighbor coordinates and center them w.r.t. q_pts.
         4) Compute (B, N_q, N_nbr, K, 3) difference → squared distances (B,N_q,N_nbr,K).
         5) If deformable: trim neighbors outside KP_extent, re‐index, etc.
         6) Compute per‐kernel influence (constant|linear|gaussian), produce (B,N_q,K,N_nbr).
         7) If ‘closest’ mode: find each neighbor’s nearest kernel, zero‐out others.
         8) Gather neighbor features from x → (B,N_q,N_nbr,C_in).
         9) Weighted sum over neighbors → (B,N_q,K,C_in).
        10) Per‐kernel linear transform + sum → (B,N_q,C_out).
        """
        (B, N_q, _) = q_pts.shape
        (_, N_s, _) = s_pts.shape
        (_, _, N_nbr) = neighb_inds.shape
        if self.deformable:
            self.offset_features = self.offset_conv(q_pts, s_pts, neighb_inds, x) + self.offset_bias
            if self.modulated:
                raw_offsets = self.offset_features[:, :, :self.p_dim * self.K]
                raw_offsets = raw_offsets.view(B, N_q, self.K, self.p_dim)
                modulations = 2.0 * torch.sigmoid(self.offset_features[:, :, self.p_dim * self.K:])
            else:
                raw_offsets = self.offset_features.view(B, N_q, self.K, self.p_dim)
                modulations = None
            offsets = raw_offsets * self.KP_extent
        else:
            offsets = None
            modulations = None
        big_val = 1000000.0
        shadow_pt = torch.zeros((B, 1, s_pts.shape[-1]), device=s_pts.device, dtype=s_pts.dtype) + big_val
        s_pts_all = torch.cat([s_pts, shadow_pt], dim=1)
        batch_idx = torch.arange(B, device=neighb_inds.device).view(B, 1, 1).expand(B, N_q, N_nbr)
        neighbors = s_pts_all[batch_idx, neighb_inds]
        neighbors = neighbors - q_pts.unsqueeze(2)
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points.view(1, 1, self.K, self.p_dim)
        else:
            self.deformed_KP = self.kernel_points.view(1, 1, self.K, self.p_dim)
        neigh_expand = neighbors.unsqueeze(3)
        KP_expand = self.deformed_KP.unsqueeze(2)
        diffs = neigh_expand - KP_expand
        sq_distances = torch.sum(diffs * diffs, dim=-1)
        if self.deformable:
            (self.min_d2, _) = torch.min(sq_distances, dim=2)
            inside_any = (sq_distances < self.KP_extent ** 2).any(dim=-1).int()
            counts_per_query = inside_any.sum(dim=2)
            max_count = counts_per_query.max().item()
            new_inds = torch.zeros((B, N_q, max_count), dtype=torch.long, device=x.device)
            for b in range(B):
                for i in range(N_q):
                    keep_mask = inside_any[b, i]
                    valid_idxs = torch.nonzero(keep_mask, as_tuple=False).view(-1)
                    num_valid = valid_idxs.shape[0]
                    if num_valid > 0:
                        pick_count = min(num_valid, max_count)
                        new_inds[b, i, :pick_count] = neighb_inds[b, i, keep_mask[:pick_count].bool()]
            padded_mask = counts_per_query.unsqueeze(-1) <= torch.arange(max_count, device=x.device)
            new_inds = new_inds.masked_fill(padded_mask, N_s)
            neighb_inds = new_inds
            _tmp = torch.zeros((B, N_q, max_count, self.K), device=x.device)
            for b in range(B):
                for i in range(N_q):
                    keep_mask = inside_any[b, i]
                    valid_idxs = torch.nonzero(keep_mask, as_tuple=False).view(-1)
                    num_valid = valid_idxs.shape[0]
                    if num_valid > 0:
                        pick_count = min(num_valid, max_count)
                        _tmp[b, i, :pick_count, :] = sq_distances[b, i, valid_idxs[:pick_count], :]
            sq_distances = _tmp
            N_nbr = max_count
        else:
            pass
        if self.KP_influence == 'constant':
            all_weights = torch.ones_like(sq_distances)
        elif self.KP_influence == 'linear':
            root_dist = torch.sqrt(sq_distances)
            all_weights = torch.clamp(1 - root_dist / self.KP_extent, min=0.0)
        elif self.KP_influence == 'gaussian':
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
        else:
            raise ValueError(f"Unknown KP_influence='{self.KP_influence}'")
        all_weights = all_weights.permute(0, 1, 3, 2).contiguous()
        if self.aggregation_mode == 'closest':
            nearest_kern = torch.argmin(sq_distances, dim=-1)
            one_hot = F.one_hot(nearest_kern, num_classes=self.K).float()
            one_hot = one_hot.permute(0, 1, 3, 2).contiguous()
            all_weights = all_weights * one_hot
        elif self.aggregation_mode == 'sum':
            pass
        else:
            raise ValueError(f"Unknown aggregation_mode='{self.aggregation_mode}'")
        x_with_shadow = torch.cat([x, torch.zeros((B, 1, self.in_channels), dtype=x.dtype, device=x.device)], dim=1)
        batch_idx = torch.arange(B, device=neighb_inds.device).view(B, 1, 1).expand(B, N_q, N_nbr)
        neighb_feats = x_with_shadow[batch_idx, neighb_inds]
        weighted_feats = torch.einsum('bqkn,bqnc->bqkc', all_weights, neighb_feats)
        y = torch.einsum('bqmc,mco->bqo', weighted_feats, self.weights)
        return y

    def __repr__(self):
        return f'KPConv(radius: {self.radius:.2f}, in_feat: {self.in_channels}, out_feat: {self.out_channels})'
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os import makedirs
from os.path import join, exists

def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """
    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([t1 + t2 * t3, t7 - t9, t11 + t12, t7 + t9, t1 + t2 * t15, t19 - t20, t11 - t12, t19 + t20, t1 + t2 * t24], axis=1)
    return np.reshape(R, (-1, 3, 3))

def spherical_Lloyd(radius, num_cells, dimension=3, fixed='center', approximation='monte-carlo', approx_n=5000, max_iter=500, momentum=0.9, verbose=0):
    """
    Creation of kernel point via Lloyd algorithm. We use an approximation of the algorithm, and compute the Voronoi
    cell centers with discretization  of space. The exact formula is not trivial with part of the sphere as sides.
    :param radius: Radius of the kernels
    :param num_cells: Number of cell (kernel points) in the Voronoi diagram.
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param approximation: Approximation method for Lloyd's algorithm ('discretization', 'monte-carlo')
    :param approx_n: Number of point used for approximation.
    :param max_iter: Maximum nu;ber of iteration for the algorithm.
    :param momentum: Momentum of the low pass filter smoothing kernel point positions
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """
    radius0 = 1.0
    kernel_points = np.zeros((0, dimension))
    while kernel_points.shape[0] < num_cells:
        new_points = np.random.rand(num_cells, dimension) * 2 * radius0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[np.logical_and(d2 < radius0 ** 2, (0.9 * radius0) ** 2 < d2), :]
    kernel_points = kernel_points[:num_cells, :].reshape((num_cells, -1))
    if fixed == 'center':
        kernel_points[0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:3, :] *= 0
        kernel_points[1, -1] += 2 * radius0 / 3
        kernel_points[2, -1] -= 2 * radius0 / 3
    if verbose > 1:
        fig = plt.figure()
    if approximation == 'discretization':
        side_n = int(np.floor(approx_n ** (1.0 / dimension)))
        dl = 2 * radius0 / side_n
        coords = np.arange(-radius0 + dl / 2, radius0, dl)
        if dimension == 2:
            (x, y) = np.meshgrid(coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y))).T
        elif dimension == 3:
            (x, y, z) = np.meshgrid(coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T
        elif dimension == 4:
            (x, y, z, t) = np.meshgrid(coords, coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z), np.ravel(t))).T
        else:
            raise ValueError('Unsupported dimension (max is 4)')
    elif approximation == 'monte-carlo':
        X = np.zeros((0, dimension))
    else:
        raise ValueError('Wrong approximation method chosen: "{:s}"'.format(approximation))
    d2 = np.sum(np.power(X, 2), axis=1)
    X = X[d2 < radius0 * radius0, :]
    warning = False
    max_moves = np.zeros((0,))
    for iter in range(max_iter):
        if approximation == 'monte-carlo':
            X = np.random.rand(approx_n, dimension) * 2 * radius0 - radius0
            d2 = np.sum(np.power(X, 2), axis=1)
            X = X[d2 < radius0 * radius0, :]
        differences = np.expand_dims(X, 1) - kernel_points
        sq_distances = np.sum(np.square(differences), axis=2)
        cell_inds = np.argmin(sq_distances, axis=1)
        centers = []
        for c in range(num_cells):
            bool_c = cell_inds == c
            num_c = np.sum(bool_c.astype(np.int32))
            if num_c > 0:
                centers.append(np.sum(X[bool_c, :], axis=0) / num_c)
            else:
                warning = True
                centers.append(kernel_points[c])
        centers = np.vstack(centers)
        moves = (1 - momentum) * (centers - kernel_points)
        kernel_points += moves
        max_moves = np.append(max_moves, np.max(np.linalg.norm(moves, axis=1)))
        if fixed == 'center':
            kernel_points[0, :] *= 0
        if fixed == 'verticals':
            kernel_points[0, :] *= 0
            kernel_points[:3, :-1] *= 0
        if verbose:
            print('iter {:5d} / max move = {:f}'.format(iter, np.max(np.linalg.norm(moves, axis=1))))
            if warning:
                print('{:}WARNING: at least one point has no cell{:}')
        if verbose > 1:
            plt.clf()
            plt.scatter(X[:, 0], X[:, 1], c=cell_inds, s=20.0, marker='.', cmap=plt.get_cmap('tab20'))
            plt.plot(kernel_points[:, 0], kernel_points[:, 1], 'k+')
            circle = plt.Circle((0, 0), radius0, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius0 * 1.1, radius0 * 1.1))
            fig.axes[0].set_ylim((-radius0 * 1.1, radius0 * 1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
    if verbose:
        if dimension == 2:
            (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=[10.4, 4.8])
            ax1.plot(max_moves)
            ax2.scatter(X[:, 0], X[:, 1], c=cell_inds, s=20.0, marker='.', cmap=plt.get_cmap('tab20'))
            ax2.plot(kernel_points[:, 0], kernel_points[:, 1], 'k+')
            circle = plt.Circle((0, 0), radius0, color='r', fill=False)
            ax2.add_artist(circle)
            ax2.set_xlim((-radius0 * 1.1, radius0 * 1.1))
            ax2.set_ylim((-radius0 * 1.1, radius0 * 1.1))
            ax2.set_aspect('equal')
            plt.title('Check if kernel is correct.')
            plt.draw()
            plt.show()
        if dimension > 2:
            plt.figure()
            plt.plot(max_moves)
            plt.title('Check if kernel is correct.')
            plt.show()
    return kernel_points * radius

def kernel_point_optimization_debug(radius, num_points, num_kernels=1, dimension=3, fixed='center', ratio=0.66, verbose=0):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """
    radius0 = 1
    diameter0 = 2
    moving_factor = 0.01
    continuous_moving_decay = 0.9995
    thresh = 1e-05
    clip = 0.05 * radius0
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while kernel_points.shape[0] < num_kernels * num_points:
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((num_kernels, num_points, -1))
    if fixed == 'center':
        kernel_points[:, 0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3
    if verbose > 1:
        fig = plt.figure()
    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    step = -1
    while step < 10000:
        step += 1
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) + 1e-06)
        inter_grads = np.sum(inter_grads, axis=1)
        circle_grads = 10 * kernel_points
        gradients = inter_grads + circle_grads
        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[step, :] = np.max(gradients_norms, axis=1)
        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)
        if fixed == 'center':
            moving_dists[:, 0] = 0
        if fixed == 'verticals':
            moving_dists[:, 0] = 0
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-06, -1)
        if verbose:
            print('step {:5d} / max grad = {:f}'.format(step, np.max(gradients_norms[:, 3:])))
        if verbose > 1:
            plt.clf()
            plt.plot(kernel_points[0, :, 0], kernel_points[0, :, 1], '.')
            circle = plt.Circle((0, 0), radius, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_ylim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
            print(moving_factor)
        moving_factor *= continuous_moving_decay
    if step < 10000:
        saved_gradient_norms = saved_gradient_norms[:step + 1, :]
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])
    return (kernel_points * radius, saved_gradient_norms)

def load_kernels(radius, num_kpoints, dimension, fixed, lloyd=False):
    kernel_dir = str(Path(__file__).resolve().parents[2] / 'cache' / 'kernels')
    if not exists(kernel_dir):
        makedirs(kernel_dir)
    if num_kpoints > 30:
        lloyd = True
    kernel_file = join(kernel_dir, 'k_{:03d}_{:s}_{:d}D.ply'.format(num_kpoints, fixed, dimension))
    if not exists(kernel_file) or Path(kernel_file).stat().st_size == 0:
        if lloyd:
            kernel_points = spherical_Lloyd(1.0, num_kpoints, dimension=dimension, fixed=fixed, verbose=0)
        else:
            (kernel_points, grad_norms) = kernel_point_optimization_debug(1.0, num_kpoints, num_kernels=100, dimension=dimension, fixed=fixed, verbose=0)
            best_k = np.argmin(grad_norms[-1, :])
            kernel_points = kernel_points[best_k, :, :]
        write_ply(kernel_file, kernel_points, ['x', 'y', 'z'][:dimension])
    else:
        data = read_ply(kernel_file)
        kernel_points = np.vstack([data[axis] for axis in ['x', 'y', 'z'][:dimension]]).T
    R = np.eye(dimension)
    theta = np.random.rand() * 2 * np.pi
    if dimension == 2:
        if fixed != 'vertical':
            (c, s) = (np.cos(theta), np.sin(theta))
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
    elif dimension == 3:
        if fixed != 'vertical':
            (c, s) = (np.cos(theta), np.sin(theta))
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        else:
            phi = (np.random.rand() - 0.5) * np.pi
            u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
            alpha = np.random.rand() * 2 * np.pi
            R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]
            R = R.astype(np.float32)
    kernel_points = kernel_points + np.random.normal(scale=0.01, size=kernel_points.shape)
    kernel_points = radius * kernel_points
    kernel_points = np.matmul(kernel_points, R)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    if kernel_points.shape[1] == 3:
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(kernel_points[:, 0], kernel_points[:, 1], kernel_points[:, 2], s=100, c='r', marker='o')
    if kernel_points.shape[1] == 2:
        ax = fig.add_subplot(1, 2, 1)
        ax.scatter(kernel_points[:, 0], kernel_points[:, 1], s=100, c='r', marker='o')
    fig.savefig(join(kernel_dir, 'kernels_3D_{:3f}_{:03d}_{:s}_{:d}D.png'.format(radius, num_kpoints, fixed, dimension)), dpi=300)
    return kernel_points.astype(np.float32)
ply_dtypes = dict([(b'int8', 'i1'), (b'char', 'i1'), (b'uint8', 'u1'), (b'uchar', 'u1'), (b'int16', 'i2'), (b'short', 'i2'), (b'uint16', 'u2'), (b'ushort', 'u2'), (b'int32', 'i4'), (b'int', 'i4'), (b'uint32', 'u4'), (b'uint', 'u4'), (b'float32', 'f4'), (b'float', 'f4'), (b'float64', 'f8'), (b'double', 'f8')])
valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}

def parse_header(plyfile, ext):
    line = []
    properties = []
    num_points = None
    while b'end_header' not in line and line != b'':
        line = plyfile.readline()
        if b'element' in line:
            line = line.split()
            num_points = int(line[2])
        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
    return (num_points, properties)

def parse_mesh_header(plyfile, ext):
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None
    while b'end_header' not in line and line != b'':
        line = plyfile.readline()
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])
        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])
        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)
    return (num_points, num_faces, vertex_properties)

def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file

    Examples
    --------
    Store data in file

    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

    Read the file

    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])
    
    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])

    """
    with open(filename, 'rb') as plyfile:
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')
        fmt = plyfile.readline().split()[1].decode()
        if fmt == 'ascii':
            raise ValueError('The file is not binary')
        ext = valid_formats[fmt]
        if triangular_mesh:
            (num_points, num_faces, properties) = parse_mesh_header(plyfile, ext)
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)
            face_properties = [('k', ext + 'u1'), ('v1', ext + 'i4'), ('v2', ext + 'i4'), ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]
        else:
            (num_points, properties) = parse_header(plyfile, ext)
            data = np.fromfile(plyfile, dtype=properties, count=num_points)
    return data

def header_properties(field_list, field_names):
    lines = []
    lines.append('element vertex %d' % field_list[0].shape[0])
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1
    return lines

def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the 
        file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a 
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered 
        as one field. 

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of 
        fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """
    field_list = list(field_list) if type(field_list) == list or type(field_list) == tuple else list((field_list,))
    for (i, field) in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False
    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(field_names):
        print('wrong number of field names')
        return False
    if not filename.endswith('.ply'):
        filename += '.ply'
    with open(filename, 'w') as plyfile:
        header = ['ply']
        header.append('format binary_' + sys.byteorder + '_endian 1.0')
        header.extend(header_properties(field_list, field_names))
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')
        header.append('end_header')
        for line in header:
            plyfile.write('%s\n' % line)
    with open(filename, 'ab') as plyfile:
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1
        data.tofile(plyfile)
        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)
    return True

class ConfigKPConv:
    """
    Class containing the parameters you want to modify for this dataset
    """
    dataset = ''
    dataset_task = ''
    num_classes = 0
    in_points_dim = 3
    in_features_dim = 1
    in_radius = 1.0
    input_threads = 8
    architecture = []
    equivar_mode = ''
    invar_mode = ''
    first_features_dim = 64
    use_batch_norm = True
    batch_norm_momentum = 0.99
    segmentation_ratio = 1.0
    num_kernel_points = 15
    first_subsampling_dl = 0.02
    conv_radius = 2.5
    deform_radius = 5.0
    KP_extent = 1.0
    KP_influence = 'linear'
    aggregation_mode = 'sum'
    fixed_kernel_points = 'center'
    modulated = False
    n_frames = 1
    max_in_points = 0
    val_radius = 51.0
    max_val_points = 50000
    learning_rate = 0.001
    momentum = 0.9
    lr_decays = {200: 0.2, 300: 0.2}
    grad_clip_norm = 100.0
    augment_scale_anisotropic = True
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_symmetries = [False, False, False]
    augment_rotation = 'vertical'
    augment_noise = 0.005
    augment_color = 0.7
    augment_occlusion = 'none'
    augment_occlusion_ratio = 0.2
    augment_occlusion_num = 1
    weight_decay = 0.001
    segloss_balance = 'none'
    class_w = []
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0
    deform_lr_factor = 0.1
    repulse_extent = 1.0
    batch_num = 10
    val_batch_num = 10
    max_epoch = 1000
    epoch_steps = 1000
    validation_size = 100
    checkpoint_gap = 50
    saving = True
    saving_path = None

    def __init__(self):
        """
        Class Initialyser
        """
        self.num_layers = len([block for block in self.architecture if 'pool' in block or 'strided' in block]) + 1
        layer_blocks = []
        self.deform_layers = []
        arch = self.architecture
        for (block_i, block) in enumerate(arch):
            if not ('pool' in block or 'strided' in block or 'global' in block or ('upsample' in block)):
                layer_blocks += [block]
                continue
            deform_layer = False
            if layer_blocks:
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    deform_layer = True
            if 'pool' in block or 'strided' in block:
                if 'deformable' in block:
                    deform_layer = True
            self.deform_layers += [deform_layer]
            layer_blocks = []
            if 'global' in block or 'upsample' in block:
                break

    def load(self, path):
        filename = join(path, 'parameters.txt')
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line_info = line.split()
            if len(line_info) > 2 and line_info[0] != '#':
                if line_info[2] == 'None':
                    setattr(self, line_info[0], None)
                elif line_info[0] == 'lr_decay_epochs':
                    self.lr_decays = {int(b.split(':')[0]): float(b.split(':')[1]) for b in line_info[2:]}
                elif line_info[0] == 'architecture':
                    self.architecture = [b for b in line_info[2:]]
                elif line_info[0] == 'augment_symmetries':
                    self.augment_symmetries = [bool(int(b)) for b in line_info[2:]]
                elif line_info[0] == 'num_classes':
                    if len(line_info) > 3:
                        self.num_classes = [int(c) for c in line_info[2:]]
                    else:
                        self.num_classes = int(line_info[2])
                elif line_info[0] == 'class_w':
                    self.class_w = [float(w) for w in line_info[2:]]
                elif hasattr(self, line_info[0]):
                    attr_type = type(getattr(self, line_info[0]))
                    if attr_type == bool:
                        setattr(self, line_info[0], attr_type(int(line_info[2])))
                    else:
                        setattr(self, line_info[0], attr_type(line_info[2]))
        self.saving = True
        self.saving_path = path
        self.__init__()

    def save(self):
        with open(join(self.saving_path, 'parameters.txt'), 'w') as text_file:
            text_file.write('# -----------------------------------#\n')
            text_file.write('# Parameters of the training session #\n')
            text_file.write('# -----------------------------------#\n\n')
            text_file.write('# Input parameters\n')
            text_file.write('# ****************\n\n')
            text_file.write('dataset = {:s}\n'.format(self.dataset))
            text_file.write('dataset_task = {:s}\n'.format(self.dataset_task))
            if type(self.num_classes) is list:
                text_file.write('num_classes =')
                for n in self.num_classes:
                    text_file.write(' {:d}'.format(n))
                text_file.write('\n')
            else:
                text_file.write('num_classes = {:d}\n'.format(self.num_classes))
            text_file.write('in_points_dim = {:d}\n'.format(self.in_points_dim))
            text_file.write('in_features_dim = {:d}\n'.format(self.in_features_dim))
            text_file.write('in_radius = {:.6f}\n'.format(self.in_radius))
            text_file.write('input_threads = {:d}\n\n'.format(self.input_threads))
            text_file.write('# Model parameters\n')
            text_file.write('# ****************\n\n')
            text_file.write('architecture =')
            for a in self.architecture:
                text_file.write(' {:s}'.format(a))
            text_file.write('\n')
            text_file.write('equivar_mode = {:s}\n'.format(self.equivar_mode))
            text_file.write('invar_mode = {:s}\n'.format(self.invar_mode))
            text_file.write('num_layers = {:d}\n'.format(self.num_layers))
            text_file.write('first_features_dim = {:d}\n'.format(self.first_features_dim))
            text_file.write('use_batch_norm = {:d}\n'.format(int(self.use_batch_norm)))
            text_file.write('batch_norm_momentum = {:.6f}\n\n'.format(self.batch_norm_momentum))
            text_file.write('segmentation_ratio = {:.6f}\n\n'.format(self.segmentation_ratio))
            text_file.write('# KPConv parameters\n')
            text_file.write('# *****************\n\n')
            text_file.write('first_subsampling_dl = {:.6f}\n'.format(self.first_subsampling_dl))
            text_file.write('num_kernel_points = {:d}\n'.format(self.num_kernel_points))
            text_file.write('conv_radius = {:.6f}\n'.format(self.conv_radius))
            text_file.write('deform_radius = {:.6f}\n'.format(self.deform_radius))
            text_file.write('fixed_kernel_points = {:s}\n'.format(self.fixed_kernel_points))
            text_file.write('KP_extent = {:.6f}\n'.format(self.KP_extent))
            text_file.write('KP_influence = {:s}\n'.format(self.KP_influence))
            text_file.write('aggregation_mode = {:s}\n'.format(self.aggregation_mode))
            text_file.write('modulated = {:d}\n'.format(int(self.modulated)))
            text_file.write('n_frames = {:d}\n'.format(self.n_frames))
            text_file.write('max_in_points = {:d}\n\n'.format(self.max_in_points))
            text_file.write('max_val_points = {:d}\n\n'.format(self.max_val_points))
            text_file.write('val_radius = {:.6f}\n\n'.format(self.val_radius))
            text_file.write('# Training parameters\n')
            text_file.write('# *******************\n\n')
            text_file.write('learning_rate = {:f}\n'.format(self.learning_rate))
            text_file.write('momentum = {:f}\n'.format(self.momentum))
            text_file.write('lr_decay_epochs =')
            for (e, d) in self.lr_decays.items():
                text_file.write(' {:d}:{:f}'.format(e, d))
            text_file.write('\n')
            text_file.write('grad_clip_norm = {:f}\n\n'.format(self.grad_clip_norm))
            text_file.write('augment_symmetries =')
            for a in self.augment_symmetries:
                text_file.write(' {:d}'.format(int(a)))
            text_file.write('\n')
            text_file.write('augment_rotation = {:s}\n'.format(self.augment_rotation))
            text_file.write('augment_noise = {:f}\n'.format(self.augment_noise))
            text_file.write('augment_occlusion = {:s}\n'.format(self.augment_occlusion))
            text_file.write('augment_occlusion_ratio = {:.6f}\n'.format(self.augment_occlusion_ratio))
            text_file.write('augment_occlusion_num = {:d}\n'.format(self.augment_occlusion_num))
            text_file.write('augment_scale_anisotropic = {:d}\n'.format(int(self.augment_scale_anisotropic)))
            text_file.write('augment_scale_min = {:.6f}\n'.format(self.augment_scale_min))
            text_file.write('augment_scale_max = {:.6f}\n'.format(self.augment_scale_max))
            text_file.write('augment_color = {:.6f}\n\n'.format(self.augment_color))
            text_file.write('weight_decay = {:f}\n'.format(self.weight_decay))
            text_file.write('segloss_balance = {:s}\n'.format(self.segloss_balance))
            text_file.write('class_w =')
            for a in self.class_w:
                text_file.write(' {:.6f}'.format(a))
            text_file.write('\n')
            text_file.write('deform_fitting_mode = {:s}\n'.format(self.deform_fitting_mode))
            text_file.write('deform_fitting_power = {:.6f}\n'.format(self.deform_fitting_power))
            text_file.write('deform_lr_factor = {:.6f}\n'.format(self.deform_lr_factor))
            text_file.write('repulse_extent = {:.6f}\n'.format(self.repulse_extent))
            text_file.write('batch_num = {:d}\n'.format(self.batch_num))
            text_file.write('val_batch_num = {:d}\n'.format(self.val_batch_num))
            text_file.write('max_epoch = {:d}\n'.format(self.max_epoch))
            if self.epoch_steps is None:
                text_file.write('epoch_steps = None\n')
            else:
                text_file.write('epoch_steps = {:d}\n'.format(self.epoch_steps))
            text_file.write('validation_size = {:d}\n'.format(self.validation_size))
            text_file.write('checkpoint_gap = {:d}\n'.format(self.checkpoint_gap))
