import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
import math
from typing import Tuple

class MOTRQueryInteractionModule(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout, pos_dim, score_thresh, random_drop=0.0, fp_ratio=0.0):
        super().__init__()
        self.pos_dim = pos_dim
        self.score_thresh = score_thresh
        self.random_drop = random_drop
        self.fp_ratio = fp_ratio
        self.state_projection = nn.Linear(pos_dim + 1, embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim))

    def _random_drop_tracks(self, selected_mask: torch.Tensor) -> torch.Tensor:
        if self.random_drop <= 0 or not self.training:
            return selected_mask
        keep_mask = torch.rand(selected_mask.shape, device=selected_mask.device) > self.random_drop
        return selected_mask & keep_mask

    def _add_fp_tracks(self, candidate_mask: torch.Tensor, selected_mask: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.fp_ratio <= 0 or not self.training:
            return selected_mask
        fp_seeds = (torch.rand_like(scores) < self.fp_ratio) & selected_mask
        num_fp = fp_seeds.sum(dim=1, keepdim=True)
        candidate_order = torch.argsort(scores.masked_fill(~candidate_mask, float('-inf')), dim=1, descending=True)
        candidate_ranks = torch.empty_like(candidate_order)
        candidate_ranks.scatter_(1, candidate_order, torch.arange(candidate_order.shape[1], device=scores.device).unsqueeze(0).expand_as(candidate_order))
        fp_mask = candidate_mask & (candidate_ranks < num_fp)
        return selected_mask | fp_mask

    def _select_active_tracks(self, last_classification: torch.Tensor, active_mask: torch.Tensor):
        if active_mask is None:
            interaction_mask = torch.ones_like(last_classification, dtype=torch.bool)
            return (interaction_mask, interaction_mask)
        active_mask = active_mask.bool()
        scores = torch.sigmoid(last_classification)
        if self.training and (self.random_drop > 0 or self.fp_ratio > 0):
            positive_mask = active_mask & (scores > self.score_thresh)
            selected_mask = self._random_drop_tracks(positive_mask)
            fp_candidate_mask = active_mask & ~positive_mask
            interaction_mask = self._add_fp_tracks(fp_candidate_mask, selected_mask, scores)
        else:
            selected_mask = active_mask
            interaction_mask = selected_mask
        return (selected_mask, interaction_mask)

    def forward(self, track_queries, last_regression, last_classification, active_mask):
        (selected_mask, interaction_mask) = self._select_active_tracks(last_classification=last_classification, active_mask=active_mask)
        query_state = torch.cat((last_regression[..., :self.pos_dim], torch.sigmoid(last_classification).unsqueeze(-1)), dim=-1)
        fused_queries = track_queries + self.state_projection(query_state)
        key_padding_mask = None
        if interaction_mask is not None:
            key_padding_mask = ~interaction_mask.bool()
            empty_batches = key_padding_mask.all(dim=1)
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[:, 0] = key_padding_mask[:, 0] & ~empty_batches
        attn_input = self.norm1(fused_queries)
        (attn_output, _) = self.self_attn(attn_input, attn_input, attn_input, key_padding_mask=key_padding_mask)
        fused_queries = fused_queries + attn_output
        fused_queries = fused_queries + self.ffn(self.norm2(fused_queries))
        if interaction_mask is not None:
            fused_queries = torch.where(interaction_mask.unsqueeze(-1), fused_queries, track_queries)
        return (fused_queries, selected_mask, interaction_mask)

class DBSCANObjectQueryInitializer(nn.Module):
    """Build fixed-size object-query hints from hard DBSCAN clusters."""

    def __init__(self, point_dim, embed_dim, num_object_queries, eps=0.4, min_samples=4, coord_dims=2, hidden_dim=128):
        super().__init__()
        if eps <= 0:
            raise ValueError('MOTR_DBSCAN_EPS must be greater than 0')
        if min_samples < 1:
            raise ValueError('MOTR_DBSCAN_MIN_SAMPLES must be at least 1')
        if coord_dims < 1 or coord_dims > point_dim:
            raise ValueError('MOTR_DBSCAN_COORD_DIMS must be between 1 and PNT_NUM_FEATURES')
        self.num_object_queries = int(num_object_queries)
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.coord_dims = int(coord_dims)
        descriptor_dim = 2 * point_dim + embed_dim + 1
        self.projection = nn.Sequential(nn.Linear(descriptor_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim))

    def _cluster_sample(self, points):
        """Return DBSCAN cluster masks over the valid, unpadded points."""
        coordinates = points[:, :self.coord_dims]
        adjacency = torch.cdist(coordinates, coordinates, p=2) <= self.eps
        core_mask = adjacency.sum(dim=1) >= self.min_samples
        return self._clusters_from_adjacency(adjacency, core_mask)

    @staticmethod
    def _clusters_from_adjacency(adjacency, core_mask):
        """Build DBSCAN components without requiring accelerator synchronization."""
        unassigned_core = core_mask.clone()
        assigned_border = torch.zeros_like(core_mask)
        clusters = []
        while bool(unassigned_core.any()):
            seed = torch.nonzero(unassigned_core, as_tuple=False)[0, 0]
            component = torch.zeros_like(core_mask)
            component[seed] = True
            while True:
                expanded = adjacency[component].any(dim=0) & core_mask
                next_component = component | expanded
                if torch.equal(next_component, component):
                    break
                component = next_component
            border = adjacency[component].any(dim=0) & ~core_mask & ~assigned_border
            cluster_mask = component | border
            clusters.append(cluster_mask)
            unassigned_core = unassigned_core & ~component
            assigned_border = assigned_border | border
        clusters.sort(key=lambda mask: int(mask.sum().item()), reverse=True)
        return clusters

    def _cluster_batch(self, frame, valid_point_mask):
        """Cluster a batch with one device-to-host synchronization per frame."""
        coordinates = frame[..., :self.coord_dims].detach().float()
        with torch.no_grad():
            adjacency = torch.cdist(coordinates, coordinates, p=2) <= self.eps
            valid_pairs = valid_point_mask.unsqueeze(1) & valid_point_mask.unsqueeze(2)
            adjacency = adjacency & valid_pairs
            core_mask = (adjacency.sum(dim=-1) >= self.min_samples) & valid_point_mask
            adjacency_cpu = adjacency.cpu()
            core_mask_cpu = core_mask.cpu()
            cluster_masks_cpu = torch.zeros(frame.shape[0], self.num_object_queries, frame.shape[1], dtype=torch.bool)
            for batch_idx in range(frame.shape[0]):
                clusters = self._clusters_from_adjacency(adjacency_cpu[batch_idx], core_mask_cpu[batch_idx])
                for (query_idx, cluster_mask) in enumerate(clusters[:self.num_object_queries]):
                    cluster_masks_cpu[batch_idx, query_idx] = cluster_mask
        return cluster_masks_cpu.to(device=frame.device, non_blocking=True)

    def forward(self, frame, frame_tokens):
        if frame.dim() != 3 or frame_tokens.dim() != 3:
            raise ValueError('DBSCAN query initialization expects [B, N, F] tensors')
        if frame.shape[:2] != frame_tokens.shape[:2]:
            raise ValueError('Raw frame and encoded frame must have matching [B, N] dimensions')
        valid_point_mask = frame.abs().sum(dim=-1) > 1e-06
        cluster_masks = self._cluster_batch(frame, valid_point_mask)
        point_weights = cluster_masks.to(dtype=frame.dtype)
        cluster_sizes = point_weights.sum(dim=-1)
        valid_queries = cluster_sizes > 0
        cluster_denominator = cluster_sizes.clamp_min(1).unsqueeze(-1)
        point_means = torch.einsum('bqn,bnf->bqf', point_weights, frame) / cluster_denominator
        point_second_moments = torch.einsum('bqn,bnf->bqf', point_weights, frame.square()) / cluster_denominator
        point_stds = (point_second_moments - point_means.square()).clamp_min(0).sqrt()
        token_weights = cluster_masks.to(dtype=frame_tokens.dtype)
        token_means = torch.einsum('bqn,bne->bqe', token_weights, frame_tokens) / cluster_denominator.to(dtype=frame_tokens.dtype)
        valid_point_counts = valid_point_mask.sum(dim=-1).clamp_min(1).to(dtype=frame.dtype).unsqueeze(-1)
        cluster_fractions = (cluster_sizes / valid_point_counts).unsqueeze(-1)
        descriptors = torch.cat((point_means, point_stds, token_means.to(dtype=frame.dtype), cluster_fractions), dim=-1)
        labels = cluster_masks.to(dtype=torch.long).argmax(dim=1)
        labels = labels.masked_fill(~cluster_masks.any(dim=1), -1)
        query_hints = self.projection(descriptors)
        query_hints = query_hints * valid_queries.unsqueeze(-1).to(query_hints.dtype)
        return {'queries': query_hints, 'valid_mask': valid_queries, 'cluster_sizes': cluster_sizes, 'labels': labels}

class PntKPConvTbA(nn.Module):
    tracking_approach = 'motr'

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.num_features = self.config.PNT_NUM_FEATURES
        self.embed_dim = self.config.PNT_EMBED_DIM
        self.num_people = self.config.PNT_NUM_PEOPLE
        self.num_track_queries = max(self.num_people, int(getattr(self.config, 'NUM_TRACK_QUERIES', self.num_people)))
        self.num_points = self.config.PNT_NUM_POINTS
        self.dropout = self.config.PNT_DROPOUT
        self.sequence_length = int(self.config.SEQUENCE_LENGTH)
        if self.sequence_length < 2:
            raise ValueError('SEQUENCE_LENGTH must be at least 2 for PntKPConvTbA')
        self.regression_parameter = self.config.REGRESSION_PARAMETER
        self.num_object_queries = getattr(self.config, 'NUM_OBJECT_QUERIES', self.num_people)
        self.new_track_score_thresh = getattr(self.config, 'MOTR_NEW_TRACK_SCORE_THRESH', 0.5)
        self.keep_track_score_thresh = getattr(self.config, 'MOTR_KEEP_TRACK_SCORE_THRESH', 0.35)
        self.max_inactive_steps = int(getattr(self.config, 'MOTR_MAX_INACTIVE_STEPS', max(1, self.sequence_length)))
        self.promotion_suppression_radius = float(getattr(self.config, 'MOTR_PROMOTION_SUPPRESSION_RADIUS', 0.0))
        self.qim_augmentation = bool(getattr(self.config, 'MOTR_QIM_AUGMENTATION_FLAG', False))
        self.qim_random_drop = float(getattr(self.config, 'MOTR_QIM_RANDOM_DROP', 0.1)) if self.qim_augmentation else 0.0
        self.qim_fp_ratio = float(getattr(self.config, 'MOTR_QIM_FP_RATIO', 0.3)) if self.qim_augmentation else 0.0
        self.qim_score_thresh = float(getattr(self.config, 'MOTR_QIM_SCORE_THRESH', self.keep_track_score_thresh))
        self.frame_memory_size = int(getattr(self.config, 'MOTR_FRAME_MEMORY_SIZE', self.sequence_length - 1))
        if self.frame_memory_size < 1:
            raise ValueError('MOTR_FRAME_MEMORY_SIZE must be at least 1 for PntKPConvTbA')
        self.point_position_encoding = bool(getattr(self.config, 'MOTR_POINT_POSITION_ENCODING_FLAG', False))
        self.dbscan_query_init = bool(getattr(self.config, 'MOTR_DBSCAN_QUERY_INIT_FLAG', False))
        self.dbscan_hint_scale = float(getattr(self.config, 'MOTR_DBSCAN_HINT_SCALE', 1.0))
        self.transformer_embed_dim = self.embed_dim
        self.transformer_num_heads = self.config.TRANSFORMER_NUM_HEADS
        self.transformer_num_layers = self.config.TRANSFORMER_NUM_LAYERS
        self.transformer_dropout = self.config.TRANSFORMER_DROPOUT
        self.feature_extractor = Backbone(config=self.config)
        self.track_pos_dim = 2 if self.feature_extractor.use_xy_only else 3
        if self.point_position_encoding:
            position_hidden_dim = int(getattr(self.config, 'MOTR_POINT_POSITION_ENCODING_HIDDEN_DIM', self.embed_dim))
            self.point_position_encoder = nn.Sequential(nn.Linear(self.track_pos_dim, position_hidden_dim), nn.ReLU(), nn.Linear(position_hidden_dim, self.transformer_embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_embed_dim, nhead=self.transformer_num_heads, dim_feedforward=512, dropout=self.transformer_dropout, batch_first=True)
        self.point_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_num_layers)
        self.memory_norm = nn.LayerNorm(self.transformer_embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.transformer_embed_dim, nhead=self.transformer_num_heads, dim_feedforward=512, dropout=self.transformer_dropout, batch_first=True)
        self.track_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.transformer_num_layers)
        self.track_queries = nn.Parameter(torch.randn(1, self.num_track_queries, self.transformer_embed_dim))
        self.object_queries = nn.Parameter(torch.randn(1, self.num_object_queries, self.transformer_embed_dim))
        if self.dbscan_query_init:
            self.dbscan_query_initializer = DBSCANObjectQueryInitializer(point_dim=self.num_features, embed_dim=self.transformer_embed_dim, num_object_queries=self.num_object_queries, eps=float(getattr(self.config, 'MOTR_DBSCAN_EPS', 0.4)), min_samples=int(getattr(self.config, 'MOTR_DBSCAN_MIN_SAMPLES', 4)), coord_dims=int(getattr(self.config, 'MOTR_DBSCAN_COORD_DIMS', self.track_pos_dim)), hidden_dim=int(getattr(self.config, 'MOTR_DBSCAN_FCN_HIDDEN_DIM', 128)))
        self.track_norm = nn.LayerNorm(self.transformer_embed_dim)
        self.track_memory_norm = nn.LayerNorm(self.transformer_embed_dim)
        self.frame_temporal_embedding = nn.Embedding(self.frame_memory_size, self.transformer_embed_dim)
        self.query_interaction = MOTRQueryInteractionModule(embed_dim=self.transformer_embed_dim, num_heads=self.transformer_num_heads, dropout=self.transformer_dropout, pos_dim=self.track_pos_dim, score_thresh=self.qim_score_thresh, random_drop=self.qim_random_drop, fp_ratio=self.qim_fp_ratio)
        self.linear_regression = nn.Linear(self.transformer_embed_dim, self.regression_parameter)
        self.linear_classification = nn.Linear(self.transformer_embed_dim, 1)

    def _ensure_sequence_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f'Expected input with shape [B, N, F] or [B, T, N, F], got {tuple(x.shape)}')
        return x

    def init_track_queries(self, batch_size: int, device=None) -> torch.Tensor:
        track_queries = self.track_queries.expand(batch_size, -1, -1)
        if device is not None and track_queries.device != device:
            track_queries = track_queries.to(device)
        return track_queries

    def init_object_queries(self, batch_size: int, device=None) -> torch.Tensor:
        object_queries = self.object_queries.expand(batch_size, -1, -1)
        if device is not None and object_queries.device != device:
            object_queries = object_queries.to(device)
        return object_queries

    def build_object_queries(self, frame, frame_tokens):
        object_queries = self.init_object_queries(frame_tokens.shape[0], device=frame_tokens.device)
        proposal_outputs = None
        if self.dbscan_query_init:
            if frame is None:
                raise ValueError('Raw point clouds are required when MOTR_DBSCAN_QUERY_INIT_FLAG is enabled')
            proposal_outputs = self.dbscan_query_initializer(frame, frame_tokens)
            object_queries = object_queries + self.dbscan_hint_scale * proposal_outputs['queries']
        return (object_queries, proposal_outputs)

    def init_track_state(self, batch_size: int, device=None, initial_queries=None):
        if isinstance(initial_queries, dict):
            state = dict(initial_queries)
            for (key, value) in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device=device)
            if 'frame_memory' not in state or 'frame_memory_mask' not in state:
                dtype = state['queries'].dtype
                (memory, memory_mask) = self.init_frame_memory(batch_size, device=device, dtype=dtype)
                state['frame_memory'] = memory
                state['frame_memory_mask'] = memory_mask
            return state
        if initial_queries is None:
            queries = self.init_track_queries(batch_size, device=device)
            active_mask = torch.zeros(batch_size, self.num_track_queries, device=queries.device, dtype=torch.bool)
        else:
            queries = initial_queries
            if device is not None and queries.device != device:
                queries = queries.to(device)
            active_mask = torch.ones(batch_size, queries.shape[1], device=queries.device, dtype=torch.bool)
        (frame_memory, frame_memory_mask) = self.init_frame_memory(batch_size, device=queries.device, dtype=queries.dtype)
        return {'queries': queries, 'active_mask': active_mask, 'inactive_steps': torch.zeros(batch_size, queries.shape[1], device=queries.device, dtype=torch.long), 'last_regression': torch.zeros(batch_size, queries.shape[1], self.regression_parameter, device=queries.device, dtype=queries.dtype), 'last_classification': torch.full((batch_size, queries.shape[1]), -10.0, device=queries.device, dtype=queries.dtype), 'frame_memory': frame_memory, 'frame_memory_mask': frame_memory_mask}

    def init_frame_memory(self, batch_size: int, device=None, dtype=None):
        return (torch.zeros(batch_size, self.frame_memory_size, self.num_points, self.transformer_embed_dim, device=device, dtype=dtype or self.track_queries.dtype), torch.zeros(batch_size, self.frame_memory_size, device=device, dtype=torch.bool))

    def predict_from_queries(self, track_queries: torch.Tensor):
        regression = self.linear_regression(track_queries)
        classification = self.linear_classification(track_queries).squeeze(-1)
        return (regression, classification)

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dim() != 3:
            raise ValueError(f'Expected frame with shape [B, N, F], got {tuple(frame.shape)}')
        features = self.feature_extractor(frame)
        if self.point_position_encoding:
            point_positions = frame[..., :self.track_pos_dim]
            features = features + self.point_position_encoder(point_positions)
        return self.point_transformer(self.memory_norm(features))

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_sequence_input(x)
        (batch_size, seq_length, num_points, _) = x.shape
        frames = x.reshape(batch_size * seq_length, num_points, self.num_features)
        encoded = self.encode_frame(frames)
        return encoded.reshape(batch_size, seq_length, num_points, self.transformer_embed_dim)

    def _build_temporal_decoder_memory(self, frame_tokens, track_state):
        previous_memory = track_state['frame_memory']
        previous_mask = track_state['frame_memory_mask']
        memory_frames = torch.cat((previous_memory[:, 1:], frame_tokens.unsqueeze(1)), dim=1)
        memory_mask = torch.cat((previous_mask[:, 1:], torch.ones(previous_mask.shape[0], 1, device=previous_mask.device, dtype=torch.bool)), dim=1)
        frame_positions = torch.arange(self.frame_memory_size, device=frame_tokens.device, dtype=torch.long)
        ages = self.frame_memory_size - 1 - frame_positions
        temporal_bias = self.frame_temporal_embedding(ages).view(1, self.frame_memory_size, 1, self.transformer_embed_dim)
        memory_frames_with_temporal_bias = memory_frames + temporal_bias
        memory_tokens = memory_frames_with_temporal_bias.reshape(frame_tokens.shape[0], self.frame_memory_size * frame_tokens.shape[1], self.transformer_embed_dim)
        memory_key_padding_mask = (~memory_mask).unsqueeze(-1).expand(-1, -1, frame_tokens.shape[1]).reshape(frame_tokens.shape[0], self.frame_memory_size * frame_tokens.shape[1])
        return (memory_tokens, memory_key_padding_mask, memory_frames, memory_mask)

    def _reset_inactive_slots(self, track_state):
        active_mask = track_state['active_mask'].clone()
        inactive_mask = ~active_mask
        reset_queries = self.init_track_queries(track_state['queries'].shape[0], device=track_state['queries'].device)
        track_state['queries'] = torch.where(inactive_mask.unsqueeze(-1), reset_queries, track_state['queries'])
        track_state['last_regression'] = track_state['last_regression'] * active_mask.unsqueeze(-1)
        track_state['last_classification'] = track_state['last_classification'].masked_fill(inactive_mask, -10.0)
        track_state['inactive_steps'] = torch.where(active_mask, track_state['inactive_steps'], torch.zeros_like(track_state['inactive_steps']))
        return track_state

    def _promote_new_tracks(self, decoded_objects, regression_objects, classification_objects, next_track_state):
        promoted_queries = next_track_state['queries'].clone()
        promoted_regression = next_track_state['last_regression'].clone()
        promoted_classification = next_track_state['last_classification'].clone()
        promoted_active_mask = next_track_state['active_mask'].clone()
        promoted_inactive_steps = next_track_state['inactive_steps'].clone()
        promoted_object_indices = torch.full((decoded_objects.shape[0], self.num_track_queries), -1, device=decoded_objects.device, dtype=torch.long)
        object_scores = torch.sigmoid(classification_objects)
        pos_dim = min(self.track_pos_dim, regression_objects.shape[-1], promoted_regression.shape[-1])
        batch_size = decoded_objects.shape[0]
        batch_indices = torch.arange(batch_size, device=decoded_objects.device)
        candidate_order = torch.argsort(object_scores, dim=1, descending=True)
        for candidate_rank in range(decoded_objects.shape[1]):
            object_idx = candidate_order[:, candidate_rank]
            candidate_scores = object_scores[batch_indices, object_idx]
            free_mask = ~promoted_active_mask
            has_free_slot = free_mask.any(dim=1)
            eligible = has_free_slot & (candidate_scores >= self.new_track_score_thresh)
            if self.promotion_suppression_radius > 0:
                candidate_positions = regression_objects[batch_indices, object_idx, :pos_dim]
                distances = torch.linalg.vector_norm(candidate_positions.unsqueeze(1) - promoted_regression[..., :pos_dim], dim=-1)
                min_active_distance = distances.masked_fill(~promoted_active_mask, float('inf')).min(dim=1).values
                eligible = eligible & (min_active_distance >= self.promotion_suppression_radius)
            free_slot_idx = free_mask.to(torch.int64).argmax(dim=1)
            slot_mask = F.one_hot(free_slot_idx, num_classes=self.num_track_queries).bool() & eligible.unsqueeze(1)
            candidate_queries = decoded_objects[batch_indices, object_idx]
            candidate_regression = regression_objects[batch_indices, object_idx]
            candidate_classification = classification_objects[batch_indices, object_idx]
            promoted_queries = torch.where(slot_mask.unsqueeze(-1), candidate_queries.unsqueeze(1), promoted_queries)
            promoted_regression = torch.where(slot_mask.unsqueeze(-1), candidate_regression.unsqueeze(1), promoted_regression)
            promoted_classification = torch.where(slot_mask, candidate_classification.unsqueeze(1), promoted_classification)
            promoted_active_mask = promoted_active_mask | slot_mask
            promoted_inactive_steps = torch.where(slot_mask, torch.zeros_like(promoted_inactive_steps), promoted_inactive_steps)
            promoted_object_indices = torch.where(slot_mask, object_idx.unsqueeze(1), promoted_object_indices)
        next_track_state['queries'] = promoted_queries
        next_track_state['last_regression'] = promoted_regression
        next_track_state['last_classification'] = promoted_classification
        next_track_state['active_mask'] = promoted_active_mask
        next_track_state['inactive_steps'] = promoted_inactive_steps
        return (next_track_state, promoted_object_indices)

    def _select_output_tracks(self, track_state):
        if self.num_track_queries == self.num_people:
            selected_track_indices = torch.arange(self.num_people, device=track_state['queries'].device, dtype=torch.long).unsqueeze(0).expand(track_state['queries'].shape[0], -1)
        else:
            track_scores = torch.sigmoid(track_state['last_classification'])
            track_scores = track_scores.masked_fill(~track_state['active_mask'], -1.0)
            selected_track_indices = torch.topk(track_scores, k=min(self.num_people, track_scores.shape[1]), dim=1).indices
        if selected_track_indices.shape[1] < self.num_people:
            pad = selected_track_indices.new_zeros(selected_track_indices.shape[0], self.num_people - selected_track_indices.shape[1])
            selected_track_indices = torch.cat((selected_track_indices, pad), dim=1)
        gather_query_idx = selected_track_indices.unsqueeze(-1).expand(-1, -1, track_state['queries'].shape[-1])
        gather_reg_idx = selected_track_indices.unsqueeze(-1).expand(-1, -1, track_state['last_regression'].shape[-1])
        output_queries = torch.gather(track_state['queries'], 1, gather_query_idx)
        output_regression = torch.gather(track_state['last_regression'], 1, gather_reg_idx)
        output_classification = torch.gather(track_state['last_classification'], 1, selected_track_indices)
        output_active_mask = torch.gather(track_state['active_mask'], 1, selected_track_indices).clone()
        output_regression = output_regression * output_active_mask.unsqueeze(-1)
        output_classification = output_classification.masked_fill(~output_active_mask, -10.0)
        return (output_queries, output_regression, output_classification, output_active_mask, selected_track_indices)

    def forward_frame(self, frame, track_queries=None, frame_already_encoded=False, raw_frame=None):
        if frame_already_encoded:
            frame_tokens = frame
        else:
            raw_frame = frame
            frame_tokens = self.encode_frame(frame)
        batch_size = frame_tokens.shape[0]
        track_state = self.init_track_state(batch_size, device=frame_tokens.device, initial_queries=track_queries)
        (propagated_track_queries, selected_track_mask, qim_interaction_mask) = self.query_interaction(track_state['queries'], track_state['last_regression'], track_state['last_classification'], track_state['active_mask'])
        if self.qim_augmentation:
            initial_track_queries = self.init_track_queries(batch_size, device=frame_tokens.device)
            propagated_track_queries = torch.where(qim_interaction_mask.unsqueeze(-1), propagated_track_queries, initial_track_queries)
        (object_queries, proposal_outputs) = self.build_object_queries(raw_frame, frame_tokens)
        decoder_queries = torch.cat((propagated_track_queries, object_queries), dim=1)
        (decoder_memory, memory_key_padding_mask, next_frame_memory, next_frame_memory_mask) = self._build_temporal_decoder_memory(frame_tokens, track_state)
        decoded_queries = self.track_decoder(tgt=self.track_norm(decoder_queries), memory=self.track_memory_norm(decoder_memory), memory_key_padding_mask=memory_key_padding_mask)
        decoded_tracks = decoded_queries[:, :self.num_track_queries]
        decoded_objects = decoded_queries[:, self.num_track_queries:]
        (regression_all, classification_all) = self.predict_from_queries(decoded_queries)
        regression_tracks = regression_all[:, :self.num_track_queries]
        classification_tracks = classification_all[:, :self.num_track_queries]
        regression_objects = regression_all[:, self.num_track_queries:]
        classification_objects = classification_all[:, self.num_track_queries:]
        next_track_state = {'queries': track_state['queries'].clone(), 'active_mask': track_state['active_mask'].clone(), 'inactive_steps': track_state['inactive_steps'].clone(), 'last_regression': torch.zeros_like(track_state['last_regression']), 'last_classification': torch.full_like(track_state['last_classification'], -10.0), 'frame_memory': next_frame_memory, 'frame_memory_mask': next_frame_memory_mask}
        if self.qim_augmentation:
            next_track_state['queries'] = self.init_track_queries(batch_size, device=frame_tokens.device).clone()
            next_track_state['active_mask'] = torch.zeros_like(track_state['active_mask'])
            next_track_state['inactive_steps'] = torch.zeros_like(track_state['inactive_steps'])
        active_scores = torch.sigmoid(classification_tracks)
        next_track_state['queries'] = torch.where(selected_track_mask.unsqueeze(-1), decoded_tracks, next_track_state['queries'])
        next_track_state['last_regression'] = torch.where(selected_track_mask.unsqueeze(-1), regression_tracks, next_track_state['last_regression'])
        next_track_state['last_classification'] = torch.where(selected_track_mask, classification_tracks, next_track_state['last_classification'])
        next_track_state['inactive_steps'] = torch.where(selected_track_mask, torch.where(active_scores >= self.keep_track_score_thresh, torch.zeros_like(next_track_state['inactive_steps']), track_state['inactive_steps'] + 1), next_track_state['inactive_steps'])
        next_track_state['active_mask'] = torch.where(selected_track_mask, next_track_state['inactive_steps'] < self.max_inactive_steps, next_track_state['active_mask'])
        next_track_state = self._reset_inactive_slots(next_track_state)
        (next_track_state, promoted_object_indices) = self._promote_new_tracks(decoded_objects=decoded_objects, regression_objects=regression_objects, classification_objects=classification_objects, next_track_state=next_track_state)
        (output_track_queries, regression_tracks_out, classification_tracks_out, output_active_mask, selected_track_indices) = self._select_output_tracks(next_track_state)
        return {'frame_tokens': frame_tokens, 'track_state': next_track_state, 'track_queries': next_track_state['queries'], 'propagated_track_queries': propagated_track_queries, 'qim_selected_track_mask': selected_track_mask, 'decoded_track_queries': decoded_tracks, 'output_track_queries': output_track_queries, 'selected_track_indices': selected_track_indices, 'active_track_mask': next_track_state['active_mask'], 'output_active_track_mask': output_active_mask, 'promoted_object_indices': promoted_object_indices, 'object_queries': decoded_objects, 'initialized_object_queries': object_queries, 'dbscan_proposal_outputs': proposal_outputs, 'all_queries': decoded_queries, 'pred_regression': regression_tracks_out, 'pred_classification': classification_tracks_out, 'pred_regression_track': next_track_state['last_regression'], 'pred_classification_track': next_track_state['last_classification'], 'pred_regression_object': regression_objects, 'pred_classification_object': classification_objects, 'pred_regression_all': regression_all, 'pred_classification_all': classification_all, 'decoder_memory': decoder_memory, 'frame_memory': next_frame_memory, 'frame_memory_mask': next_frame_memory_mask}

    def forward(self, x, initial_track_queries=None, return_aux=False):
        x = self._ensure_sequence_input(x)
        encoded_frames = self.encode_sequence(x)
        (batch_size, seq_length) = encoded_frames.shape[:2]
        track_state = self.init_track_state(batch_size, device=encoded_frames.device, initial_queries=initial_track_queries)
        per_frame_queries = []
        per_frame_propagated_queries = []
        per_frame_object_queries = []
        per_frame_all_queries = []
        per_frame_regression = []
        per_frame_classification = []
        per_frame_regression_all = []
        per_frame_classification_all = []
        per_frame_regression_object = []
        per_frame_classification_object = []
        per_frame_qim_selected_masks = []
        per_frame_active_masks = []
        per_frame_promotions = []
        for t in range(seq_length):
            frame_out = self.forward_frame(encoded_frames[:, t], track_queries=track_state, frame_already_encoded=True, raw_frame=x[:, t])
            track_state = frame_out['track_state']
            per_frame_queries.append(track_state['queries'])
            per_frame_propagated_queries.append(frame_out['propagated_track_queries'])
            per_frame_object_queries.append(frame_out['object_queries'])
            per_frame_all_queries.append(frame_out['all_queries'])
            per_frame_regression.append(frame_out['pred_regression'])
            per_frame_classification.append(frame_out['pred_classification'])
            per_frame_regression_all.append(frame_out['pred_regression_all'])
            per_frame_classification_all.append(frame_out['pred_classification_all'])
            per_frame_regression_object.append(frame_out['pred_regression_object'])
            per_frame_classification_object.append(frame_out['pred_classification_object'])
            per_frame_qim_selected_masks.append(frame_out['qim_selected_track_mask'])
            per_frame_active_masks.append(frame_out['active_track_mask'])
            per_frame_promotions.append(frame_out['promoted_object_indices'])
        per_frame_queries = torch.stack(per_frame_queries, dim=1)
        per_frame_propagated_queries = torch.stack(per_frame_propagated_queries, dim=1)
        per_frame_object_queries = torch.stack(per_frame_object_queries, dim=1)
        per_frame_all_queries = torch.stack(per_frame_all_queries, dim=1)
        per_frame_regression = torch.stack(per_frame_regression, dim=1)
        per_frame_classification = torch.stack(per_frame_classification, dim=1)
        per_frame_regression_all = torch.stack(per_frame_regression_all, dim=1)
        per_frame_classification_all = torch.stack(per_frame_classification_all, dim=1)
        per_frame_regression_object = torch.stack(per_frame_regression_object, dim=1)
        per_frame_classification_object = torch.stack(per_frame_classification_object, dim=1)
        per_frame_qim_selected_masks = torch.stack(per_frame_qim_selected_masks, dim=1)
        per_frame_active_masks = torch.stack(per_frame_active_masks, dim=1)
        per_frame_promotions = torch.stack(per_frame_promotions, dim=1)
        final_regression = per_frame_regression[:, -1]
        final_classification = per_frame_classification[:, -1]
        y = torch.concat((final_regression.reshape(batch_size, -1), final_classification.reshape(batch_size, -1)), dim=-1)
        if not return_aux:
            return y
        aux = {'encoded_frames': encoded_frames, 'track_queries_per_frame': per_frame_queries, 'propagated_track_queries_per_frame': per_frame_propagated_queries, 'object_queries_per_frame': per_frame_object_queries, 'all_queries_per_frame': per_frame_all_queries, 'pred_regression_per_frame': per_frame_regression, 'pred_classification_per_frame': per_frame_classification, 'pred_regression_all_per_frame': per_frame_regression_all, 'pred_classification_all_per_frame': per_frame_classification_all, 'pred_regression_object_per_frame': per_frame_regression_object, 'pred_classification_object_per_frame': per_frame_classification_object, 'qim_selected_track_mask_per_frame': per_frame_qim_selected_masks, 'active_track_mask_per_frame': per_frame_active_masks, 'promoted_object_indices_per_frame': per_frame_promotions, 'final_track_queries': track_state['queries'], 'final_track_state': track_state, 'final_output': y}
        return (y, aux)

class Backbone(nn.Module):
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
        self.k_max = 3
        self.max_hops = 3
        self.tau = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.use_xy_only = True
        self.hop_weights = torch.ones(self.max_hops, dtype=torch.float32)
        self.pointnet = nn.Sequential(nn.Conv1d(8, 64, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128, 384, 1), nn.BatchNorm1d(384))
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
        return pnt

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
