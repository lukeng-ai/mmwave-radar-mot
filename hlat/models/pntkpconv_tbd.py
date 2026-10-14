import torch
import torch.nn as nn
from hlat.models.components import PointSlotEncoder, SlotUtilities

class FactorizedSlotTransformerBlock(nn.Module):

    def __init__(self, slot_dim, temporal_num_heads, slot_num_heads, dim_feedforward, dropout):
        super().__init__()
        self.temporal_encoder = nn.TransformerEncoderLayer(d_model=slot_dim, nhead=temporal_num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.slot_encoder = nn.TransformerEncoderLayer(d_model=slot_dim, nhead=slot_num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

    def forward(self, x):
        (batch_size, seq_length, num_slots, slot_dim) = x.shape
        temporal_input = x.permute(0, 2, 1, 3).contiguous().reshape(batch_size * num_slots, seq_length, slot_dim)
        temporal_output = self.temporal_encoder(temporal_input)
        x = temporal_output.reshape(batch_size, num_slots, seq_length, slot_dim).permute(0, 2, 1, 3).contiguous()
        slot_input = x.reshape(batch_size * seq_length, num_slots, slot_dim)
        slot_output = self.slot_encoder(slot_input)
        return slot_output.reshape(batch_size, seq_length, num_slots, slot_dim)

class PntKPConvTbD(nn.Module):
    """Tracking-by-detection model with factorized temporal and slot attention.

The slot head predicts positions and presence scores for downstream Kalman association."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.num_features = self.config.PNT_NUM_FEATURES
        self.embed_dim = self.config.PNT_EMBED_DIM
        self.num_people = self.config.PNT_NUM_PEOPLE
        self.num_points = self.config.PNT_NUM_POINTS
        self.dropout = self.config.PNT_DROPOUT
        self.sequence_length = self.config.SEQUENCE_LENGTH
        self.regression_parameter = self.config.REGRESSION_PARAMETER
        self.slot_sequence_reordering = bool(getattr(self.config, 'SLOT_SEQUENCE_REORDERING_FLAG', False))
        self.feature_extractor = PointSlotEncoder(config=self.config)
        self.num_slots = int(getattr(self.feature_extractor, 'k_max', self.num_people))
        if self.num_people > self.num_slots:
            raise ValueError(f'PNT_NUM_PEOPLE must not exceed the number of feature-extractor slots: {self.num_people} > {self.num_slots}.')
        self.slot_dim = int(getattr(self.feature_extractor, 'pnt_emb_dim', self.embed_dim))
        self.transformer_embed_dim = self.slot_dim
        temporal_num_heads = int(getattr(self.config, 'FAC_TESH_TEMPORAL_NUM_HEADS', self.config.TRANSFORMER_NUM_HEADS))
        slot_num_heads = int(getattr(self.config, 'FAC_TESH_SLOT_NUM_HEADS', self.config.TRANSFORMER_NUM_HEADS))
        num_layers = int(getattr(self.config, 'FAC_TESH_NUM_LAYERS', self.config.TRANSFORMER_NUM_LAYERS))
        dim_feedforward = int(getattr(self.config, 'FAC_TESH_FF_DIM', 512))
        transformer_dropout = float(getattr(self.config, 'FAC_TESH_DROPOUT', self.config.TRANSFORMER_DROPOUT))
        if self.slot_dim % temporal_num_heads != 0:
            raise ValueError(f'slot_dim must be divisible by FAC_TESH_TEMPORAL_NUM_HEADS: {self.slot_dim} % {temporal_num_heads} != 0.')
        if self.slot_dim % slot_num_heads != 0:
            raise ValueError(f'slot_dim must be divisible by FAC_TESH_SLOT_NUM_HEADS: {self.slot_dim} % {slot_num_heads} != 0.')
        temporal_encoding = SlotUtilities.get_linear_positional_encoding(self.sequence_length, self.slot_dim)
        self.temporal_positional_encoding = nn.Parameter(temporal_encoding.view(1, self.sequence_length, 1, self.slot_dim))
        self.slot_positional_encoding = nn.Parameter(torch.zeros(1, 1, self.num_slots, self.slot_dim))
        self.transformer_blocks = nn.ModuleList([FactorizedSlotTransformerBlock(slot_dim=self.slot_dim, temporal_num_heads=temporal_num_heads, slot_num_heads=slot_num_heads, dim_feedforward=dim_feedforward, dropout=transformer_dropout) for _ in range(num_layers)])
        self.linear_regression = nn.Linear(self.slot_dim, self.regression_parameter)
        self.linear_classification = nn.Linear(self.slot_dim, 1)

    @staticmethod
    def get_linear_positional_encoding(seq_length, d_model):
        return SlotUtilities.get_linear_positional_encoding(seq_length, d_model)

    @staticmethod
    def reorder_by_sequence_similarity(x):
        return SlotUtilities.reorder_by_sequence_similarity(x)

    @staticmethod
    def reorder_by_nearest_neighbor(x):
        return SlotUtilities.reorder_by_nearest_neighbor(x)

    def _select_supervised_slots(self, regression, classification):
        return SlotUtilities._select_supervised_slots(self, regression, classification)

    def _reorder_slots_for_encoder(self, slots):
        if self.slot_sequence_reordering:
            return self.reorder_by_sequence_similarity(slots)
        return slots

    def forward(self, x):
        (batch_size, seq_length) = (x.shape[0], x.shape[1])
        x = x.reshape((batch_size * seq_length, self.num_points, self.num_features))
        x = self.feature_extractor(x)
        x = x.reshape(batch_size, seq_length, self.num_slots, self.slot_dim)
        x = self._reorder_slots_for_encoder(x)
        slots = x.reshape(batch_size, -1)
        x = x + self.temporal_positional_encoding.to(x.device)
        x = x + self.slot_positional_encoding.to(x.device)
        for block in self.transformer_blocks:
            x = block(x)
        x = x[:, -1]
        all_regression = self.linear_regression(x)
        all_classification = self.linear_classification(x).squeeze(-1)
        (selected_regression, selected_classification, selected_indices) = self._select_supervised_slots(all_regression, all_classification)
        y_regression = selected_regression.reshape(batch_size, self.num_people * self.regression_parameter)
        linear_classification = selected_classification
        all_predictions = torch.concat((all_regression.reshape(batch_size, self.num_slots * self.regression_parameter), all_classification, selected_indices.to(dtype=all_regression.dtype)), dim=-1)
        y = torch.concat((y_regression, linear_classification, all_predictions, slots), dim=-1)
        return y
