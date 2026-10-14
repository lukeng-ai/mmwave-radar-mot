import torch
from dataclasses import dataclass, field
import yaml
import os

@dataclass
class Config:
    DEVICE: torch.device
    args: dict = field(default_factory=dict)
    LOAD_FROM_CONFIG_FILE_FLAG: bool = False
    CONFIG_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'config.yaml')
    DATA_PATH: str = './data/'
    DATA_POINT_FILENAME: str = 'radar_pointcloud.npy'
    DATA_TARGET_FILENAME: str = 'labels_final_kalman.npy'
    DATA_SLICE_PATH: str = './data/slice/'
    WORKDIR: str = './'
    SAVE_PATH: str = './results/'
    CHAR: str = '-'
    USE_RAM: bool = False
    EPOCHS: int = 50
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.0001
    SEED: int = 42
    NUM_WORKERS: int = 0
    SAVE_INTERVAL: int = 20
    SEQUENCE_LENGTH: int = 2
    LR_DECAY: float = 0.98
    WEIGHT_DECAY: float = 0.0001
    GRADIENT_CLIPPING_MAX_NORM: float = None
    MIXED_PRECISION_FLAG: bool = False
    MIXED_PRECISION_DTYPE: str = 'bfloat16'
    DATALOADER_PERSISTENT_WORKERS_FLAG: bool = True
    DATALOADER_PREFETCH_FACTOR: int = 2
    CRITERION: str = 'MSELossPosRegBCEClsMask'
    LOSSES: list = field(default_factory=list)
    IND_LOSSES: list = field(default_factory=list)
    TEST_LOSSES: list = field(default_factory=list)
    IND_TEST_LOSSES: list = field(default_factory=list)
    WEIGHT_REGRESSION: float = 1.0
    WEIGHT_CLASSIFICATION: float = 1.0
    PRECISION_LOSS_WEIGHT: float = 0.0
    SOFT_DETA_LOSS_WEIGHT: float = 0.0
    SOFT_DETA_FP_WEIGHT: float = 1.0
    SOFT_DETA_FN_WEIGHT: float = 1.0
    TEMPORAL_PRESENCE_LOSS_WEIGHT: float = 0.0
    COUNT_OVER_LOSS_WEIGHT: float = 0.0
    COUNT_OVER_HUBER_DELTA: float = 0.25
    LOCALIZATION_REGRESSION_LOSS_DIMS: int = None
    REGRESSION_PARAMETER: int = 6
    RETURN_SEQUENCE_TARGETS: bool = False
    PNT_GT_NUM_PEOPLE: int = None
    MOTR_EOS_COEF: float = 1.0
    MOTR_DBSCAN_QUERY_INIT_FLAG: bool = False
    MOTR_DBSCAN_EPS: float = 0.4
    MOTR_DBSCAN_MIN_SAMPLES: int = 4
    MOTR_DBSCAN_COORD_DIMS: int = 2
    MOTR_DBSCAN_FCN_HIDDEN_DIM: int = 128
    MOTR_DBSCAN_HINT_SCALE: float = 1.0
    PNT_NUM_SLOTS: int = None
    SLOT_MATCHING_LOSS_WEIGHT: float = None
    SLOT_MATCHING_REGRESSION_WEIGHT: float = 1.0
    SLOT_MATCHING_CLASSIFICATION_WEIGHT: float = 1.0
    SLOT_MATCHING_EOS_COEF: float = 0.1
    SLOT_MATCHING_MATCH_DIMS: int = 2
    SLOT_MATCHING_MATCH_COST_POS: float = 1.0
    SLOT_MATCHING_MATCH_COST_CLS: float = 0.1
    GT_TRACK_STABILIZATION_FLAG: bool = False
    GT_TRACK_STABILIZATION_THRESH: float = 0.5
    GT_TRACK_STABILIZATION_MAX_MISSING: int = 0
    GT_TRACK_STABILIZATION_NUM_SLOTS: int = 2
    MATCH_COST_POS: float = 1.0
    MATCH_COST_CLS: float = 0.5
    KALMAN_TRACKING_VALIDATION_SELECTION_METRIC: str = 'score'
    KALMAN_TRACKING_VALIDATION_NMAE_XY_WEIGHT: float = 0.1
    KALMAN_TRACKING_VALIDATION_MPE_XY_WEIGHT: float = 0.1
    KALMAN_TRACKING_VALIDATION_MPE_XY_NORMALIZATION: float = None
    MOTR_STREAMING_VALIDATION_SELECTION_METRIC: str = 'score'
    MOTR_STREAMING_VALIDATION_NMAE_XY_WEIGHT: float = 0.1
    MOTR_STREAMING_VALIDATION_MPE_XY_WEIGHT: float = 0.1
    MOTR_STREAMING_VALIDATION_MPE_XY_NORMALIZATION: float = None
    TRACKING_TEST_FLAG: bool = True
    TRACKING_TEST_Y_LIMIT: float = 4.9
    TRACKING_TEST_RESET_BETWEEN_SEQUENCES: bool = True
    TRACKING_TEST_OUTPUT_DIR: str = 'test_tracking'
    GENERATE_RESULTS_AFTER_TEST_FLAG: bool = False
    GENERATE_RESULTS_PLOT_FORMAT: str = 'pdf'
    SAVE_PREDS_OTHER_FLAG: bool = None
    DATASET_KIND: str = 'mmRadLoc'
    DATASET: dict = field(default_factory=dict)
    AUGMENTATION_FLAG = False
    AUGMENTATION_NOISE: float = field(default=None, init=False)
    AUGMENTATION_FLIP: int = field(default=None, init=False)
    AUGMENTATION_TRANSLATE: float = field(default=None, init=False)
    AUGMENTATION_YTRANSLATE: float = field(default=None, init=False)
    TEMPORAL_FRAME_SAMPLING_FLAG: bool = False
    TEMPORAL_FRAME_SAMPLING_MIN_INTERVAL: int = 1
    TEMPORAL_FRAME_SAMPLING_MAX_INTERVAL: int = 10
    PNT_INPUT_POINT_DROP_PERCENT: float = 0.0
    MODEL: str = 'PntKPConvTbA'
    TRANSFORMER_FLAG: bool = False
    TRANSFORMER_EMBED_DIM: int = field(default=None, init=False)
    TRANSFORMER_NUM_HEADS: int = field(default=None, init=False)
    TRANSFORMER_NUM_LAYERS: int = field(default=None, init=False)
    TRANSFORMER_DROPOUT: int = field(default=None, init=False)
    PNT_NUM_POINTS: int = field(default=None, init=False)
    PNT_NUM_FEATURES: int = field(default=None, init=False)
    PNT_EMBED_DIM: int = field(default=None, init=False)
    PNT_DROPOUT: list = field(default_factory=list, init=False)
    LOCALIZATION_FLAG: bool = False
    LOC_LOSS_SORT_BY_POSITION_FLAG: bool = field(default=None, init=False)
    ADAPTIONS_FLAG: bool = False
    ADAPTIONS: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.LOAD_FROM_CONFIG_FILE_FLAG:
            with open(self.CONFIG_PATH) as handle:
                values = yaml.safe_load(handle)
            for (key, value) in values.items():
                setattr(self, key.upper(), value)
        for (key, value) in self.args.items():
            if value is None or key in ('config', 'datasetdir', 'slicesdir', 'workdir', 'useramTrue', 'actionwiseTrue'):
                continue
            if isinstance(value, str):
                value = yaml.safe_load(value)
            setattr(self, key.upper(), value)
        for (key, attr) in (('datasetdir', 'DATA_PATH'), ('slicesdir', 'DATA_SLICE_PATH'), ('workdir', 'WORKDIR')):
            if self.args.get(key) is not None:
                setattr(self, attr, self.args[key])
        if self.MODEL not in ('PntKPConvTbA', 'PntKPConvTbD'):
            raise ValueError(f'Unsupported model: {self.MODEL}')
        self.RETURN_SEQUENCE_TARGETS = self.MODEL == 'PntKPConvTbA'
        if self.RETURN_SEQUENCE_TARGETS:
            self.MOTR_FRAME_MEMORY_SIZE = int(getattr(self, 'MOTR_FRAME_MEMORY_SIZE', self.SEQUENCE_LENGTH - 1))
            if self.SEQUENCE_LENGTH < 2 or self.MOTR_FRAME_MEMORY_SIZE < 1:
                raise ValueError('TbA requires SEQUENCE_LENGTH >= 2 and MOTR_FRAME_MEMORY_SIZE >= 1')
        self.LOCALIZATION_FLAG = True
        self.DATASET_KIND = 'mmRadLoc'

    def save_config(self, path):
        values = {key: value for (key, value) in vars(self).items() if key not in ('DEVICE', 'args')}
        with open(os.path.join(path, 'config.yaml'), 'w') as handle:
            yaml.safe_dump(values, handle, sort_keys=True)

    def print_config(self):
        print({key: value for (key, value) in vars(self).items() if key != 'args'})
