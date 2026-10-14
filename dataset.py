from hlat.dataloader.mmRadLoc import mmRadLoc

class Dataset:

    def __init__(self, config):
        self.config = config

    def load_dataset(self, only_test_set=False):
        cfg = self.config
        if cfg.DATASET_KIND != 'mmRadLoc':
            raise ValueError('This release supports mmRadLoc only')
        split = {'train_records': cfg.DATASET['train'], 'val_records': cfg.DATASET['val'], 'test_records': cfg.DATASET['test'], 'num_features': cfg.PNT_NUM_FEATURES, 'num_people': getattr(cfg, 'PNT_GT_NUM_PEOPLE', None) or cfg.PNT_NUM_PEOPLE}

        def load(mode):
            return mmRadLoc(mode=mode, config_dict=split, datasetPath=cfg.DATA_PATH, slicesDataPath=cfg.DATA_SLICE_PATH, seq_length=cfg.SEQUENCE_LENGTH, use_pc_components=None, augmentation=cfg.AUGMENTATION_FLAG if mode == 'train' else False, data_part=1, use_ram=cfg.USE_RAM, config=cfg)
        return (None if only_test_set else load('train'), None if only_test_set else load('val'), load('test'), split, None)
