"""Evaluate either release model using its saved test and tracking settings."""
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from model import PntKPConvTbA, PntKPConvTbD
from tester import Tester
from utils.config import Config
from utils.train_utils import get_criterion

class NullWriter:

    def add_scalar(self, *args, **kwargs):
        pass

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', required=True, help='Run directory with config.yaml and best_model.pth')
    parser.add_argument('-d', '--datasetdir', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()
    run = Path(args.input).resolve()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    cfg = Config(DEVICE=device, LOAD_FROM_CONFIG_FILE_FLAG=True, CONFIG_PATH=str(run / 'config.yaml'), args={'DATA_PATH': args.datasetdir})
    cfg.NUM_WORKERS = args.num_workers
    cfg.SAVE_PATH = str(run)
    model = {'PntKPConvTbA': PntKPConvTbA, 'PntKPConvTbD': PntKPConvTbD}[cfg.MODEL](cfg).to(device)
    model.load_state_dict(torch.load(run / 'best_model.pth', map_location=device, weights_only=True))
    (_, _, test_data, _, _) = Dataset(cfg).load_dataset(only_test_set=True)
    loader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=device.type == 'cuda')
    testing = Tester(model, loader, get_criterion(cfg.CRITERION), cfg, (logging.getLogger('evaluation'), NullWriter()), str(run), cfg.SEQUENCE_LENGTH)
    testing.test()
if __name__ == '__main__':
    main()
