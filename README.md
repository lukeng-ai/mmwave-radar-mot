# Millimeter-wave radar multi-object tracking

Companion implementation for the article *Bringing Modern Multi-Object Tracking to Millimeter-Wave Radar Target Lists for Indoor Human Tracking* by Lukas Engel, Markus Bergmann, Christoph Kammel, Ingrid Ullmann, Bjoern M. Eskofier, and Martin Vossiek. (https://ieeexplore.ieee.org/document/11741145)

The repository contains the two methods evaluated in the article:

- `PntKPConvTbD`: tracking by detection with a factorized slot transformer and subsequent Kalman tracking.
- `PntKPConvTbA`: tracking by attention with persistent track queries and frame memory.

## Dataset

The dataset is not part of this repository and is hosted on Zenodo. Add the public Zenodo record URL or DOI here before releasing the repository.

After downloading and extracting it, the expected layout is:

```text
DATA_ROOT/
  train/recording_001/points.npy
  train/recording_001/targets.npy
  ...
  valid/recording_025/points.npy
  ...
  test/recording_028/points.npy
  ...
```

`points.npy` has shape `[frames, 64, 7]`. Its first two feature columns are the XY coordinates in metres; zero rows are padding. `targets.npy` has shape `[frames, 35]`: five six-value target records `[x, y, z, vx, vy, vz]`, followed by five presence flags. Both arrays must have the same frame count.

## Installation

Use Python 3.9 or later and install the PyTorch build appropriate for your CPU/CUDA environment. Then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The standard configurations use bfloat16 mixed precision. Set `--MIXED_PRECISION_FLAG False` if the target hardware does not support it.

## Standard configurations

Only the two standard article configurations are included:

| Method | Configuration | Sequence length | Frame memory | Epochs |
| --- | --- | ---: | ---: | ---: |
| TbD | `configs/tbd_seq14.yaml` | 14 | – | 60 |
| TbA | `configs/tba_seq10_memory10.yaml` | 10 | 10 | 30 |

Both configurations use the optimization settings reported in the article: Adam, a batch size of 64, an initial learning rate of `1e-4`, multiplicative learning-rate decay of `0.98`, and weight decay of `1e-4`. The release configurations additionally fix seed 42, disable input point dropping, and use the article's 1 m XY matching gate for tracking evaluation.

## Training

Run from the repository root:

```bash
python train.py -c configs/tbd_seq14.yaml -d /path/to/DATA_ROOT -w .
python train.py -c configs/tba_seq10_memory10.yaml -d /path/to/DATA_ROOT -w .
```

Results are written below `results/PntKPConvTbD/` and `results/PntKPConvTbA/`. A run directory contains the resolved configuration, checkpoints, metrics, and TensorBoard logs. TbD checkpoint selection includes Kalman tracking validation; TbA uses stateful streaming validation.

Configuration values can be overridden at the command line, for example with `--SEED 43 --NUM_WORKERS 0`. `USE_RAM` is enabled in both standard configurations. If it is disabled, `DATA_SLICE_PATH` must point to a dedicated cache directory.

## Evaluation

Evaluate a completed run with its saved configuration and `best_model.pth`:

```bash
python evaluate.py -i results/PntKPConvTbD/RUN -d /path/to/DATA_ROOT
python evaluate.py -i results/PntKPConvTbA/RUN -d /path/to/DATA_ROOT
```

The evaluator writes `test_metrics.json` and tracking outputs to `test_tracking/` inside the run directory. TbD uses Kalman association; TbA uses stateful streaming inference.

The reported tracking values follow the fixed-distance protocol used in the article. In particular, the value named `hota` is computed as `sqrt(DetA * AssA)` at the configured 1 m matching gate; it is not averaged over multiple localization thresholds. Frames containing a valid ground-truth target beyond the configured Y limit of 4.9 m are excluded from tracking evaluation.

## Implementation notes

- Windows never cross recording boundaries. TbD supervises the final frame of each window; TbA is supervised across the sequence.
- KPConv kernel layouts are generated in `cache/kernels/` on first use. No compiled KPConv extension is required.
- Raw radar processing, calibration, annotation tooling, trained weights, and dataset files are outside this code release.

## Citation

If you use this implementation, please cite the article above and the associated Zenodo dataset. Add the public dataset DOI and the final article DOI/BibTeX here before release.

## License

No reuse license has been granted for the original code in this repository yet. Until a license file is added, it remains subject to the default copyright restrictions. Required notices for incorporated third-party code are listed in `THIRD_PARTY_NOTICES.md`.
