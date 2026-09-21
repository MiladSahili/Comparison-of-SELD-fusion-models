# Comparison of SELD Fusion Models

Audio-visual fusion for **Sound Event Localization and Detection (SELD)** on the
[STARSS23](https://dcase.community/challenge2023/task-sound-event-localization-and-detection) dataset
(DCASE 2023 Task 3), comparing fusion strategies at the **input level (early)**,
**feature level (mid)**, and **decision level (late)** against an audio-only reference —
all sharing an identical architecture except for the fusion point.

All models output Multi-ACCDOA `(B, 3, 3, 13, T)` (3 tracks, 3 axes, 13 classes, T frames)
and are trained with the ADPIT loss under an identical protocol, so that the fusion
level is the only architectural variable.

## Models

| Model | Fusion point | Params |
|---|---|---|
| `audio_only` | — (reference) | 632,757 |
| `audio_only_matched` | — (capacity control, GRU hidden 266) | 670,377 |
| `early` | input level (visual projected to frequency, appended as channel) | 747,698 |
| `mid` | feature level (64-dim embeddings concatenated before shared GRU) | 763,701 |
| `mid_shuffled` | mid fusion with batch-shuffled visual features (control) | 763,701 |
| `late` | decision level (`α·out_a + (1−α)·out_v`, α = 0.5) | 673,002 |
| `late_shuffled` | late fusion with batch-shuffled visual features (control) | 673,002 |
| `mid_transformer` | mid fusion with GRU→Transformer variant | — |

## Results (STARSS23 dev test split, best checkpoint by SELD score)

| Model | ER₂₀ ↓ | F₂₀ ↑ | LE ↓ | LR ↑ | SELD ↓ |
|---|---|---|---|---|---|
| Audio-Only | 1.028 | 12.2 % | 60.4° | 33.9 % | 0.726 |
| Early Fusion | 1.055 | 13.1 % | 61.6° | 36.5 % | 0.725 |
| Mid Fusion | 0.989 | **14.1 %** | **56.8°** | 30.4 % | 0.715 |
| Late Fusion | **0.902** | 13.4 % | 59.9° | 28.2 % | **0.704** |

Key finding: late fusion wins the overall SELD score through the lowest error rate
(conservative output averaging), *not* through visual information — destroying
audio–video correspondence leaves performance unchanged. The only measurable benefit
of visual information occurs with feature-level fusion (per-class F-scores for
person-related classes, best localization error), which requires aligned audio–video
input. Model capacity does not explain these differences.

## Repository structure

```
configs/                        # YAML experiment configs + feature/threshold JSON
dcase2022_task3_seld_metrics/   # official DCASE SELD evaluation code
src/
  data/                         # dataset, loader, ADPIT label generation
  features/                     # spectral features (STFT amplitude, phase difference)
  losses/                       # ADPIT loss
  models/                       # one package per fusion strategy
  training/                     # trainer + model dispatch
  eval/                         # validation pipeline, sliding-window inference, metrics
train.py                        # training entry point
validate.py                     # checkpoint validation (writes validation curves)
precompute_visual.py            # YOLOv8n person detection -> Gaussian (2,6,37) features
```

## Data requirements

The STARSS23 development set is expected at `/app/data` (bind-mounted):

- `foa_dev/` — FOA audio (24 kHz)
- `metadata_dev/` — CSV annotations
- `video_dev/` — synchronized 360° video (29.97 fps)
- `visual_features/` — precomputed YOLOv8 features (see below)
- `dev_train_av_78.txt`, `dev_test_78.txt` — file lists (78 train / 78 test clips;
  twelve Sony recordings without video are excluded)

## Setup and training (Docker)

```bash
# one-time: precompute visual features (YOLOv8n person detections -> Gaussian bins)
docker compose run --rm train python precompute_visual.py

# train: select the experiment by editing the config path in train.py
docker compose up train

# tensorboard
docker compose up tensorboard   # http://localhost:6006
```

Training config: Adam (lr 1e-3, wd 1e-6), batch size 16, 20,000 iterations on random
1.27 s crops, checkpoints every 1,000 iterations. Validation evaluates the last ten
checkpoints per experiment; results are written to `results/val/`.

**Note:** Experiment selection is by design explicit — `train.py` and `validate.py`
contain hardcoded config/checkpoint paths at the top of the files. See
`configs/` for all experiments.

## Dependencies

Pinned via `Dockerfile` (PyTorch 2.1.0, CUDA 12.1) and `requirements.txt`
(Ultralytics YOLOv8, librosa, torchaudio, TensorBoard, …).

## References

- Shimada et al., *STARSS23: Sony-TAu Realistic Spatial Soundscapes 2023*, DCASE 2023 Workshop
- Adavanne et al., *DCASE 2022 Task 3: Sound Event Localization and Detection*, DCASE 2022
- Shimada et al., *Sony SELD baseline*, DCASE 2023 Task 3 technical report
