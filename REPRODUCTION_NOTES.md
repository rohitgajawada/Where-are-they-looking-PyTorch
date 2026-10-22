# Gaze-following reproduction repair status

Updated 2026-10-22T06:04:58.968365+00:00. Correctness repairs are implemented and the longer GPU development run is active. Full numerical reproduction of the 2015 paper is not yet established.

## Running job and monitoring

- TensorBoard: http://localhost:6006
- Active run: `reproduction_full`, PID 65488; RTX 5080, 30 epochs, batch size 128.
- Log: `/home/rohit/sources/Where-are-they-looking-PyTorch/runs/training.log`.
- Watch: `tail -f /home/rohit/sources/Where-are-they-looking-PyTorch/runs/training.log`.
- Checkpoints: `/home/rohit/sources/Where-are-they-looking-PyTorch/savedmodels/reproduction_full/`; per epoch, `last.pth.tar`, and `best.pth.tar`.
- Exact launch arguments: `/home/rohit/sources/Where-are-they-looking-PyTorch/runs/active_training.json`.
- Train/development example counts: 119,278 / 6,279, grouped by image path using a seeded SHA256 partition. Official test data is excluded from recipe selection and epoch validation.

## Correctness changes

### Immutable annotations and valid augmentation

MATLAB cells are copied, annotation arrays are read-only, and transforms copy coordinates. Grayscale/RGBA images become RGB before cropping. Eye-centered crops use zero-based half-open bounds and actual padding. This retains the historical 30%-of-person-box head-crop heuristic; exact 2015 crop parity is unverified. Crop normalization uses the resized 240-pixel dimensions, supports multiple fixation pairs, and leaves head-local eyes unchanged. Only windows retaining all required points are sampled; otherwise the original scene is kept. Vertical flipping is opt-in. Coordinates at 1 map to the last eye-grid cell.

### Shared shifted grids

`shifted_grids.py` owns head order and offsets. Encoding uses `floor(5 * coordinate + offset / 3)`, clipped to 0..4. Decoding applies that same encoding to all 15x15 pixel centers and gathers probabilities in matching head order before resizing to 227. Tests cover all 225 centers, asymmetric coordinates and edges. Exact equivalence to the original author's undocumented shifts remains unverified.

### Architecture and pretrained compatibility

The default gaze FC layout is 100/400/200/169 with a final sigmoid. `--architecture legacy` retains the original 500-unit embedding and extra convolution. Caffe-compatible AlexNet convolutions now support the actual Places205 and BVLC ImageNet tensors: 96/256/384/384/256 channels, groups and LRN. The torchvision alternative remains explicit. Inputs are converted back from normalized RGB to 0..255 BGR with channel-mean subtraction. Spatial mean-image subtraction, historical eye-grid gain 24, saliency projection activation, and exact head-crop conventions have not been verified against the unavailable original 2015 implementation. The existing saliency ReLU is retained and its inactivity is logged.

### Metrics, TensorBoard and checkpoints

Both loops log five-head NLL, grid accuracy, per-head loss/accuracy, distance to mean fixation, and minimum fixation distance. Training also streams batch metrics, branch activation diagnostics, gradient norms and learning rates. Validation additionally logs angular error and `auc_64`. The latter is a documented 64x64 multi-hot fixation ROC AUC with tie correction, not a claim of exact 2015 AUC protocol parity. Configurations and split indices are saved with each run. Checkpoints include optimizer, configuration and RNG states. Learning-rate decay supports floating-point factors and preserves the pretrained/new-layer LR ratio. TensorBoard events were read back in an integration test.

## Optimization gates and controlled comparisons

No augmentation during memorization. Both pretrained branches were optimized.

- One real example: 6 updates, NLL 0.004768, grid accuracy 100%.
- 32 real examples: 14 updates, NLL 0.017693, grid accuracy 100%.

The short comparisons use the first 2,048 training-partition rows and first 512 development-partition rows, excluding official test examples. Each candidate starts from pretrained weights, seed 235, Adam, batch size 32 and exactly 64 updates. Pretrained-layer LR is 0.1 times new-layer LR. These small ordered subsets and a single seed limit the strength of recipe selection.

| Candidate | New-layer LR | Development distance | Development NLL |
|---|---:|---:|---:|
| calibration_paper_lr_0003 | 0.0003 | 0.298760 | 3.004062 |
| calibration_paper_lr_001 | 0.001 | 0.289147 | 2.969242 |
| calibration_legacy | 0.001 | 0.297495 | 2.965970 |
| calibration_horizontal_flip | 0.001 | 0.285017 | 2.955651 |
| calibration_flip_crop | 0.001 | 0.298644 | 2.983402 |

The initial longer-run recipe uses the 100-unit layout, LR 0.001 and horizontal flips only. Corrected random cropping remains available; this short comparison did not justify enabling it. The longer run uses all development-training examples and a larger batch size, so this is a starting recipe rather than an exact continuation of the calibration.

Label-only baselines on all 6,279 development examples: center distance 0.276251, AUC64 0.707870; unconditional training-fixation prior distance 0.276186, AUC64 0.719494. The latter differs from the paper's head-conditioned fixed-bias baseline. These full-development figures should not be directly compared to the 512-example candidate table as if they were evaluated on identical samples.

## Assets and verification

`prepare_pretrained_models.py` downloads official assets, checks known SHA256 digests, converts Caffe convolutional weights into tensor-only state dicts, and extracts S3FD. Provenance and derived hashes are in `pretrained/manifest.json`. The 2015 trained gaze checkpoint itself could not be retrieved: the original gaze-following model endpoint timed out. Required initialization and detector weights are downloaded.

```bash
cd /home/rohit/sources/Where-are-they-looking-PyTorch
.venv/bin/python prepare_pretrained_models.py
.venv/bin/python -m pytest -q
.venv/bin/python run_reproduction_diagnostics.py --data_dir data/gazefollow
.venv/bin/python compare_development_recipes.py --data_dir data/gazefollow
```

Final checks: 21 tests passed, including CPU/CUDA, checkpoint restoration, immutable augmentation, shared grids, metrics, image-disjoint splitting, real TensorBoard events and documentation/type coverage. `requirements-lock.txt` records the working Python 3.12 / PyTorch 2.14.1+cu130 environment. Training was verified alive on the host GPU; the sandbox itself does not expose CUDA.

## What remains to establish paper reproduction

Allow the development run to finish and compare against baselines on identical examples. Only after choosing the recipe should a final run use the prescribed full training partition and evaluate official test data. Multi-seed results and full numerical parity are not available yet. The original 2015 code/model endpoint remains unavailable, leaving the preprocessing/crop/activation and AUC details above unresolved. No historical failure cause has been isolated quantitatively; the annotation corruption and grid/geometry inconsistencies are demonstrated correctness bugs, while their individual historical effects remain unmeasured.

Primary references: [2015 paper](https://people.csail.mit.edu/khosla/papers/nips2015_recasens.pdf), [official Places205 downloads](http://places.csail.mit.edu/downloadCNN.html), [BVLC AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet), [S3FD release](https://github.com/clcarwin/SFD_pytorch/releases/tag/v0.1).
