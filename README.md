# Where are they looking?

This repository contains an implementation of the "Where are they looking?" paper by A. Recasens*, A. Khosla*, C. Vondrick and A. Torralba.

## Introduction

A deep neural network-based approach for gaze-following automated using a SSD face detector.

## Reproduction results

Measured on an RTX 5080 with seed 235. Higher AUC is better; lower distances,
angular error, and NLL are better. Distances use normalized image coordinates.
Metrics within each row come from the same epoch, without mixing independently
selected best values. Snapshot: 2026-10-22.

### Full training set, corrected test2 evaluation — epoch 8

The repaired run uses all **125,557 training examples** and evaluates the **4,782
corrected test2 examples**, with ten gaze annotations per person, after each
epoch.

| Model | Epoch | AUC ↑ | Mean distance ↓ | Minimum distance ↓ | Angular error (°) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Original paper, Table 1(a)](https://people.csail.mit.edu/khosla/papers/nips2015_recasens.pdf) | — | 0.878 | 0.190 | 0.113 | 24 |
| Repaired reproduction (`reproduction_full_train_test2`) | **8** | **0.8695** | **0.1831** | **0.1196** | **24.10** |

Our AUC column uses **`auc_64`**: bilinearly resize the predicted heatmap to
64×64, mark all annotated fixation cells positive, compute tie-correct ROC AUC
per example, then average across examples. Exact equivalence to the original
2015 AUC evaluator is unverified; the paper's AUC is shown for reference.

Recipe: Caffe-compatible AlexNet backbones initialized from **Places205** (scene)
and **BVLC ImageNet** (head); gaze FC widths **100/400/200/169** with final sigmoid;
horizontal flips; Adam; initial LR **0.001** for new layers and **0.0001** for
pretrained layers; batch size **128**. NLL averages the five
shifted-grid classification losses.

## Installation (current PyTorch)

Verified on 2026-10-21 with Python 3.12, PyTorch 2.14.1+cu130,
torchvision 0.29.1+cu130, NVIDIA driver 595.99.02, and an RTX 5080.
The matching releases are [PyTorch 2.14.1](https://github.com/pytorch/pytorch/releases/tag/v2.14.1)
and [torchvision 0.29.1](https://github.com/pytorch/vision/releases/tag/v0.29.1).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

The tested environment is already installed in `.venv` on this machine. For an
exact Linux/Python 3.12 package snapshot, use `requirements-lock.txt` instead.
The default Linux PyTorch wheel includes CUDA 13.0 runtime libraries; an accessible
NVIDIA GPU and compatible host driver are still required. For another GPU/driver
combination, select the appropriate build using the [official installer](https://pytorch.org/get-started/locally/).
Do not replace a working host driver merely because CUDA is hidden by a sandbox.

## Verify the environment

```bash
nvidia-smi
python check_pytorch_runtime.py --device cuda --batch-size 1
python check_pytorch_runtime.py --device cuda --batch-size 2
python -m pytest -q
```

The smoke test uses the real gaze network with random initialization and synthetic
inputs. It checks forward/backward, finite nonzero gradients in both branches,
an optimizer update, heatmap prediction, and a model/optimizer checkpoint reload.
Explicit `--device cuda` fails if CUDA is unavailable. `--device cpu` is available
for diagnostics; `--device auto` selects CUDA when accessible, otherwise CPU.
GPU execution in Codex required running outside its filesystem/device sandbox.

## Train and evaluate

Download GazeFollow's original images and `train_annotations.mat` /
`test_annotations.mat`. Obtain matching Places AlexNet weights separately.
Paths no longer require a trailing slash.

```bash
python train_gaze_model.py --device cuda --data_dir /path/to/gazefollow --placesmodelpath /path/to/places_state_dict.pth --no-withaug
python train_gaze_model.py --device cuda --data_dir /path/to/gazefollow --resume /path/to/gaze_checkpoint.pth.tar
python train_gaze_model.py --device cuda --data_dir /path/to/gazefollow --resume /path/to/gaze_checkpoint.pth.tar --testOnly
```

The default Caffe-compatible backbones use downloaded Places205 and BVLC ImageNet
weights. Prepare them with `python prepare_pretrained_models.py`. The explicit
`--backbone torchvision` alternative requires matching torchvision-format weights.
Resume/inference restores both backbones from the complete gaze checkpoint;
`--random-init` is for diagnostics only. Checkpoints use `weights_only=True` loading.

Training defaults to image-disjoint development validation. To reproduce the
full-training/test2 run above, use `--validation-split test --validation-fraction 0`.
The shared architecture and augmentation choices remain explicit CLI options.

Boolean flags now use `--withaug` / `--no-withaug`, `--nesterov` /
`--no-nesterov`, and `--testOnly` / `--no-testOnly`; do not pass the strings
`True` or `False`. Seeds are set before model initialization, with NumPy and
DataLoader worker seeding included. TensorBoard uses `torch.utils.tensorboard`.

## Inference demos

```bash
python visualize_gaze_predictions.py --device cuda --data_dir /path/to/gazefollow --resume /path/to/gaze_checkpoint.pth.tar --output-dir model_outputs
python predict_gaze_from_faces.py --device cuda --resume /path/to/gaze_checkpoint.pth.tar --face-model /path/to/s3fd_convert.pth --input-dir imgs/test --output-dir model_outputs
python -m ssd_detector.evaluate_wider_faces --help
```

S3FD uses current inference APIs, explicit softmax dimensions, device-aware tensors,
and safe state-dictionary loading. The demos no longer execute inference on import.
The historical Caffe evaluator under `ssd_detector/eval/evaluate_wider_faces_caffe.py` is archival
Caffe code, not part of this PyTorch runtime.

## Reproduction scope and remaining verification

The repaired code fixes annotation mutation, crop geometry, shifted-grid
encoding/decoding consistency, global heatmap peak extraction, and distance
metrics. It supports the paper's gaze FC widths, Caffe-compatible pretrained
backbones, AUC64, angular error, and training/validation TensorBoard diagnostics.
Batch size one is supported; regression tests cover CPU/CUDA execution and geometry.


## Source layout

Every Python source file begins with a detailed module docstring describing its
responsibilities, inputs/outputs, side effects, and known limitations. The names
below replace the earlier abbreviated filenames; use the updated commands above.

| File | Responsibility |
| --- | --- |
| `train_gaze_model.py` | CLI entry point and outer training/evaluation loop. |
| `command_line_options.py` | Shared argument parser and preserved CLI flags. |
| `gazefollow_dataset.py` | Original MATLAB annotations, RGB scenes, head crops, and batch loaders. |
| `gaze_augmentations.py` | Annotation-safe flips and geometrically valid scene crops. |
| `models/gaze_following_network.py` | Scene/head branches, grid heads, and heatmap decoder. |
| `models/caffe_alexnet_backbone.py` | Caffe AlexNet grouped convolutions, LRN, and pretrained input adaptation. |
| `shifted_grids.py` | Shared shifted-grid target encoding and heatmap decoding. |
| `prepare_pretrained_models.py` | Download, verify, and convert pretrained model assets. |
| `models/training_setup.py` | Model/loss/optimizer construction and checkpoint save/resume. |
| `models/__init__.py` | Explicit re-exports of the model setup helpers. |
| `gaze_metrics.py` | Mean/minimum gaze distances, angular error, and explicit fixation AUC. |
| `training_support.py` | Learning-rate schedule and streaming statistics. |
| `device_and_initialization.py` | Device selection, RNG seeds, and Places backbone loading. |
| `type_definitions.py` | Shared array/sample types and narrow interface protocols. |
| `check_pytorch_runtime.py` | Dataset-free CPU/CUDA forward/backward/checkpoint smoke test. |
| `visualize_gaze_predictions.py` | Render predictions for annotated GazeFollow test people. |
| `predict_gaze_from_faces.py` | Render predictions for automatically detected faces. |
| `ssd_detector/s3fd_network.py` | S3FD face-detection architecture. |
| `ssd_detector/face_detection.py` | Detector loading, box decoding, and face/eye input preparation. |
| `ssd_detector/bounding_boxes.py` | Box geometry, encoders/decoders, and NMS. |
| `ssd_detector/evaluate_wider_faces.py` | Current PyTorch WIDER prediction export CLI. |
| `ssd_detector/eval/evaluate_wider_faces_caffe.py` | Archival Caffe evaluator with legacy dependencies. |
| `testing_helpers/visualize_head_crop.py` | Historical manually configured crop plotter. |
| `testing_helpers/visualize_legacy_dataset.py` | Historical plotter requiring the absent experimental loader. |
| `tests/test_pytorch_compatibility.py` | Runtime, checkpoint, distance, detector, and CLI regression tests. |
| `tests/test_types_and_documentation.py` | Type-name resolution and detailed documentation coverage. |
| `clear_python_caches.sh` | Historical cache cleanup commands; run from the repository root. |
| `REPRODUCTION_ISSUES.md` | Original debugging notes and remaining reproduction concerns. |

`README.md`, `LICENSE`, `pytest.ini`, the requirements files, package initializer
names, dataset paths, and checkpoint field/parameter names retain their conventional
or compatibility-sensitive names. The bundled MATLAB evaluator assets are unchanged.

## Contact

Please do get in touch with us by email for any questions, comments, suggestions you have!

* rohit.gajawada@gmail.com
* haard.panchal@students.iiit.ac.in

## References

* sfzhang15's SFD detector is used for face detection (https://github.com/sfzhang15/SFD).
* Link to the NIPS 2015 paper from MIT: http://people.csail.mit.edu/khosla/papers/nips2015_recasens.pdf. Please cite them if you decide to use this project for your research.
