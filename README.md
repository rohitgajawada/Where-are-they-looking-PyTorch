# Where are they looking?

This repository contains an implementation of the "Where are they looking?" paper by A. Recasens*, A. Khosla*, C. Vondrick and A. Torralba.

## Introduction

A deep neural network-based approach for gaze-following automated using a SSD face detector.

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

ImageNet initialization uses torchvision's explicit `AlexNet_Weights.IMAGENET1K_V1`
API. New training downloads those weights on first use. Resume/inference loads
complete checkpoint weights without requiring Places files or a download.
`--random-init` bypasses both pretrained backbones and is intended for diagnostics,
not a paper reproduction.

Places loading accepts a tensor state dictionary, a `state_dict` wrapper, or
`module.`-prefixed keys. Backbone parameters are checked strictly. If the historical
Places file contains a pickled model object, load it only if trusted, by explicitly
adding `--trust-legacy-checkpoint`; that flag enables Python pickle execution for
that file. Prefer converting such files to tensor state dictionaries. Gaze
checkpoints use explicit `weights_only=True` loading and `map_location`.

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

## Migration scope and remaining reproduction work

The original network layer sizes, parameter names, and shifted-grid behavior are
preserved for existing gaze checkpoints. Validation now takes a global heatmap
maximum and correctly computes distance to the mean annotation and minimum distance;
old distance logs are therefore not directly comparable. Batch size one is supported.

This is a runtime migration, not a completed scientific reproduction. The known
annotation-mutation, crop-geometry, shifted-grid consistency, and paper-architecture
differences remain to be fixed and measured in a separate change. Use `--no-withaug`
for initial diagnostics. AUC remains unimplemented and raises an explicit error if
called. GazeFollow is now downloaded under `data/gazefollow`: 125,557 training examples
and 4,782 corrected test2 examples. All annotated image paths exist, and real
train/test batches load successfully with augmentation disabled. See
`data/gazefollow/DATASET_SETUP.md` and `DATASET_MANIFEST.json` for provenance.
No real-data training or pretrained-checkpoint accuracy has been verified yet;
pretrained weights are still required. Migration regression tests use synthetic
annotations and locally generated model weights.

## Source layout

Every Python source file begins with a detailed module docstring describing its
responsibilities, inputs/outputs, side effects, and known limitations. The names
below replace the earlier abbreviated filenames; use the updated commands above.

| File | Responsibility |
| --- | --- |
| `train_gaze_model.py` | CLI entry point and outer training/evaluation loop. |
| `command_line_options.py` | Shared argument parser and preserved CLI flags. |
| `gazefollow_dataset.py` | Original MATLAB annotations, RGB scenes, head crops, and batch loaders. |
| `gaze_augmentations.py` | Historical flips/crops and their documented pending defects. |
| `models/gaze_following_network.py` | Scene/head branches, grid heads, and heatmap decoder. |
| `models/training_setup.py` | Model/loss/optimizer construction and checkpoint save/resume. |
| `models/__init__.py` | Explicit re-exports of the model setup helpers. |
| `gaze_metrics.py` | Mean-location/minimum gaze distances and the unimplemented AUC placeholder. |
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
