# S3FD face detector

This directory contains the S3FD network and the supported PyTorch face-detection
helpers used by the gaze demo. Each Python file begins with a detailed description.

- `s3fd_network.py`: learned network layers and feature normalization.
- `bounding_boxes.py`: anchor encoding/decoding, overlap, and NMS.
- `face_detection.py`: checkpoint loading, candidate detections, and RGB face/eye inputs.
- `evaluate_wider_faces.py`: multi-scale WIDER ZIP prediction export.
- `eval/evaluate_wider_faces_caffe.py`: archival evaluator requiring Caffe and old paths.

From the repository root:

```bash
python -m ssd_detector.evaluate_wider_faces --help
python predict_gaze_from_faces.py --device cuda --resume /path/to/gaze_checkpoint.pth.tar --face-model /path/to/s3fd_convert.pth --input-dir imgs/test --output-dir model_outputs
```

WIDER text exports can be evaluated with the bundled historical MATLAB/Octave
evaluator under `eval/eval_tools_old-version`; its filenames and APIs are unchanged.
The Python exporter does not itself compute average precision.

The original converted detector weights are linked in the upstream
[SFD PyTorch release](https://github.com/clcarwin/SFD_pytorch/releases/tag/v0.1).
Weights are separate assets and have not been downloaded as part of the source rename.

Original detector: [SFD](https://github.com/sfzhang15/SFD).
