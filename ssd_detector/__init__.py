"""Package the S3FD network, face-input preparation, and detector evaluation tools.

s3fd_network.py defines the neural architecture; bounding_boxes.py contains box
geometry and NMS; face_detection.py loads weights and prepares face/eye inputs
for the gaze model. evaluate_wider_faces.py is the supported PyTorch WIDER CLI.
The eval directory also retains historical Caffe/MATLAB evaluation assets.

The package initializer intentionally does not load models, initialize CUDA,
read datasets, or re-export an ambiguous collection of helper functions.
"""

from __future__ import annotations
