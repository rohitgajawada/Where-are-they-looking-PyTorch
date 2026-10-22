# Remaining reproduction work

This file preserves the intent of the original debugging notes and points to the
current filenames. File/module renames and the PyTorch migration do not resolve
these scientific correctness issues.

1. Audit `gazefollow_dataset.py`: confirm normalized x/y conventions, person-box
   versus head-crop semantics, and that every network input receives the intended data.
2. Fix annotation aliasing and crop-coordinate math in `gaze_augmentations.py`
   and its dataset call site. Disable augmentation during initial diagnostics.
3. Audit the shifted-grid labels and `Net.predict` in
   `models/gaze_following_network.py`; make head ordering, offsets, and tensor axes
   agree. Inspect generated heatmaps with a known checkpoint when available.
4. Compare layer dimensions, activation/bias choices, and pretrained preprocessing
   against the paper and original reference implementation.
5. Implement and verify the original-paper AUC protocol in `gaze_metrics.py`, then
   establish tiny-set overfitting and controlled full-dataset training results.

The original notes specifically requested checks of data loading, prediction
heatmaps, and network biases; those concerns remain represented above.
