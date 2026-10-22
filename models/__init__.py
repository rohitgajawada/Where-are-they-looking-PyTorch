"""Expose model construction and checkpoint helpers as the package's small public API.

The implementations live in training_setup.py, while the neural architecture
lives in gaze_following_network.py. Re-exporting setup, load_model,
save_checkpoint, and resumer preserves existing models.<helper> call sites
without hiding their implementation inside a package initializer.

Importing this package defines helpers but does not instantiate networks, load
weights, or start training. Learned checkpoint parameter names are independent
of these source filenames.
"""

from .training_setup import load_model, resumer, save_checkpoint, setup

__all__ = ["load_model", "resumer", "save_checkpoint", "setup"]
