"""Device selection, reproducibility, and explicit legacy-weight compatibility."""
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import alexnet


def resolve_device(requested='auto'):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(requested)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable. Check nvidia-smi and GPU device access; use --device cpu for CPU diagnostics.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {device.index} does not exist')
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def load_places_features(opt):
    # Preserve the original feature geometry and parameter names.
    features = alexnet(weights=None).features[:-2]
    if getattr(opt, 'random_init', False):
        return features
    path = Path(opt.placesmodelpath)
    if not path.is_file():
        raise FileNotFoundError(f'Places weights not found: {path}. Supply --placesmodelpath or use --random-init for diagnostics.')
    trusted = getattr(opt, 'trust_legacy_checkpoint', False)
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=not trusted)
    except Exception as exc:
        raise RuntimeError(f'Cannot load Places weights {path}. Prefer a tensor state_dict. For a trusted old pickled model only, pass --trust-legacy-checkpoint.') from exc
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()
    if isinstance(checkpoint, Mapping):
        checkpoint = checkpoint.get('state_dict', checkpoint)
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Places checkpoint must contain an AlexNet state_dict or a trusted AlexNet module')
    expected = features.state_dict()
    state = {}
    for key, value in checkpoint.items():
        key = key.removeprefix('module.').removeprefix('features.')
        if key in expected:
            state[key] = value
    features.load_state_dict(state, strict=True)
    return features
