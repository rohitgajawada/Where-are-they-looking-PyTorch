from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io
import torch
from PIL import Image
from torch.utils.data import DataLoader

import getdata
import models
import opts
import utils
from models.gazenet import Net
from runtime import load_places_features, resolve_device, seed_everything
from ssd_detector.net_s3fd import L2Norm
from ssd_detector.tester import detect, getFaces
from train import Trainer, Validator


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(4)
    seed_everything(235)


def test_boolean_flags_and_seed():
    opt = opts.optionargparser().parse_args(['--no-withaug', '--no-nesterov', '--testOnly', '--seed', '7'])
    assert not opt.withaug and not opt.nesterov and opt.testOnly and opt.seed == 7


def test_explicit_cuda_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_device('auto').type == 'cpu'
    with pytest.raises(RuntimeError, match='CUDA requested'):
        resolve_device('cuda')


def test_places_state_dict_and_explicit_legacy_opt_in(tmp_path):
    backbone = load_places_features(SimpleNamespace(random_init=True))
    safe_path = tmp_path / 'places.pt'
    torch.save({'state_dict': {'module.features.' + k: v for k, v in backbone.state_dict().items()}}, safe_path)
    loaded = load_places_features(SimpleNamespace(placesmodelpath=safe_path))
    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])
    legacy_path = tmp_path / 'legacy.pt'
    torch.save(backbone, legacy_path)
    with pytest.raises(RuntimeError, match='trust-legacy-checkpoint'):
        load_places_features(SimpleNamespace(placesmodelpath=legacy_path))
    loaded = load_places_features(SimpleNamespace(placesmodelpath=legacy_path, trust_legacy_checkpoint=True))
    torch.testing.assert_close(backbone[0].weight, loaded[0].weight)


def test_metrics_mean_location_and_batch_invariance():
    # Prediction at (0,0); annotations at (0,0), (1,0): distance to mean=.5, min=0.
    target = torch.full((2, 100), -1.)
    target[0, :4] = torch.tensor([0., 0., 1., 0.])
    target[1, :2] = torch.tensor([.5, .5])
    pred = torch.tensor([0, 50*227+100])
    expected_second = ((100/227-.5)**2 + (50/227-.5)**2)**.5
    assert utils.euclid_dist(pred, target, 2) == pytest.approx((.5 + expected_second)/2)
    assert utils.euclid_mindist(pred[:1], target[:1], 1) == 0
    separate = sum(utils.euclid_dist(pred[i:i+1], target[i:i+1], 1) for i in range(2))/2
    assert utils.euclid_dist(pred, target, 2) == pytest.approx(separate)


class Writer:
    def add_scalar(self, *args):
        pass


def synthetic_mat_dataset(tmp_path):
    Image.fromarray(np.full((300, 300, 3), 127, dtype=np.uint8)).save(tmp_path / 'image.jpg')
    arrays = {}
    for split in ('train', 'test'):
        for field, value in [('bbox', [0., 0., 1., 1.]), ('eyes', [.5, .5]), ('gaze', [.7, .3]), ('meta', [1.])]:
            cells = np.empty((1, 1), dtype=object)
            cells[0, 0] = np.array([value])
            arrays[f'{split}_{field}'] = cells
        cells = np.empty((1, 1), dtype=object)
        cells[0, 0] = 'image.jpg'
        arrays[f'{split}_path'] = cells
        scipy.io.savemat(tmp_path / f'{split}_annotations.mat', {k: v for k, v in arrays.items() if k.startswith(split)})


@pytest.mark.parametrize('device_name', ['cpu', 'cuda'])
def test_real_data_trainer_validator_checkpoint_batch_one(tmp_path, device_name):
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not exposed in this execution environment')
    synthetic_mat_dataset(tmp_path)
    opt = opts.optionargparser().parse_args(['--device', device_name, '--random-init', '--no-withaug', '--workers', '0', '--batch-size', '1', '--testbatchsize', '1', '--data_dir', str(tmp_path), '--store', str(tmp_path / 'checkpoints')])
    data = getdata.GazeFollow(opt)
    model, criterion, optimizer = models.setup(models.load_model(opt), opt)
    trainer = Trainer(model, criterion, optimizer, opt, Writer())
    trainer.train(data.train_loader, 0, opt)
    assert np.isfinite(trainer.losses.avg)
    error = Validator(model, criterion, opt, Writer()).validate(data.val_loader, 0, opt)
    assert np.isfinite(error) and 0 <= error <= 2**.5
    models.save_checkpoint(opt, model, optimizer, error, 0)
    opt.resume = str(next((tmp_path / 'checkpoints').glob('*.pth.tar')))
    model, optimizer, opt, best = models.resumer(opt, model, optimizer)
    assert opt.start_epoch == 1 and best == error
    assert next(model.parameters()).device.type == device_name


def test_s3fd_empty_detections_and_norm():
    class EmptyDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()))

        def forward(self, image):
            logits = image.new_tensor([100., -100.]).reshape(1, 2, 1, 1)
            return [logits, image.new_zeros((1, 4, 1, 1))]
    net = EmptyDetector()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert detect(net, image).shape == (0, 5)
    faces, grids, eyes = getFaces(image, net=net)
    assert faces.shape == (0, 3, 227, 227) and grids.shape == (0, 13, 13) and eyes == []
    layer = L2Norm(3, 10)
    torch.testing.assert_close(layer.weight, torch.full((3,), 10.))
    assert torch.isfinite(layer(torch.zeros(1, 3, 2, 2))).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not exposed')
def test_cuda_smoke_batch_two():
    from smoke_test import run
    assert run('cuda', 2)['device'] == 'cuda'


@pytest.mark.parametrize('entry', [
    ['main.py'], ['modeltester.py'], ['modeltester_withssd.py'],
    ['-m', 'ssd_detector.wider_eval_pytorch'],
])
def test_cli_help_has_no_asset_or_cuda_requirement(entry):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, *entry, '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--device' in result.stdout
