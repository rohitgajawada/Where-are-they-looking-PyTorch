"""Exercise the gaze model, gradients, prediction, and checkpoint round trip."""
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.gazenet import Net
from runtime import resolve_device, seed_everything


def run(device_name='cpu', batch_size=1):
    device = resolve_device(device_name)
    seed_everything(235)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = Net(SimpleNamespace(random_init=True)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scene = torch.randn(batch_size, 3, 227, 227, device=device)
    head = torch.randn_like(scene)
    eyes = torch.zeros(batch_size, 13, 13, device=device)
    eyes[:, 6, 6] = 1
    target = torch.arange(batch_size, device=device) % 25
    before = model.fc_0_0.weight.detach().clone()
    outputs = model(scene, head, eyes)
    assert len(outputs) == 5
    assert all(out.shape == (batch_size, 25) for out in outputs)
    loss = torch.stack([F.nll_loss(out, target) for out in outputs]).mean()
    assert torch.isfinite(loss)
    loss.backward()
    for branch in (model.salpath, model.gazepath):
        gradients = [p.grad for p in branch.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
    optimizer.step()
    assert not torch.equal(before, model.fc_0_0.weight)
    model.eval()
    with torch.inference_mode():
        heatmap = model.predict(scene, head, eyes)
        assert heatmap.shape == (batch_size, 227, 227)
        assert torch.isfinite(heatmap).all()
        expected = model(scene, head, eyes)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'checkpoint.pt'
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        with torch.inference_mode():
            actual = model(scene, head, eyes)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    result = {'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'device': str(device), 'batch_size': batch_size, 'loss': loss.item(),
              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'checks': ['forward', 'backward', 'both_branch_gradients', 'optimizer_step', 'predict', 'checkpoint_round_trip']}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    run(args.device, args.batch_size)
