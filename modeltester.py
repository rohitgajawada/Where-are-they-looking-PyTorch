"""Render annotated GazeFollow predictions from a trained checkpoint."""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import getdata
import opts
from models.gazenet import Net
from runtime import resolve_device, seed_everything


def load_checkpoint(opt):
    if opt.resume == 'none':
        raise ValueError('Supply a trained gaze checkpoint with --resume')
    device = resolve_device(opt.device)
    # A complete trained checkpoint supplies both backbones: no download required.
    opt.random_init = True
    model = Net(opt).to(device)
    checkpoint = torch.load(opt.resume, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint))
    return model.eval(), device


def display_image(tensor):
    mean = tensor.new_tensor([.485, .456, .406])[:, None, None]
    std = tensor.new_tensor([.229, .224, .225])[:, None, None]
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()


def main():
    opt = opts.optionargparser().parse_args()
    seed_everything(opt.seed)
    model, device = load_checkpoint(opt)
    opt.device = str(device)
    data = getdata.GazeFollow(opt)
    scene, head, grid, _, eyes, names, _, _ = next(iter(data.val_loader))
    with torch.inference_mode():
        heatmaps = model.predict(scene.to(device), head.to(device), grid.to(device)).cpu()
    directory = Path(opt.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(min(16, len(scene))):
        index = heatmaps[i].argmax().item()
        y, x = divmod(index, heatmaps.shape[-1])
        fig, ax = plt.subplots()
        ax.imshow(display_image(scene[i]))
        ax.plot([eyes[i, 0].item() * 227, x], [eyes[i, 1].item() * 227, y])
        fig.savefig(directory / f'{int(names[i])}.jpg')
        plt.close(fig)
        plt.imsave(directory / f'{int(names[i])}_heat.jpg', heatmaps[i].numpy())


if __name__ == '__main__':
    main()
