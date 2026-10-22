"""Predict gaze for S3FD-detected faces in a directory of images."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

import opts
from modeltester import display_image, load_checkpoint
from runtime import seed_everything
from ssd_detector.tester import getFaces, load_detector


def main():
    opt = opts.optionargparser().parse_args()
    seed_everything(opt.seed)
    model, device = load_checkpoint(opt)
    detector = load_detector(opt.face_model, device)
    directory = Path(opt.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(opt.input_dir).iterdir()):
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            continue
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            faces, grids, eyes = getFaces(np.asarray(rgb), net=detector)
            scene = TF.to_tensor(TF.resize(rgb, [227, 227]))
        if not len(faces):
            continue
        scene = TF.normalize(scene, [.485, .456, .406], [.229, .224, .225])
        faces = TF.normalize(faces, [.485, .456, .406], [.229, .224, .225])
        with torch.inference_mode():
            outputs = model.predict(scene.unsqueeze(0).expand(len(faces), -1, -1, -1).to(device), faces.to(device), grids.to(device)).cpu()
        fig, ax = plt.subplots()
        ax.imshow(display_image(scene))
        for heatmap, (ex, ey) in zip(outputs, eyes):
            y, x = divmod(heatmap.argmax().item(), 227)
            ax.plot([ex*227, x], [ey*227, y])
        fig.savefig(directory / f'{path.stem}_gaze.jpg')
        plt.close(fig)


if __name__ == '__main__':
    main()
