"""Evaluate S3FD on a WIDER validation ZIP with current PyTorch.

Run as: python -m ssd_detector.wider_eval_pytorch --help
"""
import argparse
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import scipy.io

from runtime import resolve_device
from .bbox import nms
from .tester import detect, load_detector


def flip_detect(net, img):
    boxes = detect(net, img[:, ::-1])
    result = boxes.copy()
    result[:, 0] = img.shape[1] - boxes[:, 2]
    result[:, 2] = img.shape[1] - boxes[:, 0]
    return result


def scale_detect(net, img, scale=2., facesize=100):
    height, width = img.shape[:2]
    scaled = np.array(Image.fromarray(img).resize((max(1, round(width*scale)), max(1, round(height*scale))), Image.Resampling.BILINEAR))
    boxes = detect(net, scaled)
    boxes[:, :4] /= scale
    sizes = boxes[:, 2:4] - boxes[:, :2] + 1
    keep = sizes.min(1) < facesize if scale > 1 else sizes.max(1) > facesize
    return boxes[keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', default='ssd_detector/eval/wider_face_val.mat')
    parser.add_argument('--dataset', required=True, help='WIDER_val.zip')
    parser.add_argument('--model', required=True, help='S3FD tensor state_dict')
    parser.add_argument('--output', default='ssd_detector/eval/sfd_val_pytorch')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    annotations = scipy.io.loadmat(args.annotations)
    net = load_detector(args.model, resolve_device(args.device))
    with zipfile.ZipFile(args.dataset) as archive:
        for event, filelist in zip(annotations['event_list'], annotations['file_list']):
            event_name = str(event[0][0])
            directory = Path(args.output) / event_name
            directory.mkdir(parents=True, exist_ok=True)
            for file in filelist[0]:
                name = str(file[0][0])
                with Image.open(BytesIO(archive.read(f'WIDER_val/images/{event_name}/{name}.jpg'))) as image:
                    img = np.array(image.convert('RGB'))[..., ::-1].copy()
                boxes = [detect(net, img), flip_detect(net, img), scale_detect(net, img, .5)]
                if img.shape[0]*img.shape[1]*4 <= 3000*3000:
                    boxes.append(scale_detect(net, img, 2.))
                boxes = np.concatenate(boxes)
                boxes = boxes[nms(boxes, .3)[:750]]
                with (directory / f'{name}.txt').open('w') as output:
                    output.write(f'{event_name}/{name}.jpg\n{len(boxes)}\n')
                    for x1, y1, x2, y2, score in boxes:
                        output.write(f'{x1:.1f} {y1:.1f} {x2-x1+1:.1f} {y2-y1+1:.1f} {score:.3f}\n')
                print(f'{event_name}/{name}: {len(boxes)} detections')


if __name__ == '__main__':
    main()
