"""S3FD inference with current PyTorch; no CLI parsing or model loading at import."""
from pathlib import Path

import numpy as np
import torch
from skimage import transform

from .net_s3fd import s3fd
from .bbox import decode, nms


def load_detector(model_path, device):
    net = s3fd().to(device)
    state = torch.load(Path(model_path), map_location=device, weights_only=True)
    net.load_state_dict(state.get('state_dict', state))
    return net.eval()


@torch.inference_mode()
def detect(net, img):
    """Detect boxes in a BGR image with 0..255 values."""
    device = next(net.parameters()).device
    image = np.asarray(img, dtype=np.float32) - np.array([104, 117, 123], dtype=np.float32)
    image = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    outputs = net(image)
    boxes = []
    for i in range(len(outputs) // 2):
        classes = outputs[2*i].softmax(dim=1).cpu()
        offsets = outputs[2*i+1].cpu()
        _, _, height, width = classes.shape
        stride = 2**(i+2)
        for row, col in torch.nonzero(classes[0, 1] >= .05).tolist():
            score = classes[0, 1, row, col].item()
            anchor = torch.tensor([[stride/2 + col*stride, stride/2 + row*stride, stride*4, stride*4]])
            box = decode(offsets[0, :, row, col].reshape(1, 4), anchor, [.1, .2])[0]
            boxes.append([*box.tolist(), score])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 5)


def getFaces(img, net=None, model_path='s3fd_convert.pth', device='cpu'):
    """Return normalized-range RGB face tensors, eye grids, and eye locations."""
    if net is None:
        net = load_detector(model_path, device)
    boxes = detect(net, np.asarray(img)[..., ::-1])
    if len(boxes):
        boxes = boxes[nms(boxes, .3)]
    height, width = img.shape[:2]
    faces, grids, eyes = [], [], []
    for x1, y1, x2, y2, score in boxes:
        if score < .5:
            continue
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(width, int(x2)), min(height, int(y2))
        if right <= left or bottom <= top:
            continue
        face = transform.resize(img[top:bottom, left:right], (227, 227))
        faces.append(torch.from_numpy(face.transpose(2, 0, 1).copy()).float())
        ex, ey = np.clip([(x1+x2)/(2*width), (y1+y2)/(2*height)], 0, np.nextafter(1., 0))
        grid = torch.zeros(13, 13)
        grid[int(ey*13), int(ex*13)] = 1
        grids.append(grid)
        eyes.append((ex, ey))
    if not faces:
        return torch.empty(0, 3, 227, 227), torch.empty(0, 13, 13), []
    return torch.stack(faces), torch.stack(grids), eyes
