"""Download and convert the pretrained assets required by this reproduction.

Official MIT Places205 and BVLC ImageNet Caffe archives supply the convolutional
backbones; the S3FD author's release supplies the optional face detector. Known
SHA256 digests validate the exact files used in the repaired run. Caffe protobuf
parsing extracts only convolutional tensors, which are saved as safe state dicts.
Run after installing requirements-lock.txt. Generated Caffe bindings and all
large assets stay under ignored pretrained/. No Caffe installation is required.
"""

from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

import numpy as np
import py7zr
import torch

ASSETS: dict[str, tuple[str, str]] = {
    "places205_alexnet.tar.gz": (
        "http://places.csail.mit.edu/model/placesCNN_upgraded.tar.gz",
        "e846a9485ff92228aedd91a9154328d70b06c8e3820750c67b292e7ca248c250",
    ),
    "imagenet_caffe_alexnet.caffemodel": (
        "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel",
        "4bff209a9837298157915ef50a4831a59636a6ca1a6b8ebacd990c3a5f3053e0",
    ),
    "imagenet_alexnet.pth": (
        "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
    ),
    "s3fd_convert.7z": (
        "https://github.com/clcarwin/SFD_pytorch/releases/download/v0.1/s3fd_convert.7z",
        "77e5ee5b827a929ac1d5838fda98471d793d20863e9e584ad0e732a4b00fcf99",
    ),
}


def digest(path: Path) -> str:
    """Compute SHA256 without loading the full model archive into memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    """Download verified assets, convert Caffe convolutions, and write provenance metadata."""
    root = Path("pretrained")
    root.mkdir(exist_ok=True)
    manifest: list[dict[str, str | int]] = []
    for name, (url, expected_hash) in ASSETS.items():
        path = root / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open(
                "wb"
            ) as destination:
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
            temporary.replace(path)
        actual_hash = digest(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Unexpected SHA256 for {path}: {actual_hash}")
        manifest.append(
            {"file": str(path), "url": url, "bytes": path.stat().st_size, "sha256": actual_hash}
        )
        print(f"Verified {path}", flush=True)
    places = root / "places205"
    places.mkdir(exist_ok=True)
    with tarfile.open(root / "places205_alexnet.tar.gz") as archive:
        for name in [
            "places205CNN_deploy_upgraded.prototxt",
            "places205CNN_iter_300000_upgraded.caffemodel",
            "places_mean.mat",
        ]:
            source = archive.extractfile(name)
            if source is None:
                raise ValueError(f"Missing archive member {name}")
            with source, (places / name).open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
    proto = root / "caffe.proto"
    proto_url = (
        "https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto"
    )
    if not proto.exists():
        proto.write_bytes(urllib.request.urlopen(proto_url, timeout=30).read())
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{root}",
            f"--python_out={root}",
            str(proto),
        ],
        check=True,
    )
    sys.path.insert(0, str(root.resolve()))
    caffe = importlib.import_module("caffe_pb2")
    for name, source_path in [
        ("places205_features", places / "places205CNN_iter_300000_upgraded.caffemodel"),
        ("imagenet_caffe_features", root / "imagenet_caffe_alexnet.caffemodel"),
    ]:
        network = caffe.NetParameter()
        network.ParseFromString(source_path.read_bytes())
        state: dict[str, torch.Tensor] = {}
        for layer in list(network.layers) + list(network.layer):
            if layer.name not in ["conv1", "conv2", "conv3", "conv4", "conv5"]:
                continue
            for suffix, blob in zip(["weight", "bias"], layer.blobs):
                shape = list(blob.shape.dim) or [
                    blob.num,
                    blob.channels,
                    blob.height,
                    blob.width,
                ]
                array = np.asarray(blob.data, dtype=np.float32).reshape(shape)
                if suffix == "bias":
                    array = array.reshape(-1)
                state[f"{layer.name}.{suffix}"] = torch.from_numpy(array.copy())
        output = root / (name + ".pt")
        torch.save(state, output)
        manifest.append(
            {
                "file": str(output),
                "source": str(source_path),
                "bytes": output.stat().st_size,
                "sha256": digest(output),
            }
        )
    with py7zr.SevenZipFile(root / "s3fd_convert.7z") as archive:
        archive.extract(path=root / "s3fd", targets=["s3fd_convert.pth"])
    detector = root / "s3fd/s3fd_convert.pth"
    manifest.append(
        {
            "file": str(detector),
            "source": "s3fd_convert.7z",
            "bytes": detector.stat().st_size,
            "sha256": digest(detector),
        }
    )
    manifest.append({"file": str(proto), "url": proto_url, "sha256": digest(proto)})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
