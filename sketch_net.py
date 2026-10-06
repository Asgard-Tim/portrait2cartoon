"""Line drawing from the Informative Drawings generator.

Caroline Chan, Frédo Durand, and Phillip Isola, CVPR 2022. At test time the
paper runs one fully convolutional generator: the shorter side is resized to
256 with bicubic interpolation, pixels are scaled to [0, 1], and the sigmoid
output is the line drawing (white paper, dark strokes). Geometry and semantic
losses are used only while training that generator.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "sk_model.pth"
# Official test.py default: match the shorter side to this size.
TEST_SIZE = 256


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class LineGenerator(nn.Module):
    """Generator from Chan et al. Three residual blocks, sigmoid output in [0, 1]."""

    def __init__(self, input_nc=3, output_nc=1, n_residual_blocks=3):
        super().__init__()
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.model1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.model2 = nn.Sequential(*[ResidualBlock(256) for _ in range(n_residual_blocks)])
        self.model3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.model4 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        return self.model4(out)


def default_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_generator(path):
    net = LineGenerator()
    state = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(state)
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net


def _test_size(height, width, short=TEST_SIZE):
    """Shorter side becomes `short`, both sides stay divisible by 4."""
    scale = short / min(height, width)
    out_h = max(4, int(round(height * scale)))
    out_w = max(4, int(round(width * scale)))
    out_h += (4 - out_h % 4) % 4
    out_w += (4 - out_w % 4) % 4
    return out_h, out_w


class PortraitSketchNet(nn.Module):
    def __init__(self, weights=WEIGHTS):
        super().__init__()
        self.generator = _load_generator(weights)

    def forward(self, x):
        return self.generator(x).repeat(1, 3, 1, 1)


def sketch_bgr(net, bgr, device):
    height, width = bgr.shape[:2]
    out_h, out_w = _test_size(height, width)
    small = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    with torch.no_grad():
        line = net.generator(tensor.to(device)).cpu()
    arr = line[0, 0].mul(255).clamp(0, 255).byte().numpy()
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def main():
    src = ROOT / "input" / "demo.png"
    dst = ROOT / "output" / "demo_sketch.png"
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {src}")
    device = default_device()
    net = PortraitSketchNet().to(device).eval()
    os.makedirs(dst.parent, exist_ok=True)
    cv2.imwrite(str(dst), sketch_bgr(net, bgr, device))
    print(f"image saved: {dst}")


if __name__ == "__main__":
    main()
