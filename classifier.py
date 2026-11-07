"""
MalariaVision - Stage 9 (Module 5): transfer learning with a pre-trained CNN.

MobileNetV2 was already trained by its authors on ImageNet (1.2 million
everyday photos), so its early layers already know how to detect edges,
blobs and textures. We keep that knowledge and only re-train the last
layers to answer one new question: "Parasitized or Uninfected?"
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models

from pipeline import load_image, preprocess

CLASSES = ["Uninfected", "Parasitized"]  # label 1 = Parasitized (positive class)
MODEL_PATH = Path(__file__).parent / "models" / "malariavision_mobilenetv2.pt"

# ImageNet colour statistics: the pre-trained network expects inputs
# normalised the same way as the photos it originally learned from.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(pretrained=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    # Replace the 1000-class ImageNet output layer with a 2-class layer.
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    return model


def freeze_backbone(model, trainable_blocks=0):
    """Freeze the pre-trained feature layers. trainable_blocks > 0 unfreezes
    only the last few blocks for gentle fine-tuning."""
    for p in model.features.parameters():
        p.requires_grad = False
    if trainable_blocks:
        for block in model.features[-trainable_blocks:]:
            for p in block.parameters():
                p.requires_grad = True


def to_tensor(rgb):
    """128x128x3 uint8 image -> normalised 3x128x128 float tensor."""
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def augment(rgb):
    """Random flips and 90-degree rotations (Module 2: rotation, flipping).
    A cell looks the same upside-down, so this gives the network more
    varied examples for free."""
    if random.random() < 0.5:
        rgb = cv2.flip(rgb, 1)
    if random.random() < 0.5:
        rgb = cv2.flip(rgb, 0)
    k = random.randint(0, 3)
    if k:
        rgb = cv2.rotate(rgb, [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                               cv2.ROTATE_90_COUNTERCLOCKWISE][k - 1])
    return rgb


class CellDataset(Dataset):
    def __init__(self, paths, labels, train=False):
        self.paths, self.labels, self.train = list(paths), list(labels), train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = preprocess(load_image(self.paths[i]))  # Stages 1-5
        if self.train:
            img = augment(img)
        return to_tensor(img), self.labels[i]


def load_trained(path=MODEL_PATH, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model, cnn_input_rgb):
    """Returns (class name, probability of Parasitized)."""
    x = to_tensor(cnn_input_rgb).unsqueeze(0)
    probs = torch.softmax(model(x), dim=1)[0]
    p_parasitized = float(probs[1])
    return CLASSES[int(p_parasitized >= 0.5)], p_parasitized
