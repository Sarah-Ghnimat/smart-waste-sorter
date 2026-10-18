"""Torch-dependent helpers shared by app.py and train.py (MobileNetV3-Small, CPU friendly)."""
import torch
import torch.nn as nn
from PIL import ImageOps
from torchvision import models, transforms

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(num_classes, pretrained=True):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v3_small(weights=weights)  # pretrained weights download on first use
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


def eval_transform():
    return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["classes"], {k: v for k, v in ck.items() if k != "state_dict"}


def predict(model, pil_image):
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    x = eval_transform()(img).unsqueeze(0)
    with torch.inference_mode():
        return torch.softmax(model(x), dim=1)[0].tolist()
