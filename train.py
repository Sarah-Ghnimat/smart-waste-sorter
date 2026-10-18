"""Fine-tune MobileNetV3-Small on folders of labeled images, evaluate on a held-out test split.

Usage:  python train.py                      (uses data/trashnet and data/custom)
        python train.py --epochs-head 3 --epochs-ft 6
Folder layout: data/<source>/<class_name>/*.jpg ; same class name in several sources is merged.
"""
import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

import model_utils as mu
import waste_core as wc

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect(data_dirs, max_per_class, seed):
    files = {}
    for d in data_dirs:
        d = Path(d)
        if not d.is_dir():
            print(f"[skip] folder not found: {d}")
            continue
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            imgs = [p for p in sub.rglob("*") if p.suffix.lower() in EXTS]
            if imgs:
                files.setdefault(sub.name, []).extend(imgs)
    if max_per_class:
        rng = random.Random(seed)
        for k, v in files.items():
            v.sort()
            rng.shuffle(v)
            files[k] = v[:max_per_class]
    return files


class ImgDS(Dataset):
    def __init__(self, items, class_to_idx, tf):
        self.items, self.c2i, self.tf = items, class_to_idx, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, cls = self.items[i]
        im = Image.open(path)
        im.draft("RGB", (mu.IMG_SIZE * 2, mu.IMG_SIZE * 2))  # faster JPEG decoding
        im = ImageOps.exif_transpose(im).convert("RGB")
        return self.tf(im), self.c2i[cls]


def run_epoch(model, loader, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    preds, labels = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            p = out.argmax(1)
            correct += (p == y).sum().item()
            tot += len(y)
            loss_sum += loss.item() * len(y)
            preds += p.tolist()
            labels += y.tolist()
    return loss_sum / tot, correct / tot, preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="+", default=["data/trashnet", "data/custom"])
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-ft", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-images", type=int, default=10, help="skip classes with fewer images")
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = use all (set e.g. 300 for a quick run)")
    ap.add_argument("--no-pretrained", action="store_true", help="do not download ImageNet weights (testing only)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    files = collect(args.data_dirs, args.max_per_class, args.seed)
    small = {k: len(v) for k, v in files.items() if len(v) < args.min_images}
    if small:
        print(f"[warning] skipping classes with < {args.min_images} images: {small}")
    files = {k: v for k, v in files.items() if len(v) >= args.min_images}
    if len(files) < 2:
        raise SystemExit("Need at least 2 classes with images. Run prepare_data.bat and/or add images to data/custom/<class>/")
    classes = sorted(files)
    c2i = {c: i for i, c in enumerate(classes)}
    counts = {c: len(files[c]) for c in classes}
    print("Classes and image counts:", counts)

    splits = wc.split_files(files, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits.json").write_text(json.dumps(splits, indent=1, ensure_ascii=False), encoding="utf-8")
    print({k: len(v) for k, v in splits.items()})

    mk = lambda items, tf, shuffle: DataLoader(ImgDS(items, c2i, tf), batch_size=args.batch, shuffle=shuffle, num_workers=0)
    tr = mk(splits["train"], mu.train_transform(), True)
    va = mk(splits["val"], mu.eval_transform(), False)
    te = mk(splits["test"], mu.eval_transform(), False)

    model = mu.build_model(len(classes), pretrained=not args.no_pretrained)
    n = np.array([counts[c] for c in classes], dtype=np.float32)
    weights = torch.tensor(n.sum() / (len(n) * n), dtype=torch.float32)  # balance rare classes
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_acc, best_state = -1.0, None
    phases = [("head", args.epochs_head, 1e-3), ("finetune", args.epochs_ft, 1e-4)]
    for phase, epochs, lr in phases:
        for p in model.features.parameters():
            p.requires_grad = (phase == "finetune")
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tl, ta, _, _ = run_epoch(model, tr, loss_fn, opt)
            vl, vacc, _, _ = run_epoch(model, va, loss_fn)
            print(f"[{phase} {ep}/{epochs}] train_acc={ta:.3f} val_acc={vacc:.3f} val_loss={vl:.3f} ({time.time()-t0:.0f}s)")
            if vacc > best_acc:
                best_acc, best_state = vacc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    _, test_acc, preds, labels = run_epoch(model, te, loss_fn)
    report = classification_report(labels, preds, labels=list(range(len(classes))), target_names=classes, digits=3, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    print(f"\nTEST accuracy (images never used for training or model selection): {test_acc:.3f}\n{report}")
    (out / "classification_report.txt").write_text(f"test_accuracy={test_acc:.4f}\n\n{report}", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"test_accuracy": test_acc, "val_accuracy": best_acc,
        "n_test": len(splits["test"]), "class_counts": counts, "confusion_matrix": cm.tolist(), "classes": classes}, indent=1), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(classes), 1.0 + 0.8 * len(classes)))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=130)

    torch.save({"state_dict": model.state_dict(), "classes": classes, "img_size": mu.IMG_SIZE,
                "test_accuracy": float(test_acc), "n_test": len(splits["test"]), "class_counts": counts,
                "trained_at": time.strftime("%Y-%m-%d %H:%M")}, out / "waste_model.pt")
    print(f"Saved {out/'waste_model.pt'}, classification_report.txt, confusion_matrix.png, metrics.json")


if __name__ == "__main__":
    main()
