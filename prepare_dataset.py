"""Download the public TrashNet dataset from Hugging Face into data/trashnet/<class>/*.jpg.

Needs internet + `pip install -r requirements-data.txt`. Free, no API key.
If this fails, download any 'garbage classification' dataset manually and arrange it as
data/<anything>/<class_name>/*.jpg (class names such as paper, cardboard, plastic, metal, glass, trash).
Check the dataset's license/terms on its page before redistributing it.
"""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="garythung/trashnet", help="Hugging Face dataset id")
    ap.add_argument("--out", default="data/trashnet")
    args = ap.parse_args()
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing package. Run: pip install -r requirements-data.txt")
    try:
        ds = load_dataset(args.hf_id, split="train")
    except Exception as e:  # network, wrong id, etc.
        sys.exit(f"Could not download '{args.hf_id}': {e}\nSee README 'Dataset' section for the manual alternative.")
    img_col = next((k for k, v in ds.features.items() if type(v).__name__ == "Image"), None)
    lab_col = next((k for k, v in ds.features.items() if type(v).__name__ == "ClassLabel"), None)
    if img_col is None or lab_col is None:
        sys.exit(f"Unexpected dataset columns: {list(ds.features)}. Arrange images manually (see README).")
    names = [n.lower().replace(" ", "_") for n in ds.features[lab_col].names]
    out = Path(args.out)
    counts = {}
    for i, row in enumerate(ds):
        cls = names[row[lab_col]]
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        img = row[img_col].convert("RGB")
        img.thumbnail((512, 512))
        img.save(d / f"{i:05d}.jpg", quality=90)
        counts[cls] = counts.get(cls, 0) + 1
    print("Saved images per class:", counts)
    print(f"Next: run train_model.bat (or python train.py) - folder: {out}")


if __name__ == "__main__":
    main()
