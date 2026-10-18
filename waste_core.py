"""Pure-Python logic (no torch needed): config, uncertainty decision, Arabic messages, data split."""
import json
import random
from pathlib import Path

REQUIRED_FIELDS = ("ar_name", "this", "put", "bin")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_config(cfg):
    """Return a list of problems (empty list = OK)."""
    problems = []
    s = cfg.get("settings", {})
    for key in ("min_confidence", "min_margin", "default_advice", "uncertain_message", "no_item_message"):
        if key not in s:
            problems.append(f"settings.{key} missing")
    for name, c in cfg.get("classes", {}).items():
        if c.get("no_item"):
            continue
        for fld in REQUIRED_FIELDS:
            if not c.get(fld):
                problems.append(f"{name}.{fld} missing")
        advice = c.get("advice", s.get("default_advice", ""))
        try:
            advice.format(put=c.get("put", ""), bin=c.get("bin", ""))
        except (KeyError, IndexError, ValueError) as e:
            problems.append(f"{name}.advice has a bad placeholder: {e}")
    return problems


def decide(probs, names, min_conf, min_margin, groups=None):
    """Turn probabilities into a decision. Never forces a category.

    status: 'ok' | 'uncertain'
    If the top-2 classes belong to the same group (e.g. paper/cardboard, same bin),
    their probabilities are added for the confidence check and the margin check is skipped.
    """
    if len(probs) != len(names) or not probs:
        raise ValueError("probs and names must have the same non-zero length")
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    top = order[0]
    p1 = probs[top]
    second = order[1] if len(order) > 1 else None
    p2 = probs[second] if second is not None else 0.0
    groups = groups or {}
    same_group = (second is not None and groups.get(names[top]) is not None
                  and groups.get(names[top]) == groups.get(names[second]))
    if same_group:
        ok = (p1 + p2) >= min_conf
    else:
        ok = p1 >= min_conf and (p1 - p2) >= min_margin
    return {
        "status": "ok" if ok else "uncertain",
        "label": names[top],
        "confidence": p1,
        "top": [(names[i], probs[i]) for i in order[:3]],
    }


def group_map(cfg):
    return {n: c.get("group") for n, c in cfg["classes"].items() if c.get("group")}


def build_result(decision, cfg):
    """Return dict(kind, title, bin, color, spoken) in Arabic for display and speech."""
    s = cfg["settings"]
    if decision["status"] != "ok":
        return {"kind": "uncertain", "title": "غير متأكد", "bin": None, "color": "#9e9e9e",
                "spoken": s["uncertain_message"]}
    label = decision["label"]
    c = cfg["classes"].get(label)
    if c is None:  # model has a class that classes.json does not know
        return {"kind": "uncertain", "title": label, "bin": None, "color": "#9e9e9e",
                "spoken": s["uncertain_message"]}
    if c.get("no_item"):
        return {"kind": "no_item", "title": c["ar_name"], "bin": None, "color": "#9e9e9e",
                "spoken": s["no_item_message"]}
    advice = c.get("advice", s["default_advice"]).format(put=c["put"], bin=c["bin"])
    return {"kind": "ok", "title": c["ar_name"], "bin": c["bin"], "color": c.get("color", "#2e7d32"),
            "spoken": f"يبدو أن {c['this']} {c['ar_name']}. {advice}"}


def split_files(files_by_class, seed=42, ratios=(0.70, 0.15, 0.15)):
    """Stratified, deterministic train/val/test split. Returns dict split -> list[(path, class)]."""
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for cls in sorted(files_by_class):
        items = sorted(str(p) for p in files_by_class[cls])
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, round(n * ratios[1]))
        n_test = max(1, round(n * ratios[2]))
        out["test"] += [(p, cls) for p in items[:n_test]]
        out["val"] += [(p, cls) for p in items[n_test:n_test + n_val]]
        out["train"] += [(p, cls) for p in items[n_test + n_val:]]
    return out
