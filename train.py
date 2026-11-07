"""
MalariaVision - train the CNN (Stage 9) and evaluate it (Stage 10).

    python train.py                  # prototype run on a balanced subset
    python train.py --per-class 0    # use all 27,558 images

Data split: 70% train / 15% validation / 15% test (test images are never
seen during training - they are only used for the final scores).

Training happens in two short phases (transfer learning):
  Phase A - pre-trained layers frozen, only the new output layer learns.
  Phase B - the last few pre-trained blocks are unfrozen and fine-tuned
            with a small learning rate.
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from classifier import CLASSES, MODEL_PATH, CellDataset, build_model, freeze_backbone
from pipeline import load_image, preprocess

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
SAMPLES = ROOT / "samples"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=ROOT / "data" / "cell_images", type=Path)
    p.add_argument("--per-class", default=3000, type=int, help="images per class, 0 = all")
    p.add_argument("--epochs-head", default=2, type=int, help="Phase A epochs")
    p.add_argument("--epochs-finetune", default=5, type=int, help="Phase B epochs")
    p.add_argument("--finetune-blocks", default=8, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def list_dataset(data_dir, per_class, seed):
    paths, labels = [], []
    for label, name in enumerate(CLASSES):
        files = sorted((data_dir / name).glob("*.png"))
        if not files:
            raise SystemExit(f"No images found in {data_dir / name}")
        if per_class:
            random.Random(seed).shuffle(files)
            files = files[:per_class]
        paths += files
        labels += [label] * len(files)
    return paths, labels


def set_train_mode(model):
    """Train mode, but frozen BatchNorm layers keep their pre-trained statistics."""
    model.train()
    for m in model.features.modules():
        if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
            m.eval()


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    set_train_mode(model) if training else model.eval()
    total_loss, labels, probs = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            labels += y.tolist()
            probs += torch.softmax(logits, 1)[:, 1].tolist()
    preds = (np.array(probs) >= 0.5).astype(int)
    return total_loss / len(labels), accuracy_score(labels, preds), np.array(labels), np.array(probs)


def train_phase(name, model, epochs, lr, loaders, device, criterion, history, best):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    n_trainable = sum(p.numel() for p in params)
    print(f"\n{name}: {epochs} epoch(s), lr={lr}, trainable parameters={n_trainable:,}")
    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, loaders["train"], device, criterion, optimizer)
        va_loss, va_acc, _, _ = run_epoch(model, loaders["val"], device, criterion)
        history.append(dict(phase=name, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc))
        print(f"  epoch {epoch + 1}/{epochs}  train loss {tr_loss:.3f} acc {tr_acc:.3f}"
              f" | val loss {va_loss:.3f} acc {va_acc:.3f}  ({time.time() - t0:.0f}s)")
        if va_acc > best["val_acc"]:
            best.update(val_acc=va_acc,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})


def plot_history(history, path):
    ep = range(1, len(history) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, lab in (("train", "Train"), ("val", "Validation")):
        ax[0].plot(ep, [h[f"{key}_loss"] for h in history], "o-", label=lab)
        ax[1].plot(ep, [h[f"{key}_acc"] for h in history], "o-", label=lab)
    n_head = sum(h["phase"].startswith("Phase A") for h in history)
    for a, title in zip(ax, ("Loss (lower is better)", "Accuracy (higher is better)")):
        a.axvline(n_head + 0.5, color="gray", ls="--", lw=1)
        a.set_xticks(list(ep))
        a.set_title(title)
        a.set_xlabel("Epoch  (dashed line: start of fine-tuning)")
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False,
                                                            values_format="d")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_predictions(paths, labels, probs, path, n=12, seed=0):
    rng = np.random.default_rng(seed)
    preds = (probs >= 0.5).astype(int)
    wrong = np.where(preds != labels)[0]
    right = np.where(preds == labels)[0]
    pick = list(rng.choice(wrong, min(2, len(wrong)), replace=False)) if len(wrong) else []
    pick += list(rng.choice(right, n - len(pick), replace=False))
    fig, axes = plt.subplots(2, n // 2, figsize=(2.0 * n // 2, 4.8))
    for ax, i in zip(axes.flat, pick):
        ax.imshow(preprocess(load_image(paths[i])))
        ok = preds[i] == labels[i]
        conf = probs[i] if preds[i] else 1 - probs[i]
        ax.set_title(f"Pred: {CLASSES[preds[i]]}\n{conf:.0%}  (true: {CLASSES[labels[i]]})",
                     fontsize=8, color="green" if ok else "red")
        ax.axis("off")
    fig.suptitle("Sample test predictions (green = correct, red = wrong)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def copy_demo_samples(paths, labels, k=6, seed=1):
    """Copy a few TEST images (never seen in training) for the demo app."""
    SAMPLES.mkdir(exist_ok=True)
    rng = random.Random(seed)
    for label, name in enumerate(CLASSES):
        idx = [i for i, y in enumerate(labels) if y == label]
        for j, i in enumerate(rng.sample(idx, k), 1):
            shutil.copy(paths[i], SAMPLES / f"{name.lower()}_{j}.png")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    paths, labels = list_dataset(args.data, args.per_class, args.seed)
    tr_p, rest_p, tr_y, rest_y = train_test_split(paths, labels, test_size=0.30,
                                                  stratify=labels, random_state=args.seed)
    va_p, te_p, va_y, te_y = train_test_split(rest_p, rest_y, test_size=0.50,
                                              stratify=rest_y, random_state=args.seed)
    print(f"Device: {device} | images: {len(paths)} -> train {len(tr_p)}, "
          f"val {len(va_p)}, test {len(te_p)}")

    loaders = {
        "train": DataLoader(CellDataset(tr_p, tr_y, train=True), args.batch_size, shuffle=True),
        "val": DataLoader(CellDataset(va_p, va_y), args.batch_size),
        "test": DataLoader(CellDataset(te_p, te_y), args.batch_size),
    }
    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    history, best = [], {"val_acc": -1.0, "state": None}
    t_start = time.time()

    freeze_backbone(model)
    train_phase("Phase A (frozen backbone)", model, args.epochs_head, 1e-3,
                loaders, device, criterion, history, best)
    freeze_backbone(model, trainable_blocks=args.finetune_blocks)
    train_phase(f"Phase B (fine-tune last {args.finetune_blocks} blocks)", model,
                args.epochs_finetune, 1e-4, loaders, device, criterion, history, best)
    train_minutes = (time.time() - t_start) / 60

    # ---------------- Stage 10: evaluation on the unseen test set ----------------
    model.load_state_dict(best["state"])
    _, _, y_true, probs = run_epoch(model, loaders["test"], device, criterion)
    y_pred = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist(),
                             "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "dataset": {"train": len(tr_p), "val": len(va_p), "test": len(te_p),
                    "per_class": args.per_class or "all"},
        "best_val_acc": best["val_acc"],
        "train_minutes": round(train_minutes, 1),
        "device": device,
        "settings": {k: str(v) for k, v in vars(args).items()},
        "history": history,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save({"state_dict": best["state"], "classes": CLASSES, "val_acc": best["val_acc"]},
               MODEL_PATH)

    plot_history(history, OUT / "training_curves.png")
    plot_confusion(cm, OUT / "confusion_matrix.png")
    plot_predictions(te_p, np.array(te_y), probs, OUT / "sample_predictions.png")
    copy_demo_samples(te_p, te_y)

    print(f"\nTest set ({len(te_p)} images, never seen in training):")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"  {k:<9} {metrics[k]:.4f}")
    print(f"  confusion matrix [[TN FP] [FN TP]] = {cm.tolist()}")
    print(f"  missed infections (false negatives): {fn}")
    print(f"Training time: {train_minutes:.1f} min. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
