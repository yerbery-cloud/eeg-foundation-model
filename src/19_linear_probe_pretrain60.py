from pathlib import Path
import csv
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from masked_eeg_model import MaskedEEGTransformer

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
MIN_DELTA = 1e-4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "eegmat_fp1_fp2" / "classification"
ENCODER_PATH = PROJECT_ROOT / "checkpoints" / "pretrain60" / "best_encoder.pt"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "linear_probe60"
RESULT_DIR = PROJECT_ROOT / "results" / "linear_probe60"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = DATA_DIR / "eegmat_X.npy"
Y_PATH = DATA_DIR / "eegmat_y.npy"
METADATA_PATH = DATA_DIR / "eegmat_metadata.npy"
SPLIT_PATH = DATA_DIR / "eegmat_subject_split.npz"
BEST_PATH = CHECKPOINT_DIR / "best.pt"
SUMMARY_PATH = RESULT_DIR / "linear_probe60_test_summary.csv"
HISTORY_PATH = RESULT_DIR / "linear_probe60_history.csv"
CURVE_PATH = RESULT_DIR / "linear_probe60_loss_curve.png"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EEGClassificationDataset(Dataset):
    def __init__(self, X, y, metadata, indices):
        self.X = X
        self.y = y
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.tensor(int(self.y[idx]), dtype=torch.long),
            torch.from_numpy(self.metadata[idx]).float(),
        )


class LinearProbeClassifier(nn.Module):
    def __init__(self, backbone, embed_dim=128):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, x):
        tokens = self.backbone.patch_embedding(x)
        reps = self.backbone.encoder(tokens)
        reps = self.backbone.norm(reps)
        features = reps.mean(dim=1)
        return self.classifier(features), features


def metrics(targets, preds):
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    tn = int(np.sum((targets == 0) & (preds == 0)))
    fp = int(np.sum((targets == 0) & (preds == 1)))
    fn = int(np.sum((targets == 1) & (preds == 0)))
    tp = int(np.sum((targets == 1) & (preds == 1)))
    acc = (tn + tp) / max(len(targets), 1)
    r0 = tn / max(tn + fp, 1)
    r1 = tp / max(tp + fn, 1)
    p0 = tn / max(tn + fn, 1)
    p1 = tp / max(tp + fp, 1)
    f0 = 2 * p0 * r0 / max(p0 + r0, 1e-12)
    f1 = 2 * p1 * r1 / max(p1 + r1, 1e-12)
    return {
        "accuracy": acc,
        "balanced_accuracy": (r0 + r1) / 2,
        "macro_f1": (f0 + f1) / 2,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    n = 0
    ys, ps, metas = [], [], []
    with torch.no_grad():
        for eeg, labels, meta in loader:
            eeg = eeg.to(device)
            labels = labels.to(device)
            logits, _ = model(eeg)
            loss = criterion(logits, labels)
            b = eeg.shape[0]
            loss_sum += float(loss.item()) * b
            n += b
            pred = logits.argmax(dim=1)
            ys.extend(labels.cpu().numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())
            metas.append(meta.numpy())
    return loss_sum / max(n, 1), metrics(ys, ps), np.asarray(ys), np.asarray(ps), np.concatenate(metas)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    metadata = np.load(METADATA_PATH)
    split = np.load(SPLIT_PATH)

    train_ds = EEGClassificationDataset(X, y, metadata, split["train_indices"])
    val_ds = EEGClassificationDataset(X, y, metadata, split["val_indices"])
    test_ds = EEGClassificationDataset(X, y, metadata, split["test_indices"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    enc_ckpt = torch.load(ENCODER_PATH, map_location=device)
    config = enc_ckpt["config"]
    backbone = MaskedEEGTransformer(**config).to(device)
    load_result = backbone.load_state_dict(enc_ckpt["encoder_state_dict"], strict=False)
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)

    for p in backbone.parameters():
        p.requires_grad = False

    model = LinearProbeClassifier(backbone, embed_dim=config["embed_dim"]).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 258:
        print(f"[WARN] expected 258 trainable params, got {trainable}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []

    print("=" * 76)
    print("Linear Probe with 60-subject pretrained encoder")
    print("Train/Val/Test:", len(train_ds), len(val_ds), len(test_ds))
    print("Trainable parameters:", trainable)
    print("=" * 76)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        ys, ps = [], []
        for eeg, labels, _ in train_loader:
            eeg = eeg.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(eeg)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            b = eeg.shape[0]
            loss_sum += float(loss.item()) * b
            n += b
            ys.extend(labels.cpu().numpy().tolist())
            ps.extend(logits.detach().argmax(dim=1).cpu().numpy().tolist())

        train_loss = loss_sum / max(n, 1)
        train_m = metrics(ys, ps)
        val_loss, val_m, *_ = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_m["accuracy"],
            "val_loss": val_loss,
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "seconds": time.time() - t0,
        })

        improved = val_loss < best_val - MIN_DELTA
        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_m["accuracy"],
                "val_macro_f1": val_m["macro_f1"],
                "trainable_parameters": trainable,
                "config": config,
            }, BEST_PATH)
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_m['accuracy']:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_m['accuracy']:.4f} F1 {val_m['macro_f1']:.4f}"
            + (" | ✓ BEST" if improved else "")
        )

        if no_improve >= PATIENCE:
            print("Early stopping.")
            break

    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)

    plt.figure(figsize=(8, 5))
    plt.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], marker="o", label="Train Loss")
    plt.plot([r["epoch"] for r in history], [r["val_loss"] for r in history], marker="o", label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Cross Entropy Loss")
    plt.title("Linear Probe - 60-Subject Pretrained Encoder")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=180); plt.close()

    best = torch.load(BEST_PATH, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    test_loss, test_m, y_true, y_pred, test_meta = evaluate(model, test_loader, criterion, device)

    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["method", "Linear Probe - Pretrain60"])
        writer.writerow(["pretrain_subjects", 60])
        writer.writerow(["trainable_parameters", trainable])
        writer.writerow(["best_epoch", best_epoch])
        writer.writerow(["test_loss", test_loss])
        writer.writerow(["test_accuracy", test_m["accuracy"]])
        writer.writerow(["test_balanced_accuracy", test_m["balanced_accuracy"]])
        writer.writerow(["test_macro_f1", test_m["macro_f1"]])

    print("\n===== Test =====")
    print("Accuracy:", f"{test_m['accuracy']:.4f}")
    print("Balanced Accuracy:", f"{test_m['balanced_accuracy']:.4f}")
    print("Macro F1:", f"{test_m['macro_f1']:.4f}")
    print("Summary:", SUMMARY_PATH)
    print("Next: python src/31_compare_pretraining_scale.py")


if __name__ == "__main__":
    main()
