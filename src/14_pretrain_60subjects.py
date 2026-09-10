from pathlib import Path
import csv
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from eeg_masking import random_patch_mask
from masked_eeg_model import MaskedEEGTransformer, masked_reconstruction_loss

SEED = 42
VAL_MASK_SEED = 2026

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "eegmmidb_fp1_fp2_60"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "pretrain60"
RESULT_DIR = PROJECT_ROOT / "results" / "pretrain60"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = DATA_ROOT / "pretrain_60subjects_windows.npy"
METADATA_PATH = DATA_ROOT / "pretrain_60subjects_metadata.npy"
SPLIT_PATH = DATA_ROOT / "pretrain_60subjects_split.npz"

BEST_PATH = CHECKPOINT_DIR / "best.pt"
LATEST_PATH = CHECKPOINT_DIR / "latest.pt"
BEST_ENCODER_PATH = CHECKPOINT_DIR / "best_encoder.pt"
HISTORY_CSV = RESULT_DIR / "training_history.csv"
HISTORY_NPZ = RESULT_DIR / "training_history.npz"
LOSS_FIG = RESULT_DIR / "loss_curve.png"

BATCH_SIZE = 64
MAX_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MASK_RATIO = 0.50
PATIENCE = 8
MIN_DELTA = 1e-4

CONFIG = {
    "num_channels": 2,
    "time_points": 640,
    "patch_size": 40,
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 4,
    "feedforward_dim": 256,
    "dropout": 0.1,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EEGDataset(Dataset):
    def __init__(self, X, metadata, indices):
        self.X = X
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return (
        torch.from_numpy(self.X[idx].copy()).float(),
        torch.from_numpy(self.metadata[idx].copy()).long(),
)


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    loss_sum = 0.0
    sample_count = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for eeg, _ in loader:
            eeg = eeg.to(device, non_blocking=True)
            masked_eeg, target_patches, mask = random_patch_mask(
                eeg, patch_size=CONFIG["patch_size"], mask_ratio=MASK_RATIO
            )
            predictions, _ = model(masked_eeg)
            loss = masked_reconstruction_loss(predictions, target_patches, mask)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            b = eeg.shape[0]
            loss_sum += float(loss.item()) * b
            sample_count += b

    return loss_sum / max(sample_count, 1)


def validate_fixed_mask(model, loader, device):
    devices = [] if device.type == "cpu" else [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(VAL_MASK_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(VAL_MASK_SEED)
        return run_epoch(model, loader, None, device, train=False)


def save_history(history):
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "lr", "seconds"]
        )
        writer.writeheader()
        writer.writerows(history)

    np.savez(
        HISTORY_NPZ,
        epoch=np.asarray([r["epoch"] for r in history]),
        train_loss=np.asarray([r["train_loss"] for r in history]),
        val_loss=np.asarray([r["val_loss"] for r in history]),
        lr=np.asarray([r["lr"] for r in history]),
    )

    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_loss"] for r in history], marker="o", label="Train Loss")
    plt.plot(epochs, [r["val_loss"] for r in history], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Masked Reconstruction MSE")
    plt.title("60-Subject Masked EEG Pretraining")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_FIG, dpi=180)
    plt.close()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.load(X_PATH, mmap_mode="r")
    metadata = np.load(METADATA_PATH)
    split = np.load(SPLIT_PATH)

    train_ds = EEGDataset(X, metadata, split["train_indices"])
    val_ds = EEGDataset(X, metadata, split["val_indices"])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    model = MaskedEEGTransformer(**CONFIG).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []

    if LATEST_PATH.exists():
        ckpt = torch.load(LATEST_PATH, map_location=device)
        if ckpt.get("config") == CONFIG:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val = float(ckpt["best_val_loss"])
            best_epoch = int(ckpt.get("best_epoch", 0))
            no_improve = int(ckpt.get("epochs_without_improvement", 0))
            history = ckpt.get("history", [])
            print(f"Resume from epoch {start_epoch}")

    print("=" * 80)
    print("60-subject Masked EEG Pretraining")
    print("Device:", device)
    print("Train windows:", len(train_ds))
    print("Validation windows:", len(val_ds))
    print("Mask ratio:", MASK_RATIO)
    print("=" * 80)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = validate_fixed_mask(model, val_loader, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        seconds = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "seconds": seconds,
        })

        improved = val_loss < best_val - MIN_DELTA
        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "best_val_loss": best_val,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": CONFIG,
                "mask_ratio": MASK_RATIO,
            }, BEST_PATH)

            encoder_state = {
                k: v.cpu()
                for k, v in model.state_dict().items()
                if not k.startswith("reconstruction_head.")
            }
            torch.save({
                "best_epoch": epoch,
                "best_val_loss": best_val,
                "encoder_state_dict": encoder_state,
                "config": CONFIG,
                "mask_ratio": MASK_RATIO,
                "pretrain_subjects": 60,
            }, BEST_ENCODER_PATH)

        else:
            no_improve += 1

        torch.save({
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "epochs_without_improvement": no_improve,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": CONFIG,
            "history": history,
        }, LATEST_PATH)

        save_history(history)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"LR {lr:.2e} | {seconds:.1f}s"
            + (" | ✓ BEST" if improved else "")
        )

        if no_improve >= PATIENCE:
            print(f"Early stopping: {PATIENCE} epochs without validation improvement.")
            break

    print("\nBest epoch:", best_epoch)
    print("Best val loss:", f"{best_val:.6f}")
    print("Best encoder:", BEST_ENCODER_PATH)
    print("Next: python src/15_evaluate_pretrain60.py")


if __name__ == "__main__":
    main()
