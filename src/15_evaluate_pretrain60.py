from pathlib import Path
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from eeg_masking import random_patch_mask
from masked_eeg_model import MaskedEEGTransformer, masked_reconstruction_loss

SEED = 42
TEST_MASK_SEED = 2027
MASK_RATIO = 0.50
BATCH_SIZE = 64

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "eegmmidb_fp1_fp2_60"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "pretrain60" / "best.pt"
RESULT_DIR = PROJECT_ROOT / "results" / "pretrain60" / "test"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = DATA_ROOT / "pretrain_60subjects_windows.npy"
METADATA_PATH = DATA_ROOT / "pretrain_60subjects_metadata.npy"
SPLIT_PATH = DATA_ROOT / "pretrain_60subjects_split.npz"
SUMMARY_PATH = RESULT_DIR / "test_summary.csv"
FIGURE_PATH = RESULT_DIR / "test_reconstruction_example.png"


class EEGDataset(Dataset):
    def __init__(self, X, metadata, indices):
        self.X = X
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.metadata[idx]).long()


def eval_model(model, loader, device, zero_baseline=False):
    model.eval()
    loss_sum = 0.0
    sample_count = 0

    devices = [] if device.type == "cpu" else [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(TEST_MASK_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(TEST_MASK_SEED)

        with torch.no_grad():
            for eeg, _ in loader:
                eeg = eeg.to(device)
                masked_eeg, target_patches, mask = random_patch_mask(
                    eeg, patch_size=40, mask_ratio=MASK_RATIO
                )
                if zero_baseline:
                    predictions = torch.zeros_like(target_patches)
                else:
                    predictions, _ = model(masked_eeg)
                loss = masked_reconstruction_loss(predictions, target_patches, mask)
                b = eeg.shape[0]
                loss_sum += float(loss.item()) * b
                sample_count += b

    return loss_sum / max(sample_count, 1)


def patch_to_signal(patches, channels=2, patch_size=40):
    b, tokens, p = patches.shape
    patches_per_channel = tokens // channels
    return patches.reshape(b, channels, patches_per_channel, p).reshape(b, channels, -1)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.load(X_PATH, mmap_mode="r")
    metadata = np.load(METADATA_PATH)
    split = np.load(SPLIT_PATH)

    test_ds = EEGDataset(X, metadata, split["test_indices"])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    config = ckpt["config"]

    pretrained = MaskedEEGTransformer(**config).to(device)
    pretrained.load_state_dict(ckpt["model_state_dict"])

    torch.manual_seed(SEED + 999)
    random_model = MaskedEEGTransformer(**config).to(device)

    pretrained_mse = eval_model(pretrained, test_loader, device)
    random_mse = eval_model(random_model, test_loader, device)
    zero_mse = eval_model(pretrained, test_loader, device, zero_baseline=True)

    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["pretrain_subjects", 60])
        writer.writerow(["test_subjects", "55-60"])
        writer.writerow(["mask_ratio", MASK_RATIO])
        writer.writerow(["pretrained_mse", pretrained_mse])
        writer.writerow(["random_mse", random_mse])
        writer.writerow(["zero_mse", zero_mse])

    # deterministic example: first held-out window
    eeg, meta = test_ds[0]
    eeg = eeg.unsqueeze(0).to(device)
    devices = [] if device.type == "cpu" else [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(TEST_MASK_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(TEST_MASK_SEED)
        masked_eeg, target_patches, mask = random_patch_mask(eeg, patch_size=40, mask_ratio=MASK_RATIO)

    with torch.no_grad():
        pred_pre, _ = pretrained(masked_eeg)
        pred_rand, _ = random_model(masked_eeg)

    original = eeg.detach().cpu().numpy()[0]
    masked = masked_eeg.detach().cpu().numpy()[0]
    pre_signal = patch_to_signal(pred_pre.cpu())[0].numpy()
    rand_signal = patch_to_signal(pred_rand.cpu())[0].numpy()

    # only replace masked regions; keep visible parts identical to original
    mask_np = mask.cpu().numpy()[0].reshape(2, 16)
    recon_pre = original.copy()
    recon_rand = original.copy()
    for c in range(2):
        for p in range(16):
            if mask_np[c, p]:
                a, b = p * 40, (p + 1) * 40
                recon_pre[c, a:b] = pre_signal[c, a:b]
                recon_rand[c, a:b] = rand_signal[c, a:b]

    t = np.arange(640) / 160.0
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    names = ["Fp1", "Fp2"]
    for c, ax in enumerate(axes):
        ax.plot(t, original[c], label="Original", linewidth=1.1)
        ax.plot(t, masked[c], label="Masked", linewidth=1.0)
        ax.plot(t, recon_pre[c], label="Pretrained Reconstruction", linewidth=1.0)
        ax.set_ylabel("Normalized amplitude")
        ax.set_title(f"{names[c]} - Test Reconstruction")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"60-Subject Pretraining | Test S{int(meta[0]):03d} R{int(meta[1]):02d} | "
        f"MSE={pretrained_mse:.4f}"
    )
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=180)
    plt.close()

    print("=" * 72)
    print("60-subject pretraining test")
    print("Pretrained MSE:", f"{pretrained_mse:.6f}")
    print("Random MSE:    ", f"{random_mse:.6f}")
    print("Zero MSE:      ", f"{zero_mse:.6f}")
    print("Summary:", SUMMARY_PATH)
    print("Figure:", FIGURE_PATH)
    print("Next: python src/19_linear_probe_pretrain60.py")


if __name__ == "__main__":
    main()
