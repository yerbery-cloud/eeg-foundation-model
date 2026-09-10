from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from eeg_dataset import (
    EEGPretrainDataset,
    split_by_subject
)

from eeg_patch_embedding import (
    EEGPatchEmbedding
)

from eeg_masking import (
    random_patch_mask
)


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eegmmidb_fp1_fp2"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 加载数据
# ============================================================

data = np.load(
    PROCESSED_DIR
    / "pilot_5subjects_windows.npy"
)

metadata = np.load(
    PROCESSED_DIR
    / "pilot_5subjects_metadata.npy"
)


# ============================================================
# 3. Subject Split
# ============================================================

train_indices, _, _ = (
    split_by_subject(
        metadata=metadata,

        train_subjects=[
            1,
            2,
            3
        ],

        val_subjects=[
            4
        ],

        test_subjects=[
            5
        ]
    )
)


# ============================================================
# 4. Dataset
# ============================================================

train_dataset = EEGPretrainDataset(
    data=data,
    metadata=metadata,
    indices=train_indices
)


# ============================================================
# 5. DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)


# ============================================================
# 6. 取一个 EEG Batch
# ============================================================

eeg_batch, meta_batch = next(
    iter(train_loader)
)


print("\n===== 原始 EEG =====")

print(
    "EEG shape：",
    eeg_batch.shape
)


# ============================================================
# 7. Random Mask
# ============================================================

PATCH_SIZE = 40

MASK_RATIO = 0.5


(
    masked_eeg,
    target_patches,
    mask
) = random_patch_mask(
    eeg_batch,
    patch_size=PATCH_SIZE,
    mask_ratio=MASK_RATIO
)


# ============================================================
# 8. 输出 Shape
# ============================================================

print("\n===== Masking 结果 =====")

print(
    "Masked EEG shape：",
    masked_eeg.shape
)

print(
    "Target Patch shape：",
    target_patches.shape
)

print(
    "Mask shape：",
    mask.shape
)


# ============================================================
# 9. 检查每个样本 Mask 数量
# ============================================================

masked_counts = mask.sum(
    dim=1
)


print("\n===== Mask 数量 =====")

print(
    "第一个样本 Mask Token 数：",
    masked_counts[0].item()
)

print(
    "整个 Batch 最少 Mask：",
    masked_counts.min().item()
)

print(
    "整个 Batch 最多 Mask：",
    masked_counts.max().item()
)


# ============================================================
# 10. 检查 Mask 比例
# ============================================================

actual_mask_ratio = (
    mask.float().mean().item()
)


print(
    "\n实际 Mask Ratio：",
    actual_mask_ratio
)


# ============================================================
# 11. 检查被 Mask 的 Patch 是否真的变成 0
# ============================================================

# 将 Masked EEG 再切成 Patch
# 用于检查

batch_size = masked_eeg.shape[0]

channels = masked_eeg.shape[1]

time_points = masked_eeg.shape[2]


num_patches_per_channel = (
    time_points
    // PATCH_SIZE
)


masked_patches_check = (
    masked_eeg.reshape(
        batch_size,
        channels,
        num_patches_per_channel,
        PATCH_SIZE
    )
)


masked_patches_check = (
    masked_patches_check.reshape(
        batch_size,
        -1,
        PATCH_SIZE
    )
)


# 取所有被 Mask Patch 的绝对值最大值
max_masked_value = (
    masked_patches_check[
        mask
    ]
    .abs()
    .max()
    .item()
)


print(
    "\n被 Mask Patch 的最大绝对值：",
    max_masked_value
)


# 应该为 0


# ============================================================
# 12. 送入 Patch Embedding
# ============================================================

patch_embedding = EEGPatchEmbedding(
    num_channels=2,
    time_points=640,
    patch_size=40,
    embed_dim=128
)


tokens = patch_embedding(
    masked_eeg
)


print("\n===== Masked EEG → Embedding =====")

print(
    "Token shape：",
    tokens.shape
)


# ============================================================
# 13. 数值检查
# ============================================================

print(
    "Token 是否全部有限：",
    torch.isfinite(tokens)
    .all()
    .item()
)


# ============================================================
# 14. 绘制一个 Mask 示例
# ============================================================

sample_index = 0

sfreq = 160.0

times = (
    np.arange(640)
    / sfreq
)


original_sample = (
    eeg_batch[
        sample_index
    ]
    .numpy()
)


masked_sample = (
    masked_eeg[
        sample_index
    ]
    .numpy()
)


fig, axes = plt.subplots(
    2,
    1,
    figsize=(13, 7),
    sharex=True
)


# Fp1
axes[0].plot(
    times,
    original_sample[0],
    label="Original"
)

axes[0].plot(
    times,
    masked_sample[0],
    label="Masked",
    alpha=0.8
)

axes[0].set_title(
    "Fp1: Original vs Masked"
)

axes[0].set_ylabel(
    "Normalized amplitude"
)

axes[0].legend()

axes[0].grid(
    alpha=0.3
)


# Fp2
axes[1].plot(
    times,
    original_sample[1],
    label="Original"
)

axes[1].plot(
    times,
    masked_sample[1],
    label="Masked",
    alpha=0.8
)

axes[1].set_title(
    "Fp2: Original vs Masked"
)

axes[1].set_xlabel(
    "Time (s)"
)

axes[1].set_ylabel(
    "Normalized amplitude"
)

axes[1].legend()

axes[1].grid(
    alpha=0.3
)


plt.suptitle(
    "EEG Random Patch Masking"
)

plt.tight_layout()


figure_path = (
    RESULT_DIR
    / "masking_example.png"
)


plt.savefig(
    figure_path,
    dpi=150
)


print(
    "\nMask 示例图保存到："
)

print(
    figure_path
)


plt.show(block=True)


# ============================================================
# 15. 最终总结
# ============================================================

print("\n")

print("=" * 60)

print(
    "Masking Check 完成"
)

print("=" * 60)


print(
    "\nOriginal EEG：",
    eeg_batch.shape
)

print(
    "Masked EEG：",
    masked_eeg.shape
)

print(
    "Target：",
    target_patches.shape
)

print(
    "Mask：",
    mask.shape
)

print(
    "Tokens：",
    tokens.shape
)