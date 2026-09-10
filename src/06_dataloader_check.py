from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader

from eeg_dataset import (
    EEGPretrainDataset,
    split_by_subject
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


# ============================================================
# 2. 数据路径
# ============================================================

DATA_PATH = (
    PROCESSED_DIR
    / "pilot_5subjects_windows.npy"
)

METADATA_PATH = (
    PROCESSED_DIR
    / "pilot_5subjects_metadata.npy"
)


# ============================================================
# 3. 加载 NumPy 数据
# ============================================================

print("正在加载数据……")

data = np.load(
    DATA_PATH
)

metadata = np.load(
    METADATA_PATH
)


print("\n===== 原始数据 =====")

print(
    "EEG shape：",
    data.shape
)

print(
    "Metadata shape：",
    metadata.shape
)


# ============================================================
# 4. Subject-level Split
# ============================================================

TRAIN_SUBJECTS = [
    1,
    2,
    3
]

VAL_SUBJECTS = [
    4
]

TEST_SUBJECTS = [
    5
]


(
    train_indices,
    val_indices,
    test_indices
) = split_by_subject(
    metadata=metadata,
    train_subjects=TRAIN_SUBJECTS,
    val_subjects=VAL_SUBJECTS,
    test_subjects=TEST_SUBJECTS
)


print("\n===== Subject Split =====")

print(
    "Train subjects：",
    TRAIN_SUBJECTS
)

print(
    "Validation subjects：",
    VAL_SUBJECTS
)

print(
    "Test subjects：",
    TEST_SUBJECTS
)


print("\n===== 样本数量 =====")

print(
    "Train samples：",
    len(train_indices)
)

print(
    "Validation samples：",
    len(val_indices)
)

print(
    "Test samples：",
    len(test_indices)
)


# ============================================================
# 5. 创建 PyTorch Dataset
# ============================================================

train_dataset = EEGPretrainDataset(
    data=data,
    metadata=metadata,
    indices=train_indices
)

val_dataset = EEGPretrainDataset(
    data=data,
    metadata=metadata,
    indices=val_indices
)

test_dataset = EEGPretrainDataset(
    data=data,
    metadata=metadata,
    indices=test_indices
)


print("\n===== Dataset =====")

print(
    "Train Dataset：",
    len(train_dataset)
)

print(
    "Validation Dataset：",
    len(val_dataset)
)

print(
    "Test Dataset：",
    len(test_dataset)
)


# ============================================================
# 6. 检查单个样本
# ============================================================

eeg_sample, meta_sample = (
    train_dataset[0]
)


print("\n===== 单个样本 =====")

print(
    "EEG Tensor shape：",
    eeg_sample.shape
)

print(
    "EEG dtype：",
    eeg_sample.dtype
)

print(
    "Metadata：",
    meta_sample
)


# ============================================================
# 7. 创建 DataLoader
# ============================================================

BATCH_SIZE = 32


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,

    # Train 要打乱
    shuffle=True,

    # Windows 初学阶段先设 0
    # 避免多进程引起额外问题
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 8. 从 DataLoader 中取出第一个 Batch
# ============================================================

eeg_batch, meta_batch = next(
    iter(train_loader)
)


print("\n===== 第一个 Batch =====")

print(
    "EEG Batch shape：",
    eeg_batch.shape
)

print(
    "Metadata Batch shape：",
    meta_batch.shape
)

print(
    "EEG dtype：",
    eeg_batch.dtype
)


# ============================================================
# 9. 检查 Batch 数值
# ============================================================

print("\n===== Batch 数值检查 =====")


# 检查 NaN / Inf
all_finite = torch.isfinite(
    eeg_batch
).all()


print(
    "全部为有限数值：",
    all_finite.item()
)


# ============================================================
# 10. 检查每个样本的 Z-score
# ============================================================

# eeg_batch:
#
# [B, C, T]
#
# axis=2 / dim=2：
# 沿时间维度检查


batch_means = eeg_batch.mean(
    dim=2
)


batch_stds = eeg_batch.std(
    dim=2,
    unbiased=False
)


print(
    "Batch |mean| 最大值：",
    batch_means.abs().max().item()
)

print(
    "Batch std 平均值：",
    batch_stds.mean().item()
)


# ============================================================
# 11. 检查 Subject 是否正确
# ============================================================

train_batch_subjects = torch.unique(
    meta_batch[:, 0]
)


print("\n===== Batch Subject =====")

print(
    "当前 Train Batch 中出现的 Subjects：",
    train_batch_subjects.tolist()
)


# ============================================================
# 12. 保存 Split
# ============================================================

SPLIT_PATH = (
    PROCESSED_DIR
    / "pilot_subject_split.npz"
)


np.savez(
    SPLIT_PATH,

    train_indices=train_indices,

    val_indices=val_indices,

    test_indices=test_indices
)


print("\nSubject Split 已保存：")

print(
    SPLIT_PATH
)


# ============================================================
# 13. 最终结果
# ============================================================

print("\n")
print("=" * 60)

print(
    "PyTorch DataLoader Check 完成"
)

print("=" * 60)


print(
    "\n最终模型输入 Batch："
)

print(
    eeg_batch.shape
)

print(
    "\n即："
)

print(
    "[Batch, Channels, Time]"
)

print(
    "=",
    f"[{eeg_batch.shape[0]}, "
    f"{eeg_batch.shape[1]}, "
    f"{eeg_batch.shape[2]}]"
)