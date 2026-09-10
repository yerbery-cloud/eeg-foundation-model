from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader

from eeg_dataset import (
    EEGPretrainDataset,
    split_by_subject
)

from eeg_patch_embedding import (
    EEGPatchEmbedding
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


print("EEG 数据 shape：")
print(data.shape)


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
# 6. 取一个 Batch
# ============================================================

eeg_batch, meta_batch = next(
    iter(train_loader)
)


print("\n===== 输入 EEG =====")

print(
    "EEG Batch shape：",
    eeg_batch.shape
)


# ============================================================
# 7. 创建 Patch Embedding
# ============================================================

patch_embedding = EEGPatchEmbedding(
    num_channels=2,

    time_points=640,

    patch_size=40,

    embed_dim=128
)


print("\n===== Patch 参数 =====")

print(
    "每通道 Patch 数：",
    patch_embedding.num_patches_per_channel
)

print(
    "总 Token 数：",
    patch_embedding.num_tokens
)

print(
    "Embedding Dimension：",
    patch_embedding.embed_dim
)


# ============================================================
# 8. Forward
# ============================================================

tokens = patch_embedding(
    eeg_batch
)


# ============================================================
# 9. 输出结果
# ============================================================

print("\n===== Patch Embedding 输出 =====")

print(
    "Token shape：",
    tokens.shape
)


print("\n应该得到：")

print(
    "[Batch, Tokens, Embedding]"
)

print(
    "=",
    f"[{tokens.shape[0]}, "
    f"{tokens.shape[1]}, "
    f"{tokens.shape[2]}]"
)


# ============================================================
# 10. 检查 NaN / Inf
# ============================================================

finite = torch.isfinite(
    tokens
).all()


print("\n全部为有限数值：")

print(
    finite.item()
)


# ============================================================
# 11. 查看第一个 Token
# ============================================================

first_token = tokens[
    0,
    0
]


print("\n===== 第一个 Token =====")

print(
    "shape：",
    first_token.shape
)

print(
    "前 10 个数值："
)

print(
    first_token[:10]
)


# ============================================================
# 12. 参数量
# ============================================================

total_params = sum(
    p.numel()
    for p in patch_embedding.parameters()
)


trainable_params = sum(
    p.numel()
    for p in patch_embedding.parameters()
    if p.requires_grad
)


print("\n===== Patch Embedding 参数 =====")

print(
    "总参数量：",
    total_params
)

print(
    "可训练参数量：",
    trainable_params
)


# ============================================================
# 13. 完成
# ============================================================

print("\n")

print("=" * 60)

print(
    "Patch Embedding Check 完成"
)

print("=" * 60)