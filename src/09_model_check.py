from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader

from eeg_dataset import (
    EEGPretrainDataset,
    split_by_subject
)

from eeg_masking import (
    random_patch_mask
)

from masked_eeg_model import (
    MaskedEEGTransformer,
    masked_reconstruction_loss
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
# 2. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("\n当前 Device：")
print(device)


# ============================================================
# 3. 加载数据
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
# 4. Subject Split
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
# 5. Dataset / DataLoader
# ============================================================

train_dataset = EEGPretrainDataset(
    data=data,
    metadata=metadata,
    indices=train_indices
)


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


eeg_batch = eeg_batch.to(
    device
)


print("\n===== 原始 EEG =====")

print(
    "EEG Batch：",
    eeg_batch.shape
)


# ============================================================
# 7. Random Mask
# ============================================================

(
    masked_eeg,
    target_patches,
    mask
) = random_patch_mask(
    eeg_batch,
    patch_size=40,
    mask_ratio=0.5
)


print("\n===== Mask =====")

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


# ============================================================
# 8. 创建模型
# ============================================================

model = MaskedEEGTransformer(

    num_channels=2,

    time_points=640,

    patch_size=40,

    embed_dim=128,

    num_heads=4,

    num_layers=4,

    feedforward_dim=256,

    dropout=0.1
)


model = model.to(
    device
)


# ============================================================
# 9. Forward
# ============================================================

predictions, representations = model(
    masked_eeg
)


print("\n===== Transformer 输出 =====")

print(
    "Prediction：",
    predictions.shape
)

print(
    "Representation：",
    representations.shape
)


# ============================================================
# 10. Loss
# ============================================================

loss = masked_reconstruction_loss(
    predictions=predictions,

    targets=target_patches,

    mask=mask
)


print("\n===== Reconstruction Loss =====")

print(
    "Loss：",
    loss.item()
)


# ============================================================
# 11. 检查数值
# ============================================================

print("\n===== 数值检查 =====")


print(
    "Prediction 全部有限：",
    torch.isfinite(
        predictions
    ).all().item()
)


print(
    "Representation 全部有限：",
    torch.isfinite(
        representations
    ).all().item()
)


print(
    "Loss 有限：",
    torch.isfinite(
        loss
    ).item()
)


# ============================================================
# 12. 检查 Mask 后参与 Loss 的数量
# ============================================================

num_masked_tokens = (
    mask.sum().item()
)


print("\n===== Masked Token =====")

print(
    "整个 Batch 被 Mask Token 数：",
    num_masked_tokens
)


print(
    "理论值：",
    eeg_batch.shape[0] * 16
)


# ============================================================
# 13. 模型参数量
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)


trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("\n===== 模型参数 =====")

print(
    "总参数量：",
    total_params
)

print(
    "可训练参数量：",
    trainable_params
)


print(
    "参数量（Million）：",
    total_params / 1e6
)


# ============================================================
# 14. 试一次反向传播
# ============================================================

print("\n===== Backward Test =====")


model.zero_grad()


loss.backward()


# 检查至少有一个参数得到梯度

has_gradient = False


for parameter in model.parameters():

    if parameter.grad is not None:

        has_gradient = True

        break


print(
    "模型成功得到 Gradient：",
    has_gradient
)


# ============================================================
# 15. 完成
# ============================================================

print("\n")

print("=" * 60)

print(
    "Masked EEG Transformer Check 完成"
)

print("=" * 60)