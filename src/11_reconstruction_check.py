from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from eeg_dataset import EEGPretrainDataset

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

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "pilot"
    / "best.pt"
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
# 2. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device：", device)


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

split = np.load(
    PROCESSED_DIR
    / "pilot_subject_split.npz"
)

test_indices = split[
    "test_indices"
]


print("\nEEG shape：")
print(data.shape)

print(
    "Test samples：",
    len(test_indices)
)


# ============================================================
# 4. 创建 Test Dataset
# ============================================================

test_dataset = EEGPretrainDataset(
    data=data,
    metadata=metadata,
    indices=test_indices
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 5. 加载 Best Checkpoint
# ============================================================

print("\n正在加载：")
print(CHECKPOINT_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

config = checkpoint[
    "config"
]


print("\n===== Best Model =====")

print(
    "Epoch：",
    checkpoint["epoch"]
)

print(
    "Train Loss：",
    checkpoint["train_loss"]
)

print(
    "Val Loss：",
    checkpoint["val_loss"]
)


# ============================================================
# 6. 根据 Checkpoint 参数创建模型
# ============================================================

model = MaskedEEGTransformer(

    num_channels=config[
        "num_channels"
    ],

    time_points=config[
        "time_points"
    ],

    patch_size=config[
        "patch_size"
    ],

    embed_dim=config[
        "embed_dim"
    ],

    num_heads=config[
        "num_heads"
    ],

    num_layers=config[
        "num_layers"
    ],

    feedforward_dim=config[
        "feedforward_dim"
    ],

    dropout=config[
        "dropout"
    ]

).to(device)


# 加载训练后的参数
model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

# 推理模式
model.eval()


print(
    "\nBest Model 加载成功！"
)


# ============================================================
# 7. 取一条 Test EEG
# ============================================================

eeg_batch, meta_batch = next(
    iter(test_loader)
)

eeg_batch = eeg_batch.to(
    device
)


subject = int(
    meta_batch[0, 0]
)

run = int(
    meta_batch[0, 1]
)

window_index = int(
    meta_batch[0, 2]
)


print("\n===== Test Sample =====")

print(
    "Subject：",
    subject
)

print(
    "Run：",
    run
)

print(
    "Window：",
    window_index
)

print(
    "EEG shape：",
    eeg_batch.shape
)


# ============================================================
# 8. Random Mask
# ============================================================

PATCH_SIZE = config[
    "patch_size"
]

MASK_RATIO = config[
    "mask_ratio"
]


(
    masked_eeg,
    target_patches,
    mask
) = random_patch_mask(

    eeg_batch,

    patch_size=PATCH_SIZE,

    mask_ratio=MASK_RATIO
)


print("\n===== Mask =====")

print(
    "Masked Tokens：",
    mask.sum().item()
)

print(
    "Mask Ratio：",
    mask.float().mean().item()
)


# ============================================================
# 9. 模型 Reconstruction
# ============================================================

with torch.no_grad():

    predictions, representations = model(
        masked_eeg
    )


print("\n===== Model Output =====")

print(
    "Prediction shape：",
    predictions.shape
)

print(
    "Representation shape：",
    representations.shape
)


# ============================================================
# 10. 计算 Test Sample Reconstruction Loss
# ============================================================

loss = masked_reconstruction_loss(

    predictions=predictions,

    targets=target_patches,

    mask=mask
)


print(
    "\n该 Test Sample 的 Masked MSE：",
    loss.item()
)


# ============================================================
# 11. 计算一个简单 Zero Baseline
# ============================================================

# 假设模型完全不会预测，
# 对所有被 Mask Patch 都直接预测 0。

zero_predictions = torch.zeros_like(
    predictions
)


zero_loss = masked_reconstruction_loss(

    predictions=zero_predictions,

    targets=target_patches,

    mask=mask
)


print(
    "Zero Baseline MSE：",
    zero_loss.item()
)


print(
    "Model / Zero Baseline：",
    loss.item()
    / zero_loss.item()
)


# ============================================================
# 12. 构建完整的 Reconstruction
# ============================================================

# predictions:
#
# [1, 32, 40]
#
# target_patches:
#
# [1, 32, 40]
#
# mask:
#
# [1, 32]


# 先复制原始 Patch
reconstructed_patches = (
    target_patches.clone()
)


# 对被 Mask 的位置，
# 用模型预测值替换
reconstructed_patches[
    mask
] = predictions[
    mask
]


# ============================================================
# 13. 还原成 EEG
# ============================================================

num_channels = 2

num_patches_per_channel = (
    640
    // PATCH_SIZE
)


reconstructed_eeg = (
    reconstructed_patches.reshape(
        1,
        num_channels,
        num_patches_per_channel,
        PATCH_SIZE
    )
)


reconstructed_eeg = (
    reconstructed_eeg.reshape(
        1,
        num_channels,
        640
    )
)


# ============================================================
# 14. Tensor → NumPy
# ============================================================

original = (
    eeg_batch[
        0
    ]
    .detach()
    .cpu()
    .numpy()
)


masked = (
    masked_eeg[
        0
    ]
    .detach()
    .cpu()
    .numpy()
)


reconstructed = (
    reconstructed_eeg[
        0
    ]
    .detach()
    .cpu()
    .numpy()
)


# ============================================================
# 15. 时间轴
# ============================================================

SFREQ = 160.0

times = (
    np.arange(
        original.shape[1]
    )
    / SFREQ
)


# ============================================================
# 16. 绘制 Original / Masked / Reconstructed
# ============================================================

fig, axes = plt.subplots(

    2,
    1,

    figsize=(14, 8),

    sharex=True
)


channel_names = [
    "Fp1",
    "Fp2"
]


for channel in range(2):

    axes[channel].plot(

        times,

        original[channel],

        label="Original",

        linewidth=1.5

    )


    axes[channel].plot(

        times,

        masked[channel],

        label="Masked",

        alpha=0.7

    )


    axes[channel].plot(

        times,

        reconstructed[channel],

        label="Reconstructed",

        linewidth=1.3

    )


    axes[channel].set_title(

        f"{channel_names[channel]} "
        "- Original vs Masked vs Reconstructed"

    )


    axes[channel].set_ylabel(
        "Normalized amplitude"
    )


    axes[channel].grid(
        alpha=0.3
    )


    axes[channel].legend()


axes[-1].set_xlabel(
    "Time (s)"
)


plt.suptitle(

    f"Pilot EEG Reconstruction | "
    f"S{subject:03d} "
    f"R{run:02d} "
    f"Window {window_index}"

)


plt.tight_layout()


# ============================================================
# 17. 保存图片
# ============================================================

figure_path = (

    RESULT_DIR
    / "pilot_reconstruction_test.png"

)


plt.savefig(

    figure_path,

    dpi=150

)


print("\nReconstruction 图保存到：")

print(
    figure_path
)


plt.show(
    block=True
)


# ============================================================
# 18. 完成
# ============================================================

print("\n")

print("=" * 60)

print(
    "Reconstruction Check 完成"
)

print("=" * 60)