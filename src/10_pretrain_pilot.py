from pathlib import Path
import random
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from eeg_dataset import EEGPretrainDataset

from eeg_masking import random_patch_mask

from masked_eeg_model import (
    MaskedEEGTransformer,
    masked_reconstruction_loss
)


# ============================================================
# 1. 随机种子
# ============================================================

SEED = 42


def set_seed(seed):
    """
    设置随机种子，使实验尽量可复现。
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 2. 项目路径
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


CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "pilot"
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. 训练参数
# ============================================================

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

PATCH_SIZE = 40

MASK_RATIO = 0.5


# ============================================================
# 4. 模型参数
# ============================================================

NUM_CHANNELS = 2

TIME_POINTS = 640

EMBED_DIM = 128

NUM_HEADS = 4

NUM_LAYERS = 4

FEEDFORWARD_DIM = 256

DROPOUT = 0.1


# ============================================================
# 5. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 60)

print("Pilot Pretraining")

print("=" * 60)


print("\nDevice：")
print(device)


# ============================================================
# 6. 加载 EEG 数据
# ============================================================

DATA_PATH = (
    PROCESSED_DIR
    / "pilot_5subjects_windows.npy"
)


METADATA_PATH = (
    PROCESSED_DIR
    / "pilot_5subjects_metadata.npy"
)


SPLIT_PATH = (
    PROCESSED_DIR
    / "pilot_subject_split.npz"
)


print("\n正在加载 EEG 数据……")


data = np.load(
    DATA_PATH
)


metadata = np.load(
    METADATA_PATH
)


split = np.load(
    SPLIT_PATH
)


train_indices = split[
    "train_indices"
]


val_indices = split[
    "val_indices"
]


test_indices = split[
    "test_indices"
]


print(
    "EEG shape：",
    data.shape
)


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
# 7. 创建 Dataset
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


# ============================================================
# 8. 创建 DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=(
        device.type == "cuda"
    )
)


val_loader = DataLoader(
    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=(
        device.type == "cuda"
    )
)


print(
    "\nTrain batches：",
    len(train_loader)
)


print(
    "Validation batches：",
    len(val_loader)
)


# ============================================================
# 9. 创建模型
# ============================================================

model = MaskedEEGTransformer(

    num_channels=NUM_CHANNELS,

    time_points=TIME_POINTS,

    patch_size=PATCH_SIZE,

    embed_dim=EMBED_DIM,

    num_heads=NUM_HEADS,

    num_layers=NUM_LAYERS,

    feedforward_dim=FEEDFORWARD_DIM,

    dropout=DROPOUT

).to(device)


# ============================================================
# 10. 模型参数量
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


print(
    "\nModel parameters：",
    total_params
)


print(
    "Trainable parameters：",
    trainable_params
)


print(
    "Model size：",
    f"{total_params / 1e6:.3f} M"
)


# ============================================================
# 11. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 12. 保存训练记录
# ============================================================

train_losses = []

val_losses = []


best_val_loss = float(
    "inf"
)


best_epoch = 0


# ============================================================
# 13. 开始训练
# ============================================================

print("\n")

print("=" * 60)

print("开始训练")

print("=" * 60)


training_start_time = time.time()


for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start_time = time.time()


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()


    train_loss_sum = 0.0

    train_sample_count = 0


    for batch_index, (
        eeg_batch,
        _
    ) in enumerate(
        train_loader,
        start=1
    ):


        # ----------------------------------------------------
        # EEG → Device
        # ----------------------------------------------------

        eeg_batch = eeg_batch.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # 每个 Batch 重新随机 Mask
        # ----------------------------------------------------

        (
            masked_eeg,
            target_patches,
            mask
        ) = random_patch_mask(

            eeg_batch,

            patch_size=PATCH_SIZE,

            mask_ratio=MASK_RATIO

        )


        # ----------------------------------------------------
        # 清空上一轮 Gradient
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        predictions, _ = model(
            masked_eeg
        )


        # ----------------------------------------------------
        # Reconstruction Loss
        # ----------------------------------------------------

        loss = masked_reconstruction_loss(

            predictions=predictions,

            targets=target_patches,

            mask=mask

        )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Gradient Clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0

        )


        # ----------------------------------------------------
        # 更新模型参数
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # 累计 Loss
        # ----------------------------------------------------

        current_batch_size = (
            eeg_batch.shape[0]
        )


        train_loss_sum += (
            loss.item()
            * current_batch_size
        )


        train_sample_count += (
            current_batch_size
        )


        # ----------------------------------------------------
        # 每 20 Batch 显示一次
        # ----------------------------------------------------

        if batch_index % 20 == 0:

            print(

                f"Epoch {epoch:02d} | "

                f"Batch "
                f"{batch_index:03d}/"
                f"{len(train_loader):03d} | "

                f"Loss "
                f"{loss.item():.4f}"

            )


    # ========================================================
    # Train Epoch Loss
    # ========================================================

    train_epoch_loss = (

        train_loss_sum

        /

        train_sample_count

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    val_loss_sum = 0.0

    val_sample_count = 0


    with torch.no_grad():


        for eeg_batch, _ in val_loader:


            eeg_batch = eeg_batch.to(
                device,
                non_blocking=True
            )


            # Validation 同样需要 Mask
            (
                masked_eeg,
                target_patches,
                mask
            ) = random_patch_mask(

                eeg_batch,

                patch_size=PATCH_SIZE,

                mask_ratio=MASK_RATIO

            )


            predictions, _ = model(
                masked_eeg
            )


            loss = masked_reconstruction_loss(

                predictions=predictions,

                targets=target_patches,

                mask=mask

            )


            current_batch_size = (
                eeg_batch.shape[0]
            )


            val_loss_sum += (

                loss.item()

                * current_batch_size

            )


            val_sample_count += (
                current_batch_size
            )


    # ========================================================
    # Validation Epoch Loss
    # ========================================================

    val_epoch_loss = (

        val_loss_sum

        /

        val_sample_count

    )


    # ========================================================
    # 保存历史
    # ========================================================

    train_losses.append(
        train_epoch_loss
    )


    val_losses.append(
        val_epoch_loss
    )


    # ========================================================
    # Epoch 时间
    # ========================================================

    epoch_time = (
        time.time()
        - epoch_start_time
    )


    print("\n")

    print(
        f"Epoch "
        f"{epoch:02d}/{EPOCHS}"
    )


    print(
        f"Train Loss："
        f"{train_epoch_loss:.6f}"
    )


    print(
        f"Val Loss：  "
        f"{val_epoch_loss:.6f}"
    )


    print(
        f"Time：      "
        f"{epoch_time:.1f} s"
    )


    # ========================================================
    # 保存当前 Epoch
    # ========================================================

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_epoch_loss,

        "val_loss":
            val_epoch_loss,

        "config": {

            "num_channels":
                NUM_CHANNELS,

            "time_points":
                TIME_POINTS,

            "patch_size":
                PATCH_SIZE,

            "embed_dim":
                EMBED_DIM,

            "num_heads":
                NUM_HEADS,

            "num_layers":
                NUM_LAYERS,

            "feedforward_dim":
                FEEDFORWARD_DIM,

            "dropout":
                DROPOUT,

            "mask_ratio":
                MASK_RATIO

        }

    }


    latest_path = (
        CHECKPOINT_DIR
        / "latest.pt"
    )


    torch.save(
        checkpoint,
        latest_path
    )


    # ========================================================
    # 保存最佳模型
    # ========================================================

    if (
        val_epoch_loss
        < best_val_loss
    ):


        best_val_loss = (
            val_epoch_loss
        )


        best_epoch = (
            epoch
        )


        best_path = (
            CHECKPOINT_DIR
            / "best.pt"
        )


        torch.save(
            checkpoint,
            best_path
        )


        print(
            "✓ 保存新的 Best Model"
        )


    print("-" * 60)


# ============================================================
# 14. 总训练时间
# ============================================================

training_time = (
    time.time()
    - training_start_time
)


print("\n")

print("=" * 60)

print("训练完成")

print("=" * 60)


print(
    "Best Epoch：",
    best_epoch
)


print(
    "Best Validation Loss：",
    best_val_loss
)


print(
    "Total Training Time：",
    f"{training_time:.1f} s"
)


# ============================================================
# 15. 保存 Loss
# ============================================================

history_path = (
    RESULT_DIR
    / "pilot_pretrain_history.npz"
)


np.savez(

    history_path,

    train_loss=np.asarray(
        train_losses
    ),

    val_loss=np.asarray(
        val_losses
    )

)


print(
    "\nLoss 数据保存到："
)

print(
    history_path
)


# ============================================================
# 16. 绘制 Loss Curve
# ============================================================

epochs = np.arange(
    1,
    EPOCHS + 1
)


plt.figure(
    figsize=(9, 6)
)


plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Train Loss"
)


plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Masked Reconstruction MSE"
)


plt.title(
    "Pilot Masked EEG Pretraining"
)


plt.grid(
    alpha=0.3
)


plt.legend()


plt.tight_layout()


loss_figure_path = (
    RESULT_DIR
    / "pilot_pretrain_loss_curve.png"
)


plt.savefig(
    loss_figure_path,
    dpi=150
)


print(
    "\nLoss Curve 保存到："
)

print(
    loss_figure_path
)


plt.show(
    block=True
)


print(
    "\n10_pretrain_pilot.py 执行完成！"
)