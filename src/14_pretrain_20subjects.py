from pathlib import Path
import csv
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
# 1. Random Seed
# ============================================================

SEED = 42
VAL_MASK_SEED = 2026


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 2. 项目目录
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eegmmidb_fp1_fp2"
    / "pretrain_20subjects"
)


CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "pretrain20"
)


RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "pretrain20"
)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. 数据路径
# ============================================================

DATA_PATH = (
    DATA_DIR
    / "pretrain_20subjects_windows.npy"
)


METADATA_PATH = (
    DATA_DIR
    / "pretrain_20subjects_metadata.npy"
)


SPLIT_PATH = (
    DATA_DIR
    / "pretrain_20subjects_split.npz"
)


# ============================================================
# 4. 输出路径
# ============================================================

LATEST_PATH = (
    CHECKPOINT_DIR
    / "latest.pt"
)


BEST_PATH = (
    CHECKPOINT_DIR
    / "best.pt"
)


ENCODER_PATH = (
    CHECKPOINT_DIR
    / "best_encoder.pt"
)


VAL_MASK_PATH = (
    CHECKPOINT_DIR
    / "fixed_validation_masks.npy"
)


HISTORY_CSV_PATH = (
    RESULT_DIR
    / "training_history.csv"
)


HISTORY_NPZ_PATH = (
    RESULT_DIR
    / "training_history.npz"
)


LOSS_FIGURE_PATH = (
    RESULT_DIR
    / "loss_curve.png"
)


# ============================================================
# 5. Training Config
# ============================================================

BATCH_SIZE = 64

MAX_EPOCHS = 30

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

MASK_RATIO = 0.50


# Early Stopping

PATIENCE = 6

MIN_DELTA = 1e-4


# 是否自动从 latest.pt 继续

RESUME = True


# ============================================================
# 6. Model Config
# ============================================================

NUM_CHANNELS = 2

TIME_POINTS = 640

PATCH_SIZE = 40

EMBED_DIM = 128

NUM_HEADS = 4

NUM_LAYERS = 4

FEEDFORWARD_DIM = 256

DROPOUT = 0.1


# ============================================================
# 7. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)

print("20-Subject Masked EEG Pretraining")

print("=" * 70)


print(
    "\nDevice:",
    device
)


# ============================================================
# 8. 加载数据
# ============================================================

print(
    "\n正在加载 20-subject dataset ..."
)


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
    "\nEEG shape:",
    data.shape
)


print(
    "Train samples:",
    len(train_indices)
)


print(
    "Validation samples:",
    len(val_indices)
)


print(
    "Test samples:",
    len(test_indices)
)


# ============================================================
# 9. Dataset
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
# 10. DataLoader
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
    "\nTrain batches:",
    len(train_loader)
)


print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# 11. 固定 Validation Mask
# ============================================================

TOTAL_TOKENS = (

    NUM_CHANNELS

    * (
        TIME_POINTS
        // PATCH_SIZE
    )

)


NUM_MASKED_TOKENS = int(

    TOTAL_TOKENS
    * MASK_RATIO

)


def build_fixed_masks(
    num_samples,
    total_tokens,
    num_masked_tokens,
    seed
):

    """
    生成固定的 Validation Mask。

    输出：

        [num_samples, total_tokens]

    True  = Mask
    False = Visible
    """


    generator = torch.Generator(
        device="cpu"
    )


    generator.manual_seed(
        seed
    )


    random_values = torch.rand(

        num_samples,

        total_tokens,

        generator=generator

    )


    random_indices = torch.argsort(

        random_values,

        dim=1

    )


    mask_indices = random_indices[
        :,
        :num_masked_tokens
    ]


    masks = torch.zeros(

        num_samples,

        total_tokens,

        dtype=torch.bool

    )


    masks.scatter_(

        dim=1,

        index=mask_indices,

        value=True

    )


    return masks


# ------------------------------------------------------------
# 已经生成过 → 直接读取
# ------------------------------------------------------------

if VAL_MASK_PATH.exists():

    fixed_val_masks = torch.from_numpy(

        np.load(
            VAL_MASK_PATH
        )

    ).bool()


    if fixed_val_masks.shape != (

        len(val_dataset),

        TOTAL_TOKENS

    ):

        raise RuntimeError(

            "已有 Validation Mask "
            "shape 与当前数据不一致"

        )


    print(
        "\n✓ 读取已有固定 Validation Mask"
    )


else:

    fixed_val_masks = build_fixed_masks(

        num_samples=len(
            val_dataset
        ),

        total_tokens=TOTAL_TOKENS,

        num_masked_tokens=NUM_MASKED_TOKENS,

        seed=VAL_MASK_SEED

    )


    np.save(

        VAL_MASK_PATH,

        fixed_val_masks.numpy()

    )


    print(
        "\n✓ 创建并保存固定 Validation Mask"
    )


print(
    "Validation Mask shape:",
    fixed_val_masks.shape
)


print(
    "每条 Validation EEG Mask Token:",
    fixed_val_masks[0].sum().item()
)


# ============================================================
# 12. 根据已经给定的 Mask 遮挡 EEG
# ============================================================

def apply_given_mask(
    eeg,
    mask,
    patch_size
):

    """
    eeg:
        [B, C, T]

    mask:
        [B, N]

    返回：

        masked_eeg:
            [B, C, T]

        target_patches:
            [B, N, patch_size]
    """


    batch_size = eeg.shape[0]

    channels = eeg.shape[1]

    time_points = eeg.shape[2]


    num_patches_per_channel = (

        time_points
        // patch_size

    )


    total_tokens = (

        channels
        * num_patches_per_channel

    )


    # --------------------------------------------------------
    # EEG → Patch
    # --------------------------------------------------------

    patches = eeg.reshape(

        batch_size,

        channels,

        num_patches_per_channel,

        patch_size

    )


    # --------------------------------------------------------
    # [B,C,P,L]
    # →
    # [B,N,L]
    # --------------------------------------------------------

    flat_patches = patches.reshape(

        batch_size,

        total_tokens,

        patch_size

    )


    target_patches = (
        flat_patches.clone()
    )


    masked_patches = (
        flat_patches.clone()
    )


    # --------------------------------------------------------
    # Mask 的 Patch 设置为 0
    # --------------------------------------------------------

    masked_patches[
        mask
    ] = 0.0


    # --------------------------------------------------------
    # 恢复连续 EEG
    # --------------------------------------------------------

    masked_eeg = (
        masked_patches.reshape(

            batch_size,

            channels,

            num_patches_per_channel,

            patch_size

        )
        .reshape(

            batch_size,

            channels,

            time_points

        )
    )


    return (
        masked_eeg,
        target_patches
    )


# ============================================================
# 13. Model
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
# 14. 参数量
# ============================================================

total_params = sum(

    parameter.numel()

    for parameter
    in model.parameters()

)


trainable_params = sum(

    parameter.numel()

    for parameter
    in model.parameters()

    if parameter.requires_grad

)


print(
    "\nTotal parameters:",
    total_params
)


print(
    "Trainable parameters:",
    trainable_params
)


print(
    "Model size:",
    f"{total_params / 1e6:.3f} M"
)


# ============================================================
# 15. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


# ============================================================
# 16. Learning Rate Scheduler
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=2,

        min_lr=1e-6

    )
)


# ============================================================
# 17. Training State
# ============================================================

start_epoch = 1


best_val_loss = float(
    "inf"
)


best_epoch = 0


epochs_without_improvement = 0


history_epoch = []

history_train_loss = []

history_val_loss = []

history_lr = []

history_time = []


# ============================================================
# 18. Resume
# ============================================================

if (
    RESUME
    and LATEST_PATH.exists()
):

    print(
        "\n发现 latest.pt，正在恢复训练 ..."
    )


    checkpoint = torch.load(

        LATEST_PATH,

        map_location=device

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    optimizer.load_state_dict(

        checkpoint[
            "optimizer_state_dict"
        ]

    )


    if (
        "scheduler_state_dict"
        in checkpoint
    ):

        scheduler.load_state_dict(

            checkpoint[
                "scheduler_state_dict"
            ]

        )


    start_epoch = (

        checkpoint[
            "epoch"
        ]

        + 1

    )


    best_val_loss = checkpoint.get(

        "best_val_loss",

        checkpoint[
            "val_loss"
        ]

    )


    best_epoch = checkpoint.get(

        "best_epoch",

        checkpoint[
            "epoch"
        ]

    )


    epochs_without_improvement = (
        checkpoint.get(
            "epochs_without_improvement",
            0
        )
    )


    history_epoch = checkpoint.get(
        "history_epoch",
        []
    )


    history_train_loss = checkpoint.get(
        "history_train_loss",
        []
    )


    history_val_loss = checkpoint.get(
        "history_val_loss",
        []
    )


    history_lr = checkpoint.get(
        "history_lr",
        []
    )


    history_time = checkpoint.get(
        "history_time",
        []
    )


    print(
        "✓ 从 Epoch",
        start_epoch,
        "继续"
    )


# ============================================================
# 19. 保存 History
# ============================================================

def save_history():

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with open(

        HISTORY_CSV_PATH,

        "w",

        newline="",

        encoding="utf-8"

    ) as file:


        writer = csv.writer(
            file
        )


        writer.writerow(

            [
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rate",
                "epoch_seconds"
            ]

        )


        for values in zip(

            history_epoch,

            history_train_loss,

            history_val_loss,

            history_lr,

            history_time

        ):

            writer.writerow(
                values
            )


    # --------------------------------------------------------
    # NPZ
    # --------------------------------------------------------

    np.savez(

        HISTORY_NPZ_PATH,

        epoch=np.asarray(
            history_epoch
        ),

        train_loss=np.asarray(
            history_train_loss
        ),

        val_loss=np.asarray(
            history_val_loss
        ),

        learning_rate=np.asarray(
            history_lr
        ),

        epoch_seconds=np.asarray(
            history_time
        )

    )


# ============================================================
# 20. 保存 Loss Curve
# ============================================================

def save_loss_figure():

    if len(
        history_epoch
    ) == 0:

        return


    plt.figure(
        figsize=(9, 6)
    )


    plt.plot(

        history_epoch,

        history_train_loss,

        marker="o",

        label="Train Loss"

    )


    plt.plot(

        history_epoch,

        history_val_loss,

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
        "20-Subject Masked EEG Pretraining"
    )


    plt.grid(
        alpha=0.3
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        LOSS_FIGURE_PATH,

        dpi=150

    )


    plt.close()


# ============================================================
# 21. Config
# ============================================================

CONFIG = {

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


# ============================================================
# 22. Training
# ============================================================

print("\n")

print("=" * 70)

print("开始正式训练")

print("=" * 70)


for epoch in range(

    start_epoch,

    MAX_EPOCHS + 1

):

    epoch_start = time.time()


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()


    train_loss_sum = 0.0

    train_samples = 0


    for batch_index, (
        eeg_batch,
        _
    ) in enumerate(

        train_loader,

        start=1

    ):


        eeg_batch = eeg_batch.to(

            device,

            non_blocking=True

        )


        # ----------------------------------------------------
        # TRAIN 使用随机 Mask
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


        optimizer.zero_grad()


        predictions, _ = model(
            masked_eeg
        )


        loss = masked_reconstruction_loss(

            predictions=predictions,

            targets=target_patches,

            mask=mask

        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0

        )


        optimizer.step()


        current_batch_size = (
            eeg_batch.shape[0]
        )


        train_loss_sum += (

            loss.item()

            * current_batch_size

        )


        train_samples += (
            current_batch_size
        )


        if (
            batch_index % 50 == 0
        ):

            print(

                f"Epoch {epoch:02d} | "

                f"Batch "
                f"{batch_index:03d}/"
                f"{len(train_loader):03d} | "

                f"Loss "
                f"{loss.item():.4f}"

            )


    train_loss = (

        train_loss_sum
        / train_samples

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    val_loss_sum = 0.0

    val_samples = 0


    # 记录目前走到 Validation 第几个样本
    val_offset = 0


    with torch.no_grad():


        for eeg_batch, _ in val_loader:


            current_batch_size = (
                eeg_batch.shape[0]
            )


            eeg_batch = eeg_batch.to(

                device,

                non_blocking=True

            )


            # ------------------------------------------------
            # 取对应固定 Mask
            # ------------------------------------------------

            mask = fixed_val_masks[

                val_offset
                :
                val_offset
                + current_batch_size

            ].to(device)


            val_offset += (
                current_batch_size
            )


            # ------------------------------------------------
            # 应用固定 Mask
            # ------------------------------------------------

            (
                masked_eeg,
                target_patches

            ) = apply_given_mask(

                eeg_batch,

                mask,

                PATCH_SIZE

            )


            predictions, _ = model(
                masked_eeg
            )


            loss = masked_reconstruction_loss(

                predictions=predictions,

                targets=target_patches,

                mask=mask

            )


            val_loss_sum += (

                loss.item()

                * current_batch_size

            )


            val_samples += (
                current_batch_size
            )


    val_loss = (

        val_loss_sum
        / val_samples

    )


    # ========================================================
    # Scheduler
    # ========================================================

    scheduler.step(
        val_loss
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # ========================================================
    # Early Stopping / Best Model
    # ========================================================

    improved = (

        val_loss

        < (
            best_val_loss
            - MIN_DELTA
        )

    )


    if improved:

        best_val_loss = (
            val_loss
        )


        best_epoch = (
            epoch
        )


        epochs_without_improvement = 0


    else:

        epochs_without_improvement += 1


    # ========================================================
    # 时间
    # ========================================================

    epoch_seconds = (

        time.time()
        - epoch_start

    )


    # ========================================================
    # History
    # ========================================================

    history_epoch.append(
        epoch
    )


    history_train_loss.append(
        train_loss
    )


    history_val_loss.append(
        val_loss
    )


    history_lr.append(
        current_lr
    )


    history_time.append(
        epoch_seconds
    )


    # ========================================================
    # Checkpoint
    # ========================================================

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        "best_val_loss":
            best_val_loss,

        "best_epoch":
            best_epoch,

        "epochs_without_improvement":
            epochs_without_improvement,

        "history_epoch":
            history_epoch,

        "history_train_loss":
            history_train_loss,

        "history_val_loss":
            history_val_loss,

        "history_lr":
            history_lr,

        "history_time":
            history_time,

        "config":
            CONFIG

    }


    # 最新模型
    torch.save(

        checkpoint,

        LATEST_PATH

    )


    # 最佳模型
    if improved:

        torch.save(

            checkpoint,

            BEST_PATH

        )


    # ========================================================
    # 保存训练记录
    # ========================================================

    save_history()

    save_loss_figure()


    # ========================================================
    # 输出 Epoch
    # ========================================================

    print("\n")

    print(
        f"Epoch "
        f"{epoch:02d}/"
        f"{MAX_EPOCHS}"
    )


    print(
        f"Train Loss: "
        f"{train_loss:.6f}"
    )


    print(
        f"Val Loss:   "
        f"{val_loss:.6f}"
    )


    print(
        f"LR:         "
        f"{current_lr:.8f}"
    )


    print(
        f"Time:       "
        f"{epoch_seconds:.1f} s"
    )


    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )


    print(
        f"Best Val:   "
        f"{best_val_loss:.6f}"
    )


    print(
        "No improvement:",
        epochs_without_improvement,
        "/",
        PATIENCE
    )


    if improved:

        print(
            "✓ 保存新的 Best Model"
        )


    print(
        "-" * 70
    )


    # ========================================================
    # Early Stop
    # ========================================================

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print("\n")

        print(
            "触发 Early Stopping"
        )

        break


# ============================================================
# 23. 保存 Best Encoder
# ============================================================

if BEST_PATH.exists():

    best_checkpoint = torch.load(

        BEST_PATH,

        map_location="cpu"

    )


    full_state_dict = (
        best_checkpoint[
            "model_state_dict"
        ]
    )


    # Reconstruction Head
    # 下游任务不再需要
    encoder_state_dict = {

        name: value

        for name, value
        in full_state_dict.items()

        if not name.startswith(
            "reconstruction_head."
        )

    }


    torch.save(

        {

            "encoder_state_dict":
                encoder_state_dict,

            "config":
                best_checkpoint[
                    "config"
                ],

            "best_epoch":
                best_checkpoint[
                    "best_epoch"
                ],

            "best_val_loss":
                best_checkpoint[
                    "best_val_loss"
                ]

        },

        ENCODER_PATH

    )


# ============================================================
# 24. Final
# ============================================================

print("\n")

print("=" * 70)

print("20-Subject Pretraining 完成")

print("=" * 70)


print(
    "Best Epoch:",
    best_epoch
)


print(
    "Best Validation Loss:",
    best_val_loss
)


print(
    "\nBest Full Model:"
)

print(
    BEST_PATH
)


print(
    "\nBest Encoder:"
)

print(
    ENCODER_PATH
)


print(
    "\nLoss Curve:"
)

print(
    LOSS_FIGURE_PATH
)


print(
    "\nTest Set 尚未使用：",
    len(test_indices),
    "samples"
)