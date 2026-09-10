from pathlib import Path
import csv
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from eeg_dataset import EEGPretrainDataset

from masked_eeg_model import (
    MaskedEEGTransformer,
    masked_reconstruction_loss
)


# ============================================================
# 1. Random Seed
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. 项目路径
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

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "pretrain20"
    / "best.pt"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "pretrain20"
    / "test"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. 文件路径
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

TEST_MASK_PATH = (
    RESULT_DIR
    / "fixed_test_masks.npy"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "test_summary.csv"
)

SAMPLE_METRICS_PATH = (
    RESULT_DIR
    / "test_sample_metrics.npz"
)

FIGURE_PATH = (
    RESULT_DIR
    / "test_reconstruction_example.png"
)


# ============================================================
# 4. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("20-Subject Pretraining Evaluation")
print("=" * 70)

print("\nDevice:")
print(device)


# ============================================================
# 5. 加载数据
# ============================================================

print("\n正在加载 Test Dataset ...")

data = np.load(
    DATA_PATH
)

metadata = np.load(
    METADATA_PATH
)

split = np.load(
    SPLIT_PATH
)

test_indices = split[
    "test_indices"
]

test_subjects = split[
    "test_subjects"
]


print(
    "\n完整 EEG shape:",
    data.shape
)

print(
    "Test samples:",
    len(test_indices)
)

print(
    "Test subjects:",
    test_subjects.tolist()
)


# ============================================================
# 6. Test Dataset / DataLoader
# ============================================================

test_dataset = EEGPretrainDataset(
    data=data,
    metadata=metadata,
    indices=test_indices
)

BATCH_SIZE = 64

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(
        device.type == "cuda"
    )
)


# ============================================================
# 7. 加载 Best Checkpoint
# ============================================================

print("\n正在加载 Best Model：")
print(CHECKPOINT_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

config = checkpoint[
    "config"
]


print("\n===== Pretraining Information =====")

print(
    "Best Epoch:",
    checkpoint["best_epoch"]
)

print(
    "Best Validation Loss:",
    checkpoint["best_val_loss"]
)


# ============================================================
# 8. 模型配置
# ============================================================

NUM_CHANNELS = config[
    "num_channels"
]

TIME_POINTS = config[
    "time_points"
]

PATCH_SIZE = config[
    "patch_size"
]

EMBED_DIM = config[
    "embed_dim"
]

NUM_HEADS = config[
    "num_heads"
]

NUM_LAYERS = config[
    "num_layers"
]

FEEDFORWARD_DIM = config[
    "feedforward_dim"
]

DROPOUT = config[
    "dropout"
]

MASK_RATIO = config[
    "mask_ratio"
]


NUM_PATCHES_PER_CHANNEL = (
    TIME_POINTS
    // PATCH_SIZE
)

TOTAL_TOKENS = (
    NUM_CHANNELS
    * NUM_PATCHES_PER_CHANNEL
)

NUM_MASKED_TOKENS = int(
    TOTAL_TOKENS
    * MASK_RATIO
)


print("\n===== Model Config =====")

print(
    "Channels:",
    NUM_CHANNELS
)

print(
    "Time points:",
    TIME_POINTS
)

print(
    "Patch size:",
    PATCH_SIZE
)

print(
    "Total tokens:",
    TOTAL_TOKENS
)

print(
    "Masked tokens:",
    NUM_MASKED_TOKENS
)


# ============================================================
# 9. 创建 Pretrained Model
# ============================================================

pretrained_model = MaskedEEGTransformer(

    num_channels=NUM_CHANNELS,

    time_points=TIME_POINTS,

    patch_size=PATCH_SIZE,

    embed_dim=EMBED_DIM,

    num_heads=NUM_HEADS,

    num_layers=NUM_LAYERS,

    feedforward_dim=FEEDFORWARD_DIM,

    dropout=DROPOUT

).to(device)


pretrained_model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

pretrained_model.eval()


print(
    "\n✓ Pretrained Model 加载成功"
)


# ============================================================
# 10. 创建 Random Initialization Model
# ============================================================

# 注意：
#
# 架构和 Pretrained 完全相同，
# 但是不加载任何训练后的参数。
#
# 用于回答：
#
# “效果是不是仅仅因为 Transformer 架构？”

torch.manual_seed(
    SEED + 100
)


random_model = MaskedEEGTransformer(

    num_channels=NUM_CHANNELS,

    time_points=TIME_POINTS,

    patch_size=PATCH_SIZE,

    embed_dim=EMBED_DIM,

    num_heads=NUM_HEADS,

    num_layers=NUM_LAYERS,

    feedforward_dim=FEEDFORWARD_DIM,

    dropout=DROPOUT

).to(device)


random_model.eval()


print(
    "✓ Random Initialization Model 创建成功"
)


# ============================================================
# 11. 创建固定 Test Mask
# ============================================================

def build_fixed_masks(
    num_samples,
    total_tokens,
    num_masked_tokens,
    seed
):

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

    indices = torch.argsort(

        random_values,

        dim=1

    )

    mask_indices = indices[
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


TEST_MASK_SEED = 20260828


if TEST_MASK_PATH.exists():

    fixed_test_masks = torch.from_numpy(

        np.load(
            TEST_MASK_PATH
        )

    ).bool()

    print(
        "\n✓ 读取已有 Test Mask"
    )

else:

    fixed_test_masks = build_fixed_masks(

        num_samples=len(
            test_dataset
        ),

        total_tokens=TOTAL_TOKENS,

        num_masked_tokens=NUM_MASKED_TOKENS,

        seed=TEST_MASK_SEED

    )

    np.save(

        TEST_MASK_PATH,

        fixed_test_masks.numpy()

    )

    print(
        "\n✓ 创建固定 Test Mask"
    )


if fixed_test_masks.shape != (

    len(test_dataset),

    TOTAL_TOKENS

):

    raise RuntimeError(
        "Test Mask shape 不正确"
    )


print(
    "Test Mask shape:",
    fixed_test_masks.shape
)


# ============================================================
# 12. 根据给定 Mask 遮挡 EEG
# ============================================================

def apply_given_mask(
    eeg,
    mask,
    patch_size
):

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


    patches = eeg.reshape(

        batch_size,

        channels,

        num_patches_per_channel,

        patch_size

    )


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


    masked_patches[
        mask
    ] = 0.0


    masked_eeg = (
        masked_patches
        .reshape(

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
# 13. 逐 Sample 计算 Masked MSE
# ============================================================

def per_sample_masked_mse(
    predictions,
    targets,
    mask
):

    """
    predictions:
        [B,N,L]

    targets:
        [B,N,L]

    mask:
        [B,N]

    返回：
        [B]
    """


    squared_error = (

        predictions
        - targets

    ) ** 2


    # Patch 内 40 个点先取平均
    patch_mse = squared_error.mean(
        dim=2
    )


    # 非 Mask 位置不参与统计
    masked_patch_mse = (

        patch_mse
        * mask.float()

    )


    sample_mse = (

        masked_patch_mse.sum(
            dim=1
        )

        /

        mask.sum(
            dim=1
        ).float()

    )


    return sample_mse


# ============================================================
# 14. 正式 Test
# ============================================================

print("\n")

print("=" * 70)

print("开始 Test Evaluation")

print("=" * 70)


pretrained_losses = []

random_losses = []

zero_losses = []


test_offset = 0


# 保存第一条样本用于之后画图
example = None


with torch.no_grad():


    for batch_index, (
        eeg_batch,
        meta_batch

    ) in enumerate(

        test_loader,

        start=1

    ):


        current_batch_size = (
            eeg_batch.shape[0]
        )


        eeg_batch = eeg_batch.to(
            device
        )


        mask = fixed_test_masks[

            test_offset
            :
            test_offset
            + current_batch_size

        ].to(device)


        test_offset += (
            current_batch_size
        )


        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        (
            masked_eeg,
            target_patches

        ) = apply_given_mask(

            eeg_batch,

            mask,

            PATCH_SIZE

        )


        # ----------------------------------------------------
        # Pretrained Model
        # ----------------------------------------------------

        pretrained_predictions, _ = (

            pretrained_model(
                masked_eeg
            )

        )


        # ----------------------------------------------------
        # Random Model
        # ----------------------------------------------------

        random_predictions, _ = (

            random_model(
                masked_eeg
            )

        )


        # ----------------------------------------------------
        # Zero Baseline
        # ----------------------------------------------------

        zero_predictions = torch.zeros_like(
            pretrained_predictions
        )


        # ----------------------------------------------------
        # 每个 Sample MSE
        # ----------------------------------------------------

        pretrained_batch_loss = (
            per_sample_masked_mse(

                pretrained_predictions,

                target_patches,

                mask

            )
        )


        random_batch_loss = (
            per_sample_masked_mse(

                random_predictions,

                target_patches,

                mask

            )
        )


        zero_batch_loss = (
            per_sample_masked_mse(

                zero_predictions,

                target_patches,

                mask

            )
        )


        pretrained_losses.extend(

            pretrained_batch_loss
            .cpu()
            .numpy()
            .tolist()

        )


        random_losses.extend(

            random_batch_loss
            .cpu()
            .numpy()
            .tolist()

        )


        zero_losses.extend(

            zero_batch_loss
            .cpu()
            .numpy()
            .tolist()

        )


        # ----------------------------------------------------
        # 保存第一条 Test EEG
        # ----------------------------------------------------

        if example is None:

            example = {

                "original":
                    eeg_batch[
                        0
                    ]
                    .cpu()
                    .numpy(),

                "masked":
                    masked_eeg[
                        0
                    ]
                    .cpu()
                    .numpy(),

                "target_patches":
                    target_patches[
                        0
                    ]
                    .cpu(),

                "prediction":
                    pretrained_predictions[
                        0
                    ]
                    .cpu(),

                "mask":
                    mask[
                        0
                    ]
                    .cpu(),

                "meta":
                    meta_batch[
                        0
                    ]
                    .numpy()

            }


        if (
            batch_index % 5 == 0
        ):

            print(

                f"Batch "
                f"{batch_index:02d}/"
                f"{len(test_loader):02d}"

            )


# ============================================================
# 15. NumPy
# ============================================================

pretrained_losses = np.asarray(
    pretrained_losses,
    dtype=np.float32
)

random_losses = np.asarray(
    random_losses,
    dtype=np.float32
)

zero_losses = np.asarray(
    zero_losses,
    dtype=np.float32
)


# ============================================================
# 16. 汇总结果
# ============================================================

pretrained_mean = float(
    pretrained_losses.mean()
)

pretrained_std = float(
    pretrained_losses.std()
)


random_mean = float(
    random_losses.mean()
)

random_std = float(
    random_losses.std()
)


zero_mean = float(
    zero_losses.mean()
)

zero_std = float(
    zero_losses.std()
)


# 相比 Zero baseline 降低多少误差

zero_improvement = (

    1.0

    - (
        pretrained_mean
        / zero_mean
    )

) * 100


# 相比 Random Model 降低多少误差

random_improvement = (

    1.0

    - (
        pretrained_mean
        / random_mean
    )

) * 100


print("\n")

print("=" * 70)

print("TEST RESULT")

print("=" * 70)


print(
    "\nPretrained Transformer:"
)

print(
    f"  MSE = "
    f"{pretrained_mean:.6f} "
    f"± {pretrained_std:.6f}"
)


print(
    "\nRandom Initialization:"
)

print(
    f"  MSE = "
    f"{random_mean:.6f} "
    f"± {random_std:.6f}"
)


print(
    "\nZero Baseline:"
)

print(
    f"  MSE = "
    f"{zero_mean:.6f} "
    f"± {zero_std:.6f}"
)


print(
    "\nPretrained vs Zero:"
)

print(
    f"  Error reduction = "
    f"{zero_improvement:.2f}%"
)


print(
    "\nPretrained vs Random:"
)

print(
    f"  Error reduction = "
    f"{random_improvement:.2f}%"
)


# ============================================================
# 17. 保存每个 Test Sample 的结果
# ============================================================

np.savez(

    SAMPLE_METRICS_PATH,

    pretrained_mse=pretrained_losses,

    random_mse=random_losses,

    zero_mse=zero_losses

)


# ============================================================
# 18. 保存 Summary CSV
# ============================================================

with open(

    SUMMARY_PATH,

    "w",

    newline="",

    encoding="utf-8"

) as file:


    writer = csv.writer(
        file
    )


    writer.writerow(

        [
            "method",
            "mean_mse",
            "std_mse"
        ]

    )


    writer.writerow(

        [
            "Pretrained Transformer",
            pretrained_mean,
            pretrained_std
        ]

    )


    writer.writerow(

        [
            "Random Initialization",
            random_mean,
            random_std
        ]

    )


    writer.writerow(

        [
            "Zero Baseline",
            zero_mean,
            zero_std
        ]

    )


# ============================================================
# 19. 构建 Reconstruction 示例
# ============================================================

target_patches = example[
    "target_patches"
].clone()


prediction = example[
    "prediction"
]


example_mask = example[
    "mask"
]


# 可见区域：
# 保留 Original
#
# Mask 区域：
# 替换成模型 Prediction

reconstructed_patches = (
    target_patches.clone()
)


reconstructed_patches[
    example_mask
] = prediction[
    example_mask
]


reconstructed = (

    reconstructed_patches

    .reshape(

        NUM_CHANNELS,

        NUM_PATCHES_PER_CHANNEL,

        PATCH_SIZE

    )

    .reshape(

        NUM_CHANNELS,

        TIME_POINTS

    )

    .numpy()

)


original = example[
    "original"
]


masked = example[
    "masked"
]


meta = example[
    "meta"
]


subject = int(
    meta[0]
)

run = int(
    meta[1]
)

start_time = float(
    meta[2]
)


# ============================================================
# 20. Reconstruction 图
# ============================================================

SFREQ = 160.0

times = (

    np.arange(
        TIME_POINTS
    )

    / SFREQ

)


fig, axes = plt.subplots(

    2,

    1,

    figsize=(14, 8),

    sharex=True

)


CHANNEL_NAMES = [
    "Fp1",
    "Fp2"
]


for channel in range(
    NUM_CHANNELS
):


    axes[channel].plot(

        times,

        original[
            channel
        ],

        label="Original",

        linewidth=1.4

    )


    axes[channel].plot(

        times,

        masked[
            channel
        ],

        label="Masked",

        alpha=0.65

    )


    axes[channel].plot(

        times,

        reconstructed[
            channel
        ],

        label="Pretrained Reconstruction",

        linewidth=1.2

    )


    axes[channel].set_title(

        f"{CHANNEL_NAMES[channel]} "
        "- Test Reconstruction"

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

    f"20-Subject Pretraining Test | "
    f"S{subject:03d} "
    f"R{run:02d} "
    f"Start {start_time:.1f}s"

)


plt.tight_layout()


plt.savefig(

    FIGURE_PATH,

    dpi=150

)


plt.show(
    block=True
)


# ============================================================
# 21. 最终输出
# ============================================================

print("\n")

print("=" * 70)

print("Evaluation 完成")

print("=" * 70)


print(
    "\nSummary:"
)

print(
    SUMMARY_PATH
)


print(
    "\nPer-sample Metrics:"
)

print(
    SAMPLE_METRICS_PATH
)


print(
    "\nReconstruction Figure:"
)

print(
    FIGURE_PATH
)
