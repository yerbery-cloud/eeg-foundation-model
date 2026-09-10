from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eegmat_fp1_fp2"
    / "classification"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eegmat_cbramod"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 输入文件
# ============================================================

X_PATH = (
    SOURCE_DIR
    / "eegmat_X.npy"
)


Y_PATH = (
    SOURCE_DIR
    / "eegmat_y.npy"
)


METADATA_PATH = (
    SOURCE_DIR
    / "eegmat_metadata.npy"
)


SPLIT_PATH = (
    SOURCE_DIR
    / "eegmat_subject_split.npz"
)


# ============================================================
# 3. 输出文件
# ============================================================

OUTPUT_X_PATH = (
    OUTPUT_DIR
    / "eegmat_cbramod_X.npy"
)


OUTPUT_Y_PATH = (
    OUTPUT_DIR
    / "eegmat_cbramod_y.npy"
)


OUTPUT_METADATA_PATH = (
    OUTPUT_DIR
    / "eegmat_cbramod_metadata.npy"
)


OUTPUT_SPLIT_PATH = (
    OUTPUT_DIR
    / "eegmat_cbramod_split.npz"
)


# ============================================================
# 4. Sampling Config
# ============================================================

SOURCE_SFREQ = 160

TARGET_SFREQ = 200


SOURCE_TIMES = 640

TARGET_TIMES = 800


# ============================================================
# 5. 加载数据
# ============================================================

print("=" * 70)

print("Prepare EEGMAT for CBraMod")

print("=" * 70)


X = np.load(
    X_PATH
)


y = np.load(
    Y_PATH
)


metadata = np.load(
    METADATA_PATH
)


split = np.load(
    SPLIT_PATH
)


print(
    "\nOriginal X:",
    X.shape
)


print(
    "y:",
    y.shape
)


# ============================================================
# 6. 输入检查
# ============================================================

if X.shape[1:] != (
    2,
    SOURCE_TIMES
):

    raise RuntimeError(

        f"Original EEG shape 异常："
        f"{X.shape}"

    )


if not np.isfinite(
    X
).all():

    raise RuntimeError(
        "输入存在 NaN / Inf"
    )


# ============================================================
# 7. Resample
#
# 160 → 200 Hz
#
# 比例：
#
# 200 / 160
# =
# 5 / 4
# ============================================================

print(
    "\nResampling "
    "160 Hz → 200 Hz ..."
)


X_200 = resample_poly(

    X,

    up=5,

    down=4,

    axis=-1

)


# ============================================================
# 8. Shape Check
# ============================================================

print(
    "\nResampled X:",
    X_200.shape
)


if X_200.shape[1:] != (
    2,
    TARGET_TIMES
):

    raise RuntimeError(

        f"CBraMod EEG shape 错误："
        f"{X_200.shape}"

    )


if not np.isfinite(
    X_200
).all():

    raise RuntimeError(

        "Resample 后存在 NaN / Inf"

    )


# ============================================================
# 9. Float32
# ============================================================

X_200 = X_200.astype(
    np.float32
)


# ============================================================
# 10. 归一化检查
#
# 原来的 EEGMAT 已经：
#
# per-window
# per-channel
# Z-score
#
# 重采样以后只验证统计量，
# 暂时不再次 Z-score，
# 避免重复改变信号。
# ============================================================

means = X_200.mean(
    axis=-1
)


stds = X_200.std(
    axis=-1
)


print(
    "\nMean |mean|:",
    np.mean(
        np.abs(
            means
        )
    )
)


print(
    "Mean std:",
    np.mean(
        stds
    )
)


# ============================================================
# 11. 保存
# ============================================================

np.save(

    OUTPUT_X_PATH,

    X_200

)


np.save(

    OUTPUT_Y_PATH,

    y

)


np.save(

    OUTPUT_METADATA_PATH,

    metadata

)


np.savez(

    OUTPUT_SPLIT_PATH,

    train_indices=split[
        "train_indices"
    ],

    val_indices=split[
        "val_indices"
    ],

    test_indices=split[
        "test_indices"
    ],

    train_subjects=split[
        "train_subjects"
    ],

    val_subjects=split[
        "val_subjects"
    ],

    test_subjects=split[
        "test_subjects"
    ]

)


# ============================================================
# 12. Split Check
# ============================================================

print("\n")

print("=" * 70)

print("Split")

print("=" * 70)


print(
    "Train:",
    len(
        split[
            "train_indices"
        ]
    )
)


print(
    "Validation:",
    len(
        split[
            "val_indices"
        ]
    )
)


print(
    "Test:",
    len(
        split[
            "test_indices"
        ]
    )
)


# ============================================================
# 13. Class Check
# ============================================================

print("\n")

print("=" * 70)

print("Labels")

print("=" * 70)


print(
    "REST:",
    int(
        np.sum(
            y == 0
        )
    )
)


print(
    "TASK:",
    int(
        np.sum(
            y == 1
        )
    )
)


# ============================================================
# 14. Final
# ============================================================

print("\n")

print("=" * 70)

print(
    "CBraMod Dataset 准备完成"
)

print("=" * 70)


print(
    "\nX:"
)

print(
    OUTPUT_X_PATH
)


print(
    "\ny:"
)

print(
    OUTPUT_Y_PATH
)


print(
    "\nmetadata:"
)

print(
    OUTPUT_METADATA_PATH
)


print(
    "\nsplit:"
)

print(
    OUTPUT_SPLIT_PATH
)