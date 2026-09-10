from pathlib import Path

import numpy as np


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eegmmidb_fp1_fp2"
)


SUBJECT_DIR = (
    BASE_DIR
    / "subjects_20"
)


OUTPUT_DIR = (
    BASE_DIR
    / "pretrain_20subjects"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Subject 划分
# ============================================================

TRAIN_SUBJECTS = list(
    range(1, 17)
)

VAL_SUBJECTS = [
    17,
    18
]

TEST_SUBJECTS = [
    19,
    20
]


ALL_SUBJECTS = (
    TRAIN_SUBJECTS
    + VAL_SUBJECTS
    + TEST_SUBJECTS
)


# ============================================================
# 3. 保存所有 Subject 数据
# ============================================================

all_windows = []

all_metadata = []


# QC 总统计
total_candidate = 0

total_rejected_nan = 0

total_rejected_flat = 0

total_rejected_amplitude = 0

total_kept = 0


# 保存每个被试样本数量
subject_counts = {}


print("=" * 65)

print("Merge 20 Subjects")

print("=" * 65)


# ============================================================
# 4. 遍历 20 位被试
# ============================================================

for subject in ALL_SUBJECTS:

    subject_name = (
        f"S{subject:03d}"
    )


    data_path = (
        SUBJECT_DIR
        / f"{subject_name}_windows.npy"
    )


    metadata_path = (
        SUBJECT_DIR
        / f"{subject_name}_metadata.npy"
    )


    qc_path = (
        SUBJECT_DIR
        / f"{subject_name}_qc.npz"
    )


    # ========================================================
    # 检查文件是否都存在
    # ========================================================

    if not data_path.exists():

        raise FileNotFoundError(
            f"缺少文件：\n{data_path}"
        )


    if not metadata_path.exists():

        raise FileNotFoundError(
            f"缺少文件：\n{metadata_path}"
        )


    if not qc_path.exists():

        raise FileNotFoundError(
            f"缺少文件：\n{qc_path}"
        )


    # ========================================================
    # 加载 EEG
    # ========================================================

    windows = np.load(
        data_path
    )


    metadata = np.load(
        metadata_path
    )


    qc = np.load(
        qc_path
    )


    # ========================================================
    # Shape 检查
    # ========================================================

    if windows.ndim != 3:

        raise RuntimeError(
            f"{subject_name} EEG 不是 3 维"
        )


    if windows.shape[1:] != (
        2,
        640
    ):

        raise RuntimeError(

            f"{subject_name} EEG shape 错误："
            f"{windows.shape}"

        )


    if metadata.shape[0] != windows.shape[0]:

        raise RuntimeError(

            f"{subject_name} "
            "EEG 和 Metadata 数量不一致"

        )


    if metadata.shape[1] != 3:

        raise RuntimeError(

            f"{subject_name} "
            f"Metadata shape 异常："
            f"{metadata.shape}"

        )


    # ========================================================
    # 检查 Metadata 中 Subject ID
    # ========================================================

    metadata_subjects = np.unique(
        metadata[:, 0]
    )


    if (
        len(metadata_subjects) != 1
        or int(metadata_subjects[0]) != subject
    ):

        raise RuntimeError(

            f"{subject_name} "
            "Metadata Subject ID 异常"

        )


    # ========================================================
    # NaN / Inf 二次检查
    # ========================================================

    if not np.isfinite(
        windows
    ).all():

        raise RuntimeError(

            f"{subject_name} "
            "仍然存在 NaN / Inf"

        )


    # ========================================================
    # 累计
    # ========================================================

    all_windows.append(
        windows.astype(
            np.float32
        )
    )


    all_metadata.append(
        metadata.astype(
            np.float32
        )
    )


    subject_counts[
        subject
    ] = windows.shape[0]


    # ========================================================
    # QC 统计
    # ========================================================

    candidate = int(
        qc["candidate"]
    )

    rejected_nan = int(
        qc["rejected_nan"]
    )

    rejected_flat = int(
        qc["rejected_flat"]
    )

    rejected_amplitude = int(
        qc["rejected_amplitude"]
    )

    kept = int(
        qc["kept"]
    )


    total_candidate += candidate

    total_rejected_nan += rejected_nan

    total_rejected_flat += rejected_flat

    total_rejected_amplitude += (
        rejected_amplitude
    )

    total_kept += kept


    print(
        f"{subject_name}: "
        f"{windows.shape[0]} windows"
    )


# ============================================================
# 5. 合并
# ============================================================

all_windows = np.concatenate(
    all_windows,
    axis=0
)


all_metadata = np.concatenate(
    all_metadata,
    axis=0
)


print("\n")

print("=" * 65)

print("合并完成")

print("=" * 65)


print(
    "EEG shape：",
    all_windows.shape
)


print(
    "Metadata shape：",
    all_metadata.shape
)


# ============================================================
# 6. 根据 Subject ID 生成索引
# ============================================================

subject_ids = (
    all_metadata[:, 0]
    .astype(np.int32)
)


train_mask = np.isin(
    subject_ids,
    TRAIN_SUBJECTS
)


val_mask = np.isin(
    subject_ids,
    VAL_SUBJECTS
)


test_mask = np.isin(
    subject_ids,
    TEST_SUBJECTS
)


train_indices = np.where(
    train_mask
)[0]


val_indices = np.where(
    val_mask
)[0]


test_indices = np.where(
    test_mask
)[0]


# ============================================================
# 7. 数据泄漏检查
# ============================================================

train_subject_set = set(
    subject_ids[
        train_indices
    ]
)


val_subject_set = set(
    subject_ids[
        val_indices
    ]
)


test_subject_set = set(
    subject_ids[
        test_indices
    ]
)


if (
    train_subject_set
    & val_subject_set
):

    raise RuntimeError(
        "Train 与 Validation 存在 Subject 泄漏"
    )


if (
    train_subject_set
    & test_subject_set
):

    raise RuntimeError(
        "Train 与 Test 存在 Subject 泄漏"
    )


if (
    val_subject_set
    & test_subject_set
):

    raise RuntimeError(
        "Validation 与 Test 存在 Subject 泄漏"
    )


# ============================================================
# 8. 检查索引是否覆盖全部数据
# ============================================================

total_split_samples = (

    len(train_indices)
    + len(val_indices)
    + len(test_indices)

)


if (
    total_split_samples
    != all_windows.shape[0]
):

    raise RuntimeError(
        "Train/Val/Test 没有完整覆盖数据"
    )


# ============================================================
# 9. 输出 Split
# ============================================================

print("\n===== Subject Split =====")


print(
    "Train Subjects：",
    TRAIN_SUBJECTS
)


print(
    "Validation Subjects：",
    VAL_SUBJECTS
)


print(
    "Test Subjects：",
    TEST_SUBJECTS
)


print("\n===== Sample Split =====")


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


print(
    "Total samples：",
    total_split_samples
)


# ============================================================
# 10. 输出 QC 总统计
# ============================================================

print("\n===== QC Summary =====")


print(
    "Candidate：",
    total_candidate
)


print(
    "NaN / Inf rejected：",
    total_rejected_nan
)


print(
    "Flat rejected：",
    total_rejected_flat
)


print(
    "Amplitude rejected：",
    total_rejected_amplitude
)


print(
    "Kept：",
    total_kept
)


if total_candidate > 0:

    rejection_rate = (

        (
            total_candidate
            - total_kept
        )

        / total_candidate

    )


    print(
        "Total rejection rate：",
        f"{rejection_rate * 100:.2f}%"
    )


# ============================================================
# 11. 每个 Subject 的样本数量
# ============================================================

print("\n===== Per-subject samples =====")


for subject in ALL_SUBJECTS:

    print(

        f"S{subject:03d}: "
        f"{subject_counts[subject]}"

    )


# ============================================================
# 12. 保存总数据
# ============================================================

DATA_PATH = (
    OUTPUT_DIR
    / "pretrain_20subjects_windows.npy"
)


METADATA_PATH = (
    OUTPUT_DIR
    / "pretrain_20subjects_metadata.npy"
)


SPLIT_PATH = (
    OUTPUT_DIR
    / "pretrain_20subjects_split.npz"
)


np.save(
    DATA_PATH,
    all_windows
)


np.save(
    METADATA_PATH,
    all_metadata
)


np.savez(

    SPLIT_PATH,

    train_indices=train_indices,

    val_indices=val_indices,

    test_indices=test_indices,

    train_subjects=np.asarray(
        TRAIN_SUBJECTS,
        dtype=np.int32
    ),

    val_subjects=np.asarray(
        VAL_SUBJECTS,
        dtype=np.int32
    ),

    test_subjects=np.asarray(
        TEST_SUBJECTS,
        dtype=np.int32
    )

)


# ============================================================
# 13. 保存 QC Summary
# ============================================================

QC_SUMMARY_PATH = (
    OUTPUT_DIR
    / "pretrain_20subjects_qc_summary.npz"
)


np.savez(

    QC_SUMMARY_PATH,

    candidate=total_candidate,

    rejected_nan=total_rejected_nan,

    rejected_flat=total_rejected_flat,

    rejected_amplitude=(
        total_rejected_amplitude
    ),

    kept=total_kept

)


# ============================================================
# 14. 最终输出
# ============================================================

print("\n")

print("=" * 65)

print("保存完成")

print("=" * 65)


print(
    "\nEEG："
)

print(
    DATA_PATH
)


print(
    "\nMetadata："
)

print(
    METADATA_PATH
)


print(
    "\nSplit："
)

print(
    SPLIT_PATH
)


print(
    "\nQC Summary："
)

print(
    QC_SUMMARY_PATH
)


print("\n")

print("=" * 65)

print(
    "13_merge_and_split.py 执行完成"
)

print("=" * 65)