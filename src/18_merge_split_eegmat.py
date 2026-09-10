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


SUBJECT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eegmat_fp1_fp2"
    / "subjects"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eegmat_fp1_fp2"
    / "classification"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Subject Split
# ============================================================

TRAIN_SUBJECTS = list(
    range(0, 28)
)

VAL_SUBJECTS = list(
    range(28, 32)
)

TEST_SUBJECTS = list(
    range(32, 36)
)


ALL_SUBJECTS = (

    TRAIN_SUBJECTS
    + VAL_SUBJECTS
    + TEST_SUBJECTS

)


print("=" * 70)

print(
    "EEGMAT Merge + Subject-Level Split"
)

print("=" * 70)


# ============================================================
# 3. 保存所有数据
# ============================================================

all_X = []

all_y = []

all_metadata = []


# 每个 Subject 的统计
subject_stats = {}


# QC 总统计
qc_total = {

    "rest_candidate": 0,

    "rest_kept": 0,

    "rest_nan_inf": 0,

    "rest_flat": 0,

    "rest_amplitude": 0,

    "task_candidate": 0,

    "task_kept": 0,

    "task_nan_inf": 0,

    "task_flat": 0,

    "task_amplitude": 0

}


# ============================================================
# 4. 遍历 36 Subjects
# ============================================================

for subject in ALL_SUBJECTS:


    subject_name = (
        f"Subject{subject:02d}"
    )


    x_path = (

        SUBJECT_DIR
        / f"{subject_name}_X.npy"

    )


    y_path = (

        SUBJECT_DIR
        / f"{subject_name}_y.npy"

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
    # 文件检查
    # ========================================================

    for path in [

        x_path,
        y_path,
        metadata_path,
        qc_path

    ]:

        if not path.exists():

            raise FileNotFoundError(

                f"缺少文件：\n"
                f"{path}"

            )


    # ========================================================
    # 加载
    # ========================================================

    X = np.load(
        x_path
    )


    y = np.load(
        y_path
    )


    metadata = np.load(
        metadata_path
    )


    qc = np.load(
        qc_path
    )


    # ========================================================
    # 基础 Shape 检查
    # ========================================================

    if X.ndim != 3:

        raise RuntimeError(

            f"{subject_name} "
            f"X 维度异常："
            f"{X.shape}"

        )


    if X.shape[1:] != (
        2,
        640
    ):

        raise RuntimeError(

            f"{subject_name} "
            f"X shape 异常："
            f"{X.shape}"

        )


    if y.ndim != 1:

        raise RuntimeError(

            f"{subject_name} "
            f"y shape 异常："
            f"{y.shape}"

        )


    if metadata.ndim != 2:

        raise RuntimeError(

            f"{subject_name} "
            f"metadata 异常："
            f"{metadata.shape}"

        )


    if metadata.shape[1] != 4:

        raise RuntimeError(

            f"{subject_name} "
            f"metadata 列数异常："
            f"{metadata.shape}"

        )


    if not (

        X.shape[0]
        == y.shape[0]
        == metadata.shape[0]

    ):

        raise RuntimeError(

            f"{subject_name} "
            "X/y/metadata 数量不一致"

        )


    # ========================================================
    # NaN / Inf
    # ========================================================

    if not np.isfinite(
        X
    ).all():

        raise RuntimeError(

            f"{subject_name} "
            "X 存在 NaN / Inf"

        )


    # ========================================================
    # Label 检查
    # ========================================================

    unique_labels = set(
        np.unique(
            y
        ).tolist()
    )


    if not unique_labels.issubset(
        {
            0,
            1
        }
    ):

        raise RuntimeError(

            f"{subject_name} "
            f"出现非法 label："
            f"{unique_labels}"

        )


    # ========================================================
    # Metadata Subject 检查
    # ========================================================

    meta_subjects = np.unique(

        metadata[
            :,
            0
        ]

    )


    if (

        len(
            meta_subjects
        ) != 1

        or int(
            meta_subjects[0]
        ) != subject

    ):

        raise RuntimeError(

            f"{subject_name} "
            "Metadata Subject ID 错误"

        )


    # ========================================================
    # Metadata Label 检查
    # ========================================================

    metadata_labels = (

        metadata[
            :,
            1
        ]
        .astype(
            np.int64
        )

    )


    if not np.array_equal(

        metadata_labels,

        y.astype(
            np.int64
        )

    ):

        raise RuntimeError(

            f"{subject_name} "
            "metadata label 与 y 不一致"

        )


    # ========================================================
    # 当前 Subject 类别数量
    # ========================================================

    rest_count = int(
        np.sum(
            y == 0
        )
    )


    task_count = int(
        np.sum(
            y == 1
        )
    )


    subject_stats[
        subject
    ] = {

        "total":
            len(y),

        "rest":
            rest_count,

        "task":
            task_count

    }


    print(

        f"{subject_name}: "

        f"Total={len(y):3d} | "

        f"REST={rest_count:2d} | "

        f"TASK={task_count:2d}"

    )


    # ========================================================
    # 合并
    # ========================================================

    all_X.append(

        X.astype(
            np.float32
        )

    )


    all_y.append(

        y.astype(
            np.int64
        )

    )


    all_metadata.append(

        metadata.astype(
            np.float32
        )

    )


    # ========================================================
    # QC 累计
    # ========================================================

    for key in qc_total.keys():

        qc_total[
            key
        ] += int(
            qc[
                key
            ]
        )


# ============================================================
# 5. 合并 36 Subjects
# ============================================================

X = np.concatenate(

    all_X,

    axis=0

)


y = np.concatenate(

    all_y,

    axis=0

)


metadata = np.concatenate(

    all_metadata,

    axis=0

)


print("\n")

print("=" * 70)

print(
    "全部数据合并完成"
)

print("=" * 70)


print(
    "X shape:",
    X.shape
)


print(
    "y shape:",
    y.shape
)


print(
    "metadata shape:",
    metadata.shape
)


print(
    "dtype X:",
    X.dtype
)


print(
    "dtype y:",
    y.dtype
)


# ============================================================
# 6. 总类别数量
# ============================================================

total_rest = int(

    np.sum(
        y == 0
    )

)


total_task = int(

    np.sum(
        y == 1
    )

)


print("\n===== Overall Class Distribution =====")


print(
    "REST:",
    total_rest
)


print(
    "TASK:",
    total_task
)


print(
    "Total:",
    len(y)
)


if total_task > 0:

    print(
        "REST / TASK:",
        f"{total_rest / total_task:.3f}"
    )


# ============================================================
# 7. Subject ID
# ============================================================

subject_ids = (

    metadata[
        :,
        0
    ]

    .astype(
        np.int32
    )

)


# ============================================================
# 8. Split Mask
# ============================================================

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
# 9. Subject Leakage Check
# ============================================================

train_subject_set = set(

    subject_ids[
        train_indices
    ]
    .tolist()

)


val_subject_set = set(

    subject_ids[
        val_indices
    ]
    .tolist()

)


test_subject_set = set(

    subject_ids[
        test_indices
    ]
    .tolist()

)


if (

    train_subject_set
    & val_subject_set

):

    raise RuntimeError(

        "Train / Validation "
        "存在 Subject 泄漏"

    )


if (

    train_subject_set
    & test_subject_set

):

    raise RuntimeError(

        "Train / Test "
        "存在 Subject 泄漏"

    )


if (

    val_subject_set
    & test_subject_set

):

    raise RuntimeError(

        "Validation / Test "
        "存在 Subject 泄漏"

    )


# ============================================================
# 10. 完整覆盖检查
# ============================================================

total_split_samples = (

    len(
        train_indices
    )

    + len(
        val_indices
    )

    + len(
        test_indices
    )

)


if (

    total_split_samples
    != len(y)

):

    raise RuntimeError(

        "Train/Val/Test "
        "没有覆盖全部样本"

    )


# ============================================================
# 11. Split 类别统计函数
# ============================================================

def print_split_stats(
    name,
    indices
):

    split_y = y[
        indices
    ]


    rest = int(

        np.sum(
            split_y == 0
        )

    )


    task = int(

        np.sum(
            split_y == 1
        )

    )


    total = len(
        split_y
    )


    print(
        f"\n{name}"
    )


    print(
        "  Samples:",
        total
    )


    print(
        "  REST:",
        rest
    )


    print(
        "  TASK:",
        task
    )


    if task > 0:

        print(
            "  REST / TASK:",
            f"{rest / task:.3f}"
        )


    if total > 0:

        print(
            "  REST %:",
            f"{rest / total * 100:.2f}%"
        )


        print(
            "  TASK %:",
            f"{task / total * 100:.2f}%"
        )


# ============================================================
# 12. 输出 Split
# ============================================================

print("\n")

print("=" * 70)

print(
    "Subject-Level Split"
)

print("=" * 70)


print(
    "Train Subjects:"
)

print(
    TRAIN_SUBJECTS
)


print(
    "\nValidation Subjects:"
)

print(
    VAL_SUBJECTS
)


print(
    "\nTest Subjects:"
)

print(
    TEST_SUBJECTS
)


print_split_stats(

    "TRAIN",

    train_indices

)


print_split_stats(

    "VALIDATION",

    val_indices

)


print_split_stats(

    "TEST",

    test_indices

)


# ============================================================
# 13. QC Summary
# ============================================================

print("\n")

print("=" * 70)

print(
    "QC Summary"
)

print("=" * 70)


print(
    "\nREST Candidate:",
    qc_total[
        "rest_candidate"
    ]
)


print(
    "REST Kept:",
    qc_total[
        "rest_kept"
    ]
)


print(
    "REST NaN/Inf:",
    qc_total[
        "rest_nan_inf"
    ]
)


print(
    "REST Flat:",
    qc_total[
        "rest_flat"
    ]
)


print(
    "REST Amplitude rejected:",
    qc_total[
        "rest_amplitude"
    ]
)


print(
    "\nTASK Candidate:",
    qc_total[
        "task_candidate"
    ]
)


print(
    "TASK Kept:",
    qc_total[
        "task_kept"
    ]
)


print(
    "TASK NaN/Inf:",
    qc_total[
        "task_nan_inf"
    ]
)


print(
    "TASK Flat:",
    qc_total[
        "task_flat"
    ]
)


print(
    "TASK Amplitude rejected:",
    qc_total[
        "task_amplitude"
    ]
)


# ============================================================
# 14. 保存路径
# ============================================================

X_PATH = (

    OUTPUT_DIR
    / "eegmat_X.npy"

)


Y_PATH = (

    OUTPUT_DIR
    / "eegmat_y.npy"

)


METADATA_PATH = (

    OUTPUT_DIR
    / "eegmat_metadata.npy"

)


SPLIT_PATH = (

    OUTPUT_DIR
    / "eegmat_subject_split.npz"

)


QC_SUMMARY_PATH = (

    OUTPUT_DIR
    / "eegmat_qc_summary.npz"

)


# ============================================================
# 15. 保存
# ============================================================

np.save(

    X_PATH,

    X

)


np.save(

    Y_PATH,

    y

)


np.save(

    METADATA_PATH,

    metadata

)


np.savez(

    SPLIT_PATH,

    train_indices=(
        train_indices
    ),

    val_indices=(
        val_indices
    ),

    test_indices=(
        test_indices
    ),

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


np.savez(

    QC_SUMMARY_PATH,

    **qc_total

)


# ============================================================
# 16. 最终检查
# ============================================================

if X.shape[1:] != (
    2,
    640
):

    raise RuntimeError(

        f"最终 X shape 错误："
        f"{X.shape}"

    )


if not np.isfinite(
    X
).all():

    raise RuntimeError(

        "最终 X 存在 NaN/Inf"

    )


if not set(

    np.unique(
        y
    ).tolist()

).issubset(
    {
        0,
        1
    }
):

    raise RuntimeError(

        "最终 y 存在非法 Label"

    )


# ============================================================
# 17. 完成
# ============================================================

print("\n")

print("=" * 70)

print(
    "EEGMAT Dataset 构建完成"
)

print("=" * 70)


print(
    "\nX:"
)

print(
    X_PATH
)


print(
    "\ny:"
)

print(
    Y_PATH
)


print(
    "\nmetadata:"
)

print(
    METADATA_PATH
)


print(
    "\nsplit:"
)

print(
    SPLIT_PATH
)


print(
    "\nQC:"
)

print(
    QC_SUMMARY_PATH
)