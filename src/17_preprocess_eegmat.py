from pathlib import Path
import time

import mne
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


RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "eegmat"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eegmat_fp1_fp2"
    / "subjects"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Dataset Config
# ============================================================

SUBJECTS = list(
    range(36)
)


TARGET_CHANNELS = [
    "Fp1",
    "Fp2"
]


TARGET_SFREQ = 160.0


LOW_FREQ = 1.0

HIGH_FREQ = 45.0


# 每种状态取 60 秒

REST_SECONDS = 60.0

TASK_SECONDS = 60.0


# Window

WINDOW_SECONDS = 4.0

STRIDE_SECONDS = 2.0


# QC

MAX_AMPLITUDE_UV = 500.0


# ============================================================
# 3. Channel 名称匹配
# ============================================================

def find_channel(
    raw,
    target
):

    """
    例如：

    EEG Fp1
    EEG Fp2

    →

    Fp1
    Fp2
    """

    target_clean = (
        target
        .lower()
        .replace(" ", "")
    )


    for channel in raw.ch_names:

        cleaned = (
            channel
            .lower()
            .replace("eeg", "")
            .replace(".", "")
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )


        if cleaned == target_clean:

            return channel


    raise RuntimeError(

        f"找不到通道：{target}\n"
        f"已有通道：{raw.ch_names}"

    )


# ============================================================
# 4. 读取并处理 EDF
# ============================================================

def load_eeg(
    path
):

    raw = mne.io.read_raw_edf(

        path,

        preload=True,

        verbose=False

    )


    # --------------------------------------------------------
    # 找到 Fp1 / Fp2
    # --------------------------------------------------------

    fp1 = find_channel(
        raw,
        "Fp1"
    )

    fp2 = find_channel(
        raw,
        "Fp2"
    )


    raw.pick(
        [
            fp1,
            fp2
        ]
    )


    # --------------------------------------------------------
    # 统一 Channel Name
    # --------------------------------------------------------

    rename_map = {}


    if fp1 != "Fp1":

        rename_map[
            fp1
        ] = "Fp1"


    if fp2 != "Fp2":

        rename_map[
            fp2
        ] = "Fp2"


    if rename_map:

        raw.rename_channels(
            rename_map
        )


    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    raw.resample(

        TARGET_SFREQ,

        npad="auto"

    )


    # --------------------------------------------------------
    # 1–45 Hz
    # --------------------------------------------------------

    raw.filter(

        l_freq=LOW_FREQ,

        h_freq=HIGH_FREQ,

        verbose=False

    )


    return raw


# ============================================================
# 5. Window + QC
# ============================================================

def create_windows(
    segment,
    label,
    subject,
    file_id,
    segment_start_seconds
):

    """
    segment:

        [2, T]

    label:

        REST = 0
        TASK = 1

    metadata 每行：

        [
            subject,
            label,
            file_id,
            original_start_seconds
        ]
    """


    window_samples = int(

        WINDOW_SECONDS
        * TARGET_SFREQ

    )


    stride_samples = int(

        STRIDE_SECONDS
        * TARGET_SFREQ

    )


    windows = []

    labels = []

    metadata = []


    stats = {

        "candidate": 0,

        "nan_inf": 0,

        "flat": 0,

        "amplitude": 0,

        "kept": 0

    }


    for start in range(

        0,

        segment.shape[1]
        - window_samples
        + 1,

        stride_samples

    ):


        stats[
            "candidate"
        ] += 1


        end = (

            start
            + window_samples

        )


        window = segment[
            :,
            start:end
        ]


        # ====================================================
        # QC 1: NaN / Inf
        # ====================================================

        if not np.isfinite(
            window
        ).all():

            stats[
                "nan_inf"
            ] += 1

            continue


        # ====================================================
        # QC 2: Flat Channel
        # ====================================================

        channel_std = np.std(

            window,

            axis=1

        )


        if np.any(

            channel_std
            < 1e-8

        ):

            stats[
                "flat"
            ] += 1

            continue


        # ====================================================
        # QC 3: Amplitude
        # ====================================================

        max_amplitude_uv = (

            np.max(
                np.abs(window)
            )

            * 1e6

        )


        if (

            max_amplitude_uv
            > MAX_AMPLITUDE_UV

        ):

            stats[
                "amplitude"
            ] += 1

            continue


        # ====================================================
        # Z-score
        # ====================================================

        mean = np.mean(

            window,

            axis=1,

            keepdims=True

        )


        std = np.std(

            window,

            axis=1,

            keepdims=True

        )


        normalized = (

            window
            - mean

        ) / (

            std
            + 1e-8

        )


        normalized = normalized.astype(
            np.float32
        )


        windows.append(
            normalized
        )


        labels.append(
            label
        )


        # Window 在原始 EDF 中的起始时间

        start_seconds = (

            segment_start_seconds

            + (
                start
                / TARGET_SFREQ
            )

        )


        metadata.append(

            [
                subject,
                label,
                file_id,
                start_seconds
            ]

        )


        stats[
            "kept"
        ] += 1


    if len(
        windows
    ) == 0:

        return (
            None,
            None,
            None,
            stats
        )


    return (

        np.stack(
            windows
        ),

        np.asarray(
            labels,
            dtype=np.int64
        ),

        np.asarray(
            metadata,
            dtype=np.float32
        ),

        stats

    )


# ============================================================
# 6. 主循环
# ============================================================

print("=" * 70)

print(
    "EEGMAT 36-Subject Preprocessing"
)

print("=" * 70)


for subject in SUBJECTS:


    subject_name = (

        f"Subject{subject:02d}"

    )


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    rest_path = (

        RAW_DIR
        / f"{subject_name}_1.edf"

    )


    task_path = (

        RAW_DIR
        / f"{subject_name}_2.edf"

    )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    x_path = (

        OUTPUT_DIR
        / f"{subject_name}_X.npy"

    )


    y_path = (

        OUTPUT_DIR
        / f"{subject_name}_y.npy"

    )


    metadata_path = (

        OUTPUT_DIR
        / f"{subject_name}_metadata.npy"

    )


    qc_path = (

        OUTPUT_DIR
        / f"{subject_name}_qc.npz"

    )


    # ========================================================
    # 已完成则跳过
    # ========================================================

    if (

        x_path.exists()

        and y_path.exists()

        and metadata_path.exists()

        and qc_path.exists()

    ):

        print(

            f"\n✓ "
            f"{subject_name} "
            f"已完成，跳过"

        )

        continue


    print("\n")

    print("=" * 70)

    print(
        f"Processing "
        f"{subject_name}"
    )

    print("=" * 70)


    # ========================================================
    # 检查 EDF
    # ========================================================

    if not rest_path.exists():

        raise FileNotFoundError(

            f"找不到：\n"
            f"{rest_path}"

        )


    if not task_path.exists():

        raise FileNotFoundError(

            f"找不到：\n"
            f"{task_path}"

        )


    # ========================================================
    # 读取
    # ========================================================

    rest_raw = load_eeg(
        rest_path
    )


    task_raw = load_eeg(
        task_path
    )


    # ========================================================
    # NumPy
    # ========================================================

    rest_data = (
        rest_raw
        .get_data()
    )


    task_data = (
        task_raw
        .get_data()
    )


    # ========================================================
    # 检查长度
    # ========================================================

    rest_required = int(

        REST_SECONDS
        * TARGET_SFREQ

    )


    task_required = int(

        TASK_SECONDS
        * TARGET_SFREQ

    )


    if (

        rest_data.shape[1]
        < rest_required

    ):

        raise RuntimeError(

            f"{subject_name} "
            f"REST 不足 60 秒"

        )


    if (

        task_data.shape[1]
        < task_required

    ):

        raise RuntimeError(

            f"{subject_name} "
            f"TASK 不足 60 秒"

        )


    # ========================================================
    # REST：最后 60 秒
    # ========================================================

    rest_start_sample = (

        rest_data.shape[1]
        - rest_required

    )


    rest_segment = rest_data[

        :,

        rest_start_sample:

    ]


    rest_segment_start_seconds = (

        rest_start_sample

        / TARGET_SFREQ

    )


    # ========================================================
    # TASK：前 60 秒
    # ========================================================

    task_segment = task_data[

        :,

        :task_required

    ]


    task_segment_start_seconds = (
        0.0
    )


    # ========================================================
    # REST Window
    # ========================================================

    (

        rest_x,
        rest_y,
        rest_metadata,
        rest_stats

    ) = create_windows(

        segment=rest_segment,

        label=0,

        subject=subject,

        file_id=1,

        segment_start_seconds=(
            rest_segment_start_seconds
        )

    )


    # ========================================================
    # TASK Window
    # ========================================================

    (

        task_x,
        task_y,
        task_metadata,
        task_stats

    ) = create_windows(

        segment=task_segment,

        label=1,

        subject=subject,

        file_id=2,

        segment_start_seconds=(
            task_segment_start_seconds
        )

    )


    if rest_x is None:

        raise RuntimeError(

            f"{subject_name} "
            f"REST 全部被 QC 删除"

        )


    if task_x is None:

        raise RuntimeError(

            f"{subject_name} "
            f"TASK 全部被 QC 删除"

        )


    # ========================================================
    # 合并 REST + TASK
    # ========================================================

    X = np.concatenate(

        [
            rest_x,
            task_x
        ],

        axis=0

    )


    y = np.concatenate(

        [
            rest_y,
            task_y
        ],

        axis=0

    )


    metadata = np.concatenate(

        [
            rest_metadata,
            task_metadata
        ],

        axis=0

    )


    # ========================================================
    # 最终检查
    # ========================================================

    if X.shape[1:] != (
        2,
        640
    ):

        raise RuntimeError(

            f"最终 shape 异常："
            f"{X.shape}"

        )


    if not (

        X.shape[0]
        == y.shape[0]
        == metadata.shape[0]

    ):

        raise RuntimeError(

            "X / y / metadata "
            "数量不一致"

        )


    if not np.isfinite(
        X
    ).all():

        raise RuntimeError(

            "最终数据仍存在 "
            "NaN / Inf"

        )


    # ========================================================
    # 保存
    # ========================================================

    np.save(
        x_path,
        X
    )


    np.save(
        y_path,
        y
    )


    np.save(
        metadata_path,
        metadata
    )


    np.savez(

        qc_path,

        rest_candidate=(
            rest_stats[
                "candidate"
            ]
        ),

        rest_kept=(
            rest_stats[
                "kept"
            ]
        ),

        rest_nan_inf=(
            rest_stats[
                "nan_inf"
            ]
        ),

        rest_flat=(
            rest_stats[
                "flat"
            ]
        ),

        rest_amplitude=(
            rest_stats[
                "amplitude"
            ]
        ),

        task_candidate=(
            task_stats[
                "candidate"
            ]
        ),

        task_kept=(
            task_stats[
                "kept"
            ]
        ),

        task_nan_inf=(
            task_stats[
                "nan_inf"
            ]
        ),

        task_flat=(
            task_stats[
                "flat"
            ]
        ),

        task_amplitude=(
            task_stats[
                "amplitude"
            ]
        )

    )


    # ========================================================
    # 输出
    # ========================================================

    print(
        "\nREST:"
    )

    print(
        "  Candidate:",
        rest_stats[
            "candidate"
        ]
    )

    print(
        "  Kept:",
        rest_stats[
            "kept"
        ]
    )

    print(
        "  Amplitude rejected:",
        rest_stats[
            "amplitude"
        ]
    )


    print(
        "\nTASK:"
    )

    print(
        "  Candidate:",
        task_stats[
            "candidate"
        ]
    )

    print(
        "  Kept:",
        task_stats[
            "kept"
        ]
    )

    print(
        "  Amplitude rejected:",
        task_stats[
            "amplitude"
        ]
    )


    print(
        "\nFinal:"
    )

    print(
        "  X:",
        X.shape
    )

    print(
        "  y:",
        y.shape
    )

    print(
        "  REST samples:",
        np.sum(
            y == 0
        )
    )

    print(
        "  TASK samples:",
        np.sum(
            y == 1
        )
    )


    print(
        f"\n✓ "
        f"{subject_name} "
        f"完成"
    )


print("\n")

print("=" * 70)

print(
    "EEGMAT 36 Subjects "
    "预处理完成"
)

print("=" * 70)