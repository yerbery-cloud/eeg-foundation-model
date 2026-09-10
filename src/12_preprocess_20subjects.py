from pathlib import Path
import time

import mne
import numpy as np
from mne.datasets import eegbci


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "eegmmidb"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eegmmidb_fp1_fp2"
    / "subjects_20"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 实验参数
# ============================================================

SUBJECTS = list(
    range(1, 21)
)

RUNS = list(
    range(1, 15)
)

CHANNELS = [
    "Fp1",
    "Fp2"
]

LOW_FREQ = 1.0
HIGH_FREQ = 45.0

WINDOW_SECONDS = 4.0
STRIDE_SECONDS = 2.0

MAX_AMPLITUDE_UV = 500.0

DOWNLOAD_RETRIES = 3


# ============================================================
# 3. Window 函数
# ============================================================

def create_windows(
    data,
    sfreq
):

    window_samples = int(
        WINDOW_SECONDS * sfreq
    )

    stride_samples = int(
        STRIDE_SECONDS * sfreq
    )

    windows = []

    metadata = []

    stats = {
        "candidate": 0,
        "nan": 0,
        "flat": 0,
        "amplitude": 0
    }


    for start in range(
        0,
        data.shape[1] - window_samples + 1,
        stride_samples
    ):

        stats["candidate"] += 1

        end = (
            start
            + window_samples
        )

        window = data[
            :,
            start:end
        ]


        # -----------------------------------------------
        # QC 1：NaN / Inf
        # -----------------------------------------------

        if not np.isfinite(
            window
        ).all():

            stats["nan"] += 1
            continue


        # -----------------------------------------------
        # QC 2：Flat Channel
        # -----------------------------------------------

        channel_std = np.std(
            window,
            axis=1
        )

        if np.any(
            channel_std < 1e-8
        ):

            stats["flat"] += 1
            continue


        # -----------------------------------------------
        # QC 3：Amplitude
        # -----------------------------------------------

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

            stats["amplitude"] += 1
            continue


        # -----------------------------------------------
        # Z-score
        # -----------------------------------------------

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
            (window - mean)
            /
            (std + 1e-8)
        )

        windows.append(
            normalized.astype(
                np.float32
            )
        )

        # 保存窗口在原始 Run 中的起始时间
        metadata.append(
            start / sfreq
        )


    if len(windows) == 0:

        return (
            None,
            None,
            stats
        )


    windows = np.stack(
        windows
    )

    metadata = np.asarray(
        metadata,
        dtype=np.float32
    )


    return (
        windows,
        metadata,
        stats
    )


# ============================================================
# 4. 开始 Subject 循环
# ============================================================

print("=" * 65)
print("20-Subject EEG Preprocessing")
print("=" * 65)


for subject in SUBJECTS:

    subject_name = (
        f"S{subject:03d}"
    )

    data_path = (
        OUTPUT_DIR
        / f"{subject_name}_windows.npy"
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
    # 已完成 → 跳过
    # ========================================================

    if (
        data_path.exists()
        and metadata_path.exists()
        and qc_path.exists()
    ):

        print(
            f"\n✓ {subject_name} 已完成，跳过"
        )

        continue


    print("\n")
    print("=" * 65)

    print(
        f"正在处理 {subject_name}"
    )

    print("=" * 65)


    subject_windows = []

    subject_metadata = []


    total_candidate = 0
    total_nan = 0
    total_flat = 0
    total_amplitude = 0


    # ========================================================
    # Run
    # ========================================================

    for run in RUNS:

        print(
            f"  Run {run:02d}",
            end=""
        )


        # ====================================================
        # 下载（带 Retry）
        # ====================================================

        files = None


        for attempt in range(
            1,
            DOWNLOAD_RETRIES + 1
        ):

            try:

                files = eegbci.load_data(

                    subjects=[
                        subject
                    ],

                    runs=[
                        run
                    ],

                    path=RAW_DATA_DIR,

                    update_path=False
                )

                break


            except Exception as error:

                print(
                    f"\n    下载失败 "
                    f"({attempt}/{DOWNLOAD_RETRIES})"
                )

                print(
                    f"    {error}"
                )

                if (
                    attempt
                    < DOWNLOAD_RETRIES
                ):

                    time.sleep(
                        3
                    )


        if files is None:

            raise RuntimeError(
                f"{subject_name} "
                f"Run {run:02d} "
                f"连续下载失败"
            )


        # ====================================================
        # 读取
        # ====================================================

        raw = mne.io.read_raw_edf(

            files[0],

            preload=True,

            verbose=False
        )


        eegbci.standardize(
            raw
        )


        # ====================================================
        # Channel 检查
        # ====================================================

        missing = [

            channel

            for channel in CHANNELS

            if channel
            not in raw.ch_names

        ]


        if missing:

            raise RuntimeError(

                f"{subject_name} "
                f"Run {run:02d} "
                f"缺少通道：{missing}"

            )


        raw.pick(
            CHANNELS
        )


        # ====================================================
        # Filter
        # ====================================================

        raw.filter(

            l_freq=LOW_FREQ,

            h_freq=HIGH_FREQ,

            verbose=False
        )


        data = raw.get_data()

        sfreq = float(
            raw.info["sfreq"]
        )


        # ====================================================
        # Window + QC
        # ====================================================

        (
            windows,
            starts,
            stats
        ) = create_windows(
            data,
            sfreq
        )


        total_candidate += (
            stats["candidate"]
        )

        total_nan += (
            stats["nan"]
        )

        total_flat += (
            stats["flat"]
        )

        total_amplitude += (
            stats["amplitude"]
        )


        if windows is None:

            print(
                " → 0 windows"
            )

            continue


        # ====================================================
        # Metadata
        # ====================================================

        # 每一行：
        #
        # subject
        # run
        # start_time_seconds

        run_metadata = np.column_stack(
            [

                np.full(
                    windows.shape[0],
                    subject,
                    dtype=np.float32
                ),

                np.full(
                    windows.shape[0],
                    run,
                    dtype=np.float32
                ),

                starts

            ]
        )


        subject_windows.append(
            windows
        )

        subject_metadata.append(
            run_metadata
        )


        print(
            f" → "
            f"{windows.shape[0]} windows"
        )


    # ========================================================
    # Subject 合并
    # ========================================================

    subject_windows = np.concatenate(

        subject_windows,

        axis=0
    )


    subject_metadata = np.concatenate(

        subject_metadata,

        axis=0
    )


    # ========================================================
    # 保存
    # ========================================================

    np.save(
        data_path,
        subject_windows
    )

    np.save(
        metadata_path,
        subject_metadata
    )


    np.savez(

        qc_path,

        candidate=total_candidate,

        rejected_nan=total_nan,

        rejected_flat=total_flat,

        rejected_amplitude=total_amplitude,

        kept=subject_windows.shape[0]

    )


    print(
        f"\n✓ {subject_name} 完成"
    )


    print(
        "  Shape：",
        subject_windows.shape
    )


    print(
        "  Candidate：",
        total_candidate
    )


    print(
        "  Amplitude rejected：",
        total_amplitude
    )


    print(
        "  Kept：",
        subject_windows.shape[0]
    )


print("\n")

print("=" * 65)

print(
    "S001–S020 全部预处理完成"
)

print("=" * 65)