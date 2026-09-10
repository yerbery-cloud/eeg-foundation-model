from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "eegmmidb"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eegmmidb_fp1_fp2"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 数据参数
# ============================================================

# 第一阶段只测试 5 位被试
SUBJECTS = [1, 2, 3, 4, 5]

# EEGMMIDB 每个被试 14 个 Run
RUNS = list(range(1, 15))

CHANNELS = [
    "Fp1",
    "Fp2"
]

LOW_FREQ = 1.0
HIGH_FREQ = 45.0

WINDOW_SECONDS = 4.0
STRIDE_SECONDS = 2.0

MAX_AMPLITUDE_UV = 500.0


# ============================================================
# 3. 定义：把一段连续 EEG 切成窗口
# ============================================================

def create_windows(
    data,
    sfreq
):
    """
    输入：
        data:
            shape = [channels, time]

        sfreq:
            EEG 采样率

    输出：
        windows:
            shape = [N, channels, time]
    """

    window_samples = int(
        WINDOW_SECONDS * sfreq
    )

    stride_samples = int(
        STRIDE_SECONDS * sfreq
    )

    windows = []

    stats = {
        "total": 0,
        "nan": 0,
        "flat": 0,
        "amplitude": 0
    }


    for start in range(
        0,
        data.shape[1] - window_samples + 1,
        stride_samples
    ):

        stats["total"] += 1

        end = (
            start
            + window_samples
        )

        window = data[
            :,
            start:end
        ]


        # ----------------------------------------------------
        # QC 1：NaN / Inf
        # ----------------------------------------------------

        if not np.isfinite(
            window
        ).all():

            stats["nan"] += 1

            continue


        # ----------------------------------------------------
        # QC 2：Flat channel
        # ----------------------------------------------------

        channel_std = np.std(
            window,
            axis=1
        )

        if np.any(
            channel_std < 1e-8
        ):

            stats["flat"] += 1

            continue


        # ----------------------------------------------------
        # QC 3：异常大幅值
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Z-score normalization
        # ----------------------------------------------------

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

        normalized_window = (
            (window - mean)
            /
            (std + 1e-8)
        )


        windows.append(
            normalized_window.astype(
                np.float32
            )
        )


    # 如果一个窗口都没有保留下来
    if len(windows) == 0:

        return None, stats


    windows = np.stack(
        windows,
        axis=0
    )


    return windows, stats


# ============================================================
# 4. 总体统计
# ============================================================

all_windows = []

metadata = []


total_candidate = 0
total_rejected_nan = 0
total_rejected_flat = 0
total_rejected_amplitude = 0


# ============================================================
# 5. 遍历 Subject
# ============================================================

for subject in SUBJECTS:

    print("\n")
    print("=" * 60)
    print(
        f"正在处理 Subject {subject:03d}"
    )
    print("=" * 60)


    # --------------------------------------------------------
    # 下载该 Subject 的 14 个 Run
    # --------------------------------------------------------

    files = eegbci.load_data(
        subjects=[subject],
        runs=RUNS,
        path=RAW_DATA_DIR,
        update_path=False
    )


    # --------------------------------------------------------
    # 遍历每个 Run
    # --------------------------------------------------------

    for run_index, edf_file in enumerate(
        files,
        start=1
    ):

        print(
            f"\n处理 "
            f"S{subject:03d} "
            f"R{run_index:02d}"
        )


        # ----------------------------------------------------
        # 读取 EDF
        # ----------------------------------------------------

        raw = mne.io.read_raw_edf(
            edf_file,
            preload=True,
            verbose=False
        )


        # ----------------------------------------------------
        # 标准化 channel name
        # ----------------------------------------------------

        eegbci.standardize(
            raw
        )


        # ----------------------------------------------------
        # 检查是否存在需要的通道
        # ----------------------------------------------------

        missing_channels = [
            ch
            for ch in CHANNELS
            if ch not in raw.ch_names
        ]


        if missing_channels:

            print(
                "跳过：缺少通道",
                missing_channels
            )

            continue


        # ----------------------------------------------------
        # 只保留 Fp1 / Fp2
        # ----------------------------------------------------

        raw.pick(
            CHANNELS
        )


        # ----------------------------------------------------
        # 滤波
        # ----------------------------------------------------

        raw.filter(
            l_freq=LOW_FREQ,
            h_freq=HIGH_FREQ,
            verbose=False
        )


        # ----------------------------------------------------
        # NumPy
        # ----------------------------------------------------

        data = raw.get_data()

        sfreq = raw.info[
            "sfreq"
        ]


        # ----------------------------------------------------
        # 切窗 + QC
        # ----------------------------------------------------

        windows, stats = (
            create_windows(
                data,
                sfreq
            )
        )


        # ----------------------------------------------------
        # 累计 QC
        # ----------------------------------------------------

        total_candidate += (
            stats["total"]
        )

        total_rejected_nan += (
            stats["nan"]
        )

        total_rejected_flat += (
            stats["flat"]
        )

        total_rejected_amplitude += (
            stats["amplitude"]
        )


        if windows is None:

            print(
                "该 Run 没有有效窗口"
            )

            continue


        print(
            "保留窗口：",
            windows.shape[0]
        )


        # ----------------------------------------------------
        # 保存 Metadata
        # ----------------------------------------------------

        for window_index in range(
            windows.shape[0]
        ):

            metadata.append(
                [
                    subject,
                    run_index,
                    window_index
                ]
            )


        all_windows.append(
            windows
        )


# ============================================================
# 6. 合并所有数据
# ============================================================

all_windows = np.concatenate(
    all_windows,
    axis=0
)

metadata = np.asarray(
    metadata,
    dtype=np.int32
)


# ============================================================
# 7. 输出总体结果
# ============================================================

print("\n")
print("=" * 60)
print("批量预处理完成")
print("=" * 60)


print(
    "\n最终数据 shape："
)

print(
    all_windows.shape
)


print(
    "\n含义："
)

print(
    "[samples, channels, time]"
)


print(
    "\n候选窗口：",
    total_candidate
)

print(
    "NaN / Inf 删除：",
    total_rejected_nan
)

print(
    "Flat channel 删除：",
    total_rejected_flat
)

print(
    "异常幅值删除：",
    total_rejected_amplitude
)

print(
    "最终保留：",
    all_windows.shape[0]
)


# ============================================================
# 8. 保存
# ============================================================

data_path = (
    PROCESSED_DIR
    / "pilot_5subjects_windows.npy"
)

metadata_path = (
    PROCESSED_DIR
    / "pilot_5subjects_metadata.npy"
)


np.save(
    data_path,
    all_windows
)

np.save(
    metadata_path,
    metadata
)


print(
    "\nEEG 保存到："
)

print(
    data_path
)


print(
    "\nMetadata 保存到："
)

print(
    metadata_path
)


print(
    "\n程序执行完成！"
)