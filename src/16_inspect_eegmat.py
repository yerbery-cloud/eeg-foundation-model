from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 项目目录
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
    / "raw"
    / "eegmat"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eegmat"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Subject 00
# ============================================================

REST_PATH = (
    DATA_DIR
    / "Subject00_1.edf"
)

TASK_PATH = (
    DATA_DIR
    / "Subject00_2.edf"
)


if not REST_PATH.exists():
    raise FileNotFoundError(
        f"找不到：{REST_PATH}"
    )

if not TASK_PATH.exists():
    raise FileNotFoundError(
        f"找不到：{TASK_PATH}"
    )


# ============================================================
# 3. 读取 EDF
# ============================================================

print("正在读取 Subject00 ...")


rest_raw = mne.io.read_raw_edf(
    REST_PATH,
    preload=True,
    verbose=False
)


task_raw = mne.io.read_raw_edf(
    TASK_PATH,
    preload=True,
    verbose=False
)


# ============================================================
# 4. 打印基本信息
# ============================================================

print("\n===== REST =====")

print(
    "Sampling Rate:",
    rest_raw.info["sfreq"]
)

print(
    "Channels:",
    len(rest_raw.ch_names)
)

print(
    "Duration:",
    rest_raw.times[-1],
    "s"
)

print(
    "Channel Names:"
)

print(
    rest_raw.ch_names
)


print("\n===== MENTAL ARITHMETIC =====")

print(
    "Sampling Rate:",
    task_raw.info["sfreq"]
)

print(
    "Channels:",
    len(task_raw.ch_names)
)

print(
    "Duration:",
    task_raw.times[-1],
    "s"
)

print(
    "Channel Names:"
)

print(
    task_raw.ch_names
)


# ============================================================
# 5. 自动寻找 Fp1 / Fp2
# ============================================================

def find_channel(
    raw,
    target
):

    target = target.lower()

    for channel in raw.ch_names:

        cleaned = (
            channel
            .lower()
            .replace(".", "")
            .replace(" ", "")
        )

        if target in cleaned:

            return channel


    raise RuntimeError(
        f"找不到通道 {target}"
    )


rest_fp1 = find_channel(
    rest_raw,
    "fp1"
)

rest_fp2 = find_channel(
    rest_raw,
    "fp2"
)


task_fp1 = find_channel(
    task_raw,
    "fp1"
)

task_fp2 = find_channel(
    task_raw,
    "fp2"
)


print("\n===== 找到的通道 =====")

print(
    "REST:",
    rest_fp1,
    rest_fp2
)

print(
    "TASK:",
    task_fp1,
    task_fp2
)


# ============================================================
# 6. 只保留 Fp1 / Fp2
# ============================================================

rest_raw.pick(
    [
        rest_fp1,
        rest_fp2
    ]
)


task_raw.pick(
    [
        task_fp1,
        task_fp2
    ]
)


# 统一名称

rest_raw.rename_channels(
    {
        rest_fp1: "Fp1",
        rest_fp2: "Fp2"
    }
)


task_raw.rename_channels(
    {
        task_fp1: "Fp1",
        task_fp2: "Fp2"
    }
)


print("\n统一后 REST channels:")

print(
    rest_raw.ch_names
)


print(
    "统一后 TASK channels:"
)

print(
    task_raw.ch_names
)


# ============================================================
# 7. 重采样为 160 Hz
# ============================================================

TARGET_SFREQ = 160.0


rest_raw.resample(
    TARGET_SFREQ
)


task_raw.resample(
    TARGET_SFREQ
)


print("\n===== Resample =====")

print(
    "REST:",
    rest_raw.info["sfreq"],
    "Hz"
)

print(
    "TASK:",
    task_raw.info["sfreq"],
    "Hz"
)


# ============================================================
# 8. 统一为 1–45 Hz
# ============================================================

rest_raw.filter(
    l_freq=1.0,
    h_freq=45.0,
    verbose=False
)


task_raw.filter(
    l_freq=1.0,
    h_freq=45.0,
    verbose=False
)


print(
    "\n✓ 1–45 Hz filtering 完成"
)


# ============================================================
# 9. 取 10 秒 EEG 看一下
# ============================================================

PLOT_SECONDS = 10

N_SAMPLES = int(
    PLOT_SECONDS
    * TARGET_SFREQ
)


rest_data = (
    rest_raw
    .get_data()
    [:, :N_SAMPLES]
)


task_data = (
    task_raw
    .get_data()
    [:, :N_SAMPLES]
)


# MNE:
# V → μV

rest_uv = (
    rest_data
    * 1e6
)


task_uv = (
    task_data
    * 1e6
)


times = (
    np.arange(
        N_SAMPLES
    )
    / TARGET_SFREQ
)


# ============================================================
# 10. 绘制 Fp1
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(13, 7),
    sharex=True
)


axes[0].plot(
    times,
    rest_uv[0]
)

axes[0].set_title(
    "Subject00 - REST - Fp1"
)

axes[0].set_ylabel(
    "Amplitude (μV)"
)

axes[0].grid(
    alpha=0.3
)


axes[1].plot(
    times,
    task_uv[0]
)

axes[1].set_title(
    "Subject00 - Mental Arithmetic - Fp1"
)

axes[1].set_xlabel(
    "Time (s)"
)

axes[1].set_ylabel(
    "Amplitude (μV)"
)

axes[1].grid(
    alpha=0.3
)


plt.suptitle(
    "EEGMAT Dataset Inspection"
)


plt.tight_layout()


FIGURE_PATH = (
    RESULT_DIR
    / "Subject00_rest_vs_task.png"
)


plt.savefig(
    FIGURE_PATH,
    dpi=150
)


print(
    "\nFigure 保存到："
)

print(
    FIGURE_PATH
)


plt.show(
    block=True
)


# ============================================================
# 11. 最终检查
# ============================================================

print("\n")

print("=" * 60)

print(
    "EEGMAT Inspection 完成"
)

print("=" * 60)


print(
    "REST shape:",
    rest_raw.get_data().shape
)


print(
    "TASK shape:",
    task_raw.get_data().shape
)


print(
    "Channels:",
    rest_raw.ch_names
)


print(
    "Sampling Rate:",
    rest_raw.info["sfreq"]
)