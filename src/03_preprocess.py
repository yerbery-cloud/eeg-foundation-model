from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt
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

RESULT_DIR = PROJECT_ROOT / "results"


# 自动创建不存在的文件夹
PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 实验参数
# ============================================================

# 当前只测试 Subject 1 / Run 1
SUBJECT = 1
RUN = 1

# 和 LoongBrain 保持一致
CHANNELS = [
    "Fp1",
    "Fp2"
]

# EEG 带通滤波范围
LOW_FREQ = 1.0
HIGH_FREQ = 45.0

# 每个训练样本长度：4 秒
WINDOW_SECONDS = 4.0

# 每次向后移动：2 秒
STRIDE_SECONDS = 2.0

# 极端幅值 QC 阈值
# 如果一个窗口中出现超过 ±500 μV 的数据，
# 暂时认为这个窗口存在明显伪迹
MAX_AMPLITUDE_UV = 500.0


# ============================================================
# 3. 找到并读取 EEG 文件
# ============================================================

files = eegbci.load_data(
    subjects=[SUBJECT],
    runs=[RUN],
    path=RAW_DATA_DIR,
    update_path=False
)

edf_file = files[0]


print("正在读取 EEG 文件：")
print(edf_file)


raw = mne.io.read_raw_edf(
    edf_file,
    preload=True,
    verbose=False
)


# ============================================================
# 4. 标准化通道名称
# ============================================================

# EEGBCI 原始通道名称可能类似：
# Fp1.
# Fp2.
#
# standardize() 后变成：
# Fp1
# Fp2

eegbci.standardize(raw)


# ============================================================
# 5. 检查并提取 Fp1 / Fp2
# ============================================================

missing_channels = [
    ch
    for ch in CHANNELS
    if ch not in raw.ch_names
]


if missing_channels:
    raise RuntimeError(
        f"缺少需要的通道：{missing_channels}"
    )


raw.pick(CHANNELS)


print("\n===== 当前 EEG 信息 =====")

print("通道：")
print(raw.ch_names)

print("\n采样率：")
print(raw.info["sfreq"], "Hz")


# ============================================================
# 6. 保存一份原始 EEG
# ============================================================

# copy() 很重要：
# 保留未经滤波的 Raw EEG，
# 之后才能和滤波结果比较。

raw_original = raw.copy()


# ============================================================
# 7. 1–45 Hz 带通滤波
# ============================================================

raw_filtered = raw.copy()

raw_filtered.filter(
    l_freq=LOW_FREQ,
    h_freq=HIGH_FREQ,
    verbose=False
)


print(
    f"\n已完成 {LOW_FREQ}–{HIGH_FREQ} Hz 带通滤波"
)


# ============================================================
# 8. 转成 NumPy 数组
# ============================================================

data_raw = raw_original.get_data()

data_filtered = raw_filtered.get_data()

sfreq = float(
    raw.info["sfreq"]
)


print("\n===== EEG 数据 =====")

print("原始数据 shape：")
print(data_raw.shape)

print("\n滤波后 shape：")
print(data_filtered.shape)


# ============================================================
# 9. 绘制 Raw vs Filtered
# ============================================================

# 只显示前 10 秒
PLOT_SECONDS = 10

plot_samples = int(
    PLOT_SECONDS * sfreq
)

# 防止数据不足 10 秒
plot_samples = min(
    plot_samples,
    data_raw.shape[1]
)


# 时间轴
times = (
    np.arange(plot_samples)
    / sfreq
)


# MNE 中 EEG 默认单位是 V
# 为了方便观察，转换成 μV
raw_uv = (
    data_raw[:, :plot_samples]
    * 1e6
)

filtered_uv = (
    data_filtered[:, :plot_samples]
    * 1e6
)


# 创建两幅子图
fig, axes = plt.subplots(
    2,
    1,
    figsize=(13, 7),
    sharex=True
)


# ------------------------------------------------------------
# Fp1
# ------------------------------------------------------------

axes[0].plot(
    times,
    raw_uv[0],
    label="Raw"
)

axes[0].plot(
    times,
    filtered_uv[0],
    label="1-45 Hz filtered",
    alpha=0.8
)

axes[0].set_title(
    "Fp1: Raw vs Filtered"
)

axes[0].set_ylabel(
    "Amplitude (μV)"
)

axes[0].legend()

axes[0].grid(
    alpha=0.3
)


# ------------------------------------------------------------
# Fp2
# ------------------------------------------------------------

axes[1].plot(
    times,
    raw_uv[1],
    label="Raw"
)

axes[1].plot(
    times,
    filtered_uv[1],
    label="1-45 Hz filtered",
    alpha=0.8
)

axes[1].set_title(
    "Fp2: Raw vs Filtered"
)

axes[1].set_xlabel(
    "Time (s)"
)

axes[1].set_ylabel(
    "Amplitude (μV)"
)

axes[1].legend()

axes[1].grid(
    alpha=0.3
)


plt.suptitle(
    f"Subject {SUBJECT} - Run {RUN} - EEG Preprocessing"
)

plt.tight_layout()


# ============================================================
# 10. 保存滤波对比图
# ============================================================

figure_path = (
    RESULT_DIR
    / f"S{SUBJECT:03d}_R{RUN:02d}_raw_vs_filtered.png"
)


plt.savefig(
    figure_path,
    dpi=150
)


print("\n滤波对比图已保存：")
print(figure_path)


# 显示图片
plt.show(block=True)


# ============================================================
# 11. 设置切窗参数
# ============================================================

window_samples = int(
    WINDOW_SECONDS * sfreq
)

stride_samples = int(
    STRIDE_SECONDS * sfreq
)


print("\n===== Window 参数 =====")

print(
    "Window:",
    WINDOW_SECONDS,
    "秒"
)

print(
    "每个 Window:",
    window_samples,
    "samples"
)

print(
    "Stride:",
    STRIDE_SECONDS,
    "秒"
)

print(
    "每次移动:",
    stride_samples,
    "samples"
)


# ============================================================
# 12. 初始化切窗与 QC 统计
# ============================================================

windows = []


# 一共有多少候选窗口
total_window_count = 0

# 因 NaN / Inf 被删除
rejected_nan = 0

# 因通道几乎完全平坦被删除
rejected_flat = 0

# 因异常大幅值被删除
rejected_amplitude = 0


# ============================================================
# 13. 连续 EEG → 多个 4 秒窗口
# ============================================================

total_samples = data_filtered.shape[1]


for start in range(
    0,
    total_samples - window_samples + 1,
    stride_samples
):

    total_window_count += 1


    # 当前窗口结束位置
    end = start + window_samples


    # shape:
    # [2, 640]
    window = data_filtered[
        :,
        start:end
    ]


    # ========================================================
    # QC 1：检查 NaN / Inf
    # ========================================================

    if not np.isfinite(
        window
    ).all():

        rejected_nan += 1

        continue


    # ========================================================
    # QC 2：检查 Flat Channel
    # ========================================================

    # 分别计算 Fp1 和 Fp2 的标准差
    channel_std = np.std(
        window,
        axis=1
    )


    # 如果某个通道几乎完全没有变化，
    # 说明可能没有正常记录
    if np.any(
        channel_std < 1e-8
    ):

        rejected_flat += 1

        continue


    # ========================================================
    # QC 3：检查极端幅值
    # ========================================================

    # 找到当前窗口中最大的绝对幅值
    # MNE 数据单位为 V
    #
    # × 1e6 转换为 μV

    max_amplitude_uv = (
        np.max(
            np.abs(window)
        )
        * 1e6
    )


    # 例如：
    #
    # 正常：
    # 50 μV
    # 80 μV
    #
    # 明显异常：
    # 600 μV
    #
    # 当前暂时设置阈值为 500 μV

    if (
        max_amplitude_uv
        > MAX_AMPLITUDE_UV
    ):

        rejected_amplitude += 1

        continue


    # ========================================================
    # 14. 每个通道分别进行 Z-score Normalization
    # ========================================================

    # shape：
    # mean = [2, 1]

    mean = np.mean(
        window,
        axis=1,
        keepdims=True
    )


    # shape：
    # std = [2, 1]

    std = np.std(
        window,
        axis=1,
        keepdims=True
    )


    # Z-score：
    #
    # x' = (x - mean) / std
    #
    # 处理之后：
    #
    # mean ≈ 0
    # std  ≈ 1

    normalized_window = (
        (window - mean)
        /
        (std + 1e-8)
    )


    # 使用 float32
    # 可以减少之后训练模型时的内存占用

    normalized_window = (
        normalized_window.astype(
            np.float32
        )
    )


    windows.append(
        normalized_window
    )


# ============================================================
# 15. 输出 QC 统计
# ============================================================

print("\n===== QC 统计 =====")

print(
    "候选窗口数量：",
    total_window_count
)

print(
    "NaN / Inf 删除：",
    rejected_nan
)

print(
    "平坦通道删除：",
    rejected_flat
)

print(
    "异常幅值删除：",
    rejected_amplitude
)

print(
    "最终保留：",
    len(windows)
)


# ============================================================
# 16. List → NumPy Array
# ============================================================

# 正常情况下至少应该留下一个窗口
if len(windows) == 0:

    raise RuntimeError(
        "QC 后没有剩余有效窗口，请检查阈值或原始 EEG 数据。"
    )


windows = np.stack(
    windows,
    axis=0
)


# ============================================================
# 17. 输出最终数据 Shape
# ============================================================

print("\n===== 切窗完成 =====")

print(
    "所有 Window shape："
)

print(
    windows.shape
)


print(
    "\nShape 含义："
)

print(
    "[样本数量, 通道数, 时间点数]"
)


# ============================================================
# 18. 检查第一个样本
# ============================================================

first_window = windows[0]


print("\n===== 第一个样本检查 =====")

print(
    "第一个样本 shape："
)

print(
    first_window.shape
)


print(
    "\nFp1 mean：",
    first_window[0].mean()
)

print(
    "Fp1 std：",
    first_window[0].std()
)


print(
    "\nFp2 mean：",
    first_window[1].mean()
)

print(
    "Fp2 std：",
    first_window[1].std()
)


# ============================================================
# 19. 保存处理后的 EEG
# ============================================================

save_path = (
    PROCESSED_DIR
    / f"S{SUBJECT:03d}_R{RUN:02d}_windows.npy"
)


np.save(
    save_path,
    windows
)


print("\n===== 保存结果 =====")

print(
    "处理后的 EEG 已保存："
)

print(
    save_path
)


print(
    "\n最终保存的数据 shape：",
    windows.shape
)


print(
    "\n03_preprocess.py 执行完成！"
)