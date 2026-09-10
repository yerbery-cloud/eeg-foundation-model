from pathlib import Path

import mne
from mne.datasets import eegbci


# =========================
# 1. 找到项目目录
# =========================

# 当前文件：
# D:/eeg_foundation/src/02_inspect.py
#
# .parent       -> D:/eeg_foundation/src
# .parent.parent -> D:/eeg_foundation

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "eegmmidb"


# =========================
# 2. 找到 Subject 1 / Run 1
# =========================

# Subject 1
# Run 1：睁眼静息
files = eegbci.load_data(
    subjects=[1],
    runs=[1],
    path=DATA_DIR,
    update_path=False
)

# files 是一个文件路径列表
# 因为这里只有一个被试、一个 run，
# 所以取第 0 个文件即可
edf_file = files[0]


print("正在读取 EEG 文件：")
print(edf_file)


# =========================
# 3. 读取 EDF
# =========================

raw = mne.io.read_raw_edf(
    edf_file,
    preload=True,
    verbose=False
)


# =========================
# 4. 查看原始 EEG 基本信息
# =========================

print("\n===== 原始 EEG 基本信息 =====")

print(raw)

print("\n采样率：")
print(raw.info["sfreq"])

print("\n原始通道数量：")
print(len(raw.ch_names))

print("\n原始通道名称：")
print(raw.ch_names)


# =========================
# 5. 标准化 EEGBCI 通道名称
# =========================

# EEGBCI 原始通道名可能类似：
# Fp1.
# Fp2.
#
# standardize() 会转换成更标准的：
# Fp1
# Fp2

eegbci.standardize(raw)

print("\n===== 标准化后的通道名称 =====")
print(raw.ch_names)


# =========================
# 6. 只保留 Fp1 和 Fp2
# =========================

# LoongBrain 的 EEG 正好只有：
# Fp1、Fp2
#
# 因此公开数据也只保留这两个通道，
# 让以后预训练模型和真实设备输入保持一致。

raw.pick(["Fp1", "Fp2"])


print("\n===== 提取 Fp1 / Fp2 后 =====")

print(raw)

print("\n当前通道数量：")
print(len(raw.ch_names))

print("\n当前通道：")
print(raw.ch_names)


# =========================
# 7. 转换成 NumPy 数组
# =========================

data = raw.get_data()

print("\n===== EEG NumPy 数据 =====")

print("数据 shape：")
print(data.shape)

print("\nshape 含义：")
print("[通道数, 时间采样点数]")


# =========================
# 8. 打印一些辅助信息
# =========================

sfreq = raw.info["sfreq"]

duration = data.shape[1] / sfreq

print("\n采样率：", sfreq, "Hz")
print("总采样点数：", data.shape[1])
print("EEG 时长：", duration, "秒")


# =========================
# 9. 绘制 EEG
# =========================

print("\n即将打开 EEG 波形窗口……")

# =========================
# 9. 绘制前 10 秒 EEG
# =========================

import numpy as np
import matplotlib.pyplot as plt


# 显示多少秒
plot_duration = 10

# 10 秒一共有多少采样点
n_samples = int(plot_duration * sfreq)

# 只取前 10 秒
plot_data = data[:, :n_samples]

# EEG 在 MNE 中默认单位是 V
# 为了方便观察，转换成 μV
plot_data_uv = plot_data * 1e6

# 构造时间轴
# 例如：
# 0, 1/160, 2/160, 3/160 ...
times = np.arange(n_samples) / sfreq


# 创建两个图
fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 6),
    sharex=True
)


# Fp1
axes[0].plot(times, plot_data_uv[0])
axes[0].set_title("Fp1 - Raw EEG")
axes[0].set_ylabel("Amplitude (μV)")
axes[0].grid(alpha=0.3)


# Fp2
axes[1].plot(times, plot_data_uv[1])
axes[1].set_title("Fp2 - Raw EEG")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude (μV)")
axes[1].grid(alpha=0.3)


plt.suptitle("Subject 1 - Run 1 - Raw EEG")
plt.tight_layout()


# 保存图片
RESULT_DIR = PROJECT_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

save_path = RESULT_DIR / "subject01_run01_raw_eeg.png"

plt.savefig(
    save_path,
    dpi=150
)

print("\nEEG 图片已保存到：")
print(save_path)


# 显示窗口，并阻止程序立即退出
plt.show(block=True)