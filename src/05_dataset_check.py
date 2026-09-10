from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eegmmidb_fp1_fp2"
)

RESULT_DIR = PROJECT_ROOT / "results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 数据文件路径
# ============================================================

DATA_PATH = (
    PROCESSED_DIR
    / "pilot_5subjects_windows.npy"
)

METADATA_PATH = (
    PROCESSED_DIR
    / "pilot_5subjects_metadata.npy"
)


# ============================================================
# 3. 检查文件是否存在
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"找不到 EEG 数据文件：\n{DATA_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"找不到 Metadata 文件：\n{METADATA_PATH}"
    )


# ============================================================
# 4. 加载数据
# ============================================================

print("正在加载 EEG 数据……")

data = np.load(DATA_PATH)

metadata = np.load(METADATA_PATH)


print("\n===== 基本信息 =====")

print("EEG shape：")
print(data.shape)

print("\nMetadata shape：")
print(metadata.shape)

print("\nEEG dtype：")
print(data.dtype)

print("\nMetadata dtype：")
print(metadata.dtype)


# ============================================================
# 5. 检查 EEG 和 metadata 是否一一对应
# ============================================================

print("\n===== 数据对应检查 =====")

n_samples = data.shape[0]
n_metadata = metadata.shape[0]


print("EEG 样本数量：", n_samples)
print("Metadata 数量：", n_metadata)


if n_samples != n_metadata:

    raise RuntimeError(
        "EEG 样本数量和 Metadata 数量不一致！"
    )

else:

    print("✓ EEG 与 Metadata 数量一致")


# ============================================================
# 6. 检查 EEG shape
# ============================================================

print("\n===== Shape 检查 =====")

expected_channels = 2
expected_timepoints = 640


if data.ndim != 3:

    raise RuntimeError(
        f"EEG 应该是 3 维数据，当前却是 {data.ndim} 维"
    )


if data.shape[1] != expected_channels:

    raise RuntimeError(
        f"预期 2 个通道，实际得到 {data.shape[1]}"
    )


if data.shape[2] != expected_timepoints:

    raise RuntimeError(
        f"预期 640 个时间点，实际得到 {data.shape[2]}"
    )


print(
    "✓ 数据 Shape 正确：",
    data.shape
)


# ============================================================
# 7. 检查 NaN / Inf
# ============================================================

print("\n===== NaN / Inf 检查 =====")


nan_count = np.isnan(
    data
).sum()

inf_count = np.isinf(
    data
).sum()


print("NaN 数量：", nan_count)
print("Inf 数量：", inf_count)


if nan_count == 0 and inf_count == 0:

    print("✓ 没有 NaN / Inf")

else:

    print("⚠ 数据中存在异常数值")


# ============================================================
# 8. 查看 Metadata 的含义
# ============================================================

print("\n===== Metadata 示例 =====")

print(
    "每一行含义："
)

print(
    "[subject, run, window_index]"
)


print("\n前 10 行：")

print(
    metadata[:10]
)


# ============================================================
# 9. 统计每个 Subject 的样本数量
# ============================================================

print("\n===== 每个 Subject 样本数量 =====")


subjects = np.unique(
    metadata[:, 0]
)


subject_counts = {}


for subject in subjects:

    count = np.sum(
        metadata[:, 0] == subject
    )

    subject_counts[
        int(subject)
    ] = int(count)

    print(
        f"Subject {int(subject):03d}: "
        f"{count} windows"
    )


# ============================================================
# 10. 检查每个 Run 的样本数量
# ============================================================

print("\n===== 每个 Subject / Run 的窗口数量 =====")


for subject in subjects:

    print(
        f"\nSubject {int(subject):03d}"
    )

    subject_metadata = metadata[
        metadata[:, 0] == subject
    ]

    runs = np.unique(
        subject_metadata[:, 1]
    )


    for run in runs:

        count = np.sum(
            (metadata[:, 0] == subject)
            &
            (metadata[:, 1] == run)
        )

        print(
            f"  Run {int(run):02d}: "
            f"{count}"
        )


# ============================================================
# 11. 检查 Z-score
# ============================================================

print("\n===== Z-score 检查 =====")


# data shape:
# [samples, channels, time]
#
# 沿着 time 维度计算
# 每个窗口、每个通道的 mean/std

sample_means = np.mean(
    data,
    axis=2
)

sample_stds = np.std(
    data,
    axis=2
)


print(
    "所有窗口 mean 的平均值：",
    sample_means.mean()
)

print(
    "所有窗口 |mean| 最大值：",
    np.abs(
        sample_means
    ).max()
)

print(
    "所有窗口 std 的平均值：",
    sample_stds.mean()
)

print(
    "所有窗口 std 最小值：",
    sample_stds.min()
)

print(
    "所有窗口 std 最大值：",
    sample_stds.max()
)


# ============================================================
# 12. 查看数据整体范围
# ============================================================

print("\n===== 标准化后数据范围 =====")

print(
    "最小值：",
    data.min()
)

print(
    "最大值：",
    data.max()
)

print(
    "平均值：",
    data.mean()
)

print(
    "标准差：",
    data.std()
)


# ============================================================
# 13. 随机抽取几个样本
# ============================================================

RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)

sample_indices = rng.choice(
    n_samples,
    size=min(5, n_samples),
    replace=False
)


print("\n===== 随机抽取样本 =====")

print(
    "抽取 index：",
    sample_indices
)


# ============================================================
# 14. 绘制随机样本
# ============================================================

sfreq = 160.0

times = (
    np.arange(
        expected_timepoints
    )
    / sfreq
)


fig, axes = plt.subplots(
    len(sample_indices),
    1,
    figsize=(13, 12),
    sharex=True
)


# 防止只有一个样本时 axes 不是数组
if len(sample_indices) == 1:
    axes = [axes]


for ax, sample_index in zip(
    axes,
    sample_indices
):

    sample = data[
        sample_index
    ]

    meta = metadata[
        sample_index
    ]


    subject = int(
        meta[0]
    )

    run = int(
        meta[1]
    )

    window_index = int(
        meta[2]
    )


    # 为了避免 Fp1/Fp2 完全重叠，
    # 将 Fp2 在纵轴方向人工下移 6 个单位。
    #
    # 注意：
    # 这里只是为了画图，
    # 不会修改真正的数据。

    ax.plot(
        times,
        sample[0],
        label="Fp1"
    )

    ax.plot(
        times,
        sample[1] - 6,
        label="Fp2 (offset)"
    )


    ax.set_title(
        f"Sample {sample_index} | "
        f"S{subject:03d} "
        f"R{run:02d} "
        f"Window {window_index}"
    )

    ax.set_ylabel(
        "Normalized amplitude"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        loc="upper right"
    )


axes[-1].set_xlabel(
    "Time (s)"
)


plt.suptitle(
    "Random EEG Windows After Preprocessing"
)

plt.tight_layout()


figure_path = (
    RESULT_DIR
    / "pilot_5subjects_dataset_check.png"
)


plt.savefig(
    figure_path,
    dpi=150
)


print("\n随机 EEG 图已保存：")
print(figure_path)


plt.show(block=True)


# ============================================================
# 15. 最终总结
# ============================================================

print("\n")
print("=" * 60)
print("Dataset Check 完成")
print("=" * 60)

print(
    "EEG shape：",
    data.shape
)

print(
    "Subjects：",
    subjects
)

print(
    "Samples：",
    n_samples
)

print(
    "\n如果上面没有报错，说明数据集基本结构正常。"
)