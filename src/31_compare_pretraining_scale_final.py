from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 0. Purpose
# ============================================================
#
# Final scaling summary:
#
# 20-subject pretraining
#        vs
# 60-subject pretraining
#
# Use ONLY fair comparisons:
#
# 1) Same-test EEGMMIDB reconstruction
#    - same subjects: S055-S060
#    - same windows
#    - same mask
#
# 2) EEGMAT Linear Probe
#    - same downstream dataset
#    - same subject split
#    - same frozen-encoder protocol
#
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
)

SCALING_DIR = (
    RESULT_ROOT
    / "scaling_20_vs_60"
)

SCALING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. Input files
# ============================================================

SAME_TEST_PATH = (
    SCALING_DIR
    / "same_test_reconstruction_summary.csv"
)


def first_existing(candidates):

    for path in candidates:

        if path.exists():
            return path

    raise FileNotFoundError(
        "找不到需要的结果文件。\n尝试过：\n"
        + "\n".join(str(p) for p in candidates)
    )


LINEAR20_PATH = first_existing(
    [
        RESULT_ROOT
        / "linear_probe"
        / "linear_probe_test_summary.csv",

        RESULT_ROOT
        / "linear_probe20"
        / "linear_probe20_test_summary.csv",

        RESULT_ROOT
        / "linear_probe20"
        / "linear_probe_test_summary.csv",
    ]
)


LINEAR60_PATH = first_existing(
    [
        RESULT_ROOT
        / "linear_probe60"
        / "linear_probe60_test_summary.csv",

        RESULT_ROOT
        / "linear_probe60"
        / "linear_probe_test_summary.csv",
    ]
)


# ============================================================
# 2. Output files
# ============================================================

SUMMARY_PATH = (
    SCALING_DIR
    / "final_scaling_summary.csv"
)


RECON_FIGURE_PATH = (
    SCALING_DIR
    / "scaling_same_test_reconstruction.png"
)


LINEAR_FIGURE_PATH = (
    SCALING_DIR
    / "scaling_linear_probe.png"
)


# ============================================================
# 3. Read same-test reconstruction result
# ============================================================

if not SAME_TEST_PATH.exists():

    raise FileNotFoundError(
        "找不到 32 号脚本生成的同测试集结果：\n"
        f"{SAME_TEST_PATH}\n\n"
        "请先运行：\n"
        "python src/32_compare_pretrain_same_test.py"
    )


recon20 = None
recon60 = None


with open(
    SAME_TEST_PATH,
    "r",
    encoding="utf-8-sig"
) as file:

    reader = csv.reader(file)

    rows = list(reader)


for row in rows[1:]:

    if len(row) < 3:
        continue

    metric = row[0].strip()

    if metric == "same_test_reconstruction_mse":

        recon20 = float(row[1])
        recon60 = float(row[2])


if recon20 is None or recon60 is None:

    raise RuntimeError(
        "same_test_reconstruction_summary.csv "
        "中找不到 same_test_reconstruction_mse"
    )


# ============================================================
# 4. Read metric-value summary CSV
# ============================================================

def read_metric_value_csv(path):

    result = {}

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.reader(file)

        rows = list(reader)


    # Typical format:
    #
    # metric,value
    # test_accuracy,0.x
    # test_macro_f1,0.x

    for row in rows[1:]:

        if len(row) < 2:
            continue

        key = row[0].strip()
        value = row[1].strip()

        result[key] = value


    return result


linear20 = read_metric_value_csv(
    LINEAR20_PATH
)


linear60 = read_metric_value_csv(
    LINEAR60_PATH
)


# ============================================================
# 5. Flexible metric extraction
# ============================================================

def get_float(
    result,
    keys
):

    for key in keys:

        if key in result:

            try:
                return float(
                    result[key]
                )

            except ValueError:
                pass

    raise RuntimeError(
        "找不到指标："
        + " / ".join(keys)
    )


accuracy20 = get_float(
    linear20,
    [
        "test_accuracy",
        "accuracy",
        "Test Accuracy",
    ]
)


accuracy60 = get_float(
    linear60,
    [
        "test_accuracy",
        "accuracy",
        "Test Accuracy",
    ]
)


f1_20 = get_float(
    linear20,
    [
        "test_macro_f1",
        "macro_f1",
        "Macro F1",
    ]
)


f1_60 = get_float(
    linear60,
    [
        "test_macro_f1",
        "macro_f1",
        "Macro F1",
    ]
)


# ============================================================
# 6. Calculate gains
# ============================================================

recon_abs_change = (
    recon60
    - recon20
)


recon_relative_improvement = (
    (
        recon20
        - recon60
    )
    / recon20
    * 100.0
)


accuracy_abs_gain = (
    accuracy60
    - accuracy20
)


accuracy_relative_gain = (
    accuracy_abs_gain
    / accuracy20
    * 100.0
)


f1_abs_gain = (
    f1_60
    - f1_20
)


f1_relative_gain = (
    f1_abs_gain
    / f1_20
    * 100.0
)


# ============================================================
# 7. Terminal summary
# ============================================================

print("=" * 82)

print(
    "FINAL PRETRAINING SCALING COMPARISON"
)

print("=" * 82)


print("\n")

print(
    f"{'Metric':<34}"
    f"{'20 subjects':>15}"
    f"{'60 subjects':>15}"
    f"{'Change':>15}"
)


print("-" * 79)


print(
    f"{'Same-test Reconstruction MSE':<34}"
    f"{recon20:>15.6f}"
    f"{recon60:>15.6f}"
    f"{recon_abs_change:>+15.6f}"
)


print(
    f"{'EEGMAT Linear Probe Accuracy':<34}"
    f"{accuracy20:>15.4f}"
    f"{accuracy60:>15.4f}"
    f"{accuracy_abs_gain:>+15.4f}"
)


print(
    f"{'EEGMAT Linear Probe Macro F1':<34}"
    f"{f1_20:>15.4f}"
    f"{f1_60:>15.4f}"
    f"{f1_abs_gain:>+15.4f}"
)


print("\n")

print("=" * 82)

print(
    "SCALING GAINS"
)

print("=" * 82)


print(
    "\nSame-test reconstruction MSE reduction:",
    f"{recon_relative_improvement:+.2f}%"
)


print(
    "Linear Probe Accuracy absolute gain:",
    f"{accuracy_abs_gain:+.4f}"
)


print(
    "Linear Probe Accuracy relative gain:",
    f"{accuracy_relative_gain:+.2f}%"
)


print(
    "Linear Probe Macro F1 absolute gain:",
    f"{f1_abs_gain:+.4f}"
)


print(
    "Linear Probe Macro F1 relative gain:",
    f"{f1_relative_gain:+.2f}%"
)


# ============================================================
# 8. Save final summary CSV
# ============================================================

with open(
    SUMMARY_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)


    writer.writerow(
        [
            "metric",
            "20_subjects",
            "60_subjects",
            "absolute_change",
            "relative_improvement_percent",
        ]
    )


    writer.writerow(
        [
            "same_test_reconstruction_mse",
            recon20,
            recon60,
            recon_abs_change,
            recon_relative_improvement,
        ]
    )


    writer.writerow(
        [
            "linear_probe_test_accuracy",
            accuracy20,
            accuracy60,
            accuracy_abs_gain,
            accuracy_relative_gain,
        ]
    )


    writer.writerow(
        [
            "linear_probe_test_macro_f1",
            f1_20,
            f1_60,
            f1_abs_gain,
            f1_relative_gain,
        ]
    )


# ============================================================
# 9. Figure 1:
#    Same-test reconstruction scaling
# ============================================================

labels = [
    "20 subjects",
    "60 subjects",
]


recon_values = [
    recon20,
    recon60,
]


plt.figure(
    figsize=(8, 6)
)


bars = plt.bar(
    labels,
    recon_values
)


plt.ylabel(
    "Masked Reconstruction MSE"
)


plt.title(
    "Pretraining Scale vs Same-Test Reconstruction"
)


plt.grid(
    axis="y",
    alpha=0.25
)


upper = (
    max(recon_values)
    * 1.25
)


plt.ylim(
    0,
    upper
)


for bar, value in zip(
    bars,
    recon_values
):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        value
        + upper * 0.025,

        f"{value:.4f}",

        ha="center",

        va="bottom",

        fontsize=12,

    )


plt.tight_layout()


plt.savefig(
    RECON_FIGURE_PATH,
    dpi=180
)


plt.close()


# ============================================================
# 10. Figure 2:
#     Linear Probe scaling
# ============================================================

x = np.arange(
    2
)


width = 0.32


plt.figure(
    figsize=(8.5, 6)
)


plt.bar(
    x - width / 2,

    [
        accuracy20,
        accuracy60,
    ],

    width,

    label="Accuracy"
)


plt.bar(
    x + width / 2,

    [
        f1_20,
        f1_60,
    ],

    width,

    label="Macro F1"
)


plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
    label="Chance Level"
)


plt.xticks(
    x,
    labels
)


plt.ylabel(
    "Score"
)


plt.ylim(
    0,
    0.75
)


plt.title(
    "Pretraining Scale vs EEGMAT Linear Probe"
)


plt.grid(
    axis="y",
    alpha=0.25
)


plt.legend()


# Data labels

values = [

    (
        x[0] - width / 2,
        accuracy20
    ),

    (
        x[1] - width / 2,
        accuracy60
    ),

    (
        x[0] + width / 2,
        f1_20
    ),

    (
        x[1] + width / 2,
        f1_60
    ),

]


for xpos, value in values:

    plt.text(

        xpos,

        value + 0.015,

        f"{value:.4f}",

        ha="center",

        va="bottom",

        fontsize=10,

    )


plt.tight_layout()


plt.savefig(
    LINEAR_FIGURE_PATH,
    dpi=180
)


plt.close()


# ============================================================
# 11. Final conclusion
# ============================================================

print("\n")

print("=" * 82)

print(
    "FINAL CONCLUSION"
)

print("=" * 82)


if (
    recon60 < recon20
    and f1_60 > f1_20
):

    print(

        "\n扩大预训练规模同时改善了："
        "\n1. 相同未见被试上的 EEG 重建能力"
        "\n2. EEGMAT 下游 Linear Probe 的迁移性能"

    )


elif f1_60 > f1_20:

    print(

        "\n扩大预训练规模明显改善了下游迁移性能，"
        "但重建指标未同步改善。"

    )


else:

    print(

        "\n当前实验没有观察到稳定的 scaling benefit，"
        "需要进一步分析数据规模、域差异或模型容量。"

    )


print("\n")

print(
    "Summary:"
)

print(
    SUMMARY_PATH
)


print(
    "\nReconstruction Figure:"
)

print(
    RECON_FIGURE_PATH
)


print(
    "\nLinear Probe Figure:"
)

print(
    LINEAR_FIGURE_PATH
)


print("\n")

print(
    "Scaling comparison 完成。"
)
