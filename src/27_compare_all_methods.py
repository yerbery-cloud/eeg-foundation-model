from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径
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

OUTPUT_DIR = (
    RESULT_ROOT
    / "final_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 六种方法结果
# ============================================================

METHOD_FILES = {

    "Random Probe": (
        RESULT_ROOT
        / "random_probe"
        / "random_probe_test_summary.csv"
    ),

    "Linear Probe": (
        RESULT_ROOT
        / "linear_probe"
        / "linear_probe_test_summary.csv"
    ),

    "LoRA": (
        RESULT_ROOT
        / "lora"
        / "lora_test_summary.csv"
    ),

    "Full Fine-tuning": (
        RESULT_ROOT
        / "full_finetune"
        / "full_finetune_test_summary.csv"
    ),

    "EEGNet": (
        RESULT_ROOT
        / "eegnet"
        / "eegnet_test_summary.csv"
    ),

    "CBraMod": (
        RESULT_ROOT
        / "cbramod"
        / "cbramod_test_summary.csv"
    )
}


# ============================================================
# 3. 输出
# ============================================================

CSV_PATH = (
    OUTPUT_DIR
    / "final_method_comparison.csv"
)

PERFORMANCE_PATH = (
    OUTPUT_DIR
    / "final_performance_comparison.png"
)

PARAMETER_PATH = (
    OUTPUT_DIR
    / "final_parameter_comparison.png"
)

EFFICIENCY_PATH = (
    OUTPUT_DIR
    / "final_parameter_efficiency.png"
)


# ============================================================
# 4. CSV Reader
# ============================================================

def read_summary(path):

    if not path.exists():

        raise FileNotFoundError(
            f"找不到：\n{path}"
        )

    result = {}

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.reader(file)

        rows = list(reader)

    for row in rows[1:]:

        if len(row) < 2:
            continue

        result[
            row[0].strip()
        ] = row[1].strip()

    return result


# ============================================================
# 5. 安全数字转换
# ============================================================

def get_float(
    result,
    key,
    default=np.nan
):

    try:
        return float(
            result[key]
        )

    except (
        KeyError,
        ValueError,
        TypeError
    ):

        return default


def get_int(
    result,
    key,
    default=0
):

    try:
        return int(
            float(
                result[key]
            )
        )

    except (
        KeyError,
        ValueError,
        TypeError
    ):

        return default


# ============================================================
# 6. 加载
# ============================================================

results = []


for method, path in METHOD_FILES.items():

    summary = read_summary(
        path
    )

    trainable = get_int(
        summary,
        "trainable_parameters"
    )

    total = get_int(
        summary,
        "total_parameters",
        trainable
    )

    results.append(
        {
            "method":
                method,

            "total_parameters":
                total,

            "trainable_parameters":
                trainable,

            "best_epoch":
                get_int(
                    summary,
                    "best_epoch"
                ),

            "val_accuracy":
                get_float(
                    summary,
                    "val_accuracy"
                ),

            "val_macro_f1":
                get_float(
                    summary,
                    "val_macro_f1"
                ),

            "test_accuracy":
                get_float(
                    summary,
                    "test_accuracy"
                ),

            "test_balanced_accuracy":
                get_float(
                    summary,
                    "test_balanced_accuracy"
                ),

            "test_macro_f1":
                get_float(
                    summary,
                    "test_macro_f1"
                )
        }
    )


# ============================================================
# 7. 核心指标检查
# ============================================================

for row in results:

    for key in [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1"
    ]:

        if np.isnan(
            row[key]
        ):

            raise RuntimeError(
                f"{row['method']} "
                f"缺少 {key}"
            )


# ============================================================
# 8. 保存最终 CSV
# ============================================================

fieldnames = [
    "method",
    "total_parameters",
    "trainable_parameters",
    "best_epoch",
    "val_accuracy",
    "val_macro_f1",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1"
]


with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# 9. 终端输出
# ============================================================

print("=" * 85)

print(
    "FINAL EEGMAT COMPARISON"
)

print("=" * 85)


print(
    f"\n{'Method':<22}"
    f"{'Trainable':>13}"
    f"{'Accuracy':>12}"
    f"{'Balanced':>12}"
    f"{'Macro F1':>12}"
)


print("-" * 72)


for row in results:

    print(
        f"{row['method']:<22}"

        f"{row['trainable_parameters']:>13,d}"

        f"{row['test_accuracy']:>12.4f}"

        f"{row['test_balanced_accuracy']:>12.4f}"

        f"{row['test_macro_f1']:>12.4f}"
    )


# ============================================================
# 10. Best 方法
# ============================================================

best_accuracy = max(
    results,
    key=lambda x:
    x["test_accuracy"]
)


best_balanced = max(
    results,
    key=lambda x:
    x["test_balanced_accuracy"]
)


best_f1 = max(
    results,
    key=lambda x:
    x["test_macro_f1"]
)


print("\n")

print("=" * 85)

print("BEST RESULTS")

print("=" * 85)


print(
    "\nBest Accuracy:",
    best_accuracy["method"],
    f"{best_accuracy['test_accuracy']:.4f}"
)


print(
    "Best Balanced Accuracy:",
    best_balanced["method"],
    f"{best_balanced['test_balanced_accuracy']:.4f}"
)


print(
    "Best Macro F1:",
    best_f1["method"],
    f"{best_f1['test_macro_f1']:.4f}"
)


# ============================================================
# 11. 预训练收益
# ============================================================

result_dict = {
    row["method"]: row
    for row in results
}


random_probe = (
    result_dict[
        "Random Probe"
    ]
)


linear_probe = (
    result_dict[
        "Linear Probe"
    ]
)


print("\n")

print("=" * 85)

print("PRETRAINING GAIN")

print("=" * 85)


print(
    "\nAccuracy Gain:",
    f"{linear_probe['test_accuracy'] - random_probe['test_accuracy']:+.4f}"
)


print(
    "Balanced Accuracy Gain:",
    f"{linear_probe['test_balanced_accuracy'] - random_probe['test_balanced_accuracy']:+.4f}"
)


print(
    "Macro F1 Gain:",
    f"{linear_probe['test_macro_f1'] - random_probe['test_macro_f1']:+.4f}"
)


# ============================================================
# 12. LoRA 参数效率
# ============================================================

lora = result_dict[
    "LoRA"
]


full_ft = result_dict[
    "Full Fine-tuning"
]


reduction = (

    1

    - (
        lora[
            "trainable_parameters"
        ]

        / full_ft[
            "trainable_parameters"
        ]
    )

) * 100


print("\n")

print("=" * 85)

print("LORA PARAMETER EFFICIENCY")

print("=" * 85)


print(
    "\nLoRA Params:",
    f"{lora['trainable_parameters']:,}"
)


print(
    "Full FT Params:",
    f"{full_ft['trainable_parameters']:,}"
)


print(
    "Parameter Reduction:",
    f"{reduction:.2f}%"
)


print(
    "LoRA Macro F1:",
    f"{lora['test_macro_f1']:.4f}"
)


print(
    "Full FT Macro F1:",
    f"{full_ft['test_macro_f1']:.4f}"
)


# ============================================================
# 13. 性能图
# ============================================================

methods = [
    row["method"]
    for row in results
]


accuracy = [
    row["test_accuracy"]
    for row in results
]


balanced = [
    row["test_balanced_accuracy"]
    for row in results
]


macro_f1 = [
    row["test_macro_f1"]
    for row in results
]


x = np.arange(
    len(methods)
)


width = 0.25


plt.figure(
    figsize=(13, 7)
)


plt.bar(
    x - width,
    accuracy,
    width,
    label="Accuracy"
)


plt.bar(
    x,
    balanced,
    width,
    label="Balanced Accuracy"
)


plt.bar(
    x + width,
    macro_f1,
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
    methods,
    rotation=20,
    ha="right"
)


plt.ylabel(
    "Score"
)


plt.ylim(
    0,
    1
)


plt.title(
    "EEGMAT Final Method Comparison"
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    PERFORMANCE_PATH,
    dpi=180
)


plt.close()


# ============================================================
# 14. 参数图
# ============================================================

parameters = [
    max(
        row[
            "trainable_parameters"
        ],
        1
    )
    for row in results
]


plt.figure(
    figsize=(12, 7)
)


bars = plt.bar(
    methods,
    parameters
)


plt.yscale(
    "log"
)


plt.ylabel(
    "Trainable Parameters (log scale)"
)


plt.title(
    "Final Trainable Parameter Comparison"
)


plt.xticks(
    rotation=20,
    ha="right"
)


plt.grid(
    axis="y",
    alpha=0.3
)


for bar, value in zip(
    bars,
    parameters
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        value * 1.15,

        f"{value:,}",

        ha="center",

        fontsize=9
    )


plt.tight_layout()


plt.savefig(
    PARAMETER_PATH,
    dpi=180
)


plt.close()


# ============================================================
# 15. 参数效率图
# ============================================================

plt.figure(
    figsize=(10, 7)
)


for row in results:

    parameter = max(
        row[
            "trainable_parameters"
        ],
        1
    )

    f1 = row[
        "test_macro_f1"
    ]


    plt.scatter(
        parameter,
        f1,
        s=100
    )


    plt.annotate(
        row["method"],
        (
            parameter,
            f1
        ),
        xytext=(
            6,
            6
        ),
        textcoords="offset points"
    )


plt.xscale(
    "log"
)


plt.xlabel(
    "Trainable Parameters (log scale)"
)


plt.ylabel(
    "Test Macro F1"
)


plt.title(
    "Final Performance vs Parameter Efficiency"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    EFFICIENCY_PATH,
    dpi=180
)


plt.close()


# ============================================================
# 16. 完成
# ============================================================

print("\n")

print("=" * 85)

print(
    "Final Comparison 完成"
)

print("=" * 85)


print(
    "\nCSV:"
)

print(
    CSV_PATH
)


print(
    "\nPerformance:"
)

print(
    PERFORMANCE_PATH
)


print(
    "\nParameters:"
)

print(
    PARAMETER_PATH
)


print(
    "\nEfficiency:"
)

print(
    EFFICIENCY_PATH
)