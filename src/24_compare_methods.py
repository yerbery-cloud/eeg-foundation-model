from pathlib import Path
import csv
import math

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 项目路径
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
    / "comparison"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 输入结果文件
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
    )

}


# ============================================================
# 3. 输出文件
# ============================================================

COMPARISON_CSV = (
    OUTPUT_DIR
    / "method_comparison.csv"
)


PERFORMANCE_FIGURE = (
    OUTPUT_DIR
    / "performance_comparison.png"
)


PARAMETER_FIGURE = (
    OUTPUT_DIR
    / "parameter_comparison.png"
)


EFFICIENCY_FIGURE = (
    OUTPUT_DIR
    / "parameter_efficiency.png"
)


# ============================================================
# 4. 读取 metric,value 类型 CSV
# ============================================================

def read_summary_csv(
    path
):

    if not path.exists():

        raise FileNotFoundError(
            f"找不到结果文件：\n{path}"
        )


    result = {}


    with open(

        path,

        "r",

        encoding="utf-8-sig"

    ) as file:


        reader = csv.reader(
            file
        )


        rows = list(
            reader
        )


    # --------------------------------------------------------
    # 第一行一般是：
    #
    # metric,value
    #
    # 所以从第二行开始
    # --------------------------------------------------------

    for row in rows[1:]:


        if len(row) < 2:

            continue


        key = (
            row[0]
            .strip()
        )


        value = (
            row[1]
            .strip()
        )


        result[
            key
        ] = value


    return result


# ============================================================
# 5. 数字转换
# ============================================================

def to_float(
    dictionary,
    key,
    default=np.nan
):

    if key not in dictionary:

        return default


    try:

        return float(
            dictionary[
                key
            ]
        )


    except (
        ValueError,
        TypeError
    ):

        return default


def to_int(
    dictionary,
    key,
    default=0
):

    if key not in dictionary:

        return default


    try:

        return int(
            float(
                dictionary[
                    key
                ]
            )
        )


    except (
        ValueError,
        TypeError
    ):

        return default


# ============================================================
# 6. 加载五种方法
# ============================================================

results = []


print("=" * 85)

print(
    "EEGMAT Method Comparison"
)

print("=" * 85)


for method_name, file_path in (
    METHOD_FILES.items()
):


    summary = read_summary_csv(
        file_path
    )


    # ========================================================
    # 读取参数
    # ========================================================

    trainable_parameters = to_int(

        summary,

        "trainable_parameters"

    )


    total_parameters = to_int(

        summary,

        "total_parameters",

        trainable_parameters

    )


    best_epoch = to_int(

        summary,

        "best_epoch"

    )


    # ========================================================
    # Validation
    # ========================================================

    val_accuracy = to_float(

        summary,

        "val_accuracy"

    )


    val_balanced_accuracy = to_float(

        summary,

        "val_balanced_accuracy"

    )


    val_macro_f1 = to_float(

        summary,

        "val_macro_f1"

    )


    # ========================================================
    # Test
    # ========================================================

    test_accuracy = to_float(

        summary,

        "test_accuracy"

    )


    test_balanced_accuracy = to_float(

        summary,

        "test_balanced_accuracy"

    )


    test_macro_f1 = to_float(

        summary,

        "test_macro_f1"

    )


    test_loss = to_float(

        summary,

        "test_loss"

    )


    results.append(

        {

            "method":
                method_name,

            "total_parameters":
                total_parameters,

            "trainable_parameters":
                trainable_parameters,

            "best_epoch":
                best_epoch,

            "val_accuracy":
                val_accuracy,

            "val_balanced_accuracy":
                val_balanced_accuracy,

            "val_macro_f1":
                val_macro_f1,

            "test_loss":
                test_loss,

            "test_accuracy":
                test_accuracy,

            "test_balanced_accuracy":
                test_balanced_accuracy,

            "test_macro_f1":
                test_macro_f1

        }

    )


# ============================================================
# 7. 检查核心指标
# ============================================================

for row in results:


    for key in [

        "test_accuracy",

        "test_balanced_accuracy",

        "test_macro_f1"

    ]:


        if np.isnan(
            row[
                key
            ]
        ):

            raise RuntimeError(

                f"{row['method']} "
                f"缺少指标：{key}"

            )


# ============================================================
# 8. 保存统一 CSV
# ============================================================

fieldnames = [

    "method",

    "total_parameters",

    "trainable_parameters",

    "best_epoch",

    "val_accuracy",

    "val_balanced_accuracy",

    "val_macro_f1",

    "test_loss",

    "test_accuracy",

    "test_balanced_accuracy",

    "test_macro_f1"

]


with open(

    COMPARISON_CSV,

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
# 9. 终端打印统一表格
# ============================================================

print("\n")

print(
    f"{'Method':<22}"
    f"{'Trainable':>12}"
    f"{'Accuracy':>12}"
    f"{'Balanced':>12}"
    f"{'Macro F1':>12}"
)


print("-" * 70)


for row in results:


    print(

        f"{row['method']:<22}"

        f"{row['trainable_parameters']:>12,d}"

        f"{row['test_accuracy']:>12.4f}"

        f"{row['test_balanced_accuracy']:>12.4f}"

        f"{row['test_macro_f1']:>12.4f}"

    )


# ============================================================
# 10. 找最佳方法
# ============================================================

best_accuracy_method = max(

    results,

    key=lambda x:
    x["test_accuracy"]

)


best_balanced_method = max(

    results,

    key=lambda x:
    x["test_balanced_accuracy"]

)


best_f1_method = max(

    results,

    key=lambda x:
    x["test_macro_f1"]

)


print("\n")

print("=" * 85)

print(
    "Best Results"
)

print("=" * 85)


print(

    "\nBest Accuracy:"

)


print(

    best_accuracy_method[
        "method"
    ],

    f"= "
    f"{best_accuracy_method['test_accuracy']:.4f}"

)


print(

    "\nBest Balanced Accuracy:"

)


print(

    best_balanced_method[
        "method"
    ],

    f"= "
    f"{best_balanced_method['test_balanced_accuracy']:.4f}"

)


print(

    "\nBest Macro F1:"

)


print(

    best_f1_method[
        "method"
    ],

    f"= "
    f"{best_f1_method['test_macro_f1']:.4f}"

)


# ============================================================
# 11. 找 Random / Linear
#
# 用于直接评价 Pretraining Gain
# ============================================================

result_dict = {

    row[
        "method"
    ]:
    row

    for row
    in results

}


random_result = result_dict[
    "Random Probe"
]


linear_result = result_dict[
    "Linear Probe"
]


# ============================================================
# 12. Pretraining Improvement
# ============================================================

accuracy_gain = (

    linear_result[
        "test_accuracy"
    ]

    -

    random_result[
        "test_accuracy"
    ]

)


balanced_gain = (

    linear_result[
        "test_balanced_accuracy"
    ]

    -

    random_result[
        "test_balanced_accuracy"
    ]

)


f1_gain = (

    linear_result[
        "test_macro_f1"
    ]

    -

    random_result[
        "test_macro_f1"
    ]

)


print("\n")

print("=" * 85)

print(
    "Pretraining Gain"
)

print("=" * 85)


print(

    "\nLinear Probe "
    "vs Random Probe"

)


print(

    "Accuracy Gain:       ",

    f"{accuracy_gain:+.4f}"

)


print(

    "Balanced Acc Gain:   ",

    f"{balanced_gain:+.4f}"

)


print(

    "Macro F1 Gain:       ",

    f"{f1_gain:+.4f}"

)


# ============================================================
# 13. LoRA vs Full FT
# ============================================================

lora_result = result_dict[
    "LoRA"
]


full_result = result_dict[
    "Full Fine-tuning"
]


parameter_reduction = (

    1

    -

    (
        lora_result[
            "trainable_parameters"
        ]

        /

        full_result[
            "trainable_parameters"
        ]

    )

) * 100


print("\n")

print("=" * 85)

print(
    "LoRA Parameter Efficiency"
)

print("=" * 85)


print(

    "\nLoRA Trainable Params:",

    f"{lora_result['trainable_parameters']:,}"

)


print(

    "Full FT Trainable Params:",

    f"{full_result['trainable_parameters']:,}"

)


print(

    "Parameter Reduction:",

    f"{parameter_reduction:.2f}%"

)


print(

    "\nLoRA Macro F1:",

    f"{lora_result['test_macro_f1']:.4f}"

)


print(

    "Full FT Macro F1:",

    f"{full_result['test_macro_f1']:.4f}"

)


# ============================================================
# 14. Performance Comparison
# ============================================================

methods = [

    row[
        "method"
    ]

    for row
    in results

]


accuracy_values = [

    row[
        "test_accuracy"
    ]

    for row
    in results

]


balanced_values = [

    row[
        "test_balanced_accuracy"
    ]

    for row
    in results

]


f1_values = [

    row[
        "test_macro_f1"
    ]

    for row
    in results

]


x = np.arange(
    len(
        methods
    )
)


width = 0.25


plt.figure(
    figsize=(12, 7)
)


plt.bar(

    x - width,

    accuracy_values,

    width,

    label="Accuracy"

)


plt.bar(

    x,

    balanced_values,

    width,

    label="Balanced Accuracy"

)


plt.bar(

    x + width,

    f1_values,

    width,

    label="Macro F1"

)


plt.axhline(

    y=0.5,

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
    "EEGMAT Downstream Method Comparison"
)


plt.grid(

    axis="y",

    alpha=0.3

)


plt.legend()


plt.tight_layout()


plt.savefig(

    PERFORMANCE_FIGURE,

    dpi=180

)


plt.close()


# ============================================================
# 15. Parameter Comparison
# ============================================================

trainable_values = [

    row[
        "trainable_parameters"
    ]

    for row
    in results

]


plt.figure(
    figsize=(11, 6)
)


bars = plt.bar(

    methods,

    trainable_values

)


plt.yscale(
    "log"
)


plt.ylabel(
    "Trainable Parameters (log scale)"
)


plt.title(
    "Trainable Parameter Comparison"
)


plt.xticks(

    rotation=20,

    ha="right"

)


plt.grid(

    axis="y",

    alpha=0.3

)


# 参数数值
for bar, value in zip(

    bars,

    trainable_values

):


    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        value * 1.15,

        f"{value:,}",

        ha="center",

        va="bottom",

        fontsize=9

    )


plt.tight_layout()


plt.savefig(

    PARAMETER_FIGURE,

    dpi=180

)


plt.close()


# ============================================================
# 16. Parameter Efficiency
#
# X = 可训练参数
# Y = Macro F1
# ============================================================

plt.figure(
    figsize=(9, 7)
)


for row in results:


    parameters = max(

        row[
            "trainable_parameters"
        ],

        1

    )


    macro_f1 = row[
        "test_macro_f1"
    ]


    plt.scatter(

        parameters,

        macro_f1,

        s=100

    )


    plt.annotate(

        row[
            "method"
        ],

        (
            parameters,
            macro_f1
        ),

        xytext=(
            7,
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
    "Performance vs Parameter Efficiency"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    EFFICIENCY_FIGURE,

    dpi=180

)


plt.close()


# ============================================================
# 17. 最终输出
# ============================================================

print("\n")

print("=" * 85)

print(
    "Method Comparison 完成"
)

print("=" * 85)


print(
    "\nComparison CSV:"
)


print(
    COMPARISON_CSV
)


print(
    "\nPerformance Figure:"
)


print(
    PERFORMANCE_FIGURE
)


print(
    "\nParameter Figure:"
)


print(
    PARAMETER_FIGURE
)


print(
    "\nEfficiency Figure:"
)


print(
    EFFICIENCY_FIGURE
)