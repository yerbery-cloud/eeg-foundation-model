from pathlib import Path
from collections import deque
import csv
import math
import time

import mne
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    sosfiltfilt,
    resample_poly
)

from masked_eeg_model import (
    MaskedEEGTransformer
)


# ============================================================
# 0. 关闭 Transformer Fast Path
#
# 和 LoRA 训练时保持一致。
# ============================================================

if (
    hasattr(torch.backends, "mha")
    and hasattr(
        torch.backends.mha,
        "set_fastpath_enabled"
    )
):
    torch.backends.mha.set_fastpath_enabled(
        False
    )


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


LORA_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "lora"
    / "best.pt"
)


RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "realtime_simulation"
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CSV_PATH = (
    RESULT_DIR
    / "simulation_results.csv"
)


FIGURE_PATH = (
    RESULT_DIR
    / "realtime_probability.png"
)


# ============================================================
# 2. 模拟 Subject
#
# Subject32 属于我们之前的 Test Set。
# ============================================================

SUBJECT = 32


REST_PATH = (
    RAW_DIR
    / f"Subject{SUBJECT:02d}_1.edf"
)


TASK_PATH = (
    RAW_DIR
    / f"Subject{SUBJECT:02d}_2.edf"
)


# ============================================================
# 3. 实时系统参数
# ============================================================

DEVICE_SFREQ = 250

MODEL_SFREQ = 160


# 每个预测使用 4 秒

WINDOW_SECONDS = 4.0


DEVICE_WINDOW_SAMPLES = int(

    DEVICE_SFREQ
    * WINDOW_SECONDS

)


MODEL_WINDOW_SAMPLES = int(

    MODEL_SFREQ
    * WINDOW_SECONDS

)


# ============================================================
# 4. 模拟 Streaming 参数
# ============================================================

# 每次收到 0.2 秒数据
#
# 250 Hz × 0.2 s = 50 samples

CHUNK_SECONDS = 0.2

CHUNK_SAMPLES = int(

    DEVICE_SFREQ
    * CHUNK_SECONDS

)


# 每 1 秒进行一次模型预测

INFERENCE_STEP_SECONDS = 1.0

INFERENCE_STEP_SAMPLES = int(

    DEVICE_SFREQ
    * INFERENCE_STEP_SECONDS

)


# 每种状态模拟多少秒

STATE_SECONDS = 30.0


# ============================================================
# 5. 是否真的按现实时间等待
#
# False：
# 快速模拟，几十秒数据几秒内跑完。
#
# True：
# 0.2 秒数据就真实等待 0.2 秒。
# ============================================================

REAL_TIME_SLEEP = False


# ============================================================
# 6. Artifact QC
# ============================================================

MAX_AMPLITUDE_UV = 500.0


# ============================================================
# 7. Probability Smoothing
#
# 连续 5 次预测做移动平均。
# ============================================================

SMOOTHING_WINDOWS = 5


# ============================================================
# 8. Device
# ============================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


print("=" * 75)

print("EEG Foundation Model Realtime Simulation")

print("=" * 75)


print(
    "\nDevice:",
    device
)


# ============================================================
# 9. LoRA Layer
# ============================================================

class LoRALinear(
    nn.Module
):

    def __init__(
        self,
        base_linear,
        rank=8,
        alpha=16,
        dropout=0.05
    ):

        super().__init__()


        self.in_features = (
            base_linear.in_features
        )


        self.out_features = (
            base_linear.out_features
        )


        self.rank = rank

        self.alpha = alpha


        self.scaling = (

            alpha
            / rank

        )


        self.base_linear = (
            base_linear
        )


        self.lora_dropout = nn.Dropout(
            dropout
        )


        self.lora_A = nn.Linear(

            self.in_features,

            rank,

            bias=False

        )


        self.lora_B = nn.Linear(

            rank,

            self.out_features,

            bias=False

        )


        nn.init.kaiming_uniform_(

            self.lora_A.weight,

            a=math.sqrt(5)

        )


        nn.init.zeros_(

            self.lora_B.weight

        )


    @property
    def weight(self):

        return (
            self.base_linear.weight
        )


    @property
    def bias(self):

        return (
            self.base_linear.bias
        )


    def forward(
        self,
        x
    ):

        base_output = (
            self.base_linear(
                x
            )
        )


        lora_output = (

            self.lora_B(

                self.lora_A(

                    self.lora_dropout(
                        x
                    )

                )

            )

            * self.scaling

        )


        return (

            base_output
            + lora_output

        )


# ============================================================
# 10. LoRA Classifier
# ============================================================

class EEGLoraClassifier(
    nn.Module
):

    def __init__(
        self,
        backbone,
        embed_dim,
        num_classes=2
    ):

        super().__init__()


        self.backbone = backbone


        self.classifier = nn.Linear(

            embed_dim,

            num_classes

        )


    def forward(
        self,
        x
    ):

        # ----------------------------------------------------
        # Patch Embedding
        # ----------------------------------------------------

        tokens = (
            self.backbone
            .patch_embedding(
                x
            )
        )


        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        representations = (
            self.backbone
            .encoder(
                tokens
            )
        )


        # ----------------------------------------------------
        # LayerNorm
        # ----------------------------------------------------

        representations = (
            self.backbone
            .norm(
                representations
            )
        )


        # ----------------------------------------------------
        # Mean Pooling
        # ----------------------------------------------------

        features = (
            representations
            .mean(
                dim=1
            )
        )


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            features
        )


        return logits


# ============================================================
# 11. 加载 LoRA Checkpoint
# ============================================================

print(
    "\nLoading LoRA Foundation Model ..."
)


checkpoint = torch.load(

    LORA_PATH,

    map_location=device

)


config = checkpoint[
    "config"
]


LORA_RANK = checkpoint.get(
    "lora_rank",
    8
)


LORA_ALPHA = checkpoint.get(
    "lora_alpha",
    16
)


LORA_DROPOUT = checkpoint.get(
    "lora_dropout",
    0.05
)


# ============================================================
# 12. 创建 Backbone
# ============================================================

backbone = MaskedEEGTransformer(

    num_channels=config[
        "num_channels"
    ],

    time_points=config[
        "time_points"
    ],

    patch_size=config[
        "patch_size"
    ],

    embed_dim=config[
        "embed_dim"
    ],

    num_heads=config[
        "num_heads"
    ],

    num_layers=config[
        "num_layers"
    ],

    feedforward_dim=config[
        "feedforward_dim"
    ],

    dropout=config[
        "dropout"
    ]

)


# ============================================================
# 13. 注入 LoRA
# ============================================================

for layer in backbone.encoder.layers:


    layer.linear1 = LoRALinear(

        layer.linear1,

        rank=LORA_RANK,

        alpha=LORA_ALPHA,

        dropout=LORA_DROPOUT

    )


    layer.linear2 = LoRALinear(

        layer.linear2,

        rank=LORA_RANK,

        alpha=LORA_ALPHA,

        dropout=LORA_DROPOUT

    )


# ============================================================
# 14. 创建分类模型
# ============================================================

model = EEGLoraClassifier(

    backbone=backbone,

    embed_dim=config[
        "embed_dim"
    ],

    num_classes=2

)


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]

)


model = model.to(
    device
)


model.eval()


print(
    "✓ LoRA Model 加载完成"
)


print(
    "LoRA Rank:",
    LORA_RANK
)


# ============================================================
# 15. Channel 匹配
# ============================================================

def find_channel(
    raw,
    target
):

    target = (
        target
        .lower()
        .replace(
            " ",
            ""
        )
    )


    for channel in raw.ch_names:


        cleaned = (

            channel
            .lower()
            .replace(
                "eeg",
                ""
            )
            .replace(
                " ",
                ""
            )
            .replace(
                ".",
                ""
            )

        )


        if cleaned == target:

            return channel


    raise RuntimeError(

        f"找不到通道："
        f"{target}"

    )


# ============================================================
# 16. 从 EDF 创建“模拟 LoongBrain 数据流”
# ============================================================

def load_device_like_signal(
    path,
    state
):

    raw = mne.io.read_raw_edf(

        path,

        preload=True,

        verbose=False

    )


    fp1 = find_channel(
        raw,
        "fp1"
    )


    fp2 = find_channel(
        raw,
        "fp2"
    )


    raw.pick(
        [
            fp1,
            fp2
        ]
    )


    # --------------------------------------------------------
    # 500 Hz
    #
    # ↓
    #
    # 模拟 LoongBrain
    #
    # 250 Hz
    # --------------------------------------------------------

    raw.resample(

        DEVICE_SFREQ,

        npad="auto"

    )


    data = raw.get_data()


    required_samples = int(

        STATE_SECONDS
        * DEVICE_SFREQ

    )


    if data.shape[1] < required_samples:

        raise RuntimeError(

            f"{state} 数据不足 "
            f"{STATE_SECONDS} 秒"

        )


    # --------------------------------------------------------
    # REST：
    #
    # 和我们训练数据类似，
    # 取记录末尾。
    # --------------------------------------------------------

    if state == "REST":

        segment = data[
            :,
            -required_samples:
        ]


    # --------------------------------------------------------
    # TASK：
    #
    # 取记录开头。
    # --------------------------------------------------------

    else:

        segment = data[
            :,
            :required_samples
        ]


    return segment


# ============================================================
# 17. 加载模拟数据
# ============================================================

rest_stream = load_device_like_signal(

    REST_PATH,

    "REST"

)


task_stream = load_device_like_signal(

    TASK_PATH,

    "TASK"

)


print("\n")

print("=" * 75)

print("Simulated Device Stream")

print("=" * 75)


print(
    "REST:",
    rest_stream.shape
)


print(
    "TASK:",
    task_stream.shape
)


print(
    "Sampling Rate:",
    DEVICE_SFREQ,
    "Hz"
)


# ============================================================
# 18. 实时 Filter
# ============================================================

FILTER_SOS = butter(

    N=4,

    Wn=[
        1.0,
        45.0
    ],

    btype="bandpass",

    fs=DEVICE_SFREQ,

    output="sos"

)


# ============================================================
# 19. 一个 4 秒 Window 的实时预处理
# ============================================================

def preprocess_realtime_window(
    window
):

    # --------------------------------------------------------
    # Shape
    #
    # [2,1000] @250Hz
    # --------------------------------------------------------

    if window.shape != (
        2,
        DEVICE_WINDOW_SAMPLES
    ):

        raise RuntimeError(

            f"Realtime window shape 错误："
            f"{window.shape}"

        )


    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    if not np.isfinite(
        window
    ).all():

        return (
            None,
            "NaN/Inf",
            np.nan
        )


    # --------------------------------------------------------
    # Artifact
    # --------------------------------------------------------

    max_uv = (

        np.max(
            np.abs(
                window
            )
        )

        * 1e6

    )


    if max_uv > MAX_AMPLITUDE_UV:

        return (
            None,
            "Artifact",
            max_uv
        )


    # --------------------------------------------------------
    # 1–45 Hz
    # --------------------------------------------------------

    filtered = sosfiltfilt(

        FILTER_SOS,

        window,

        axis=-1

    )


    # --------------------------------------------------------
    # 250 Hz
    #
    # ↓
    #
    # 160 Hz
    #
    # 16 / 25
    # --------------------------------------------------------

    resampled = resample_poly(

        filtered,

        up=16,

        down=25,

        axis=-1

    )


    if resampled.shape != (
        2,
        MODEL_WINDOW_SAMPLES
    ):

        raise RuntimeError(

            f"Resample shape 错误："
            f"{resampled.shape}"

        )


    # --------------------------------------------------------
    # Per-window per-channel Z-score
    #
    # 和训练完全一致。
    # --------------------------------------------------------

    mean = np.mean(

        resampled,

        axis=1,

        keepdims=True

    )


    std = np.std(

        resampled,

        axis=1,

        keepdims=True

    )


    normalized = (

        resampled
        - mean

    ) / (

        std
        + 1e-8

    )


    normalized = normalized.astype(
        np.float32
    )


    return (

        normalized,
        "OK",
        max_uv

    )


# ============================================================
# 20. 模型推理
# ============================================================

def infer(
    eeg
):

    # --------------------------------------------------------
    # [2,640]
    #
    # →
    #
    # [1,2,640]
    # --------------------------------------------------------

    tensor = torch.from_numpy(

        eeg

    ).unsqueeze(
        0
    ).to(device)


    model_start = (
        time.perf_counter()
    )


    with torch.no_grad():


        logits = model(
            tensor
        )


        probabilities = torch.softmax(

            logits,

            dim=1

        )


    model_latency = (

        time.perf_counter()
        - model_start

    ) * 1000


    rest_probability = float(

        probabilities[
            0,
            0
        ].item()

    )


    task_probability = float(

        probabilities[
            0,
            1
        ].item()

    )


    return (

        rest_probability,

        task_probability,

        model_latency

    )


# ============================================================
# 21. 模拟一个状态
# ============================================================

def simulate_state(
    signal,
    state_name,
    true_label,
    global_offset
):

    print("\n")

    print("=" * 75)

    print(
        f"Streaming State: "
        f"{state_name}"
    )

    print("=" * 75)


    # Rolling Buffer

    buffer = np.empty(

        (
            2,
            0
        ),

        dtype=np.float64

    )


    new_samples_since_prediction = 0


    probability_history = deque(

        maxlen=SMOOTHING_WINDOWS

    )


    records = []


    # ========================================================
    # Streaming chunks
    # ========================================================

    for start in range(

        0,

        signal.shape[1],

        CHUNK_SAMPLES

    ):


        end = min(

            start
            + CHUNK_SAMPLES,

            signal.shape[1]

        )


        chunk = signal[
            :,
            start:end
        ]


        # ----------------------------------------------------
        # 模拟实时接收
        # ----------------------------------------------------

        buffer = np.concatenate(

            [
                buffer,
                chunk
            ],

            axis=1

        )


        # 只保留最近 4 秒

        if buffer.shape[1] > DEVICE_WINDOW_SAMPLES:

            buffer = buffer[
                :,
                -DEVICE_WINDOW_SAMPLES:
            ]


        new_samples_since_prediction += (
            chunk.shape[1]
        )


        # ----------------------------------------------------
        # 可选：
        #
        # 真正模拟现实时间
        # ----------------------------------------------------

        if REAL_TIME_SLEEP:

            time.sleep(

                chunk.shape[1]
                / DEVICE_SFREQ

            )


        # ----------------------------------------------------
        # Buffer 还不到 4 秒
        # ----------------------------------------------------

        if buffer.shape[1] < DEVICE_WINDOW_SAMPLES:

            continue


        # ----------------------------------------------------
        # 每 1 秒预测一次
        # ----------------------------------------------------

        if (

            new_samples_since_prediction
            < INFERENCE_STEP_SAMPLES

        ):

            continue


        new_samples_since_prediction = 0


        pipeline_start = (
            time.perf_counter()
        )


        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        (
            processed,
            quality,
            max_uv

        ) = preprocess_realtime_window(

            buffer.copy()

        )


        current_time = (

            global_offset

            + (
                end
                / DEVICE_SFREQ
            )

        )


        # ----------------------------------------------------
        # Artifact
        # ----------------------------------------------------

        if processed is None:

            print(

                f"{current_time:6.1f}s | "

                f"{state_name:<4} | "

                f"{quality:<8} | "

                f"Max = "
                f"{max_uv:.1f} μV"

            )


            continue


        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        (
            rest_prob,

            task_prob,

            model_latency

        ) = infer(
            processed
        )


        probability_history.append(
            task_prob
        )


        smoothed_task_prob = float(

            np.mean(
                probability_history
            )

        )


        predicted_label = int(

            smoothed_task_prob
            >= 0.5

        )


        predicted_name = (

            "TASK"

            if predicted_label == 1

            else "REST"

        )


        pipeline_latency = (

            time.perf_counter()

            - pipeline_start

        ) * 1000


        # ----------------------------------------------------
        # Terminal Realtime Output
        # ----------------------------------------------------

        print(

            f"{current_time:6.1f}s | "

            f"True={state_name:<4} | "

            f"Pred={predicted_name:<4} | "

            f"P(Task)="
            f"{smoothed_task_prob:.3f} | "

            f"Latency="
            f"{pipeline_latency:.1f} ms"

        )


        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        records.append(

            {

                "time_seconds":
                    current_time,

                "true_label":
                    true_label,

                "true_state":
                    state_name,

                "predicted_label":
                    predicted_label,

                "predicted_state":
                    predicted_name,

                "raw_task_probability":
                    task_prob,

                "smoothed_task_probability":
                    smoothed_task_prob,

                "model_latency_ms":
                    model_latency,

                "pipeline_latency_ms":
                    pipeline_latency,

                "max_abs_uv":
                    max_uv

            }

        )


    return records


# ============================================================
# 22. 正式模拟
# ============================================================

all_records = []


# REST

rest_records = simulate_state(

    rest_stream,

    state_name="REST",

    true_label=0,

    global_offset=0.0

)


all_records.extend(
    rest_records
)


# TASK

task_records = simulate_state(

    task_stream,

    state_name="TASK",

    true_label=1,

    global_offset=STATE_SECONDS

)


all_records.extend(
    task_records
)


# ============================================================
# 23. 结果检查
# ============================================================

if len(
    all_records
) == 0:

    raise RuntimeError(

        "没有生成有效实时预测。"
        "可能全部窗口被 Artifact QC 删除。"

    )


# ============================================================
# 24. 保存 CSV
# ============================================================

with open(

    CSV_PATH,

    "w",

    newline="",

    encoding="utf-8"

) as file:


    writer = csv.DictWriter(

        file,

        fieldnames=list(
            all_records[
                0
            ].keys()
        )

    )


    writer.writeheader()


    writer.writerows(
        all_records
    )


# ============================================================
# 25. 汇总
# ============================================================

true_labels = np.asarray(

    [
        row[
            "true_label"
        ]
        for row
        in all_records
    ]

)


predicted_labels = np.asarray(

    [
        row[
            "predicted_label"
        ]
        for row
        in all_records
    ]

)


pipeline_latencies = np.asarray(

    [
        row[
            "pipeline_latency_ms"
        ]
        for row
        in all_records
    ]

)


model_latencies = np.asarray(

    [
        row[
            "model_latency_ms"
        ]
        for row
        in all_records
    ]

)


accuracy = np.mean(

    true_labels
    == predicted_labels

)


print("\n")

print("=" * 75)

print("REALTIME SIMULATION RESULT")

print("=" * 75)


print(
    "\nPredictions:",
    len(
        all_records
    )
)


print(
    "Accuracy:",
    f"{accuracy:.4f}"
)


print(
    "\nMean Model Latency:",
    f"{model_latencies.mean():.2f} ms"
)


print(
    "Mean Pipeline Latency:",
    f"{pipeline_latencies.mean():.2f} ms"
)


print(
    "95th Percentile Latency:",
    f"{np.percentile(pipeline_latencies, 95):.2f} ms"
)


# ============================================================
# 26. Probability Figure
# ============================================================

times = np.asarray(

    [
        row[
            "time_seconds"
        ]
        for row
        in all_records
    ]

)


task_probs = np.asarray(

    [
        row[
            "smoothed_task_probability"
        ]
        for row
        in all_records
    ]

)


plt.figure(
    figsize=(12, 6)
)


plt.plot(

    times,

    task_probs,

    marker="o",

    label="P(Mental Arithmetic)"

)


# Classification threshold

plt.axhline(

    0.5,

    linestyle="--",

    label="Decision Threshold"

)


# REST / TASK 分界

plt.axvline(

    STATE_SECONDS,

    linestyle="--",

    label="REST → TASK"

)


plt.xlabel(
    "Simulation Time (s)"
)


plt.ylabel(
    "Task Probability"
)


plt.ylim(
    0,
    1
)


plt.title(
    "Realtime EEG Foundation Model Simulation"
)


plt.grid(
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(

    FIGURE_PATH,

    dpi=180

)


plt.show()


# ============================================================
# 27. Final
# ============================================================

print("\n")

print("=" * 75)

print(
    "Realtime Pipeline Test 完成"
)

print("=" * 75)


print(
    "\nCSV:"
)

print(
    CSV_PATH
)


print(
    "\nProbability Figure:"
)

print(
    FIGURE_PATH
)