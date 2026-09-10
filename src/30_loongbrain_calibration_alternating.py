from pathlib import Path
from collections import deque
from fractions import Fraction
from datetime import datetime

import csv
import math
import time

import numpy as np
import torch
import torch.nn as nn

from scipy.signal import (
    butter,
    sosfiltfilt,
    resample_poly,
)

from pylsl import (
    resolve_streams,
    StreamInlet,
)

from masked_eeg_model import MaskedEEGTransformer


# ============================================================
# 0. PyTorch Transformer / LoRA Compatibility
# ============================================================

if (
    hasattr(torch.backends, "mha")
    and hasattr(torch.backends.mha, "set_fastpath_enabled")
):
    torch.backends.mha.set_fastpath_enabled(False)


# ============================================================
# 1. Project Path
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "lora"
    / "best.pt"
)


RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "loongbrain_calibration"
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. LSL Stream Config
# ============================================================

# 第一次先保持 None。
#
# 如果 29_lsl_stream_check.py 发现多个 EEG Stream，
# 把正确的 Name 填进来，例如：
#
# TARGET_STREAM_NAME = "EEG-123456"

TARGET_STREAM_NAME = None


# 如果 LSL 没有 Fp1/Fp2 channel label，
# 但是 29 号程序已经确认：
#
# channel 0 = Fp1
# channel 1 = Fp2
#
# 就设置：
#
# MANUAL_CHANNEL_INDICES = (0, 1)

MANUAL_CHANNEL_INDICES = None


# ============================================================
# 3. Model Input Config
# ============================================================

MODEL_SFREQ = 160.0

MODEL_WINDOW_SECONDS = 4.0

MODEL_WINDOW_SAMPLES = int(
    MODEL_SFREQ
    * MODEL_WINDOW_SECONDS
)


# 每 1 秒预测一次

PREDICTION_STEP_SECONDS = 1.0


# 最近 5 次预测进行平滑

SMOOTHING_WINDOWS = 5


# ============================================================
# 4. Classification Threshold
# ============================================================

# 第一次真机实验保持 0.5。
#
# 后续完成 Calibration 后，
# 可以固定成个体化阈值，例如 0.42。

TASK_THRESHOLD = 0.50


# ============================================================
# 5. QC
# ============================================================

MAX_ABS_UV = 500.0


# ============================================================
# 6. Guided Experiment Protocol
# ============================================================

SETTLE_SECONDS = 10

REST_SECONDS = 20

PREPARE_SECONDS = 5

TASK_SECONDS = 20


# 心算任务

START_NUMBER = 4387

SUBTRACTOR = 47


# ============================================================
# 7. Torch Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 80)
print("LoongBrain Personal Calibration | Alternating REST-TASK Blocks")
print("=" * 80)

print(
    "\nTorch Device:",
    device,
)


# ============================================================
# 8. Beep
# ============================================================

def beep():

    try:

        import winsound

        winsound.Beep(
            880,
            250,
        )

    except Exception:

        pass


# ============================================================
# 9. 获取 Channel Metadata
# ============================================================

def get_channel_metadata(info):

    labels = []

    units = []

    try:

        channel = (
            info
            .desc()
            .child("channels")
            .child("channel")
        )

        for _ in range(
            info.channel_count()
        ):

            label = channel.child_value(
                "label"
            )

            unit = channel.child_value(
                "unit"
            )

            labels.append(
                label if label else ""
            )

            units.append(
                unit if unit else ""
            )

            channel = (
                channel.next_sibling()
            )

    except Exception:

        labels = [
            ""
            for _ in range(
                info.channel_count()
            )
        ]

        units = [
            ""
            for _ in range(
                info.channel_count()
            )
        ]

    return labels, units


# ============================================================
# 10. Channel 名称清洗
# ============================================================

def clean_channel_name(name):

    return (
        name
        .lower()
        .replace("eeg", "")
        .replace(" ", "")
        .replace(".", "")
        .replace("-", "")
        .replace("_", "")
    )


# ============================================================
# 11. 搜索 LoongBrain EEG Stream
# ============================================================

def discover_stream():

    print(
        "\n正在搜索 LSL Stream ..."
    )

    streams = resolve_streams(
        wait_time=5.0
    )


    if len(streams) == 0:

        raise RuntimeError(

            "\n没有发现任何 LSL Stream。\n\n"
            "请检查：\n"
            "1. LoongBrain 软件已打开\n"
            "2. 蓝牙设备已连接\n"
            "3. LoongBrain 中已经启动 LSL 广播\n"

        )


    # ========================================================
    # 用户指定 Stream Name
    # ========================================================

    if TARGET_STREAM_NAME is not None:

        matches = [

            stream

            for stream in streams

            if stream.name()
            == TARGET_STREAM_NAME

        ]


        if len(matches) != 1:

            raise RuntimeError(

                f"没有唯一找到 Stream："
                f"{TARGET_STREAM_NAME}"

            )


        return matches[0]


    # ========================================================
    # 自动寻找 EEG
    # ========================================================

    candidates = []


    for stream in streams:

        text = (
            stream.name()
            + " "
            + stream.type()
        ).lower()


        if (
            stream.type().lower() == "eeg"
            or "eeg" in text
            or "loong" in text
        ):

            candidates.append(
                stream
            )


    if len(candidates) == 0:

        print(
            "\n找到的所有 Stream："
        )

        for stream in streams:

            print(
                stream.name(),
                "|",
                stream.type(),
                "|",
                stream.channel_count(),
                "channels",
            )

        raise RuntimeError(

            "\n没有自动识别到 EEG Stream。\n"
            "请先运行 29_lsl_stream_check.py。"

        )


    if len(candidates) > 1:

        print(
            "\n发现多个 EEG Candidate："
        )

        for stream in candidates:

            print(

                stream.name(),
                "| type =",
                stream.type(),
                "| channels =",
                stream.channel_count(),
                "| sfreq =",
                stream.nominal_srate(),

            )

        raise RuntimeError(

            "\n存在多个候选 EEG Stream。\n"
            "请把正确的 Stream Name "
            "填写到 TARGET_STREAM_NAME。"

        )


    return candidates[0]


# ============================================================
# 12. 找 Fp1 / Fp2
# ============================================================

def select_channels(info):

    labels, units = (
        get_channel_metadata(
            info
        )
    )


    print(
        "\nChannel Labels:"
    )

    print(
        labels
    )


    print(
        "\nChannel Units:"
    )

    print(
        units
    )


    # ========================================================
    # 手动指定
    # ========================================================

    if MANUAL_CHANNEL_INDICES is not None:

        indices = list(
            MANUAL_CHANNEL_INDICES
        )

        print(
            "\n使用手动 Channel Indices：",
            indices,
        )

        return (
            indices,
            labels,
            units,
        )


    # ========================================================
    # 自动寻找
    # ========================================================

    cleaned = [

        clean_channel_name(
            label
        )

        for label in labels

    ]


    fp1_index = None

    fp2_index = None


    for index, channel in enumerate(
        cleaned
    ):

        if channel == "fp1":

            fp1_index = index

        elif channel == "fp2":

            fp2_index = index


    if (
        fp1_index is not None
        and fp2_index is not None
    ):

        print(
            "\n自动识别："
        )

        print(
            "Fp1 index:",
            fp1_index,
        )

        print(
            "Fp2 index:",
            fp2_index,
        )

        return (

            [
                fp1_index,
                fp2_index,
            ],

            labels,
            units,

        )


    # ========================================================
    # 两通道且无 Label
    # ========================================================

    if (
        info.channel_count() == 2
        and all(
            label == ""
            for label in labels
        )
    ):

        print(
            "\n⚠ LSL Stream 没有 channel label。"
        )

        print(
            "当前只有两个通道，"
            "暂按："
        )

        print(
            "channel 0 = Fp1"
        )

        print(
            "channel 1 = Fp2"
        )

        return (

            [0, 1],

            labels,

            units,

        )


    raise RuntimeError(

        "\n无法自动确定 Fp1 / Fp2。\n"
        "请运行 29_lsl_stream_check.py，"
        "确认通道顺序后填写 "
        "MANUAL_CHANNEL_INDICES。"

    )


# ============================================================
# 13. 判断 LSL 信号单位
# ============================================================

def determine_scale_to_volts(
    selected_units,
    example_data,
):

    unit_text = " ".join(
        selected_units
    ).lower()


    # μV

    if (
        "uv" in unit_text
        or "µv" in unit_text
        or "microvolt" in unit_text
    ):

        print(
            "\n检测到 EEG 单位：μV"
        )

        return 1e-6


    # mV

    if "mv" in unit_text:

        print(
            "\n检测到 EEG 单位：mV"
        )

        return 1e-3


    # V

    stripped = unit_text.strip()

    if stripped in {
        "v",
        "volt",
        "volts",
    }:

        print(
            "\n检测到 EEG 单位：V"
        )

        return 1.0


    # ========================================================
    # Metadata 没有单位 → 根据幅值估计
    # ========================================================

    p95 = np.percentile(

        np.abs(
            example_data
        ),

        95,

    )


    print(
        "\nLSL 没有明确单位。"
    )

    print(
        "Raw amplitude P95:",
        p95,
    )


    # 数值几十、几百，一般更像 μV

    if p95 > 1.0:

        print(
            "自动按 μV 处理。"
        )

        return 1e-6


    print(
        "自动按 V 处理。"
    )

    return 1.0


# ============================================================
# 14. LoRA Linear
# ============================================================

class LoRALinear(
    nn.Module
):

    def __init__(
        self,
        base_linear,
        rank=8,
        alpha=16,
        dropout=0.05,
    ):

        super().__init__()


        self.base_linear = (
            base_linear
        )


        self.rank = rank

        self.alpha = alpha


        self.scaling = (
            alpha
            / rank
        )


        self.lora_dropout = nn.Dropout(
            dropout
        )


        self.lora_A = nn.Linear(

            base_linear.in_features,

            rank,

            bias=False,

        )


        self.lora_B = nn.Linear(

            rank,

            base_linear.out_features,

            bias=False,

        )


        nn.init.kaiming_uniform_(

            self.lora_A.weight,

            a=math.sqrt(5),

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
        x,
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
# 15. LoRA EEG Classifier
# ============================================================

class EEGLoraClassifier(
    nn.Module
):

    def __init__(
        self,
        backbone,
        embed_dim,
    ):

        super().__init__()


        self.backbone = (
            backbone
        )


        self.classifier = nn.Linear(

            embed_dim,

            2,

        )


    def forward(
        self,
        x,
    ):

        # Patch Embedding

        tokens = (

            self.backbone
            .patch_embedding(
                x
            )

        )


        # Transformer

        representations = (

            self.backbone
            .encoder(
                tokens
            )

        )


        # LayerNorm

        representations = (

            self.backbone
            .norm(
                representations
            )

        )


        # Mean Pool

        features = (

            representations
            .mean(
                dim=1
            )

        )


        # Classification

        logits = self.classifier(
            features
        )


        return logits


# ============================================================
# 16. 加载训练好的 LoRA 模型
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"\n找不到 LoRA checkpoint：\n"
            f"{MODEL_PATH}"

        )


    print(
        "\n正在加载 LoRA Model ..."
    )


    checkpoint = torch.load(

        MODEL_PATH,

        map_location=device,

    )


    config = checkpoint[
        "config"
    ]


    rank = checkpoint.get(
        "lora_rank",
        8,
    )


    alpha = checkpoint.get(
        "lora_alpha",
        16,
    )


    dropout = checkpoint.get(
        "lora_dropout",
        0.05,
    )


    # ========================================================
    # Backbone
    # ========================================================

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
        ],

    )


    # ========================================================
    # 注入 LoRA
    # ========================================================

    for layer in (
        backbone
        .encoder
        .layers
    ):

        layer.linear1 = LoRALinear(

            layer.linear1,

            rank=rank,

            alpha=alpha,

            dropout=dropout,

        )


        layer.linear2 = LoRALinear(

            layer.linear2,

            rank=rank,

            alpha=alpha,

            dropout=dropout,

        )


    # ========================================================
    # Classifier
    # ========================================================

    model = EEGLoraClassifier(

        backbone=backbone,

        embed_dim=config[
            "embed_dim"
        ],

    )


    # ========================================================
    # Checkpoint
    # ========================================================

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
        rank,
    )


    print(
        "LoRA Alpha:",
        alpha,
    )


    return model


# ============================================================
# 17. 一个实时 Window 的预处理
# ============================================================

def preprocess_window(
    window_volts,
    source_sfreq,
    filter_sos,
):

    # ========================================================
    # NaN / Inf
    # ========================================================

    if not np.isfinite(
        window_volts
    ).all():

        return (
            None,
            "NaN/Inf",
            np.nan,
        )


    # ========================================================
    # 1–45 Hz
    # ========================================================
    #
    # 真机 EEG 可能带有较大的 DC offset。
    # 因此先带通滤波，再在有效 EEG 频段内进行幅值 QC。
    # ========================================================

    filtered = sosfiltfilt(

        filter_sos,

        window_volts,

        axis=-1,

    )


    # ========================================================
    # Artifact QC（对滤波后的 EEG 判断）
    # ========================================================

    max_uv = (

        np.max(
            np.abs(
                filtered
            )
        )

        * 1e6

    )


    if max_uv > MAX_ABS_UV:

        return (
            None,
            "Artifact",
            max_uv,
        )


    # ========================================================
    # source sampling rate → 160 Hz
    # ========================================================

    ratio = Fraction(

        MODEL_SFREQ
        / source_sfreq

    ).limit_denominator(
        1000
    )


    resampled = resample_poly(

        filtered,

        up=ratio.numerator,

        down=ratio.denominator,

        axis=-1,

    )


    # ========================================================
    # 保证严格 [2,640]
    # ========================================================

    if (
        resampled.shape[1]
        > MODEL_WINDOW_SAMPLES
    ):

        resampled = resampled[
            :,
            :MODEL_WINDOW_SAMPLES
        ]


    elif (
        resampled.shape[1]
        < MODEL_WINDOW_SAMPLES
    ):

        missing = (

            MODEL_WINDOW_SAMPLES

            - resampled.shape[1]

        )


        resampled = np.pad(

            resampled,

            (
                (0, 0),
                (0, missing),
            ),

            mode="edge",

        )


    # ========================================================
    # per-window / per-channel Z-score
    # ========================================================

    mean = np.mean(

        resampled,

        axis=1,

        keepdims=True,

    )


    std = np.std(

        resampled,

        axis=1,

        keepdims=True,

    )


    if np.any(
        std < 1e-8
    ):

        return (
            None,
            "Flat",
            max_uv,
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
        max_uv,
    )


# ============================================================
# 18. Model Inference
# ============================================================

def infer(
    model,
    eeg,
):

    tensor = (

        torch
        .from_numpy(
            eeg
        )
        .unsqueeze(0)
        .to(device)

    )


    start = (
        time.perf_counter()
    )


    with torch.no_grad():

        logits = model(
            tensor
        )


        probability = torch.softmax(

            logits,

            dim=1,

        )


    latency_ms = (

        time.perf_counter()
        - start

    ) * 1000


    p_rest = float(

        probability[
            0,
            0
        ].item()

    )


    p_task = float(

        probability[
            0,
            1
        ].item()

    )


    return (
        p_rest,
        p_task,
        latency_ms,
    )


# ============================================================
# 19. 实验阶段提示
# ============================================================

def show_phase(
    phase,
):

    print("\n")
    print("=" * 80)


    if phase == "SETTLE":

        print(
            "适应阶段"
        )

        print(
            f"时长：{SETTLE_SECONDS} 秒"
        )

        print(
            "受试者：闭眼、放松、保持不动。"
        )


    elif phase.startswith("REST"):

        beep()

        print(
            f"{phase} 开始"
        )

        print(
            f"时长：{REST_SECONDS} 秒"
        )

        print(
            "闭眼、放松、不做心算。"
        )


    elif phase == "PREPARE":

        beep()

        print(
            "TASK 准备阶段"
        )

        print(
            f"请记住：从 {START_NUMBER} 开始，"
            f"每次减 {SUBTRACTOR}。"
        )


    elif phase.startswith("TASK"):

        beep()

        print(
            f"{phase} 开始"
        )

        print(
            f"时长：{TASK_SECONDS} 秒"
        )

        print(
            f"持续在脑中计算："
            f"{START_NUMBER}"
            f" - {SUBTRACTOR}"
            f" - {SUBTRACTOR}"
            f" - ..."
        )

        print(
            "不要说话，不要数手指，尽量保持不动。"
        )


    print("=" * 80)


# ============================================================
# 20. 加载 Model
# ============================================================

model = load_model()


# ============================================================
# 21. 搜索 LSL
# ============================================================

stream_info = discover_stream()


print("\n")
print("=" * 80)
print("Selected EEG Stream")
print("=" * 80)


print(
    "Name:",
    stream_info.name(),
)


print(
    "Type:",
    stream_info.type(),
)


print(
    "Channel Count:",
    stream_info.channel_count(),
)


source_sfreq = float(
    stream_info.nominal_srate()
)


print(
    "Sampling Rate:",
    source_sfreq,
)


if source_sfreq <= 0:

    raise RuntimeError(

        "当前 Stream nominal_srate <= 0。\n"
        "请先运行 29 检查真实采样格式。"

    )


if source_sfreq <= 100:

    raise RuntimeError(

        f"当前 EEG 采样率只有 "
        f"{source_sfreq} Hz，"
        "无法使用 1–45 Hz 预处理。"

    )


# ============================================================
# 22. 选择 Fp1/Fp2
# ============================================================

(
    channel_indices,
    labels,
    units,

) = select_channels(
    stream_info
)


print(
    "\nSelected indices:",
    channel_indices,
)


# ============================================================
# 23. 建立 LSL Inlet
# ============================================================

inlet = StreamInlet(

    stream_info,

    max_buflen=10,

)


# ============================================================
# 24. Warm-up：检查信号单位
# ============================================================

print(
    "\n正在读取短数据进行信号检查..."
)


warm_samples, warm_timestamps = (
    inlet.pull_chunk(

        timeout=2.0,

        max_samples=max(
            int(
                source_sfreq
            ),
            100,
        ),

    )
)


if len(warm_samples) == 0:

    raise RuntimeError(

        "找到 LSL Stream，"
        "但没有收到 EEG 样本。"

    )


warm_data = np.asarray(

    warm_samples,

    dtype=np.float64,

)


warm_eeg = (

    warm_data[
        :,
        channel_indices
    ]
    .T

)


selected_units = [

    units[index]

    if index < len(units)

    else ""

    for index in channel_indices

]


scale_to_volts = (
    determine_scale_to_volts(

        selected_units,

        warm_eeg,

    )
)


print(
    "\nScale to volts:",
    scale_to_volts,
)


# ============================================================
# 25. Filter
# ============================================================

filter_sos = butter(

    N=4,

    Wn=[
        1.0,
        45.0,
    ],

    btype="bandpass",

    fs=source_sfreq,

    output="sos",

)


# ============================================================
# 26. 实时 Buffer 参数
# ============================================================

window_samples = int(

    round(

        source_sfreq

        * MODEL_WINDOW_SECONDS

    )

)


step_samples = int(

    round(

        source_sfreq

        * PREDICTION_STEP_SECONDS

    )

)


print(
    "\nRealtime Window:",
    window_samples,
    "samples",
)


print(
    "Prediction Step:",
    step_samples,
    "samples",
)


# ============================================================
# 27. 输出文件
# ============================================================

session_id = (
    datetime.now()
    .strftime(
        "%Y%m%d_%H%M%S"
    )
)


RAW_CSV_PATH = (

    RESULT_DIR

    / f"raw_eeg_{session_id}.csv"

)


PREDICTION_CSV_PATH = (

    RESULT_DIR

    / f"predictions_{session_id}.csv"

)


# ============================================================
# 28. 数据容器
# ============================================================

raw_records = []

prediction_records = []


probability_history = deque(

    maxlen=SMOOTHING_WINDOWS

)


session_time = 0.0


# ============================================================
# 29. Run Phase
# ============================================================

def run_phase(
    phase_name,
    duration_seconds,
    true_label,
):

    global session_time


    show_phase(
        phase_name
    )


    target_samples = int(

        round(

            duration_seconds

            * source_sfreq

        )

    )


    received_samples = 0


    # 每个正式阶段单独使用 buffer

    buffer = np.empty(

        (
            2,
            0,
        ),

        dtype=np.float64,

    )


    samples_since_prediction = 0


    probability_history.clear()


    # ========================================================
    # Streaming
    # ========================================================

    while (
        received_samples
        < target_samples
    ):


        samples, timestamps = (
            inlet.pull_chunk(

                timeout=0.5,

                max_samples=max(

                    int(
                        source_sfreq
                    ),

                    50,

                ),

            )
        )


        if len(samples) == 0:

            continue


        chunk = np.asarray(

            samples,

            dtype=np.float64,

        )


        timestamps = np.asarray(

            timestamps,

            dtype=np.float64,

        )


        # ====================================================
        # 不超过 phase 目标长度
        # ====================================================

        remaining = (

            target_samples

            - received_samples

        )


        if (
            chunk.shape[0]
            > remaining
        ):

            chunk = chunk[
                :remaining
            ]

            timestamps = timestamps[
                :remaining
            ]


        # ====================================================
        # Fp1 / Fp2
        # ====================================================

        eeg_raw = (

            chunk[
                :,
                channel_indices
            ]
            .T

        )


        eeg_volts = (

            eeg_raw

            * scale_to_volts

        )


        current_chunk_samples = (
            eeg_volts.shape[1]
        )


        received_samples += (
            current_chunk_samples
        )


        # ====================================================
        # 保存原始 EEG
        # ====================================================

        for sample_index in range(
            current_chunk_samples
        ):

            session_time += (
                1.0
                / source_sfreq
            )


            lsl_timestamp = (

                timestamps[
                    sample_index
                ]

                if sample_index
                < len(timestamps)

                else np.nan

            )


            raw_records.append(

                {

                    "session_time_seconds":
                        session_time,

                    "lsl_timestamp":
                        lsl_timestamp,

                    "phase":
                        phase_name,

                    "true_label":
                        (
                            ""
                            if true_label is None
                            else true_label
                        ),

                    "fp1_raw":
                        eeg_raw[
                            0,
                            sample_index
                        ],

                    "fp2_raw":
                        eeg_raw[
                            1,
                            sample_index
                        ],

                    "fp1_volts":
                        eeg_volts[
                            0,
                            sample_index
                        ],

                    "fp2_volts":
                        eeg_volts[
                            1,
                            sample_index
                        ],

                }

            )


        # ====================================================
        # SETTLE / PREPARE 不做推理
        # ====================================================

        if true_label is None:

            continue


        # ====================================================
        # Rolling Buffer
        # ====================================================

        buffer = np.concatenate(

            [
                buffer,
                eeg_volts,
            ],

            axis=1,

        )


        # 只留最近 4 秒

        if (
            buffer.shape[1]
            > window_samples
        ):

            buffer = buffer[
                :,
                -window_samples:
            ]


        samples_since_prediction += (
            current_chunk_samples
        )


        # 还没满 4 秒

        if (
            buffer.shape[1]
            < window_samples
        ):

            continue


        # 每 1 秒预测一次

        if (
            samples_since_prediction
            < step_samples
        ):

            continue


        samples_since_prediction = 0


        pipeline_start = (
            time.perf_counter()
        )


        # ====================================================
        # Preprocess
        # ====================================================

        (
            processed,
            quality,
            max_uv,

        ) = preprocess_window(

            buffer.copy(),

            source_sfreq,

            filter_sos,

        )


        # ====================================================
        # Artifact
        # ====================================================

        if processed is None:

            print(

                f"{session_time:6.1f}s | "
                f"{phase_name:<5} | "
                f"{quality:<8} | "
                f"Max={max_uv:7.1f} μV"

            )


            prediction_records.append(

                {

                    "time_seconds":
                        session_time,

                    "phase":
                        phase_name,

                    "true_label":
                        true_label,

                    "predicted_label":
                        "",

                    "predicted_state":
                        "",

                    "p_rest":
                        "",

                    "p_task_raw":
                        "",

                    "p_task_smooth":
                        "",

                    "threshold":
                        TASK_THRESHOLD,

                    "quality":
                        quality,

                    "max_abs_uv":
                        max_uv,

                    "model_latency_ms":
                        "",

                    "pipeline_latency_ms":
                        "",

                }

            )


            continue


        # ====================================================
        # Inference
        # ====================================================

        (
            p_rest,
            p_task,
            model_latency,

        ) = infer(

            model,

            processed,

        )


        probability_history.append(
            p_task
        )


        p_task_smooth = float(

            np.mean(
                probability_history
            )

        )


        predicted_label = int(

            p_task_smooth

            >= TASK_THRESHOLD

        )


        predicted_state = (

            "TASK"

            if predicted_label == 1

            else "REST"

        )


        pipeline_latency = (

            time.perf_counter()

            - pipeline_start

        ) * 1000


        # ====================================================
        # Terminal Demo
        # ====================================================

        print(

            f"{session_time:6.1f}s | "

            f"True={phase_name:<5} | "

            f"Pred={predicted_state:<4} | "

            f"P(Task)="
            f"{p_task_smooth:.3f} | "

            f"Max="
            f"{max_uv:6.1f}μV | "

            f"Latency="
            f"{pipeline_latency:.1f} ms"

        )


        # ====================================================
        # Prediction Record
        # ====================================================

        prediction_records.append(

            {

                "time_seconds":
                    session_time,

                "phase":
                    phase_name,

                "true_label":
                    true_label,

                "predicted_label":
                    predicted_label,

                "predicted_state":
                    predicted_state,

                "p_rest":
                    p_rest,

                "p_task_raw":
                    p_task,

                "p_task_smooth":
                    p_task_smooth,

                "threshold":
                    TASK_THRESHOLD,

                "quality":
                    quality,

                "max_abs_uv":
                    max_uv,

                "model_latency_ms":
                    model_latency,

                "pipeline_latency_ms":
                    pipeline_latency,

            }

        )


# ============================================================
# 30. Start Protocol
# ============================================================

print("\n")
print("=" * 80)
print("真机实验准备")
print("=" * 80)


print(
    "\n实验流程："
)


print(
    f"{SETTLE_SECONDS}s SETTLE"
)


print(
    f"{REST_SECONDS}s REST_A"
)


print(
    f"{PREPARE_SECONDS}s PREPARE"
)


print(
    f"{TASK_SECONDS}s TASK_A"
)


print(
    f"{REST_SECONDS}s REST_B"
)


print(
    f"{PREPARE_SECONDS}s PREPARE"
)


print(
    f"{TASK_SECONDS}s TASK_B"
)


print(
    "\n受试者要求："
)


print(
    "1. 坐姿放松"
)


print(
    "2. 尽量保持头部不动"
)


print(
    "3. REST_A / REST_B 阶段闭眼放松"
)


print(
    "4. TASK_A / TASK_B 阶段闭眼进行连续心算"
)


print(
    "5. 不说出答案"
)


print(
    "6. 不使用手指数数"
)


print(
    "7. 尽量避免皱眉、咬牙、大幅眨眼"
)


input(

    "\n确认 LoongBrain 阻抗正常，"
    "受试者准备好后按 Enter 开始..."

)


# ============================================================
# 31. Guided Experiment
# ============================================================

run_phase(

    phase_name="SETTLE",

    duration_seconds=SETTLE_SECONDS,

    true_label=None,

)


run_phase(

    phase_name="REST_A",

    duration_seconds=REST_SECONDS,

    true_label=0,

)


run_phase(

    phase_name="PREPARE",

    duration_seconds=PREPARE_SECONDS,

    true_label=None,

)


run_phase(

    phase_name="TASK_A",

    duration_seconds=TASK_SECONDS,

    true_label=1,

)


run_phase(

    phase_name="REST_B",

    duration_seconds=REST_SECONDS,

    true_label=0,

)


run_phase(

    phase_name="PREPARE",

    duration_seconds=PREPARE_SECONDS,

    true_label=None,

)


run_phase(

    phase_name="TASK_B",

    duration_seconds=TASK_SECONDS,

    true_label=1,

)


beep()


# ============================================================
# 32. 保存 Raw EEG
# ============================================================

if len(raw_records) > 0:

    with open(

        RAW_CSV_PATH,

        "w",

        newline="",

        encoding="utf-8",

    ) as file:


        writer = csv.DictWriter(

            file,

            fieldnames=[

                "session_time_seconds",

                "lsl_timestamp",

                "phase",

                "true_label",

                "fp1_raw",

                "fp2_raw",

                "fp1_volts",

                "fp2_volts",

            ],

        )


        writer.writeheader()


        writer.writerows(
            raw_records
        )


# ============================================================
# 33. 保存 Prediction
# ============================================================

if len(prediction_records) > 0:

    with open(

        PREDICTION_CSV_PATH,

        "w",

        newline="",

        encoding="utf-8",

    ) as file:


        writer = csv.DictWriter(

            file,

            fieldnames=[

                "time_seconds",

                "phase",

                "true_label",

                "predicted_label",

                "predicted_state",

                "p_rest",

                "p_task_raw",

                "p_task_smooth",

                "threshold",

                "quality",

                "max_abs_uv",

                "model_latency_ms",

                "pipeline_latency_ms",

            ],

        )


        writer.writeheader()


        writer.writerows(
            prediction_records
        )


# ============================================================
# 34. Session Summary
# ============================================================

valid_predictions = [

    row

    for row in prediction_records

    if row[
        "predicted_label"
    ] != ""

]


print("\n")
print("=" * 80)
print("LoongBrain Real-time Session 完成")
print("=" * 80)


print(
    "\nRaw EEG Samples:",
    len(raw_records),
)


print(
    "Valid Predictions:",
    len(valid_predictions),
)


if len(valid_predictions) > 0:

    y_true = np.asarray(

        [

            int(
                row[
                    "true_label"
                ]
            )

            for row in valid_predictions

        ]

    )


    y_pred = np.asarray(

        [

            int(
                row[
                    "predicted_label"
                ]
            )

            for row in valid_predictions

        ]

    )


    latency = np.asarray(

        [

            float(
                row[
                    "pipeline_latency_ms"
                ]
            )

            for row in valid_predictions

        ]

    )


    accuracy = np.mean(

        y_true
        == y_pred

    )


    print(
        "\nSession Accuracy:",
        f"{accuracy:.4f}",
    )


    print(
        "Mean Pipeline Latency:",
        f"{latency.mean():.2f} ms",
    )


    print(
        "95% Pipeline Latency:",
        f"{np.percentile(latency, 95):.2f} ms",
    )



# ============================================================
# 35. Calibration Analysis
# ============================================================

CALIBRATION_SUMMARY_PATH = (
    RESULT_DIR
    / f"calibration_summary_{session_id}.csv"
)


def _safe_float(value):

    try:
        return float(value)

    except Exception:
        return np.nan


calibration_valid = [

    row

    for row in prediction_records

    if (
        row["predicted_label"] != ""
        and np.isfinite(
            _safe_float(
                row["p_task_smooth"]
            )
        )
    )

]


print("\n")
print("=" * 80)
print("Calibration Analysis")
print("=" * 80)


phase_order = [
    "REST_A",
    "TASK_A",
    "REST_B",
    "TASK_B",
]


phase_stats = {}


for phase_name in phase_order:

    values = np.asarray(

        [
            _safe_float(
                row["p_task_smooth"]
            )

            for row in calibration_valid

            if row["phase"] == phase_name
        ],

        dtype=float,

    )


    values = values[
        np.isfinite(values)
    ]


    if len(values) > 0:

        phase_stats[
            phase_name
        ] = {

            "n": len(values),

            "mean": float(
                np.mean(values)
            ),

            "median": float(
                np.median(values)
            ),

            "min": float(
                np.min(values)
            ),

            "max": float(
                np.max(values)
            ),

        }


        print(
            f"{phase_name:<7} | "
            f"n={len(values):2d} | "
            f"mean={np.mean(values):.4f} | "
            f"median={np.median(values):.4f} | "
            f"range=[{np.min(values):.4f}, {np.max(values):.4f}]"
        )

    else:

        print(
            f"{phase_name:<7} | no valid predictions"
        )


rest_values = np.asarray(

    [
        _safe_float(
            row["p_task_smooth"]
        )

        for row in calibration_valid

        if int(
            row["true_label"]
        ) == 0
    ],

    dtype=float,

)


task_values = np.asarray(

    [
        _safe_float(
            row["p_task_smooth"]
        )

        for row in calibration_valid

        if int(
            row["true_label"]
        ) == 1
    ],

    dtype=float,

)


rest_values = rest_values[
    np.isfinite(rest_values)
]

task_values = task_values[
    np.isfinite(task_values)
]


suggested_threshold = np.nan
suggested_direction = ""
best_balanced_accuracy = np.nan


if (
    len(rest_values) >= 5
    and len(task_values) >= 5
):

    all_values = np.sort(

        np.unique(

            np.concatenate(
                [
                    rest_values,
                    task_values,
                ]
            )

        )

    )


    candidates = []


    if len(all_values) == 1:

        candidates = [
            float(
                all_values[0]
            )
        ]


    else:

        candidates = [

            float(
                (
                    all_values[i]
                    + all_values[i + 1]
                )
                / 2.0
            )

            for i in range(
                len(all_values) - 1
            )

        ]


    best = None


    for direction in [
        "HIGHER",
        "LOWER",
    ]:

        for threshold in candidates:

            if direction == "HIGHER":

                rest_correct = np.mean(
                    rest_values
                    < threshold
                )

                task_correct = np.mean(
                    task_values
                    >= threshold
                )


            else:

                rest_correct = np.mean(
                    rest_values
                    > threshold
                )

                task_correct = np.mean(
                    task_values
                    <= threshold
                )


            balanced_accuracy = (

                rest_correct
                + task_correct

            ) / 2.0


            candidate = (
                balanced_accuracy,
                threshold,
                direction,
                rest_correct,
                task_correct,
            )


            if (
                best is None
                or candidate[0]
                > best[0]
            ):

                best = candidate


    (
        best_balanced_accuracy,
        suggested_threshold,
        suggested_direction,
        rest_specificity,
        task_sensitivity,

    ) = best


    print("\nPooled calibration:")

    print(
        "REST mean / median:",
        f"{np.mean(rest_values):.4f} / "
        f"{np.median(rest_values):.4f}",
    )

    print(
        "TASK mean / median:",
        f"{np.mean(task_values):.4f} / "
        f"{np.median(task_values):.4f}",
    )


    print("\nSuggested rule FOR NEXT INDEPENDENT TEST:")

    if suggested_direction == "HIGHER":

        print(
            f"TASK if P(Task) >= "
            f"{suggested_threshold:.4f}"
        )

    else:

        print(
            f"TASK if P(Task) <= "
            f"{suggested_threshold:.4f}"
        )


    print(
        "Calibration Balanced Accuracy:",
        f"{best_balanced_accuracy:.4f}",
    )

    print(
        "REST specificity:",
        f"{rest_specificity:.4f}",
    )

    print(
        "TASK sensitivity:",
        f"{task_sensitivity:.4f}",
    )


    print(
        "\nIMPORTANT: "
        "这个阈值只能用于下一轮独立 Final Test，"
        "不能把本轮 calibration accuracy 当最终测试成绩。"
    )


else:

    print(
        "\n有效 REST/TASK 窗口不足，"
        "暂时不建议确定个人阈值。"
    )


with open(

    CALIBRATION_SUMMARY_PATH,

    "w",

    newline="",

    encoding="utf-8",

) as file:

    writer = csv.writer(
        file
    )


    writer.writerow(
        [
            "item",
            "value",
        ]
    )


    writer.writerow(
        [
            "rest_valid_windows",
            len(rest_values),
        ]
    )


    writer.writerow(
        [
            "task_valid_windows",
            len(task_values),
        ]
    )


    if len(rest_values) > 0:

        writer.writerow(
            [
                "rest_mean_p_task",
                float(
                    np.mean(rest_values)
                ),
            ]
        )

        writer.writerow(
            [
                "rest_median_p_task",
                float(
                    np.median(rest_values)
                ),
            ]
        )


    if len(task_values) > 0:

        writer.writerow(
            [
                "task_mean_p_task",
                float(
                    np.mean(task_values)
                ),
            ]
        )

        writer.writerow(
            [
                "task_median_p_task",
                float(
                    np.median(task_values)
                ),
            ]
        )


    writer.writerow(
        [
            "suggested_direction",
            suggested_direction,
        ]
    )


    writer.writerow(
        [
            "suggested_threshold",
            suggested_threshold,
        ]
    )


    writer.writerow(
        [
            "calibration_balanced_accuracy",
            best_balanced_accuracy,
        ]
    )


print(
    "\nCalibration Summary:"
)

print(
    CALIBRATION_SUMMARY_PATH
)



print("\n")
print("=" * 80)
print("Saved Files")
print("=" * 80)


print(
    "\nRaw EEG:"
)

print(
    RAW_CSV_PATH
)


print(
    "\nPredictions:"
)

print(
    PREDICTION_CSV_PATH
)