from pathlib import Path
from collections import deque, Counter
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
# 35_loongbrain_final_test.py
#
# 独立 Final Test：
#
# - 使用原 LoRA Encoder
# - 加载 34 号脚本生成的 Personal Linear Head
# - 不再使用 calibration 数据训练任何参数
# - 重新采集一整个新 session
# - 同时输出：
#       Personal Head 结果
#       原 Source LoRA 结果（作为实时对照）
#
# Final Test block 顺序故意与 calibration 起始顺序不同：
#
# SETTLE
# TASK_A -> REST_A -> TASK_B -> REST_B
#
# 用来降低固定 REST->TASK 顺序与时间漂移混淆。
# ============================================================


# ============================================================
# 1. PyTorch compatibility
# ============================================================

if (
    hasattr(torch.backends, "mha")
    and hasattr(
        torch.backends.mha,
        "set_fastpath_enabled",
    )
):

    torch.backends.mha.set_fastpath_enabled(
        False
    )


# ============================================================
# 2. Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "lora"
    / "best.pt"
)


PERSONAL_HEAD_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "personal_head"
    / "best.pt"
)


RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "loongbrain_final"
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. LSL config
# ============================================================

# 如果已经知道自己的 Stream Name，
# 可以直接填写：
#
# TARGET_STREAM_NAME = "EEG-xxxx"
#
# 保持 None 时：
# - 只有一条 EEG -> 自动使用
# - 多条 EEG -> 终端列出并让你输入编号

TARGET_STREAM_NAME = None


# 如果没有 Fp1/Fp2 label，
# 但已经确认 channel 0 / 1 就是 Fp1 / Fp2：
#
# MANUAL_CHANNEL_INDICES = (0, 1)

MANUAL_CHANNEL_INDICES = None


# ============================================================
# 4. Signal / model config
# ============================================================

MODEL_SFREQ = 160.0

MODEL_WINDOW_SECONDS = 4.0

MODEL_WINDOW_SAMPLES = int(
    MODEL_SFREQ
    * MODEL_WINDOW_SECONDS
)


PREDICTION_STEP_SECONDS = 1.0

SMOOTHING_WINDOWS = 5

MAX_ABS_UV = 500.0


# ============================================================
# 5. Final Test protocol
# ============================================================

SETTLE_SECONDS = 10

TRANSITION_SECONDS = 5

BLOCK_SECONDS = 20


# 心算任务

START_NUMBER = 4387

SUBTRACTOR = 47


# ============================================================
# 6. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 7. Beep
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
# 8. LoRA
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

        self.lora_dropout = (
            nn.Dropout(
                dropout
            )
        )

        self.lora_A = (
            nn.Linear(
                base_linear.in_features,
                rank,
                bias=False,
            )
        )

        self.lora_B = (
            nn.Linear(
                rank,
                base_linear.out_features,
                bias=False,
            )
        )


    @property
    def weight(
        self
    ):

        return (
            self.base_linear.weight
        )


    @property
    def bias(
        self
    ):

        return (
            self.base_linear.bias
        )


    def forward(
        self,
        x,
    ):

        return (

            self.base_linear(
                x
            )

            +

            self.lora_B(

                self.lora_A(

                    self.lora_dropout(
                        x
                    )

                )

            )

            * self.scaling

        )


# ============================================================
# 9. Source model
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

        self.classifier = (
            nn.Linear(
                embed_dim,
                2,
            )
        )


    def extract_features(
        self,
        x,
    ):

        tokens = (
            self.backbone
            .patch_embedding(
                x
            )
        )

        representations = (
            self.backbone
            .encoder(
                tokens
            )
        )

        representations = (
            self.backbone
            .norm(
                representations
            )
        )

        return (
            representations
            .mean(
                dim=1
            )
        )


    def forward(
        self,
        x,
    ):

        features = (
            self.extract_features(
                x
            )
        )

        return (
            self.classifier(
                features
            )
        )


# ============================================================
# 10. Load source model + personal head
# ============================================================

def load_models():

    if not SOURCE_MODEL_PATH.exists():

        raise FileNotFoundError(
            "\n找不到 Source LoRA：\n"
            f"{SOURCE_MODEL_PATH}"
        )


    if not PERSONAL_HEAD_PATH.exists():

        raise FileNotFoundError(

            "\n找不到 Personal Head：\n"
            f"{PERSONAL_HEAD_PATH}\n\n"
            "请先运行：\n"
            "python src\\34_train_personal_head.py"

        )


    source_checkpoint = (
        torch.load(
            SOURCE_MODEL_PATH,
            map_location="cpu",
        )
    )


    config = (
        source_checkpoint[
            "config"
        ]
    )


    backbone = (
        MaskedEEGTransformer(

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
    )


    rank = (
        source_checkpoint.get(
            "lora_rank",
            8,
        )
    )

    alpha = (
        source_checkpoint.get(
            "lora_alpha",
            16,
        )
    )

    lora_dropout = (
        source_checkpoint.get(
            "lora_dropout",
            0.05,
        )
    )


    for layer in (
        backbone
        .encoder
        .layers
    ):

        layer.linear1 = (
            LoRALinear(

                layer.linear1,

                rank=rank,
                alpha=alpha,
                dropout=lora_dropout,

            )
        )

        layer.linear2 = (
            LoRALinear(

                layer.linear2,

                rank=rank,
                alpha=alpha,
                dropout=lora_dropout,

            )
        )


    source_model = (
        EEGLoraClassifier(

            backbone=backbone,

            embed_dim=config[
                "embed_dim"
            ],

        )
    )


    source_model.load_state_dict(
        source_checkpoint[
            "model_state_dict"
        ]
    )


    source_model = (
        source_model
        .to(
            device
        )
    )

    source_model.eval()


    for parameter in (
        source_model.parameters()
    ):

        parameter.requires_grad = (
            False
        )


    # --------------------------------------------------------
    # Personal head
    # --------------------------------------------------------

    personal_checkpoint = (
        torch.load(
            PERSONAL_HEAD_PATH,
            map_location="cpu",
        )
    )


    personal_head = (
        nn.Linear(
            config[
                "embed_dim"
            ],
            2,
        )
    )


    personal_head.load_state_dict(
        personal_checkpoint[
            "personal_head_state_dict"
        ]
    )


    personal_head = (
        personal_head
        .to(
            device
        )
    )

    personal_head.eval()


    for parameter in (
        personal_head.parameters()
    ):

        parameter.requires_grad = (
            False
        )


    feature_mean = (
        personal_checkpoint[
            "feature_mean"
        ]
        .float()
        .to(
            device
        )
    )


    feature_std = (
        personal_checkpoint[
            "feature_std"
        ]
        .float()
        .to(
            device
        )
    )


    feature_mask = (
        personal_checkpoint[
            "feature_mask"
        ]
        .float()
        .to(
            device
        )
    )


    print(
        "\n✓ Source LoRA loaded"
    )

    print(
        "✓ Personal Head loaded"
    )


    print(
        "Personal Top-K:",
        personal_checkpoint.get(
            "selected_top_k",
            "unknown",
        ),
    )


    print(
        "Calibration Block-CV BA:",
        f"{personal_checkpoint.get('block_cv_balanced_accuracy', float('nan')):.4f}",
    )


    if (
        personal_checkpoint.get(
            "block_cv_balanced_accuracy",
            0.0,
        )
        < 0.55
    ):

        print(
            "⚠ Calibration Block-CV 较低；"
            "Final Test 结果应如实报告，"
            "不要预设 Personal Head 一定优于 Source LoRA。"
        )


    return (
        source_model,
        personal_head,
        feature_mean,
        feature_std,
        feature_mask,
        personal_checkpoint,
    )


# ============================================================
# 11. Channel metadata
# ============================================================

def get_channel_metadata(
    info
):

    labels = []
    units = []


    try:

        channel = (
            info
            .desc()
            .child(
                "channels"
            )
            .child(
                "channel"
            )
        )


        for _ in range(
            info.channel_count()
        ):

            label = (
                channel
                .child_value(
                    "label"
                )
            )

            unit = (
                channel
                .child_value(
                    "unit"
                )
            )


            labels.append(
                label
                if label
                else ""
            )

            units.append(
                unit
                if unit
                else ""
            )


            channel = (
                channel
                .next_sibling()
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


    return (
        labels,
        units,
    )


def clean_channel_name(
    name
):

    return (
        name
        .lower()
        .replace(
            "eeg",
            "",
        )
        .replace(
            " ",
            "",
        )
        .replace(
            ".",
            "",
        )
        .replace(
            "-",
            "",
        )
        .replace(
            "_",
            "",
        )
    )


# ============================================================
# 12. Discover stream
# ============================================================

def discover_stream():

    print(
        "\n正在搜索 LSL Stream ..."
    )


    streams = (
        resolve_streams(
            wait_time=5.0
        )
    )


    if len(streams) == 0:

        raise RuntimeError(

            "\n没有发现任何 LSL Stream。\n"
            "请检查 LoongBrain 已连接且 LSL 广播已经打开。"

        )


    if TARGET_STREAM_NAME is not None:

        matches = [

            stream

            for stream in streams

            if stream.name()
            == TARGET_STREAM_NAME

        ]


        if len(matches) != 1:

            raise RuntimeError(

                f"没有唯一找到："
                f"{TARGET_STREAM_NAME}"

            )


        return matches[0]


    candidates = []


    for stream in streams:

        text = (

            stream.name()

            + " "

            + stream.type()

        ).lower()


        if (

            stream.type().lower()
            == "eeg"

            or "eeg" in text

            or "loong" in text

        ):

            candidates.append(
                stream
            )


    if len(candidates) == 0:

        raise RuntimeError(

            "没有识别到 EEG Stream。\n"
            "请先运行 29_lsl_stream_check.py。"

        )


    if len(candidates) == 1:

        return candidates[0]


    print(
        "\n发现多个 EEG Stream："
    )


    for index, stream in enumerate(
        candidates
    ):

        print(

            f"[{index}] "
            f"{stream.name()} | "
            f"type={stream.type()} | "
            f"channels={stream.channel_count()} | "
            f"sfreq={stream.nominal_srate()} | "
            f"source_id={stream.source_id()}"

        )


    while True:

        value = input(
            "\n请输入属于你这台 LoongBrain 的编号："
        ).strip()


        try:

            index = int(
                value
            )

        except ValueError:

            print(
                "请输入整数编号。"
            )

            continue


        if (
            0
            <= index
            < len(candidates)
        ):

            return candidates[
                index
            ]


        print(
            "编号超出范围。"
        )


# ============================================================
# 13. Select Fp1 / Fp2
# ============================================================

def select_channels(
    info
):

    (
        labels,
        units,

    ) = get_channel_metadata(
        info
    )


    print(
        "\nChannel Labels:",
        labels,
    )

    print(
        "Channel Units:",
        units,
    )


    if (
        MANUAL_CHANNEL_INDICES
        is not None
    ):

        return (
            list(
                MANUAL_CHANNEL_INDICES
            ),
            labels,
            units,
        )


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

        return (
            [
                fp1_index,
                fp2_index,
            ],
            labels,
            units,
        )


    if (
        info.channel_count() == 2
        and all(
            label == ""
            for label in labels
        )
    ):

        print(
            "⚠ 无 channel label，"
            "当前只有 2 通道，暂按 0=Fp1, 1=Fp2。"
        )

        return (
            [
                0,
                1,
            ],
            labels,
            units,
        )


    raise RuntimeError(

        "无法确定 Fp1/Fp2。\n"
        "请先运行 29，"
        "再设置 MANUAL_CHANNEL_INDICES。"

    )


# ============================================================
# 14. Unit scaling
# ============================================================

def determine_scale_to_volts(
    selected_units,
    example_data,
):

    unit_text = " ".join(
        selected_units
    ).lower()


    if (
        "uv" in unit_text
        or "µv" in unit_text
        or "microvolt" in unit_text
    ):

        print(
            "检测到 EEG 单位：μV"
        )

        return 1e-6


    if "mv" in unit_text:

        print(
            "检测到 EEG 单位：mV"
        )

        return 1e-3


    if (
        unit_text.strip()
        in {
            "v",
            "volt",
            "volts",
        }
    ):

        print(
            "检测到 EEG 单位：V"
        )

        return 1.0


    p95 = (
        np.percentile(
            np.abs(
                example_data
            ),
            95,
        )
    )


    print(
        "LSL 没有明确单位。"
    )

    print(
        "Raw amplitude P95:",
        p95,
    )


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
# 15. Preprocess
# ============================================================

def preprocess_window(
    window_volts,
    source_sfreq,
    filter_sos,
):

    if not np.isfinite(
        window_volts
    ).all():

        return (
            None,
            "NaN/Inf",
            np.nan,
        )


    # First filter, then artifact QC.

    filtered = (
        sosfiltfilt(

            filter_sos,

            window_volts,

            axis=-1,

        )
    )


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


    ratio = (
        Fraction(
            MODEL_SFREQ
            / source_sfreq
        )
        .limit_denominator(
            1000
        )
    )


    resampled = (
        resample_poly(

            filtered,

            up=ratio.numerator,

            down=ratio.denominator,

            axis=-1,

        )
    )


    if (
        resampled.shape[1]
        > MODEL_WINDOW_SAMPLES
    ):

        resampled = (
            resampled[
                :,
                :MODEL_WINDOW_SAMPLES
            ]
        )


    elif (
        resampled.shape[1]
        < MODEL_WINDOW_SAMPLES
    ):

        missing = (

            MODEL_WINDOW_SAMPLES

            - resampled.shape[1]

        )


        resampled = (
            np.pad(

                resampled,

                (
                    (0, 0),
                    (0, missing),
                ),

                mode="edge",

            )
        )


    mean = (
        resampled.mean(
            axis=1,
            keepdims=True,
        )
    )


    std = (
        resampled.std(
            axis=1,
            keepdims=True,
        )
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

        (
            resampled
            - mean
        )

        /

        (
            std
            + 1e-8
        )

    ).astype(
        np.float32
    )


    return (
        normalized,
        "OK",
        max_uv,
    )


# ============================================================
# 16. Inference: Source + Personal with one encoder pass
# ============================================================

def infer_both(
    source_model,
    personal_head,
    feature_mean,
    feature_std,
    feature_mask,
    eeg,
):

    tensor = (
        torch.from_numpy(
            eeg
        )
        .unsqueeze(
            0
        )
        .to(
            device
        )
    )


    start = (
        time.perf_counter()
    )


    with torch.no_grad():

        features = (
            source_model
            .extract_features(
                tensor
            )
        )


        source_logits = (
            source_model
            .classifier(
                features
            )
        )


        normalized_features = (

            (
                features
                - feature_mean
            )

            / feature_std

        )


        normalized_features = (

            normalized_features

            * feature_mask

        )


        personal_logits = (
            personal_head(
                normalized_features
            )
        )


        source_probability = (
            torch.softmax(
                source_logits,
                dim=1,
            )
        )


        personal_probability = (
            torch.softmax(
                personal_logits,
                dim=1,
            )
        )


    model_latency_ms = (

        time.perf_counter()
        - start

    ) * 1000.0


    return (
        float(
            source_probability[
                0,
                1
            ].item()
        ),
        float(
            personal_probability[
                0,
                1
            ].item()
        ),
        model_latency_ms,
    )


# ============================================================
# 17. Metrics
# ============================================================

def binary_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )


    accuracy = float(
        np.mean(
            y_true
            == y_pred
        )
    )


    recalls = []
    f1s = []


    for label in [
        0,
        1,
    ]:

        tp = np.sum(
            (y_true == label)
            & (y_pred == label)
        )

        fn = np.sum(
            (y_true == label)
            & (y_pred != label)
        )

        fp = np.sum(
            (y_true != label)
            & (y_pred == label)
        )


        recall = (
            tp
            / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )


        precision = (
            tp
            / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )


        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )


        recalls.append(
            recall
        )

        f1s.append(
            f1
        )


    return {
        "accuracy":
            accuracy,
        "balanced_accuracy":
            float(
                np.mean(
                    recalls
                )
            ),
        "macro_f1":
            float(
                np.mean(
                    f1s
                )
            ),
    }


# ============================================================
# 18. Phase display
# ============================================================

def show_phase(
    phase
):

    print("\n")
    print("=" * 80)


    if phase == "SETTLE":

        print(
            f"SETTLE | {SETTLE_SECONDS}s"
        )

        print(
            "闭眼、放松、保持不动。"
        )


    elif phase == "PREPARE_TASK":

        beep()

        print(
            f"PREPARE TASK | {TRANSITION_SECONDS}s"
        )

        print(
            f"准备从 {START_NUMBER} 开始，"
            f"持续减 {SUBTRACTOR}。"
        )


    elif phase == "PREPARE_REST":

        beep()

        print(
            f"PREPARE REST | {TRANSITION_SECONDS}s"
        )

        print(
            "停止心算，放松额头和眼睛。"
        )


    elif phase.startswith(
        "TASK"
    ):

        beep()

        print(
            f"{phase} | {BLOCK_SECONDS}s"
        )

        print(
            f"持续在脑中计算："
            f"{START_NUMBER}"
            f" - {SUBTRACTOR}"
            f" - {SUBTRACTOR}"
            f" - ..."
        )

        print(
            "不说话、不数手指、不皱眉，尽量保持不动。"
        )


    elif phase.startswith(
        "REST"
    ):

        beep()

        print(
            f"{phase} | {BLOCK_SECONDS}s"
        )

        print(
            "闭眼、放松、不做心算。"
        )


    print("=" * 80)


# ============================================================
# 19. Main
# ============================================================

def main():

    print("=" * 80)
    print("LoongBrain Independent Final Test | Personal Head")
    print("=" * 80)

    print(
        "\nTorch Device:",
        device,
    )


    (
        source_model,
        personal_head,
        feature_mean,
        feature_std,
        feature_mask,
        personal_checkpoint,

    ) = load_models()


    # --------------------------------------------------------
    # LSL
    # --------------------------------------------------------

    stream_info = (
        discover_stream()
    )


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
        "Source ID:",
        stream_info.source_id(),
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


    if source_sfreq <= 100:

        raise RuntimeError(
            "当前采样率不足以进行 1–45 Hz 处理。"
        )


    (
        channel_indices,
        labels,
        units,

    ) = select_channels(
        stream_info
    )


    print(
        "Selected indices:",
        channel_indices,
    )


    inlet = (
        StreamInlet(
            stream_info,
            max_buflen=10,
        )
    )


    # --------------------------------------------------------
    # Warm-up + unit
    # --------------------------------------------------------

    print(
        "\n正在读取短数据进行信号检查..."
    )


    (
        warm_samples,
        _,

    ) = inlet.pull_chunk(

        timeout=2.0,

        max_samples=max(
            int(
                source_sfreq
            ),
            100,
        ),

    )


    if len(warm_samples) == 0:

        raise RuntimeError(
            "找到了 Stream，但没有收到 EEG 样本。"
        )


    warm_data = (
        np.asarray(
            warm_samples,
            dtype=np.float64,
        )
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
        "Scale to volts:",
        scale_to_volts,
    )


    # --------------------------------------------------------
    # Filter / realtime parameters
    # --------------------------------------------------------

    filter_sos = (
        butter(

            N=4,

            Wn=[
                1.0,
                45.0,
            ],

            btype="bandpass",

            fs=source_sfreq,

            output="sos",

        )
    )


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


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    session_id = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )


    raw_csv_path = (
        RESULT_DIR
        / f"raw_eeg_{session_id}.csv"
    )


    prediction_csv_path = (
        RESULT_DIR
        / f"predictions_{session_id}.csv"
    )


    summary_csv_path = (
        RESULT_DIR
        / f"final_summary_{session_id}.csv"
    )


    raw_records = []
    prediction_records = []

    session_time = 0.0


    # ========================================================
    # Run phase
    # ========================================================

    def run_phase(
        phase_name,
        duration_seconds,
        true_label,
    ):

        nonlocal session_time


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


        buffer = np.empty(
            (
                2,
                0,
            ),
            dtype=np.float64,
        )


        samples_since_prediction = 0


        personal_history = deque(
            maxlen=SMOOTHING_WINDOWS
        )

        source_history = deque(
            maxlen=SMOOTHING_WINDOWS
        )


        while (
            received_samples
            < target_samples
        ):

            (
                samples,
                timestamps,

            ) = inlet.pull_chunk(

                timeout=0.5,

                max_samples=max(
                    int(
                        source_sfreq
                    ),
                    50,
                ),

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


            remaining = (

                target_samples

                - received_samples

            )


            if (
                chunk.shape[0]
                > remaining
            ):

                chunk = (
                    chunk[
                        :remaining
                    ]
                )

                timestamps = (
                    timestamps[
                        :remaining
                    ]
                )


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


            # ------------------------------------------------
            # Save raw samples
            # ------------------------------------------------

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
                                if true_label
                                is None
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


            # Transition / settle: no prediction

            if true_label is None:

                continue


            buffer = (
                np.concatenate(
                    [
                        buffer,
                        eeg_volts,
                    ],
                    axis=1,
                )
            )


            if (
                buffer.shape[1]
                > window_samples
            ):

                buffer = (
                    buffer[
                        :,
                        -window_samples:
                    ]
                )


            samples_since_prediction += (
                current_chunk_samples
            )


            if (
                buffer.shape[1]
                < window_samples
            ):

                continue


            if (
                samples_since_prediction
                < step_samples
            ):

                continue


            samples_since_prediction = 0


            pipeline_start = (
                time.perf_counter()
            )


            (
                processed,
                quality,
                max_uv,

            ) = preprocess_window(

                buffer.copy(),

                source_sfreq,

                filter_sos,

            )


            if processed is None:

                print(

                    f"{session_time:6.1f}s | "
                    f"{phase_name:<6} | "
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

                        "personal_predicted_label":
                            "",

                        "personal_predicted_state":
                            "",

                        "personal_p_task_raw":
                            "",

                        "personal_p_task_smooth":
                            "",

                        "source_predicted_label":
                            "",

                        "source_predicted_state":
                            "",

                        "source_p_task_raw":
                            "",

                        "source_p_task_smooth":
                            "",

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


            (
                source_p_task,
                personal_p_task,
                model_latency,

            ) = infer_both(

                source_model,

                personal_head,

                feature_mean,

                feature_std,

                feature_mask,

                processed,

            )


            source_history.append(
                source_p_task
            )

            personal_history.append(
                personal_p_task
            )


            source_p_smooth = float(
                np.mean(
                    source_history
                )
            )


            personal_p_smooth = float(
                np.mean(
                    personal_history
                )
            )


            source_pred = int(
                source_p_smooth
                >= 0.5
            )


            personal_pred = int(
                personal_p_smooth
                >= 0.5
            )


            source_state = (
                "TASK"
                if source_pred == 1
                else "REST"
            )


            personal_state = (
                "TASK"
                if personal_pred == 1
                else "REST"
            )


            pipeline_latency = (

                time.perf_counter()

                - pipeline_start

            ) * 1000.0


            print(

                f"{session_time:6.1f}s | "
                f"True={phase_name:<6} | "
                f"Personal={personal_state:<4} "
                f"P={personal_p_smooth:.3f} | "
                f"Source={source_state:<4} "
                f"P={source_p_smooth:.3f} | "
                f"Max={max_uv:6.1f}μV | "
                f"Latency={pipeline_latency:.1f}ms"

            )


            prediction_records.append(
                {
                    "time_seconds":
                        session_time,

                    "phase":
                        phase_name,

                    "true_label":
                        true_label,

                    "personal_predicted_label":
                        personal_pred,

                    "personal_predicted_state":
                        personal_state,

                    "personal_p_task_raw":
                        personal_p_task,

                    "personal_p_task_smooth":
                        personal_p_smooth,

                    "source_predicted_label":
                        source_pred,

                    "source_predicted_state":
                        source_state,

                    "source_p_task_raw":
                        source_p_task,

                    "source_p_task_smooth":
                        source_p_smooth,

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


    # ========================================================
    # Start
    # ========================================================

    print("\n")
    print("=" * 80)
    print("Independent Final Test Protocol")
    print("=" * 80)

    print(
        f"{SETTLE_SECONDS}s SETTLE"
    )

    print(
        f"{TRANSITION_SECONDS}s PREPARE TASK"
    )

    print(
        f"{BLOCK_SECONDS}s TASK_A"
    )

    print(
        f"{TRANSITION_SECONDS}s PREPARE REST"
    )

    print(
        f"{BLOCK_SECONDS}s REST_A"
    )

    print(
        f"{TRANSITION_SECONDS}s PREPARE TASK"
    )

    print(
        f"{BLOCK_SECONDS}s TASK_B"
    )

    print(
        f"{TRANSITION_SECONDS}s PREPARE REST"
    )

    print(
        f"{BLOCK_SECONDS}s REST_B"
    )


    print(
        "\n注意："
    )

    print(
        "1. 这是独立 Final Test，不能再根据本轮结果调 Personal Head。"
    )

    print(
        "2. TASK 只在脑中持续心算，不说话、不皱眉。"
    )

    print(
        "3. REST 完全停止心算并放松。"
    )


    input(
        "\n确认 LoongBrain 波形/阻抗正常后按 Enter 开始..."
    )


    run_phase(
        "SETTLE",
        SETTLE_SECONDS,
        None,
    )


    run_phase(
        "PREPARE_TASK",
        TRANSITION_SECONDS,
        None,
    )


    run_phase(
        "TASK_A",
        BLOCK_SECONDS,
        1,
    )


    run_phase(
        "PREPARE_REST",
        TRANSITION_SECONDS,
        None,
    )


    run_phase(
        "REST_A",
        BLOCK_SECONDS,
        0,
    )


    run_phase(
        "PREPARE_TASK",
        TRANSITION_SECONDS,
        None,
    )


    run_phase(
        "TASK_B",
        BLOCK_SECONDS,
        1,
    )


    run_phase(
        "PREPARE_REST",
        TRANSITION_SECONDS,
        None,
    )


    run_phase(
        "REST_B",
        BLOCK_SECONDS,
        0,
    )


    beep()


    # ========================================================
    # Save CSVs
    # ========================================================

    if len(raw_records) > 0:

        with open(
            raw_csv_path,
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


    if len(prediction_records) > 0:

        with open(
            prediction_csv_path,
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
                    "personal_predicted_label",
                    "personal_predicted_state",
                    "personal_p_task_raw",
                    "personal_p_task_smooth",
                    "source_predicted_label",
                    "source_predicted_state",
                    "source_p_task_raw",
                    "source_p_task_smooth",
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


    # ========================================================
    # Summary
    # ========================================================

    valid = [

        row

        for row in prediction_records

        if row[
            "personal_predicted_label"
        ] != ""

    ]


    quality_counts = Counter(

        row[
            "quality"
        ]

        for row in prediction_records

    )


    print("\n")
    print("=" * 80)
    print("INDEPENDENT FINAL TEST RESULT")
    print("=" * 80)


    print(
        "\nRaw EEG Samples:",
        len(raw_records),
    )

    print(
        "Prediction Rows:",
        len(prediction_records),
    )

    print(
        "Valid Predictions:",
        len(valid),
    )

    print(
        "Quality Counts:",
        dict(
            quality_counts
        ),
    )


    summary_items = []


    if len(valid) > 0:

        y_true = np.asarray(

            [
                int(
                    row[
                        "true_label"
                    ]
                )

                for row in valid
            ],

            dtype=int,

        )


        personal_pred = np.asarray(

            [
                int(
                    row[
                        "personal_predicted_label"
                    ]
                )

                for row in valid
            ],

            dtype=int,

        )


        source_pred = np.asarray(

            [
                int(
                    row[
                        "source_predicted_label"
                    ]
                )

                for row in valid
            ],

            dtype=int,

        )


        personal_metrics = (
            binary_metrics(

                y_true,

                personal_pred,

            )
        )


        source_metrics = (
            binary_metrics(

                y_true,

                source_pred,

            )
        )


        latency = np.asarray(

            [
                float(
                    row[
                        "pipeline_latency_ms"
                    ]
                )

                for row in valid
            ],

            dtype=float,

        )


        print("\nPersonal Head:")

        print(
            "  Accuracy:",
            f"{personal_metrics['accuracy']:.4f}",
        )

        print(
            "  Balanced Accuracy:",
            f"{personal_metrics['balanced_accuracy']:.4f}",
        )

        print(
            "  Macro F1:",
            f"{personal_metrics['macro_f1']:.4f}",
        )


        print("\nOriginal Source LoRA:")

        print(
            "  Accuracy:",
            f"{source_metrics['accuracy']:.4f}",
        )

        print(
            "  Balanced Accuracy:",
            f"{source_metrics['balanced_accuracy']:.4f}",
        )

        print(
            "  Macro F1:",
            f"{source_metrics['macro_f1']:.4f}",
        )


        print(
            "\nPersonal Accuracy Gain:",
            f"{personal_metrics['accuracy'] - source_metrics['accuracy']:+.4f}",
        )


        print(
            "\nMean Pipeline Latency:",
            f"{latency.mean():.2f} ms",
        )

        print(
            "95% Pipeline Latency:",
            f"{np.percentile(latency, 95):.2f} ms",
        )


        print("\nPer-phase Personal Accuracy:")


        for phase_name in [
            "TASK_A",
            "REST_A",
            "TASK_B",
            "REST_B",
        ]:

            phase_rows = [

                row

                for row in valid

                if row[
                    "phase"
                ] == phase_name

            ]


            if len(phase_rows) == 0:

                print(
                    f"  {phase_name}: no valid windows"
                )

                continue


            phase_acc = float(

                np.mean(

                    [
                        int(
                            row[
                                "personal_predicted_label"
                            ]
                        )

                        == int(
                            row[
                                "true_label"
                            ]
                        )

                        for row
                        in phase_rows

                    ]

                )

            )


            print(
                f"  {phase_name}: "
                f"{phase_acc:.4f} "
                f"(n={len(phase_rows)})"
            )


        summary_items = [
            (
                "personal_accuracy",
                personal_metrics[
                    "accuracy"
                ],
            ),
            (
                "personal_balanced_accuracy",
                personal_metrics[
                    "balanced_accuracy"
                ],
            ),
            (
                "personal_macro_f1",
                personal_metrics[
                    "macro_f1"
                ],
            ),
            (
                "source_accuracy",
                source_metrics[
                    "accuracy"
                ],
            ),
            (
                "source_balanced_accuracy",
                source_metrics[
                    "balanced_accuracy"
                ],
            ),
            (
                "source_macro_f1",
                source_metrics[
                    "macro_f1"
                ],
            ),
            (
                "personal_accuracy_gain",
                personal_metrics[
                    "accuracy"
                ]
                - source_metrics[
                    "accuracy"
                ],
            ),
            (
                "mean_pipeline_latency_ms",
                float(
                    latency.mean()
                ),
            ),
            (
                "p95_pipeline_latency_ms",
                float(
                    np.percentile(
                        latency,
                        95,
                    )
                ),
            ),
        ]


    with open(
        summary_csv_path,
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
                "raw_eeg_samples",
                len(raw_records),
            ]
        )

        writer.writerow(
            [
                "prediction_rows",
                len(prediction_records),
            ]
        )

        writer.writerow(
            [
                "valid_predictions",
                len(valid),
            ]
        )

        writer.writerow(
            [
                "calibration_block_cv_ba",
                personal_checkpoint.get(
                    "block_cv_balanced_accuracy",
                    "",
                ),
            ]
        )


        for item, value in (
            summary_items
        ):

            writer.writerow(
                [
                    item,
                    value,
                ]
            )


    print("\n")
    print("=" * 80)
    print("Saved Files")
    print("=" * 80)

    print(
        "\nRaw EEG:"
    )

    print(
        raw_csv_path
    )

    print(
        "\nPredictions:"
    )

    print(
        prediction_csv_path
    )

    print(
        "\nFinal Summary:"
    )

    print(
        summary_csv_path
    )


    print(
        "\nFinal Test 已结束。"
    )

    print(
        "不要再用这轮 Final Test 调参；"
        "如果需要再次修改模型，应重新定义新的 test session。"
    )


if __name__ == "__main__":

    main()
