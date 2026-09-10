from pathlib import Path
from collections import deque, Counter
from datetime import datetime
import csv
import importlib.util
import time

import numpy as np
import torch
import torch.nn as nn

from scipy.signal import butter, welch
from pylsl import StreamInlet


# ============================================================
# 37_loongbrain_calibrated_recording_demo.py
#
# 目标：
# 为“录屏演示”提供一个透明的 subject-specific calibrated demo。
#
# 流程：
# 1. 同一受试者、同一佩戴状态下先做 3 组交替 Calibration
# 2. 冻结 LoRA Encoder
# 3. 自动训练两种轻量校准器：
#       A. Encoder feature head
#       B. EEG spectral head
# 4. 使用 leave-one-pair-out block CV 选择 ensemble 权重
# 5. 不移动头带、不重启 LSL，立即进入录屏 Demo
#
# 注意：
# 这是 calibrated demo，不是 independent held-out test。
# 汇报时应明确说明“录屏前进行了同一受试者短时校准”。
# ============================================================


# ============================================================
# 0. Project
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "recording_demo"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "recording_demo_adapter"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. User-adjustable settings
# ============================================================

# 多 EEG stream 环境下可以保持 None，
# 程序会显示候选并让你选择。
TARGET_STREAM_NAME = None

# 若 29 已确认 channel 0/1 = Fp1/Fp2，可设置 (0,1)。
MANUAL_CHANNEL_INDICES = None

# 真机数据参数
MAX_ABS_UV = 500.0
MODEL_SFREQ = 160.0
WINDOW_SECONDS = 4.0
STEP_SECONDS = 1.0

# 先让电极和受试者稳定
INITIAL_SETTLE_SECONDS = 30

# Calibration：3 组 REST/TASK，每组各 20 s
CALIBRATION_BLOCK_SECONDS = 20
TRANSITION_SECONDS = 5
CALIBRATION_PAIRS = 3

# Demo：一段 REST + 一段 TASK，适合约 1 分钟录屏
DEMO_SETTLE_SECONDS = 10
DEMO_BLOCK_SECONDS = 25

# 与此前任务保持一致；若你改，Calibration 和 Demo 必须完全一致
START_NUMBER = 4387
SUBTRACTOR = 47

# 录屏输出稳定性：最近 3 次概率取均值
SMOOTHING_WINDOWS = 3

# Calibration CV 太低时不要急着录屏。
# 这不是“刷分阈值”，而是质量控制：说明当前校准不可复现。
MIN_RECOMMENDED_CV_BA = 0.60

SEED = 20260903


# ============================================================
# 2. Dynamic import existing project helpers
# ============================================================

def load_module(
    module_name,
    path,
):

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            str(path),
        )
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


train34 = load_module(
    "train34_helpers",
    SRC_DIR
    / "34_train_personal_head.py",
)

rt35 = load_module(
    "rt35_helpers",
    SRC_DIR
    / "35_loongbrain_final_test.py",
)

rt35.TARGET_STREAM_NAME = (
    TARGET_STREAM_NAME
)

rt35.MANUAL_CHANNEL_INDICES = (
    MANUAL_CHANNEL_INDICES
)

rt35.MAX_ABS_UV = (
    MAX_ABS_UV
)


device = train34.device


# ============================================================
# 3. Spectral feature
# ============================================================

def spectral_features_one(
    eeg_160,
):

    # eeg_160: [2, 640], already 1-45 Hz + Z-score
    # 使用相对频带功率，降低幅值漂移影响。

    bands = [
        (4.0, 7.0),    # theta
        (8.0, 13.0),   # alpha
        (13.0, 30.0),  # beta
        (30.0, 45.0),  # low gamma
    ]

    features = []
    band_logs_by_channel = []


    for channel_index in range(2):

        f, pxx = welch(
            eeg_160[
                channel_index
            ],
            fs=MODEL_SFREQ,
            nperseg=256,
            noverlap=128,
        )

        valid = (
            (f >= 1.0)
            & (f <= 45.0)
        )

        total_power = (
            np.sum(
                pxx[
                    valid
                ]
            )
            + 1e-12
        )

        band_logs = []


        for low, high in bands:

            band_power = (
                np.sum(
                    pxx[
                        (f >= low)
                        & (f < high)
                    ]
                )
                / total_power
            )

            log_power = float(
                np.log(
                    band_power
                    + 1e-8
                )
            )

            band_logs.append(
                log_power
            )

            features.append(
                log_power
            )


        # normalized spectral entropy
        spectrum = (
            pxx[
                valid
            ]
            .astype(
                np.float64
            )
        )

        spectrum = (
            spectrum
            / (
                np.sum(
                    spectrum
                )
                + 1e-12
            )
        )

        entropy = float(
            -np.sum(
                spectrum
                * np.log(
                    spectrum
                    + 1e-12
                )
            )
            / np.log(
                len(
                    spectrum
                )
            )
        )

        features.append(
            entropy
        )

        band_logs_by_channel.append(
            band_logs
        )


    # Fp1-Fp2 spectral asymmetry
    difference = (
        np.asarray(
            band_logs_by_channel[0]
        )
        - np.asarray(
            band_logs_by_channel[1]
        )
    )

    features.extend(
        difference.tolist()
    )


    # theta/alpha and beta/alpha (log ratio), each channel
    for band_logs in (
        band_logs_by_channel
    ):

        theta = (
            band_logs[0]
        )

        alpha = (
            band_logs[1]
        )

        beta = (
            band_logs[2]
        )

        features.append(
            theta
            - alpha
        )

        features.append(
            beta
            - alpha
        )


    return np.asarray(
        features,
        dtype=np.float32,
    )


# ============================================================
# 4. Metrics
# ============================================================

def balanced_accuracy(
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

    recalls = []


    for label in [
        0,
        1,
    ]:

        mask = (
            y_true
            == label
        )

        if np.sum(
            mask
        ) == 0:

            continue

        recalls.append(
            float(
                np.mean(
                    y_pred[
                        mask
                    ]
                    == label
                )
            )
        )


    if not recalls:

        return np.nan


    return float(
        np.mean(
            recalls
        )
    )


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
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        f1 = (
            2.0
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
# 5. Simple standardized linear head helpers
# ============================================================

def standardize_stats(
    features,
):

    mean = (
        features.mean(
            axis=0
        )
        .astype(
            np.float32
        )
    )

    std = (
        features.std(
            axis=0
        )
        .astype(
            np.float32
        )
    )

    std = np.maximum(
        std,
        1e-4,
    ).astype(
        np.float32
    )

    return (
        mean,
        std,
    )


def train_all_feature_head(
    features,
    labels,
    l2,
    epochs=400,
    seed=SEED,
):

    mean, std = (
        standardize_stats(
            features
        )
    )

    mask = np.ones(
        features.shape[1],
        dtype=np.float32,
    )

    head, _ = (
        train34.train_head(

            features,
            labels,

            mean,
            std,
            mask,

            l2=l2,
            epochs=epochs,
            seed=seed,

        )
    )

    return (
        head,
        mean,
        std,
        mask,
    )


def predict_probability(
    head,
    features,
    mean,
    std,
    mask,
):

    _, probability = (
        train34.predict_head(

            head,
            features,

            mean,
            std,
            mask,

        )
    )

    return probability[
        :,
        1
    ]


# ============================================================
# 6. CV selection
# ============================================================

ENCODER_TOP_K_GRID = [
    4,
    8,
    16,
    32,
]

ENCODER_L2_GRID = [
    0.0,
    0.01,
    0.1,
]

SPECTRAL_L2_GRID = [
    0.0,
    0.01,
    0.1,
    1.0,
]

ENSEMBLE_WEIGHT_GRID = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.0,
]
# weight = spectral probability weight


def fit_encoder_head(
    features,
    labels,
    top_k,
    l2,
    seed,
):

    (
        mean,
        std,
        mask,
        selected_indices,

    ) = train34.feature_stats_and_mask(

        features,
        labels,
        top_k,

    )

    head, _ = (
        train34.train_head(

            features,
            labels,

            mean,
            std,
            mask,

            l2=l2,
            epochs=350,
            seed=seed,

        )
    )

    return (
        head,
        mean,
        std,
        mask,
        selected_indices,
    )


def choose_adapter(
    encoder_features,
    spectral_features,
    labels,
    pair_ids,
):

    unique_pairs = sorted(
        np.unique(
            pair_ids
        ).tolist()
    )

    if len(
        unique_pairs
    ) < 3:

        raise RuntimeError(
            "Calibration 至少需要 3 个 REST/TASK pair。"
        )


    fold_cache = []


    print("\n")
    print("=" * 80)
    print("Leave-one-pair-out Calibration CV")
    print("=" * 80)


    for fold_index, held_out_pair in enumerate(
        unique_pairs
    ):

        train_idx = np.where(
            pair_ids
            != held_out_pair
        )[0]

        valid_idx = np.where(
            pair_ids
            == held_out_pair
        )[0]


        encoder_models = {}

        for top_k in (
            ENCODER_TOP_K_GRID
        ):

            for l2 in (
                ENCODER_L2_GRID
            ):

                (
                    head,
                    mean,
                    std,
                    mask,
                    _,

                ) = fit_encoder_head(

                    encoder_features[
                        train_idx
                    ],

                    labels[
                        train_idx
                    ],

                    top_k,
                    l2,

                    seed=(
                        SEED
                        + 100
                        * fold_index
                        + top_k
                    ),

                )

                probability = (
                    predict_probability(

                        head,

                        encoder_features[
                            valid_idx
                        ],

                        mean,
                        std,
                        mask,

                    )
                )

                encoder_models[
                    (
                        top_k,
                        l2,
                    )
                ] = probability


        spectral_models = {}

        for l2 in (
            SPECTRAL_L2_GRID
        ):

            (
                head,
                mean,
                std,
                mask,

            ) = train_all_feature_head(

                spectral_features[
                    train_idx
                ],

                labels[
                    train_idx
                ],

                l2=l2,
                epochs=350,
                seed=(
                    SEED
                    + 1000
                    + fold_index
                ),

            )

            probability = (
                predict_probability(

                    head,

                    spectral_features[
                        valid_idx
                    ],

                    mean,
                    std,
                    mask,

                )
            )

            spectral_models[
                l2
            ] = probability


        fold_cache.append(
            {
                "valid_idx":
                    valid_idx,

                "encoder":
                    encoder_models,

                "spectral":
                    spectral_models,
            }
        )


    candidates = []


    for top_k in (
        ENCODER_TOP_K_GRID
    ):

        for encoder_l2 in (
            ENCODER_L2_GRID
        ):

            for spectral_l2 in (
                SPECTRAL_L2_GRID
            ):

                for spectral_weight in (
                    ENSEMBLE_WEIGHT_GRID
                ):

                    fold_scores = []


                    for fold in (
                        fold_cache
                    ):

                        valid_idx = (
                            fold[
                                "valid_idx"
                            ]
                        )

                        p_encoder = (
                            fold[
                                "encoder"
                            ][
                                (
                                    top_k,
                                    encoder_l2,
                                )
                            ]
                        )

                        p_spectral = (
                            fold[
                                "spectral"
                            ][
                                spectral_l2
                            ]
                        )

                        probability = (

                            spectral_weight
                            * p_spectral

                            +

                            (
                                1.0
                                - spectral_weight
                            )
                            * p_encoder

                        )

                        prediction = (
                            probability
                            >= 0.5
                        ).astype(
                            int
                        )

                        score = (
                            balanced_accuracy(

                                labels[
                                    valid_idx
                                ],

                                prediction,

                            )
                        )

                        fold_scores.append(
                            score
                        )


                    candidates.append(
                        {
                            "top_k":
                                top_k,

                            "encoder_l2":
                                encoder_l2,

                            "spectral_l2":
                                spectral_l2,

                            "spectral_weight":
                                spectral_weight,

                            "fold_scores":
                                fold_scores,

                            "mean_ba":
                                float(
                                    np.mean(
                                        fold_scores
                                    )
                                ),
                        }
                    )


    best = max(
        candidates,
        key=lambda item:
            item[
                "mean_ba"
            ],
    )


    print(
        "\nSelected adapter:"
    )

    print(
        "  Encoder Top-K:",
        best[
            "top_k"
        ],
    )

    print(
        "  Encoder L2:",
        best[
            "encoder_l2"
        ],
    )

    print(
        "  Spectral L2:",
        best[
            "spectral_l2"
        ],
    )

    print(
        "  Spectral ensemble weight:",
        best[
            "spectral_weight"
        ],
    )

    print(
        "  Fold BA:",
        [
            round(
                value,
                4,
            )
            for value
            in best[
                "fold_scores"
            ]
        ],
    )

    print(
        "  Mean CV BA:",
        f"{best['mean_ba']:.4f}",
    )


    return best


# ============================================================
# 7. Load source foundation model
# ============================================================

(
    source_model,
    source_checkpoint,
    source_config,

) = train34.load_source_model()


# ============================================================
# 8. LSL
# ============================================================

stream_info = (
    rt35.discover_stream()
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
    "Source ID:",
    stream_info.source_id(),
)

print(
    "Channels:",
    stream_info.channel_count(),
)

source_sfreq = float(
    stream_info.nominal_srate()
)

print(
    "Sampling Rate:",
    source_sfreq,
)


(
    channel_indices,
    labels,
    units,

) = rt35.select_channels(
    stream_info
)

print(
    "Selected indices:",
    channel_indices,
)


inlet = StreamInlet(
    stream_info,
    max_buflen=20,
)


# ============================================================
# 9. Warm-up + unit scale
# ============================================================

print(
    "\n读取短数据检查单位..."
)

warm_samples, _ = (
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

if len(
    warm_samples
) == 0:

    raise RuntimeError(
        "LSL Stream 存在，但没有收到 EEG 样本。"
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
    if index < len(
        units
    )
    else ""
    for index
    in channel_indices
]

scale_to_volts = (
    rt35.determine_scale_to_volts(
        selected_units,
        warm_eeg,
    )
)

print(
    "Scale to volts:",
    scale_to_volts,
)


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

window_samples = int(
    round(
        source_sfreq
        * WINDOW_SECONDS
    )
)

step_samples = int(
    round(
        source_sfreq
        * STEP_SECONDS
    )
)


# ============================================================
# 10. Session storage
# ============================================================

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
    / f"demo_summary_{session_id}.csv"
)

adapter_path = (
    CHECKPOINT_DIR
    / f"adapter_{session_id}.pt"
)


raw_records = []
calibration_encoder_features = []
calibration_spectral_features = []
calibration_labels = []
calibration_pair_ids = []

demo_prediction_records = []

session_time = 0.0


# ============================================================
# 11. Phase collection
# ============================================================

def print_phase(
    phase_name,
    duration_seconds,
):

    print("\n")
    print("=" * 80)
    print(
        f"{phase_name} | "
        f"{duration_seconds}s"
    )

    if phase_name.startswith(
        "REST"
    ):

        print(
            "闭眼、放松、停止心算；额头和眼睛尽量保持放松。"
        )

    elif phase_name.startswith(
        "TASK"
    ):

        print(
            f"闭眼持续心算：{START_NUMBER} - {SUBTRACTOR} - {SUBTRACTOR} - ..."
        )

        print(
            "不说话、不数手指、不皱眉，不追求速度，只保持持续计算。"
        )

    elif "PREPARE" in phase_name:

        print(
            "保持姿势不动，准备切换状态。"
        )

    else:

        print(
            "闭眼、放松、保持不动。"
        )

    print("=" * 80)


def collect_phase(
    phase_name,
    duration_seconds,
    true_label,
    pair_id=None,
    calibration_mode=False,
    demo_adapter=None,
):

    global session_time

    print_phase(
        phase_name,
        duration_seconds,
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

    demo_probability_history = deque(
        maxlen=SMOOTHING_WINDOWS
    )

    valid_count = 0
    artifact_count = 0


    while received_samples < target_samples:

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

        if len(
            samples
        ) == 0:

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

        if chunk.shape[0] > remaining:

            chunk = chunk[
                :remaining
            ]

            timestamps = timestamps[
                :remaining
            ]


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

        current_samples = (
            eeg_volts.shape[1]
        )

        received_samples += (
            current_samples
        )


        for sample_index in range(
            current_samples
        ):

            session_time += (
                1.0
                / source_sfreq
            )

            raw_records.append(
                {
                    "session_time_seconds":
                        session_time,

                    "phase":
                        phase_name,

                    "true_label":
                        (
                            ""
                            if true_label
                            is None
                            else true_label
                        ),

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


        if true_label is None:

            continue


        buffer = np.concatenate(
            [
                buffer,
                eeg_volts,
            ],
            axis=1,
        )

        if buffer.shape[1] > window_samples:

            buffer = buffer[
                :,
                -window_samples:
            ]


        samples_since_prediction += (
            current_samples
        )


        if buffer.shape[1] < window_samples:

            continue


        if samples_since_prediction < step_samples:

            continue


        samples_since_prediction = 0

        pipeline_start = (
            time.perf_counter()
        )


        processed, quality, max_uv = (
            rt35.preprocess_window(
                buffer.copy(),
                source_sfreq,
                filter_sos,
            )
        )


        if processed is None:

            artifact_count += 1

            print(
                f"{session_time:6.1f}s | "
                f"{phase_name:<8} | "
                f"{quality:<8} | "
                f"Max={max_uv:7.1f} μV"
            )

            continue


        valid_count += 1


        eeg_tensor = (
            torch.from_numpy(
                processed
            )
            .unsqueeze(
                0
            )
            .to(
                device
            )
        )


        with torch.no_grad():

            encoder_feature_tensor = (
                source_model
                .extract_features(
                    eeg_tensor
                )
            )

            source_logits = (
                source_model
                .classifier(
                    encoder_feature_tensor
                )
            )

            source_probability = (
                torch.softmax(
                    source_logits,
                    dim=1,
                )[
                    0,
                    1
                ]
                .item()
            )


        encoder_feature = (
            encoder_feature_tensor[
                0
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )


        spectral_feature = (
            spectral_features_one(
                processed
            )
        )


        if calibration_mode:

            calibration_encoder_features.append(
                encoder_feature
            )

            calibration_spectral_features.append(
                spectral_feature
            )

            calibration_labels.append(
                int(
                    true_label
                )
            )

            calibration_pair_ids.append(
                int(
                    pair_id
                )
            )

            print(
                f"{session_time:6.1f}s | "
                f"{phase_name:<8} | "
                f"CAL window OK | "
                f"Source P(Task)={source_probability:.3f} | "
                f"Max={max_uv:6.1f}μV"
            )


        elif demo_adapter is not None:

            (
                encoder_head,
                encoder_mean,
                encoder_std,
                encoder_mask,
                spectral_head,
                spectral_mean,
                spectral_std,
                spectral_mask,
                spectral_weight,

            ) = demo_adapter


            p_encoder = float(
                predict_probability(
                    encoder_head,
                    encoder_feature[
                        None,
                        :
                    ],
                    encoder_mean,
                    encoder_std,
                    encoder_mask,
                )[0]
            )


            p_spectral = float(
                predict_probability(
                    spectral_head,
                    spectral_feature[
                        None,
                        :
                    ],
                    spectral_mean,
                    spectral_std,
                    spectral_mask,
                )[0]
            )


            p_demo_raw = (

                spectral_weight
                * p_spectral

                +

                (
                    1.0
                    - spectral_weight
                )
                * p_encoder

            )


            demo_probability_history.append(
                p_demo_raw
            )


            p_demo = float(
                np.mean(
                    demo_probability_history
                )
            )


            predicted_label = int(
                p_demo
                >= 0.5
            )


            predicted_state = (
                "TASK"
                if predicted_label == 1
                else "REST"
            )


            pipeline_latency = (
                time.perf_counter()
                - pipeline_start
            ) * 1000.0


            true_state = (
                "TASK"
                if int(
                    true_label
                ) == 1
                else "REST"
            )


            is_correct = (
                predicted_label
                == int(
                    true_label
                )
            )


            marker = (
                "✓"
                if is_correct
                else "×"
            )


            print(
                f"{session_time:6.1f}s | "
                f"True={true_state:<4} | "
                f"DemoPred={predicted_state:<4} {marker} | "
                f"P(Task)={p_demo:.3f} | "
                f"Foundation={source_probability:.3f} | "
                f"Latency={pipeline_latency:.1f}ms"
            )


            demo_prediction_records.append(
                {
                    "time_seconds":
                        session_time,

                    "phase":
                        phase_name,

                    "true_label":
                        int(
                            true_label
                        ),

                    "predicted_label":
                        predicted_label,

                    "predicted_state":
                        predicted_state,

                    "p_task_demo":
                        p_demo,

                    "p_task_demo_raw":
                        p_demo_raw,

                    "p_task_encoder":
                        p_encoder,

                    "p_task_spectral":
                        p_spectral,

                    "p_task_source_lora":
                        source_probability,

                    "quality":
                        quality,

                    "max_abs_uv":
                        max_uv,

                    "pipeline_latency_ms":
                        pipeline_latency,
                }
            )


    print(
        f"\n{phase_name} 完成 | "
        f"valid={valid_count}, "
        f"rejected={artifact_count}"
    )


# ============================================================
# 12. Initial stabilization
# ============================================================

print("\n")
print("=" * 80)
print("Calibrated Recording Demo")
print("=" * 80)

print(
    "本脚本会先校准，再提示你开始录屏。"
)

print(
    "校准和录屏之间不要摘头带、不要移动电极、不要重启 LSL。"
)

input(
    "\n确认波形和阻抗正常后按 Enter 开始..."
)


collect_phase(
    "INITIAL_SETTLE",
    INITIAL_SETTLE_SECONDS,
    true_label=None,
)


# ============================================================
# 13. Calibration: R/T x 3
# ============================================================

for pair_id in range(
    1,
    CALIBRATION_PAIRS
    + 1,
):

    collect_phase(
        f"REST_{pair_id}",
        CALIBRATION_BLOCK_SECONDS,
        true_label=0,
        pair_id=pair_id,
        calibration_mode=True,
    )

    collect_phase(
        "PREPARE_TASK",
        TRANSITION_SECONDS,
        true_label=None,
    )

    collect_phase(
        f"TASK_{pair_id}",
        CALIBRATION_BLOCK_SECONDS,
        true_label=1,
        pair_id=pair_id,
        calibration_mode=True,
    )

    if pair_id < CALIBRATION_PAIRS:

        collect_phase(
            "PREPARE_REST",
            TRANSITION_SECONDS,
            true_label=None,
        )


encoder_features = np.stack(
    calibration_encoder_features
)

spectral_features = np.stack(
    calibration_spectral_features
)

calibration_y = np.asarray(
    calibration_labels,
    dtype=np.int64,
)

pair_ids = np.asarray(
    calibration_pair_ids,
    dtype=np.int64,
)


print("\n")
print("=" * 80)
print("Calibration Summary")
print("=" * 80)

print(
    "Valid windows:",
    len(
        calibration_y
    ),
)

print(
    "REST / TASK:",
    np.bincount(
        calibration_y,
        minlength=2,
    ).tolist(),
)


# ============================================================
# 14. Adapter selection
# ============================================================

best = choose_adapter(
    encoder_features,
    spectral_features,
    calibration_y,
    pair_ids,
)


# ============================================================
# 15. Fit final demo adapter on all calibration windows
# ============================================================

(
    encoder_mean,
    encoder_std,
    encoder_mask,
    selected_indices,

) = train34.feature_stats_and_mask(

    encoder_features,
    calibration_y,
    best[
        "top_k"
    ],

)


encoder_head, _ = (
    train34.train_head(

        encoder_features,
        calibration_y,

        encoder_mean,
        encoder_std,
        encoder_mask,

        l2=best[
            "encoder_l2"
        ],

        epochs=500,
        seed=SEED,

    )
)


(
    spectral_head,
    spectral_mean,
    spectral_std,
    spectral_mask,

) = train_all_feature_head(

    spectral_features,
    calibration_y,

    l2=best[
        "spectral_l2"
    ],

    epochs=500,
    seed=(
        SEED
        + 999
    ),

)


spectral_weight = float(
    best[
        "spectral_weight"
    ]
)


demo_adapter = (
    encoder_head,
    encoder_mean,
    encoder_std,
    encoder_mask,
    spectral_head,
    spectral_mean,
    spectral_std,
    spectral_mask,
    spectral_weight,
)


torch.save(
    {
        "encoder_head_state_dict":
            encoder_head
            .cpu()
            .state_dict(),

        "encoder_mean":
            torch.from_numpy(
                encoder_mean
            ),

        "encoder_std":
            torch.from_numpy(
                encoder_std
            ),

        "encoder_mask":
            torch.from_numpy(
                encoder_mask
            ),

        "spectral_head_state_dict":
            spectral_head
            .cpu()
            .state_dict(),

        "spectral_mean":
            torch.from_numpy(
                spectral_mean
            ),

        "spectral_std":
            torch.from_numpy(
                spectral_std
            ),

        "spectral_mask":
            torch.from_numpy(
                spectral_mask
            ),

        "spectral_weight":
            spectral_weight,

        "selected_top_k":
            int(
                best[
                    "top_k"
                ]
            ),

        "encoder_l2":
            float(
                best[
                    "encoder_l2"
                ]
            ),

        "spectral_l2":
            float(
                best[
                    "spectral_l2"
                ]
            ),

        "calibration_cv_ba":
            float(
                best[
                    "mean_ba"
                ]
            ),

        "calibration_fold_ba":
            best[
                "fold_scores"
            ],

        "calibration_valid_windows":
            int(
                len(
                    calibration_y
                )
            ),

    },
    adapter_path,
)


# move heads back to current device after save
encoder_head = encoder_head.to(
    device
)
spectral_head = spectral_head.to(
    device
)

demo_adapter = (
    encoder_head,
    encoder_mean,
    encoder_std,
    encoder_mask,
    spectral_head,
    spectral_mean,
    spectral_std,
    spectral_mask,
    spectral_weight,
)


# ============================================================
# 16. Recording gate
# ============================================================

print("\n")
print("=" * 80)
print("READY FOR RECORDING")
print("=" * 80)

print(
    "Calibration CV Balanced Accuracy:",
    f"{best['mean_ba']:.4f}",
)

print(
    "Valid calibration windows:",
    len(
        calibration_y
    ),
)

print(
    "Adapter:",
    f"{(1-spectral_weight):.2f} Encoder + "
    f"{spectral_weight:.2f} Spectral",
)


if best[
    "mean_ba"
] < MIN_RECOMMENDED_CV_BA:

    print(
        "\n⚠ 当前 calibration 的跨 block 稳定性偏低。"
    )

    print(
        "建议先检查电极接触、额头/眼动、受试者状态，"
        "重新做一次 calibration 后再录屏。"
    )

    print(
        "不要通过放宽 Artifact 阈值或修改真实标签来提高分数。"
    )

else:

    print(
        "\n✓ Calibration 稳定性达到建议录屏水平。"
    )


print(
    "\n现在可以开启系统录屏。"
)

print(
    "请保持头带和坐姿完全不变。"
)

input(
    "录屏开始后按 Enter 进入 Demo..."
)


# ============================================================
# 17. Demo
# ============================================================

collect_phase(
    "DEMO_SETTLE",
    DEMO_SETTLE_SECONDS,
    true_label=None,
)

collect_phase(
    "REST_DEMO",
    DEMO_BLOCK_SECONDS,
    true_label=0,
    demo_adapter=demo_adapter,
)

collect_phase(
    "PREPARE_TASK",
    TRANSITION_SECONDS,
    true_label=None,
)

collect_phase(
    "TASK_DEMO",
    DEMO_BLOCK_SECONDS,
    true_label=1,
    demo_adapter=demo_adapter,
)


# ============================================================
# 18. Save
# ============================================================

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
            "phase",
            "true_label",
            "fp1_volts",
            "fp2_volts",
        ],
    )

    writer.writeheader()

    writer.writerows(
        raw_records
    )


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
            "predicted_label",
            "predicted_state",
            "p_task_demo",
            "p_task_demo_raw",
            "p_task_encoder",
            "p_task_spectral",
            "p_task_source_lora",
            "quality",
            "max_abs_uv",
            "pipeline_latency_ms",
        ],
    )

    writer.writeheader()

    writer.writerows(
        demo_prediction_records
    )


# ============================================================
# 19. Final demo metrics
# ============================================================

print("\n")
print("=" * 80)
print("RECORDED DEMO RESULT")
print("=" * 80)


if len(
    demo_prediction_records
) == 0:

    print(
        "没有有效 Demo 预测，请检查 EEG 质量。"
    )

else:

    y_true = np.asarray(
        [
            row[
                "true_label"
            ]
            for row
            in demo_prediction_records
        ],
        dtype=int,
    )

    y_pred = np.asarray(
        [
            row[
                "predicted_label"
            ]
            for row
            in demo_prediction_records
        ],
        dtype=int,
    )

    metrics = binary_metrics(
        y_true,
        y_pred,
    )

    latency = np.asarray(
        [
            row[
                "pipeline_latency_ms"
            ]
            for row
            in demo_prediction_records
        ],
        dtype=float,
    )

    print(
        "Valid Predictions:",
        len(
            y_true
        ),
    )

    print(
        "Accuracy:",
        f"{metrics['accuracy']:.4f}",
    )

    print(
        "Balanced Accuracy:",
        f"{metrics['balanced_accuracy']:.4f}",
    )

    print(
        "Macro F1:",
        f"{metrics['macro_f1']:.4f}",
    )

    print(
        "Mean Pipeline Latency:",
        f"{latency.mean():.2f} ms",
    )

    print(
        "P95 Pipeline Latency:",
        f"{np.percentile(latency,95):.2f} ms",
    )


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
                "demo_type",
                "same-session subject-specific calibrated recording demo",
            ]
        )

        writer.writerow(
            [
                "calibration_cv_balanced_accuracy",
                best[
                    "mean_ba"
                ],
            ]
        )

        writer.writerow(
            [
                "calibration_valid_windows",
                len(
                    calibration_y
                ),
            ]
        )

        writer.writerow(
            [
                "spectral_ensemble_weight",
                spectral_weight,
            ]
        )

        writer.writerow(
            [
                "demo_valid_predictions",
                len(
                    y_true
                ),
            ]
        )

        writer.writerow(
            [
                "demo_accuracy",
                metrics[
                    "accuracy"
                ],
            ]
        )

        writer.writerow(
            [
                "demo_balanced_accuracy",
                metrics[
                    "balanced_accuracy"
                ],
            ]
        )

        writer.writerow(
            [
                "demo_macro_f1",
                metrics[
                    "macro_f1"
                ],
            ]
        )

        writer.writerow(
            [
                "mean_pipeline_latency_ms",
                float(
                    latency.mean()
                ),
            ]
        )

        writer.writerow(
            [
                "p95_pipeline_latency_ms",
                float(
                    np.percentile(
                        latency,
                        95,
                    )
                ),
            ]
        )


print("\nSaved:")
print(
    raw_csv_path
)
print(
    prediction_csv_path
)
print(
    summary_csv_path
)
print(
    adapter_path
)

print(
    "\n汇报口径：这是同一受试者、同一佩戴 session "
    "下经过短时 calibration 后的实时演示，不应称为独立测试准确率。"
)
