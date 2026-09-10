from pathlib import Path
from fractions import Fraction
import csv
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import (
    butter,
    sosfiltfilt,
    resample_poly,
)

from masked_eeg_model import MaskedEEGTransformer


# ============================================================
# 34_train_personal_head.py
#
# 用途：
# 使用一次独立 LoongBrain Calibration Session 的真机 EEG，
# 冻结原 LoRA EEG Encoder，仅训练一个 Personal Linear Head。
#
# 训练数据：
#   REST_A / TASK_A / REST_B / TASK_B
#
# 关键原则：
# 1. LoRA Encoder 完全冻结
# 2. 只训练 Linear(128 -> 2)，共 258 个参数
# 3. 使用 Block-wise CV：
#       A blocks -> B blocks
#       B blocks -> A blocks
#    来选择正则化与特征子集，避免只看训练集成绩
# 4. 本脚本生成的是 Calibration Adapter，
#    最终成绩必须由 35 号脚本在新的独立 session 上测试
# ============================================================


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 20260902

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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


CALIBRATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "loongbrain_calibration"
)


PERSONAL_HEAD_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "personal_head"
)


PERSONAL_HEAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_PATH = (
    PERSONAL_HEAD_DIR
    / "best.pt"
)


SUMMARY_PATH = (
    PERSONAL_HEAD_DIR
    / "training_summary.csv"
)


# ============================================================
# 3. Signal config
# ============================================================

MODEL_SFREQ = 160.0
MODEL_WINDOW_SECONDS = 4.0
MODEL_WINDOW_SAMPLES = 640

WINDOW_STEP_SECONDS = 1.0

LOWCUT = 1.0
HIGHCUT = 45.0
FILTER_ORDER = 4

MAX_ABS_UV = 500.0


# ============================================================
# 4. Personal-head training config
# ============================================================

PHASE_TO_LABEL = {
    "REST_A": 0,
    "TASK_A": 1,
    "REST_B": 0,
    "TASK_B": 1,
}


# 保持最终 classifier 为 Linear(128 -> 2)，258 参数。
# 但校准时允许把不稳定维度置零，仅使用一部分更稳定的
# LoRA encoder dimensions。

TOP_K_GRID = [
    2,
    8,
    16,
    32,
    64,
    128,
]


L2_GRID = [
    0.0,
    0.01,
    0.1,
    1.0,
]


CV_EPOCHS = 400
FINAL_EPOCHS = 500

LEARNING_RATE = 0.01


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 5. LoRA layers
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
# 6. Source LoRA model
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

        features = (
            representations
            .mean(
                dim=1
            )
        )

        return features


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
# 7. Load source LoRA model
# ============================================================

def load_source_model():

    if not SOURCE_MODEL_PATH.exists():

        raise FileNotFoundError(
            "\n找不到 LoRA checkpoint：\n"
            f"{SOURCE_MODEL_PATH}"
        )


    checkpoint = torch.load(
        SOURCE_MODEL_PATH,
        map_location="cpu",
    )


    config = (
        checkpoint[
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
                dropout=dropout,

            )
        )

        layer.linear2 = (
            LoRALinear(

                layer.linear2,

                rank=rank,
                alpha=alpha,
                dropout=dropout,

            )
        )


    model = (
        EEGLoraClassifier(

            backbone=backbone,

            embed_dim=config[
                "embed_dim"
            ],

        )
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    # Encoder completely frozen.

    for parameter in (
        model.parameters()
    ):

        parameter.requires_grad = (
            False
        )


    model = model.to(
        device
    )

    model.eval()


    return (
        model,
        checkpoint,
        config,
    )


# ============================================================
# 8. Find latest calibration raw EEG
# ============================================================

def find_latest_calibration_csv():

    files = sorted(
        CALIBRATION_DIR.glob(
            "raw_eeg_*.csv"
        )
    )


    if len(files) == 0:

        raise FileNotFoundError(

            "\n没有找到 Calibration Raw EEG。\n\n"
            "预期目录：\n"
            f"{CALIBRATION_DIR}\n\n"
            "请先运行：\n"
            "python src\\30_loongbrain_calibration_alternating.py"

        )


    return files[-1]


# ============================================================
# 9. Read calibration CSV
# ============================================================

def read_raw_csv(
    path
):

    rows = []


    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        required = {
            "session_time_seconds",
            "phase",
            "fp1_volts",
            "fp2_volts",
        }


        if not required.issubset(
            set(
                reader.fieldnames
                or []
            )
        ):

            raise RuntimeError(

                "Calibration CSV 缺少必须列："
                f"{required}"

            )


        for row in reader:

            rows.append(
                row
            )


    if len(rows) < 2:

        raise RuntimeError(
            "Calibration CSV 数据太少。"
        )


    times = np.asarray(

        [
            float(
                row[
                    "session_time_seconds"
                ]
            )

            for row in rows
        ],

        dtype=np.float64,

    )


    dt = np.diff(
        times
    )


    dt = dt[
        np.isfinite(
            dt
        )
        & (
            dt > 0
        )
    ]


    if len(dt) == 0:

        raise RuntimeError(
            "无法从 CSV 推断采样率。"
        )


    source_sfreq = (
        1.0
        / float(
            np.median(
                dt
            )
        )
    )


    return (
        rows,
        source_sfreq,
    )


# ============================================================
# 10. Preprocess one 4-s window
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


    # First band-pass, then artifact QC.
    # This avoids rejecting a valid EEG solely because
    # the device has a large DC offset.

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
        np.mean(
            resampled,
            axis=1,
            keepdims=True,
        )
    )


    std = (
        np.std(
            resampled,
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
# 11. Build calibration windows
# ============================================================

def build_windows(
    rows,
    source_sfreq,
):

    if (
        source_sfreq
        <= 2 * HIGHCUT
    ):

        raise RuntimeError(

            f"采样率 {source_sfreq:.2f} Hz "
            "不足以进行 1–45 Hz 处理。"

        )


    filter_sos = (
        butter(

            N=FILTER_ORDER,

            Wn=[
                LOWCUT,
                HIGHCUT,
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
            * WINDOW_STEP_SECONDS
        )
    )


    X = []
    y = []
    blocks = []

    rejected = {
        "Artifact": 0,
        "Flat": 0,
        "NaN/Inf": 0,
    }


    candidate_count = 0


    for phase_name, label in (
        PHASE_TO_LABEL.items()
    ):

        phase_rows = [

            row

            for row in rows

            if row[
                "phase"
            ] == phase_name

        ]


        if len(phase_rows) == 0:

            raise RuntimeError(
                f"CSV 中没有找到 {phase_name}"
            )


        eeg = np.asarray(

            [
                [
                    float(
                        row[
                            "fp1_volts"
                        ]
                    )

                    for row
                    in phase_rows
                ],

                [
                    float(
                        row[
                            "fp2_volts"
                        ]
                    )

                    for row
                    in phase_rows
                ],
            ],

            dtype=np.float64,

        )


        phase_valid = 0


        for start in range(

            0,

            eeg.shape[1]
            - window_samples
            + 1,

            step_samples,

        ):

            candidate_count += 1


            window = (
                eeg[
                    :,
                    start:
                    start
                    + window_samples
                ]
            )


            (
                processed,
                quality,
                _,

            ) = preprocess_window(

                window,

                source_sfreq,

                filter_sos,

            )


            if processed is None:

                rejected[
                    quality
                ] = (

                    rejected.get(
                        quality,
                        0,
                    )

                    + 1

                )

                continue


            X.append(
                processed
            )

            y.append(
                label
            )

            blocks.append(
                phase_name
            )

            phase_valid += 1


        print(
            f"{phase_name:<7}: "
            f"{phase_valid} valid windows"
        )


    if len(X) == 0:

        raise RuntimeError(
            "没有任何有效 Calibration Window。"
        )


    return (
        np.stack(
            X
        ),
        np.asarray(
            y,
            dtype=np.int64,
        ),
        np.asarray(
            blocks,
        ),
        candidate_count,
        rejected,
    )


# ============================================================
# 12. Feature extraction
# ============================================================

def extract_features(
    model,
    X,
):

    features = []

    batch_size = 64


    with torch.no_grad():

        for start in range(
            0,
            len(X),
            batch_size,
        ):

            batch = (
                torch.from_numpy(

                    X[
                        start:
                        start
                        + batch_size
                    ]

                )
                .float()
                .to(
                    device
                )
            )


            batch_features = (
                model.extract_features(
                    batch
                )
                .cpu()
                .numpy()
            )


            features.append(
                batch_features
            )


    return np.concatenate(
        features,
        axis=0,
    )


# ============================================================
# 13. Metrics
# ============================================================

def balanced_accuracy(
    y_true,
    y_pred,
):

    scores = []


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


        scores.append(

            np.mean(
                y_pred[
                    mask
                ]
                == label
            )

        )


    if len(scores) == 0:

        return np.nan


    return float(
        np.mean(
            scores
        )
    )


# ============================================================
# 14. Fisher feature selection
# ============================================================

def feature_stats_and_mask(
    features,
    labels,
    top_k,
):

    mean = (
        features.mean(
            axis=0
        )
    )


    std = (
        features.std(
            axis=0
        )
    )


    std = np.maximum(
        std,
        1e-4,
    )


    normalized = (

        (
            features
            - mean
        )

        / std

    )


    class0 = (
        normalized[
            labels == 0
        ]
    )


    class1 = (
        normalized[
            labels == 1
        ]
    )


    if (
        len(class0) == 0
        or len(class1) == 0
    ):

        raise RuntimeError(
            "训练 fold 中缺少某个类别。"
        )


    mean0 = (
        class0.mean(
            axis=0
        )
    )


    mean1 = (
        class1.mean(
            axis=0
        )
    )


    var0 = (
        class0.var(
            axis=0
        )
    )


    var1 = (
        class1.var(
            axis=0
        )
    )


    fisher = (

        (
            mean1
            - mean0
        ) ** 2

        /

        (
            var0
            + var1
            + 1e-6
        )

    )


    top_k = min(
        int(
            top_k
        ),
        features.shape[1],
    )


    selected = (
        np.argsort(
            fisher
        )[
            -top_k:
        ]
    )


    mask = np.zeros(
        features.shape[1],
        dtype=np.float32,
    )


    mask[
        selected
    ] = 1.0


    return (
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
        mask,
        selected,
    )


# ============================================================
# 15. Train one linear head
# ============================================================

def train_head(
    train_features,
    train_labels,
    feature_mean,
    feature_std,
    feature_mask,
    l2,
    epochs,
    seed,
):

    torch.manual_seed(
        seed
    )


    X_train = (

        (
            train_features
            - feature_mean
        )

        / feature_std

    )


    X_train = (

        X_train

        * feature_mask

    ).astype(
        np.float32
    )


    X_tensor = (
        torch.from_numpy(
            X_train
        )
        .float()
        .to(
            device
        )
    )


    y_tensor = (
        torch.from_numpy(
            train_labels
        )
        .long()
        .to(
            device
        )
    )


    head = (
        nn.Linear(
            train_features.shape[1],
            2,
        )
        .to(
            device
        )
    )


    counts = (
        np.bincount(
            train_labels,
            minlength=2,
        )
    )


    weights = (

        len(
            train_labels
        )

        /

        (
            2.0

            * np.maximum(
                counts,
                1,
            )
        )

    ).astype(
        np.float32
    )


    weight_tensor = (
        torch.from_numpy(
            weights
        )
        .to(
            device
        )
    )


    optimizer = (
        torch.optim.Adam(
            head.parameters(),
            lr=LEARNING_RATE,
        )
    )


    final_loss = np.nan


    for _ in range(
        epochs
    ):

        head.train()

        optimizer.zero_grad()


        logits = (
            head(
                X_tensor
            )
        )


        ce_loss = (
            F.cross_entropy(

                logits,

                y_tensor,

                weight=weight_tensor,

            )
        )


        l2_loss = (

            head.weight
            .pow(
                2
            )
            .sum()

        )


        loss = (

            ce_loss

            + float(
                l2
            )
            * l2_loss

        )


        loss.backward()

        optimizer.step()


        final_loss = float(
            loss.item()
        )


    head.eval()


    return (
        head,
        final_loss,
    )


# ============================================================
# 16. Predict with one head
# ============================================================

def predict_head(
    head,
    features,
    feature_mean,
    feature_std,
    feature_mask,
):

    X = (

        (
            features
            - feature_mean
        )

        / feature_std

    )


    X = (

        X

        * feature_mask

    ).astype(
        np.float32
    )


    tensor = (
        torch.from_numpy(
            X
        )
        .float()
        .to(
            device
        )
    )


    with torch.no_grad():

        logits = (
            head(
                tensor
            )
        )


        probability = (
            torch.softmax(
                logits,
                dim=1,
            )
            .cpu()
            .numpy()
        )


        prediction = (
            np.argmax(
                probability,
                axis=1,
            )
        )


    return (
        prediction,
        probability,
    )


# ============================================================
# 17. Block-wise CV
# ============================================================

def select_hyperparameters(
    features,
    labels,
    blocks,
):

    group_a = np.isin(
        blocks,
        [
            "REST_A",
            "TASK_A",
        ],
    )


    group_b = np.isin(
        blocks,
        [
            "REST_B",
            "TASK_B",
        ],
    )


    folds = [
        (
            np.where(
                group_a
            )[0],
            np.where(
                group_b
            )[0],
        ),
        (
            np.where(
                group_b
            )[0],
            np.where(
                group_a
            )[0],
        ),
    ]


    records = []


    print("\n")
    print("=" * 78)
    print("Block-wise Cross Validation")
    print("=" * 78)


    for top_k in TOP_K_GRID:

        for l2 in L2_GRID:

            fold_scores = []


            for fold_index, (
                train_index,
                valid_index,
            ) in enumerate(
                folds
            ):

                (
                    feature_mean,
                    feature_std,
                    feature_mask,
                    _,

                ) = feature_stats_and_mask(

                    features[
                        train_index
                    ],

                    labels[
                        train_index
                    ],

                    top_k,

                )


                (
                    head,
                    _,

                ) = train_head(

                    features[
                        train_index
                    ],

                    labels[
                        train_index
                    ],

                    feature_mean,

                    feature_std,

                    feature_mask,

                    l2=l2,

                    epochs=CV_EPOCHS,

                    seed=(
                        SEED
                        + fold_index
                    ),

                )


                (
                    prediction,
                    _,

                ) = predict_head(

                    head,

                    features[
                        valid_index
                    ],

                    feature_mean,

                    feature_std,

                    feature_mask,

                )


                score = (
                    balanced_accuracy(

                        labels[
                            valid_index
                        ],

                        prediction,

                    )
                )


                fold_scores.append(
                    score
                )


            mean_score = float(
                np.mean(
                    fold_scores
                )
            )


            records.append(
                {
                    "top_k": int(
                        top_k
                    ),
                    "l2": float(
                        l2
                    ),
                    "fold_a_to_b": float(
                        fold_scores[0]
                    ),
                    "fold_b_to_a": float(
                        fold_scores[1]
                    ),
                    "mean_balanced_accuracy":
                        mean_score,
                }
            )


            print(

                f"TopK={top_k:3d} | "
                f"L2={l2:<4g} | "
                f"A->B={fold_scores[0]:.4f} | "
                f"B->A={fold_scores[1]:.4f} | "
                f"Mean={mean_score:.4f}"

            )


    best = max(

        records,

        key=lambda item:
            item[
                "mean_balanced_accuracy"
            ],

    )


    return (
        best,
        records,
    )


# ============================================================
# 18. Main
# ============================================================

def main():

    print("=" * 78)
    print("Personal EEG Head Calibration")
    print("=" * 78)

    print(
        "\nDevice:",
        device,
    )


    calibration_csv = (
        find_latest_calibration_csv()
    )


    print(
        "\nCalibration Raw EEG:"
    )

    print(
        calibration_csv
    )


    (
        rows,
        source_sfreq,

    ) = read_raw_csv(
        calibration_csv
    )


    print(
        "\nDetected Sampling Rate:",
        f"{source_sfreq:.3f} Hz",
    )


    (
        X,
        labels,
        blocks,
        candidate_count,
        rejected,

    ) = build_windows(

        rows,
        source_sfreq,

    )


    print(
        "\nCandidate Windows:",
        candidate_count,
    )

    print(
        "Valid Windows:",
        len(X),
    )

    print(
        "Rejected:",
        rejected,
    )


    (
        source_model,
        source_checkpoint,
        config,

    ) = load_source_model()


    print(
        "\n✓ Source LoRA Encoder loaded and frozen"
    )


    features = (
        extract_features(

            source_model,
            X,

        )
    )


    print(
        "Feature shape:",
        features.shape,
    )


    (
        best,
        cv_records,

    ) = select_hyperparameters(

        features,
        labels,
        blocks,

    )


    print("\n")
    print("=" * 78)
    print("Selected Calibration Adapter")
    print("=" * 78)


    print(
        "Top-K encoder dimensions:",
        best[
            "top_k"
        ],
    )

    print(
        "L2:",
        best[
            "l2"
        ],
    )

    print(
        "Block-CV Balanced Accuracy:",
        f"{best['mean_balanced_accuracy']:.4f}",
    )

    print(
        "A -> B:",
        f"{best['fold_a_to_b']:.4f}",
    )

    print(
        "B -> A:",
        f"{best['fold_b_to_a']:.4f}",
    )


    # --------------------------------------------------------
    # Final adapter is trained on ALL calibration blocks.
    # --------------------------------------------------------

    (
        feature_mean,
        feature_std,
        feature_mask,
        selected_indices,

    ) = feature_stats_and_mask(

        features,

        labels,

        best[
            "top_k"
        ],

    )


    (
        personal_head,
        final_loss,

    ) = train_head(

        features,
        labels,

        feature_mean,
        feature_std,
        feature_mask,

        l2=best[
            "l2"
        ],

        epochs=FINAL_EPOCHS,

        seed=SEED,

    )


    (
        calibration_prediction,
        calibration_probability,

    ) = predict_head(

        personal_head,

        features,

        feature_mean,
        feature_std,
        feature_mask,

    )


    calibration_accuracy = float(

        np.mean(

            calibration_prediction

            == labels

        )

    )


    calibration_balanced_accuracy = (
        balanced_accuracy(

            labels,

            calibration_prediction,

        )
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    checkpoint = {

        "personal_head_state_dict":
            personal_head
            .cpu()
            .state_dict(),

        "feature_mean":
            torch.from_numpy(
                feature_mean
            ),

        "feature_std":
            torch.from_numpy(
                feature_std
            ),

        "feature_mask":
            torch.from_numpy(
                feature_mask
            ),

        "selected_indices":
            selected_indices.tolist(),

        "selected_top_k":
            int(
                best[
                    "top_k"
                ]
            ),

        "selected_l2":
            float(
                best[
                    "l2"
                ]
            ),

        "block_cv_balanced_accuracy":
            float(
                best[
                    "mean_balanced_accuracy"
                ]
            ),

        "fold_a_to_b":
            float(
                best[
                    "fold_a_to_b"
                ]
            ),

        "fold_b_to_a":
            float(
                best[
                    "fold_b_to_a"
                ]
            ),

        "calibration_fit_accuracy":
            calibration_accuracy,

        "calibration_fit_balanced_accuracy":
            calibration_balanced_accuracy,

        "calibration_raw_csv":
            calibration_csv.name,

        "source_model":
            str(
                SOURCE_MODEL_PATH
            ),

        "source_sfreq":
            float(
                source_sfreq
            ),

        "max_abs_uv":
            float(
                MAX_ABS_UV
            ),

        "model_sfreq":
            float(
                MODEL_SFREQ
            ),

        "window_seconds":
            float(
                MODEL_WINDOW_SECONDS
            ),

        "config":
            config,

    }


    torch.save(
        checkpoint,
        OUTPUT_PATH,
    )


    with open(
        SUMMARY_PATH,
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
                "calibration_raw_csv",
                calibration_csv.name,
            ]
        )

        writer.writerow(
            [
                "source_sfreq",
                source_sfreq,
            ]
        )

        writer.writerow(
            [
                "candidate_windows",
                candidate_count,
            ]
        )

        writer.writerow(
            [
                "valid_windows",
                len(X),
            ]
        )

        writer.writerow(
            [
                "selected_top_k",
                best[
                    "top_k"
                ],
            ]
        )

        writer.writerow(
            [
                "selected_l2",
                best[
                    "l2"
                ],
            ]
        )

        writer.writerow(
            [
                "block_cv_balanced_accuracy",
                best[
                    "mean_balanced_accuracy"
                ],
            ]
        )

        writer.writerow(
            [
                "fold_a_to_b",
                best[
                    "fold_a_to_b"
                ],
            ]
        )

        writer.writerow(
            [
                "fold_b_to_a",
                best[
                    "fold_b_to_a"
                ],
            ]
        )

        writer.writerow(
            [
                "calibration_fit_accuracy",
                calibration_accuracy,
            ]
        )

        writer.writerow(
            [
                "calibration_fit_balanced_accuracy",
                calibration_balanced_accuracy,
            ]
        )

        writer.writerow(
            [
                "final_training_loss",
                final_loss,
            ]
        )


    print("\n")
    print("=" * 78)
    print("Personal Head Saved")
    print("=" * 78)


    print(
        "\nPersonal Head:"
    )

    print(
        OUTPUT_PATH
    )


    print(
        "\nTraining Summary:"
    )

    print(
        SUMMARY_PATH
    )


    print(
        "\nCalibration fit Accuracy:",
        f"{calibration_accuracy:.4f}",
    )

    print(
        "Calibration fit Balanced Accuracy:",
        f"{calibration_balanced_accuracy:.4f}",
    )


    print("\nIMPORTANT:")

    print(
        "上面的 Calibration fit 不是最终测试成绩。"
    )

    print(
        "必须运行 35_loongbrain_final_test.py "
        "重新采集一个独立 session。"
    )


    if (
        best[
            "mean_balanced_accuracy"
        ]
        < 0.55
    ):

        print(
            "\n⚠ Block-CV < 0.55："
            "个人适配的跨 block 稳定性仍然较弱。"
        )

        print(
            "仍可运行 35 做真正独立测试，"
            "但不要预设准确率一定会提高。"
        )

    else:

        print(
            "\nBlock-CV 高于 0.55，"
            "可以进入独立 Final Test。"
        )


if __name__ == "__main__":

    main()
