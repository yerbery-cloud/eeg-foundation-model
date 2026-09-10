from pathlib import Path
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import (
    Dataset,
    DataLoader
)

from masked_eeg_model import (
    MaskedEEGTransformer
)


# ============================================================
# 1. Random Seed
# ============================================================

SEED = 42


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 2. 项目路径
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eegmat_fp1_fp2"
    / "classification"
)


ENCODER_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "pretrain20"
    / "best_encoder.pt"
)


CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "full_finetune"
)


RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "full_finetune"
)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. 输入文件
# ============================================================

X_PATH = (
    DATA_DIR
    / "eegmat_X.npy"
)


Y_PATH = (
    DATA_DIR
    / "eegmat_y.npy"
)


METADATA_PATH = (
    DATA_DIR
    / "eegmat_metadata.npy"
)


SPLIT_PATH = (
    DATA_DIR
    / "eegmat_subject_split.npz"
)


# ============================================================
# 4. 输出文件
# ============================================================

BEST_PATH = (
    CHECKPOINT_DIR
    / "best.pt"
)


HISTORY_PATH = (
    RESULT_DIR
    / "full_finetune_history.csv"
)


SUMMARY_PATH = (
    RESULT_DIR
    / "full_finetune_test_summary.csv"
)


SUBJECT_RESULT_PATH = (
    RESULT_DIR
    / "full_finetune_subject_results.csv"
)


LOSS_CURVE_PATH = (
    RESULT_DIR
    / "full_finetune_loss_curve.png"
)


ACCURACY_CURVE_PATH = (
    RESULT_DIR
    / "full_finetune_accuracy_curve.png"
)


CONFUSION_PATH = (
    RESULT_DIR
    / "full_finetune_confusion_matrix.png"
)


# ============================================================
# 5. Training Config
#
# Full FT 学习率比 Linear / LoRA 更低，
# 避免一下子破坏预训练参数。
# ============================================================

BATCH_SIZE = 64

MAX_EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 8

MIN_DELTA = 1e-4


# ============================================================
# 6. Device
# ============================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


print("=" * 70)

print(
    "EEGMAT Full Fine-tuning"
)

print("=" * 70)


print(
    "\nDevice:",
    device
)


# ============================================================
# 7. 文件检查
# ============================================================

required_files = [

    X_PATH,

    Y_PATH,

    METADATA_PATH,

    SPLIT_PATH,

    ENCODER_PATH

]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(

            f"找不到文件：\n"
            f"{path}"

        )


# ============================================================
# 8. 加载 EEGMAT
# ============================================================

print(
    "\n正在加载 EEGMAT ..."
)


X = np.load(
    X_PATH
)


y = np.load(
    Y_PATH
)


metadata = np.load(
    METADATA_PATH
)


split = np.load(
    SPLIT_PATH
)


train_indices = split[
    "train_indices"
]


val_indices = split[
    "val_indices"
]


test_indices = split[
    "test_indices"
]


print(
    "\nX shape:",
    X.shape
)


print(
    "y shape:",
    y.shape
)


print(
    "metadata shape:",
    metadata.shape
)


print(
    "\nTrain:",
    len(train_indices)
)


print(
    "Validation:",
    len(val_indices)
)


print(
    "Test:",
    len(test_indices)
)


# ============================================================
# 9. 数据检查
# ============================================================

if X.shape[1:] != (
    2,
    640
):

    raise RuntimeError(

        f"EEG shape 异常："
        f"{X.shape}"

    )


if not np.isfinite(
    X
).all():

    raise RuntimeError(

        "EEG 中存在 NaN / Inf"

    )


if not set(

    np.unique(
        y
    ).tolist()

).issubset(
    {
        0,
        1
    }
):

    raise RuntimeError(

        "Label 中存在非法值"

    )


# ============================================================
# 10. Dataset
# ============================================================

class EEGClassificationDataset(
    Dataset
):

    def __init__(
        self,
        X,
        y,
        metadata,
        indices
    ):

        self.X = X

        self.y = y

        self.metadata = metadata

        self.indices = np.asarray(
            indices
        )


    def __len__(
        self
    ):

        return len(
            self.indices
        )


    def __getitem__(
        self,
        index
    ):

        real_index = self.indices[
            index
        ]


        eeg = torch.from_numpy(

            self.X[
                real_index
            ]

        ).float()


        label = torch.tensor(

            self.y[
                real_index
            ],

            dtype=torch.long

        )


        meta = torch.from_numpy(

            self.metadata[
                real_index
            ]

        ).float()


        return (
            eeg,
            label,
            meta
        )


# ============================================================
# 11. Dataset
# ============================================================

train_dataset = (
    EEGClassificationDataset(

        X,
        y,
        metadata,
        train_indices

    )
)


val_dataset = (
    EEGClassificationDataset(

        X,
        y,
        metadata,
        val_indices

    )
)


test_dataset = (
    EEGClassificationDataset(

        X,
        y,
        metadata,
        test_indices

    )
)


# ============================================================
# 12. DataLoader
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=(
        device.type == "cuda"
    )

)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=(
        device.type == "cuda"
    )

)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=(
        device.type == "cuda"
    )

)


# ============================================================
# 13. 加载 Pretrained Encoder
# ============================================================

print(
    "\n正在加载 Pretrained Encoder："
)


print(
    ENCODER_PATH
)


encoder_checkpoint = torch.load(

    ENCODER_PATH,

    map_location=device

)


config = encoder_checkpoint[
    "config"
]


print(
    "\nPretraining Best Epoch:",
    encoder_checkpoint.get(
        "best_epoch",
        "Unknown"
    )
)


print(
    "Pretraining Best Val Loss:",
    encoder_checkpoint.get(
        "best_val_loss",
        "Unknown"
    )
)


# ============================================================
# 14. 创建 Backbone
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

).to(device)


# ============================================================
# 15. 加载 Pretrained 权重
# ============================================================

load_result = backbone.load_state_dict(

    encoder_checkpoint[
        "encoder_state_dict"
    ],

    strict=False

)


print(
    "\nMissing keys:"
)


print(
    load_result.missing_keys
)


print(
    "\nUnexpected keys:"
)


print(
    load_result.unexpected_keys
)


# 正常应该只有 Reconstruction Head 缺失

expected_missing = {

    "reconstruction_head.weight",

    "reconstruction_head.bias"

}


actual_missing = set(
    load_result.missing_keys
)


if actual_missing != expected_missing:

    print(
        "\n⚠️ Missing keys 与预期不同"
    )


if len(
    load_result.unexpected_keys
) > 0:

    raise RuntimeError(

        "Pretrained Encoder "
        "与当前模型结构不匹配"

    )


# ============================================================
# 16. Full Fine-tuning 参数设置
# ============================================================

# ------------------------------------------------------------
# Patch Embedding
# Transformer
# Norm
#
# 全部允许训练
# ------------------------------------------------------------

for parameter in (
    backbone.patch_embedding.parameters()
):

    parameter.requires_grad = True


for parameter in (
    backbone.encoder.parameters()
):

    parameter.requires_grad = True


for parameter in (
    backbone.norm.parameters()
):

    parameter.requires_grad = True


# ------------------------------------------------------------
# Reconstruction Head
#
# 下游分类完全不需要
#
# 所以冻结
# ------------------------------------------------------------

for parameter in (
    backbone
    .reconstruction_head
    .parameters()
):

    parameter.requires_grad = False


# ============================================================
# 17. Classification Model
# ============================================================

class EEGFullFinetuneClassifier(
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

        # ====================================================
        # Patch Embedding
        #
        # [B,2,640]
        # →
        # [B,32,128]
        # ====================================================

        tokens = (
            self.backbone
            .patch_embedding(
                x
            )
        )


        # ====================================================
        # Transformer Encoder
        #
        # Full Fine-tuning：
        #
        # 这里所有参数都参与梯度更新。
        # ====================================================

        representations = (
            self.backbone
            .encoder(
                tokens
            )
        )


        # ====================================================
        # LayerNorm
        # ====================================================

        representations = (
            self.backbone
            .norm(
                representations
            )
        )


        # ====================================================
        # Mean Pooling
        #
        # [B,32,128]
        # →
        # [B,128]
        # ====================================================

        features = (
            representations
            .mean(
                dim=1
            )
        )


        # ====================================================
        # Classification
        # ====================================================

        logits = self.classifier(
            features
        )


        return (
            logits,
            features
        )


# ============================================================
# 18. 创建模型
# ============================================================

model = EEGFullFinetuneClassifier(

    backbone=backbone,

    embed_dim=config[
        "embed_dim"
    ],

    num_classes=2

).to(device)


# ============================================================
# 19. 参数统计
# ============================================================

total_params = sum(

    parameter.numel()

    for parameter
    in model.parameters()

)


trainable_params = sum(

    parameter.numel()

    for parameter
    in model.parameters()

    if parameter.requires_grad

)


frozen_params = (

    total_params

    - trainable_params

)


classifier_params = sum(

    parameter.numel()

    for parameter
    in model.classifier.parameters()

)


print("\n")

print("=" * 70)

print(
    "Parameter Statistics"
)

print("=" * 70)


print(
    "\nTotal parameters:",
    total_params
)


print(
    "Trainable parameters:",
    trainable_params
)


print(
    "Frozen parameters:",
    frozen_params
)


print(
    "Classifier parameters:",
    classifier_params
)


print(
    "Trainable ratio:",
    f"{100 * trainable_params / total_params:.4f}%"
)


print(
    "\nLinear Probe trainable = 258"
)


print(
    "LoRA trainable ≈ 24834"
)


print(
    "Full Fine-tuning 应为约 538K"
)


# ============================================================
# 20. 检查 Reconstruction Head
# ============================================================

for name, parameter in (
    model.named_parameters()
):

    if (

        "reconstruction_head"
        in name

        and parameter.requires_grad

    ):

        raise RuntimeError(

            f"Reconstruction Head "
            f"不应该参与分类训练："
            f"{name}"

        )


print(
    "\n✓ Reconstruction Head 已冻结"
)


print(
    "✓ Pretrained Encoder 全参数可训练"
)


print(
    "✓ Classifier 可训练"
)


# ============================================================
# 21. Loss
# ============================================================

criterion = (
    nn.CrossEntropyLoss()
)


# ============================================================
# 22. Optimizer
# ============================================================

trainable_parameter_list = [

    parameter

    for parameter
    in model.parameters()

    if parameter.requires_grad

]


optimizer = torch.optim.AdamW(

    trainable_parameter_list,

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


# ============================================================
# 23. Metrics
# ============================================================

def calculate_metrics(
    targets,
    predictions
):

    targets = np.asarray(
        targets
    )


    predictions = np.asarray(
        predictions
    )


    # ========================================================
    # Confusion Matrix
    # ========================================================

    tn = int(

        np.sum(

            (targets == 0)

            &

            (predictions == 0)

        )

    )


    fp = int(

        np.sum(

            (targets == 0)

            &

            (predictions == 1)

        )

    )


    fn = int(

        np.sum(

            (targets == 1)

            &

            (predictions == 0)

        )

    )


    tp = int(

        np.sum(

            (targets == 1)

            &

            (predictions == 1)

        )

    )


    # ========================================================
    # Accuracy
    # ========================================================

    accuracy = (

        (tn + tp)

        / max(
            len(targets),
            1
        )

    )


    # ========================================================
    # Recall
    # ========================================================

    recall_0 = (

        tn

        / max(
            tn + fp,
            1
        )

    )


    recall_1 = (

        tp

        / max(
            tp + fn,
            1
        )

    )


    # ========================================================
    # Precision
    # ========================================================

    precision_0 = (

        tn

        / max(
            tn + fn,
            1
        )

    )


    precision_1 = (

        tp

        / max(
            tp + fp,
            1
        )

    )


    # ========================================================
    # F1
    # ========================================================

    f1_0 = (

        2
        * precision_0
        * recall_0

        / max(

            precision_0
            + recall_0,

            1e-12

        )

    )


    f1_1 = (

        2
        * precision_1
        * recall_1

        / max(

            precision_1
            + recall_1,

            1e-12

        )

    )


    macro_f1 = (

        f1_0
        + f1_1

    ) / 2


    balanced_accuracy = (

        recall_0
        + recall_1

    ) / 2


    return {

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1":
            macro_f1,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp

    }


# ============================================================
# 24. Evaluation
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()


    total_loss = 0.0

    sample_count = 0


    targets = []

    predictions = []

    metadata_list = []


    with torch.no_grad():


        for (
            eeg,
            labels,
            meta

        ) in loader:


            eeg = eeg.to(

                device,

                non_blocking=True

            )


            labels = labels.to(

                device,

                non_blocking=True

            )


            logits, _ = model(
                eeg
            )


            loss = criterion(

                logits,

                labels

            )


            current_batch_size = (
                eeg.shape[0]
            )


            total_loss += (

                loss.item()

                * current_batch_size

            )


            sample_count += (
                current_batch_size
            )


            prediction = torch.argmax(

                logits,

                dim=1

            )


            targets.extend(

                labels
                .cpu()
                .numpy()
                .tolist()

            )


            predictions.extend(

                prediction
                .cpu()
                .numpy()
                .tolist()

            )


            metadata_list.append(

                meta.numpy()

            )


    average_loss = (

        total_loss

        / sample_count

    )


    metrics = calculate_metrics(

        targets,

        predictions

    )


    return (

        average_loss,

        metrics,

        np.asarray(
            targets
        ),

        np.asarray(
            predictions
        ),

        np.concatenate(

            metadata_list,

            axis=0

        )

    )


# ============================================================
# 25. Training State
# ============================================================

best_val_loss = float(
    "inf"
)


best_epoch = 0


epochs_without_improvement = 0


history = []


# ============================================================
# 26. Training
# ============================================================

print("\n")

print("=" * 70)

print(
    "开始 Full Fine-tuning"
)

print("=" * 70)


for epoch in range(

    1,

    MAX_EPOCHS + 1

):


    epoch_start = time.time()


    model.train()


    train_loss_sum = 0.0

    train_sample_count = 0


    train_targets = []

    train_predictions = []


    # ========================================================
    # Batch Training
    # ========================================================

    for (
        eeg,
        labels,
        _
    ) in train_loader:


        eeg = eeg.to(

            device,

            non_blocking=True

        )


        labels = labels.to(

            device,

            non_blocking=True

        )


        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits, _ = model(
            eeg
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(

            logits,

            labels

        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Gradient Clipping
        #
        # Full FT 更新参数多，
        # Gradient clipping 更重要。
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(

            trainable_parameter_list,

            max_norm=1.0

        )


        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        optimizer.step()


        current_batch_size = (
            eeg.shape[0]
        )


        train_loss_sum += (

            loss.item()

            * current_batch_size

        )


        train_sample_count += (
            current_batch_size
        )


        prediction = torch.argmax(

            logits,

            dim=1

        )


        train_targets.extend(

            labels
            .cpu()
            .numpy()
            .tolist()

        )


        train_predictions.extend(

            prediction
            .detach()
            .cpu()
            .numpy()
            .tolist()

        )


    # ========================================================
    # Train Metrics
    # ========================================================

    train_loss = (

        train_loss_sum

        / train_sample_count

    )


    train_metrics = calculate_metrics(

        train_targets,

        train_predictions

    )


    # ========================================================
    # Validation
    # ========================================================

    (
        val_loss,

        val_metrics,

        _,

        _,

        _

    ) = evaluate(

        model,

        val_loader

    )


    epoch_seconds = (

        time.time()

        - epoch_start

    )


    # ========================================================
    # History
    # ========================================================

    history.append(

        {

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_metrics[
                    "accuracy"
                ],

            "train_macro_f1":
                train_metrics[
                    "macro_f1"
                ],

            "val_loss":
                val_loss,

            "val_accuracy":
                val_metrics[
                    "accuracy"
                ],

            "val_balanced_accuracy":
                val_metrics[
                    "balanced_accuracy"
                ],

            "val_macro_f1":
                val_metrics[
                    "macro_f1"
                ],

            "seconds":
                epoch_seconds

        }

    )


    # ========================================================
    # Best Model
    #
    # 与 Linear / LoRA 一致：
    #
    # 使用 Validation Loss 选择。
    # ========================================================

    improved = (

        val_loss

        < (
            best_val_loss
            - MIN_DELTA
        )

    )


    if improved:

        best_val_loss = (
            val_loss
        )


        best_epoch = (
            epoch
        )


        epochs_without_improvement = 0


        torch.save(

            {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_metrics[
                        "accuracy"
                    ],

                "val_balanced_accuracy":
                    val_metrics[
                        "balanced_accuracy"
                    ],

                "val_macro_f1":
                    val_metrics[
                        "macro_f1"
                    ],

                "trainable_parameters":
                    trainable_params,

                "total_parameters":
                    total_params,

                "learning_rate":
                    LEARNING_RATE,

                "config":
                    config

            },

            BEST_PATH

        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # Output
    # ========================================================

    print(

        f"Epoch "
        f"{epoch:02d}/"
        f"{MAX_EPOCHS} | "

        f"Train Loss "
        f"{train_loss:.4f} | "

        f"Train Acc "
        f"{train_metrics['accuracy']:.4f} | "

        f"Train F1 "
        f"{train_metrics['macro_f1']:.4f} | "

        f"Val Loss "
        f"{val_loss:.4f} | "

        f"Val Acc "
        f"{val_metrics['accuracy']:.4f} | "

        f"Val F1 "
        f"{val_metrics['macro_f1']:.4f}"

    )


    if improved:

        print(

            "  ✓ 保存新的 "
            "Best Full FT Model"

        )


    # ========================================================
    # Early Stopping
    # ========================================================

    if (

        epochs_without_improvement

        >= PATIENCE

    ):

        print("\n")

        print(
            "触发 Early Stopping"
        )


        print(

            f"连续 "
            f"{PATIENCE} "
            f"个 Epoch "
            f"Validation Loss 未改善"

        )


        break


# ============================================================
# 27. 保存 History
# ============================================================

with open(

    HISTORY_PATH,

    "w",

    newline="",

    encoding="utf-8"

) as file:


    writer = csv.DictWriter(

        file,

        fieldnames=[

            "epoch",

            "train_loss",

            "train_accuracy",

            "train_macro_f1",

            "val_loss",

            "val_accuracy",

            "val_balanced_accuracy",

            "val_macro_f1",

            "seconds"

        ]

    )


    writer.writeheader()


    writer.writerows(
        history
    )


# ============================================================
# 28. Loss Curve
# ============================================================

epochs = [

    row[
        "epoch"
    ]

    for row in history

]


train_losses = [

    row[
        "train_loss"
    ]

    for row in history

]


val_losses = [

    row[
        "val_loss"
    ]

    for row in history

]


plt.figure(
    figsize=(9, 6)
)


plt.plot(

    epochs,

    train_losses,

    marker="o",

    label="Train Loss"

)


plt.plot(

    epochs,

    val_losses,

    marker="o",

    label="Validation Loss"

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Cross Entropy Loss"
)


plt.title(
    "EEGMAT Full Fine-tuning - Loss"
)


plt.grid(
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(

    LOSS_CURVE_PATH,

    dpi=150

)


plt.close()


# ============================================================
# 29. Accuracy Curve
# ============================================================

train_accuracy = [

    row[
        "train_accuracy"
    ]

    for row in history

]


val_accuracy = [

    row[
        "val_accuracy"
    ]

    for row in history

]


plt.figure(
    figsize=(9, 6)
)


plt.plot(

    epochs,

    train_accuracy,

    marker="o",

    label="Train Accuracy"

)


plt.plot(

    epochs,

    val_accuracy,

    marker="o",

    label="Validation Accuracy"

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "EEGMAT Full Fine-tuning - Accuracy"
)


plt.grid(
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(

    ACCURACY_CURVE_PATH,

    dpi=150

)


plt.close()


# ============================================================
# 30. 加载 Best Model
# ============================================================

print("\n")

print("=" * 70)

print(
    "加载 Best Full Fine-tuning Model"
)

print("=" * 70)


if not BEST_PATH.exists():

    raise RuntimeError(

        "没有生成 Best Model"

    )


best_checkpoint = torch.load(

    BEST_PATH,

    map_location=device

)


model.load_state_dict(

    best_checkpoint[
        "model_state_dict"
    ]

)


model.eval()


print(
    "Best Epoch:",
    best_checkpoint[
        "epoch"
    ]
)


print(
    "Best Val Loss:",
    f"{best_checkpoint['val_loss']:.6f}"
)


print(
    "Best Val Accuracy:",
    f"{best_checkpoint['val_accuracy']:.4f}"
)


print(
    "Best Val Balanced Accuracy:",
    f"{best_checkpoint['val_balanced_accuracy']:.4f}"
)


print(
    "Best Val Macro F1:",
    f"{best_checkpoint['val_macro_f1']:.4f}"
)


# ============================================================
# 31. Test
# ============================================================

print("\n")

print("=" * 70)

print(
    "Full Fine-tuning Test Evaluation"
)

print("=" * 70)


(
    test_loss,

    test_metrics,

    test_targets,

    test_predictions,

    test_metadata

) = evaluate(

    model,

    test_loader

)


print(
    "\nTest Loss:",
    f"{test_loss:.6f}"
)


print(
    "Test Accuracy:",
    f"{test_metrics['accuracy']:.4f}"
)


print(
    "Test Balanced Accuracy:",
    f"{test_metrics['balanced_accuracy']:.4f}"
)


print(
    "Test Macro F1:",
    f"{test_metrics['macro_f1']:.4f}"
)


print(
    "\nTN:",
    test_metrics[
        "tn"
    ]
)


print(
    "FP:",
    test_metrics[
        "fp"
    ]
)


print(
    "FN:",
    test_metrics[
        "fn"
    ]
)


print(
    "TP:",
    test_metrics[
        "tp"
    ]
)


# ============================================================
# 32. Per-Subject Test
# ============================================================

test_subject_ids = (

    test_metadata[
        :,
        0
    ]

    .astype(
        np.int32
    )

)


subject_results = []


print("\n")

print("=" * 70)

print(
    "Per-Subject Test Results"
)

print("=" * 70)


for subject in np.unique(
    test_subject_ids
):


    subject_mask = (

        test_subject_ids
        == subject

    )


    subject_metrics = calculate_metrics(

        test_targets[
            subject_mask
        ],

        test_predictions[
            subject_mask
        ]

    )


    print(

        f"Subject{subject:02d}: "

        f"Acc = "
        f"{subject_metrics['accuracy']:.4f} | "

        f"Balanced Acc = "
        f"{subject_metrics['balanced_accuracy']:.4f} | "

        f"F1 = "
        f"{subject_metrics['macro_f1']:.4f}"

    )


    subject_results.append(

        {

            "subject":
                int(subject),

            "samples":
                int(
                    np.sum(
                        subject_mask
                    )
                ),

            "accuracy":
                subject_metrics[
                    "accuracy"
                ],

            "balanced_accuracy":
                subject_metrics[
                    "balanced_accuracy"
                ],

            "macro_f1":
                subject_metrics[
                    "macro_f1"
                ]

        }

    )


# ============================================================
# 33. 保存 Per-Subject Results
# ============================================================

with open(

    SUBJECT_RESULT_PATH,

    "w",

    newline="",

    encoding="utf-8"

) as file:


    writer = csv.DictWriter(

        file,

        fieldnames=[

            "subject",

            "samples",

            "accuracy",

            "balanced_accuracy",

            "macro_f1"

        ]

    )


    writer.writeheader()


    writer.writerows(
        subject_results
    )


# ============================================================
# 34. 保存 Summary
# ============================================================

with open(

    SUMMARY_PATH,

    "w",

    newline="",

    encoding="utf-8"

) as file:


    writer = csv.writer(
        file
    )


    writer.writerow(
        [
            "metric",
            "value"
        ]
    )


    writer.writerow(
        [
            "method",
            "Full Fine-tuning"
        ]
    )


    writer.writerow(
        [
            "learning_rate",
            LEARNING_RATE
        ]
    )


    writer.writerow(
        [
            "total_parameters",
            total_params
        ]
    )


    writer.writerow(
        [
            "trainable_parameters",
            trainable_params
        ]
    )


    writer.writerow(
        [
            "best_epoch",
            best_checkpoint[
                "epoch"
            ]
        ]
    )


    writer.writerow(
        [
            "val_loss",
            best_checkpoint[
                "val_loss"
            ]
        ]
    )


    writer.writerow(
        [
            "val_accuracy",
            best_checkpoint[
                "val_accuracy"
            ]
        ]
    )


    writer.writerow(
        [
            "val_balanced_accuracy",
            best_checkpoint[
                "val_balanced_accuracy"
            ]
        ]
    )


    writer.writerow(
        [
            "val_macro_f1",
            best_checkpoint[
                "val_macro_f1"
            ]
        ]
    )


    writer.writerow(
        [
            "test_loss",
            test_loss
        ]
    )


    writer.writerow(
        [
            "test_accuracy",
            test_metrics[
                "accuracy"
            ]
        ]
    )


    writer.writerow(
        [
            "test_balanced_accuracy",
            test_metrics[
                "balanced_accuracy"
            ]
        ]
    )


    writer.writerow(
        [
            "test_macro_f1",
            test_metrics[
                "macro_f1"
            ]
        ]
    )


# ============================================================
# 35. Confusion Matrix
# ============================================================

confusion_matrix = np.asarray(

    [

        [

            test_metrics[
                "tn"
            ],

            test_metrics[
                "fp"
            ]

        ],

        [

            test_metrics[
                "fn"
            ],

            test_metrics[
                "tp"
            ]

        ]

    ]

)


plt.figure(
    figsize=(6, 5)
)


plt.imshow(
    confusion_matrix
)


plt.xticks(

    [
        0,
        1
    ],

    [
        "REST",
        "TASK"
    ]

)


plt.yticks(

    [
        0,
        1
    ],

    [
        "REST",
        "TASK"
    ]

)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "True"
)


plt.title(
    "EEGMAT Full Fine-tuning - Test Confusion Matrix"
)


for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            str(
                confusion_matrix[
                    i,
                    j
                ]
            ),

            ha="center",

            va="center"

        )


plt.tight_layout()


plt.savefig(

    CONFUSION_PATH,

    dpi=150

)


plt.close()


# ============================================================
# 36. Final
# ============================================================

print("\n")

print("=" * 70)

print(
    "Full Fine-tuning 完成"
)

print("=" * 70)


print(
    "\nTotal Parameters:",
    total_params
)


print(
    "Trainable Parameters:",
    trainable_params
)


print(
    "\nBest Model:"
)

print(
    BEST_PATH
)


print(
    "\nHistory:"
)

print(
    HISTORY_PATH
)


print(
    "\nSummary:"
)

print(
    SUMMARY_PATH
)


print(
    "\nLoss Curve:"
)

print(
    LOSS_CURVE_PATH
)


print(
    "\nAccuracy Curve:"
)

print(
    ACCURACY_CURVE_PATH
)


print(
    "\nConfusion Matrix:"
)

print(
    CONFUSION_PATH
)