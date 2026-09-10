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

RANDOM_ENCODER_SEED = 20260828


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


PRETRAINED_ENCODER_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "pretrain20"
    / "best_encoder.pt"
)


CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "random_probe"
)


RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "random_probe"
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
# 3. Data Paths
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
# 4. Output Paths
# ============================================================

BEST_PATH = (
    CHECKPOINT_DIR
    / "best.pt"
)


HISTORY_PATH = (
    RESULT_DIR
    / "random_probe_history.csv"
)


LOSS_CURVE_PATH = (
    RESULT_DIR
    / "random_probe_loss_curve.png"
)


ACCURACY_CURVE_PATH = (
    RESULT_DIR
    / "random_probe_accuracy_curve.png"
)


SUMMARY_PATH = (
    RESULT_DIR
    / "random_probe_test_summary.csv"
)


CONFUSION_PATH = (
    RESULT_DIR
    / "random_probe_confusion_matrix.png"
)


SUBJECT_RESULT_PATH = (
    RESULT_DIR
    / "random_probe_subject_results.csv"
)


# ============================================================
# 5. Training Config
#
# 必须和 Linear Probe 保持一致
# ============================================================

BATCH_SIZE = 64

MAX_EPOCHS = 50

LEARNING_RATE = 1e-3

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
    "EEGMAT Random Encoder Linear Probe"
)

print("=" * 70)


print(
    "\nDevice:",
    device
)


# ============================================================
# 7. 检查文件
# ============================================================

required_files = [

    X_PATH,

    Y_PATH,

    METADATA_PATH,

    SPLIT_PATH,

    PRETRAINED_ENCODER_PATH

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
# 9. Dataset
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
# 10. Dataset
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
# 11. DataLoader
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
# 12. 读取模型 Config
#
# 注意：
#
# 这里只读取模型结构参数
# 不加载任何预训练权重
# ============================================================

config_checkpoint = torch.load(

    PRETRAINED_ENCODER_PATH,

    map_location="cpu"

)


config = config_checkpoint[
    "config"
]


print("\n")

print("=" * 70)

print(
    "创建随机初始化 Encoder"
)

print("=" * 70)


print(
    "\n不会加载任何 pretrained weights。"
)


# ============================================================
# 13. 固定 Random Encoder 初始化
# ============================================================

torch.manual_seed(
    RANDOM_ENCODER_SEED
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_ENCODER_SEED
    )


# ============================================================
# 14. 创建 Random Backbone
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


print(
    "\n✓ Random Encoder 创建成功"
)


# ============================================================
# 15. 冻结 Random Backbone
# ============================================================

for parameter in backbone.parameters():

    parameter.requires_grad = False


backbone.eval()


# ============================================================
# 16. Random Linear Probe
# ============================================================

class EEGLinearProbe(
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
        # Frozen Random Encoder
        # ====================================================

        with torch.no_grad():


            # -----------------------------------------------
            # Patch Embedding
            # -----------------------------------------------

            tokens = (
                self.backbone
                .patch_embedding(
                    x
                )
            )


            # -----------------------------------------------
            # Transformer
            # -----------------------------------------------

            representations = (
                self.backbone
                .encoder(
                    tokens
                )
            )


            # -----------------------------------------------
            # LayerNorm
            # -----------------------------------------------

            representations = (
                self.backbone
                .norm(
                    representations
                )
            )


            # -----------------------------------------------
            # Mean Pooling
            # -----------------------------------------------

            features = (
                representations
                .mean(
                    dim=1
                )
            )


        logits = self.classifier(
            features
        )


        return (
            logits,
            features
        )


# ============================================================
# 17. 创建 Model
# ============================================================

model = EEGLinearProbe(

    backbone=backbone,

    embed_dim=config[
        "embed_dim"
    ],

    num_classes=2

).to(device)


# ============================================================
# 18. 参数检查
# ============================================================

total_params = sum(

    p.numel()

    for p in model.parameters()

)


trainable_params = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad

)


print(
    "\nTotal parameters:",
    total_params
)


print(
    "Trainable parameters:",
    trainable_params
)


print(
    "Trainable ratio:",
    f"{100 * trainable_params / total_params:.4f}%"
)


if trainable_params != 258:

    raise RuntimeError(

        f"Random Probe 可训练参数应为 258，"
        f"实际为 {trainable_params}"

    )


print(
    "\n✓ Random Encoder 已冻结"
)


# ============================================================
# 19. Loss + Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(

    model.classifier.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


# ============================================================
# 20. Metrics
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


    accuracy = (

        (tp + tn)

        / max(
            len(targets),
            1
        )

    )


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
# 21. Evaluate
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    model.backbone.eval()


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
                device
            )


            labels = labels.to(
                device
            )


            logits, _ = model(
                eeg
            )


            loss = criterion(

                logits,

                labels

            )


            batch_size = (
                eeg.shape[0]
            )


            total_loss += (

                loss.item()
                * batch_size

            )


            sample_count += (
                batch_size
            )


            pred = torch.argmax(

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

                pred
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
# 22. Training State
# ============================================================

best_val_loss = float(
    "inf"
)


epochs_without_improvement = 0


history = []


# ============================================================
# 23. Train
# ============================================================

print("\n")

print("=" * 70)

print(
    "开始 Random Encoder Linear Probe"
)

print("=" * 70)


for epoch in range(

    1,

    MAX_EPOCHS + 1

):


    epoch_start = time.time()


    model.train()

    model.backbone.eval()


    train_loss_sum = 0.0

    train_sample_count = 0


    train_targets = []

    train_predictions = []


    for (
        eeg,
        labels,
        _
    ) in train_loader:


        eeg = eeg.to(
            device
        )


        labels = labels.to(
            device
        )


        optimizer.zero_grad()


        logits, _ = model(
            eeg
        )


        loss = criterion(

            logits,

            labels

        )


        loss.backward()


        optimizer.step()


        batch_size = (
            eeg.shape[0]
        )


        train_loss_sum += (

            loss.item()

            * batch_size

        )


        train_sample_count += (
            batch_size
        )


        pred = torch.argmax(

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

            pred
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
    # Best
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

                "random_encoder_seed":
                    RANDOM_ENCODER_SEED,

                "config":
                    config

            },

            BEST_PATH

        )


    else:

        epochs_without_improvement += 1


    print(

        f"Epoch "
        f"{epoch:02d}/"
        f"{MAX_EPOCHS} | "

        f"Train Loss "
        f"{train_loss:.4f} | "

        f"Train Acc "
        f"{train_metrics['accuracy']:.4f} | "

        f"Val Loss "
        f"{val_loss:.4f} | "

        f"Val Acc "
        f"{val_metrics['accuracy']:.4f} | "

        f"Val F1 "
        f"{val_metrics['macro_f1']:.4f}"

    )


    if improved:

        print(
            "  ✓ 保存新的 Best Random Probe"
        )


    if (

        epochs_without_improvement
        >= PATIENCE

    ):

        print(
            "\n触发 Early Stopping"
        )

        break


# ============================================================
# 24. 保存 History
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
# 25. 曲线
# ============================================================

epochs = [

    row["epoch"]

    for row in history

]


train_loss_values = [

    row["train_loss"]

    for row in history

]


val_loss_values = [

    row["val_loss"]

    for row in history

]


train_acc_values = [

    row["train_accuracy"]

    for row in history

]


val_acc_values = [

    row["val_accuracy"]

    for row in history

]


plt.figure(
    figsize=(9, 6)
)


plt.plot(

    epochs,

    train_loss_values,

    marker="o",

    label="Train Loss"

)


plt.plot(

    epochs,

    val_loss_values,

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
    "Random Encoder Probe - Loss"
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


plt.figure(
    figsize=(9, 6)
)


plt.plot(

    epochs,

    train_acc_values,

    marker="o",

    label="Train Accuracy"

)


plt.plot(

    epochs,

    val_acc_values,

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
    "Random Encoder Probe - Accuracy"
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
# 26. Load Best
# ============================================================

print("\n")

print("=" * 70)

print(
    "加载 Best Random Probe"
)

print("=" * 70)


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

model.backbone.eval()


print(
    "Best Epoch:",
    best_checkpoint[
        "epoch"
    ]
)


print(
    "Best Val Accuracy:",
    f"{best_checkpoint['val_accuracy']:.4f}"
)


print(
    "Best Val Macro F1:",
    f"{best_checkpoint['val_macro_f1']:.4f}"
)


# ============================================================
# 27. Test
# ============================================================

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


print("\n")

print("=" * 70)

print(
    "Random Probe Test Result"
)

print("=" * 70)


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
# 28. Per Subject
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


print(
    "\n===== Per-Subject ====="
)


for subject in np.unique(
    test_subject_ids
):


    mask = (

        test_subject_ids
        == subject

    )


    metrics = calculate_metrics(

        test_targets[
            mask
        ],

        test_predictions[
            mask
        ]

    )


    print(

        f"Subject{subject:02d}: "

        f"Acc = "
        f"{metrics['accuracy']:.4f} | "

        f"F1 = "
        f"{metrics['macro_f1']:.4f}"

    )


    subject_results.append(

        {

            "subject":
                int(subject),

            "accuracy":
                metrics[
                    "accuracy"
                ],

            "balanced_accuracy":
                metrics[
                    "balanced_accuracy"
                ],

            "macro_f1":
                metrics[
                    "macro_f1"
                ]

        }

    )


# ============================================================
# 29. 保存 Subject Results
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
# 30. 保存 Summary
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
            "Random Encoder Linear Probe"
        ]
    )


    writer.writerow(
        [
            "random_encoder_seed",
            RANDOM_ENCODER_SEED
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
            "trainable_parameters",
            trainable_params
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
# 31. Confusion Matrix
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
    "Random Encoder Probe - Test Confusion Matrix"
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
# 32. Final
# ============================================================

print("\n")

print("=" * 70)

print(
    "Random Encoder Probe 完成"
)

print("=" * 70)


print(
    "\nTrainable Parameters:",
    trainable_params
)


print(
    "\nBest Model:"
)

print(
    BEST_PATH
)


print(
    "\nSummary:"
)

print(
    SUMMARY_PATH
)