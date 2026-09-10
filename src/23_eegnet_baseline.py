from pathlib import Path
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


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
# 2. Path
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


CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "eegnet"
)


RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eegnet"
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
# 3. Input
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
# 4. Output
# ============================================================

BEST_PATH = (
    CHECKPOINT_DIR
    / "best.pt"
)


HISTORY_PATH = (
    RESULT_DIR
    / "eegnet_history.csv"
)


SUMMARY_PATH = (
    RESULT_DIR
    / "eegnet_test_summary.csv"
)


SUBJECT_RESULT_PATH = (
    RESULT_DIR
    / "eegnet_subject_results.csv"
)


LOSS_CURVE_PATH = (
    RESULT_DIR
    / "eegnet_loss_curve.png"
)


ACCURACY_CURVE_PATH = (
    RESULT_DIR
    / "eegnet_accuracy_curve.png"
)


CONFUSION_PATH = (
    RESULT_DIR
    / "eegnet_confusion_matrix.png"
)


# ============================================================
# 5. Training Config
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
print("EEGMAT EEGNet-style Baseline")
print("=" * 70)

print(
    "\nDevice:",
    device
)


# ============================================================
# 7. Load Data
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
    "\nX:",
    X.shape
)


print(
    "y:",
    y.shape
)


print(
    "Train:",
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
# 8. Dataset
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


    def __len__(self):

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


train_dataset = EEGClassificationDataset(

    X,
    y,
    metadata,
    train_indices

)


val_dataset = EEGClassificationDataset(

    X,
    y,
    metadata,
    val_indices

)


test_dataset = EEGClassificationDataset(

    X,
    y,
    metadata,
    test_indices

)


# ============================================================
# 9. DataLoader
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
# 10. EEGNet-style Model
# ============================================================

class EEGNetBaseline(
    nn.Module
):

    def __init__(
        self,
        num_channels=2,
        num_classes=2
    ):

        super().__init__()


        # ----------------------------------------------------
        # 输入：
        #
        # [B, 2, 640]
        #
        # Conv2d 需要：
        #
        # [B, 1, 2, 640]
        # ----------------------------------------------------


        # ====================================================
        # Block 1
        #
        # Temporal Convolution
        # ====================================================

        self.temporal_conv = nn.Conv2d(

            in_channels=1,

            out_channels=16,

            kernel_size=(
                1,
                64
            ),

            padding=(
                0,
                32
            ),

            bias=False

        )


        self.bn1 = nn.BatchNorm2d(
            16
        )


        # ====================================================
        # Block 2
        #
        # Spatial / Channel Mixing
        #
        # kernel height = 2
        #
        # 一次覆盖 Fp1 / Fp2
        # ====================================================

        self.spatial_conv = nn.Conv2d(

            in_channels=16,

            out_channels=32,

            kernel_size=(
                num_channels,
                1
            ),

            groups=16,

            bias=False

        )


        self.bn2 = nn.BatchNorm2d(
            32
        )


        self.activation1 = nn.ELU()


        self.pool1 = nn.AvgPool2d(

            kernel_size=(
                1,
                4
            )

        )


        self.dropout1 = nn.Dropout(
            0.5
        )


        # ====================================================
        # Block 3
        #
        # Depthwise Temporal Conv
        # ====================================================

        self.depthwise_temporal = nn.Conv2d(

            in_channels=32,

            out_channels=32,

            kernel_size=(
                1,
                16
            ),

            padding=(
                0,
                8
            ),

            groups=32,

            bias=False

        )


        # ====================================================
        # Pointwise Conv
        # ====================================================

        self.pointwise = nn.Conv2d(

            in_channels=32,

            out_channels=32,

            kernel_size=(
                1,
                1
            ),

            bias=False

        )


        self.bn3 = nn.BatchNorm2d(
            32
        )


        self.activation2 = nn.ELU()


        self.pool2 = nn.AvgPool2d(

            kernel_size=(
                1,
                8
            )

        )


        self.dropout2 = nn.Dropout(
            0.5
        )


        # ====================================================
        # Adaptive Pool
        #
        # 避免手算 Flatten size
        # ====================================================

        self.global_pool = (
            nn.AdaptiveAvgPool2d(
                (
                    1,
                    1
                )
            )
        )


        # ====================================================
        # Classifier
        # ====================================================

        self.classifier = nn.Linear(

            32,

            num_classes

        )


    def forward(
        self,
        x
    ):

        # ----------------------------------------------------
        # [B,2,640]
        #
        # →
        #
        # [B,1,2,640]
        # ----------------------------------------------------

        x = x.unsqueeze(
            1
        )


        # ====================================================
        # Temporal Conv
        # ====================================================

        x = self.temporal_conv(
            x
        )


        x = self.bn1(
            x
        )


        # ====================================================
        # Spatial Conv
        # ====================================================

        x = self.spatial_conv(
            x
        )


        x = self.bn2(
            x
        )


        x = self.activation1(
            x
        )


        x = self.pool1(
            x
        )


        x = self.dropout1(
            x
        )


        # ====================================================
        # Separable Temporal Conv
        # ====================================================

        x = self.depthwise_temporal(
            x
        )


        x = self.pointwise(
            x
        )


        x = self.bn3(
            x
        )


        x = self.activation2(
            x
        )


        x = self.pool2(
            x
        )


        x = self.dropout2(
            x
        )


        # ====================================================
        # Global Average Pool
        # ====================================================

        x = self.global_pool(
            x
        )


        # [B,32,1,1]
        #
        # →
        #
        # [B,32]

        features = x.flatten(
            1
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
# 11. Create Model
# ============================================================

model = EEGNetBaseline(

    num_channels=2,

    num_classes=2

).to(device)


# ============================================================
# 12. Parameter Statistics
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


print("\n")

print("=" * 70)
print("Parameter Statistics")
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
    "Trainable ratio:",
    f"{100 * trainable_params / total_params:.2f}%"
)


print(
    "\n说明：EEGNet-style baseline "
    "从随机初始化开始训练。"
)


print(
    "不使用任何 EEGMMIDB 预训练权重。"
)


# ============================================================
# 13. Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 14. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


# ============================================================
# 15. Metrics
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

        (tn + tp)

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
# 16. Evaluation
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
# 17. Training State
# ============================================================

best_val_loss = float(
    "inf"
)


best_epoch = 0


epochs_without_improvement = 0


history = []


# ============================================================
# 18. Training
# ============================================================

print("\n")

print("=" * 70)
print("开始 EEGNet Baseline Training")
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


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0

        )


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
    # Train Results
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
    # Best Model
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


        best_epoch = epoch


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

                "total_parameters":
                    total_params,

                "trainable_parameters":
                    trainable_params

            },

            BEST_PATH

        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # Print
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
            "  ✓ 保存新的 Best EEGNet"
        )


    if (

        epochs_without_improvement

        >= PATIENCE

    ):

        print("\n")

        print(
            "触发 Early Stopping"
        )

        break


# ============================================================
# 19. 保存 History
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
# 20. Curves
# ============================================================

epochs = [

    row["epoch"]

    for row in history

]


train_losses = [

    row["train_loss"]

    for row in history

]


val_losses = [

    row["val_loss"]

    for row in history

]


train_accuracy = [

    row["train_accuracy"]

    for row in history

]


val_accuracy = [

    row["val_accuracy"]

    for row in history

]


# Loss

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
    "EEGMAT EEGNet Baseline - Loss"
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


# Accuracy

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
    "EEGMAT EEGNet Baseline - Accuracy"
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
# 21. Load Best
# ============================================================

print("\n")

print("=" * 70)
print("加载 Best EEGNet Baseline")
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
    "Best Val Macro F1:",
    f"{best_checkpoint['val_macro_f1']:.4f}"
)


# ============================================================
# 22. Test
# ============================================================

print("\n")

print("=" * 70)
print("EEGNet Test Evaluation")
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
    test_metrics["tn"]
)


print(
    "FP:",
    test_metrics["fp"]
)


print(
    "FN:",
    test_metrics["fn"]
)


print(
    "TP:",
    test_metrics["tp"]
)


# ============================================================
# 23. Per Subject
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
print("Per-Subject Test Results")
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
# 24. Save Subject Results
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
# 25. Save Summary
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
            "EEGNet-style Baseline"
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
# 26. Confusion Matrix
# ============================================================

confusion_matrix = np.asarray(

    [

        [

            test_metrics["tn"],

            test_metrics["fp"]

        ],

        [

            test_metrics["fn"],

            test_metrics["tp"]

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

    [0, 1],

    [
        "REST",
        "TASK"
    ]

)


plt.yticks(

    [0, 1],

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
    "EEGNet Baseline - Test Confusion Matrix"
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
# 27. Final
# ============================================================

print("\n")

print("=" * 70)
print("EEGNet Baseline 完成")
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