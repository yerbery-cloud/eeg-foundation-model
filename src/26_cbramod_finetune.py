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

from braindecode.models import CBraMod


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
    / "eegmat_cbramod"
)


CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "cbramod"
)


RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "cbramod"
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
    / "eegmat_cbramod_X.npy"
)


Y_PATH = (
    DATA_DIR
    / "eegmat_cbramod_y.npy"
)


METADATA_PATH = (
    DATA_DIR
    / "eegmat_cbramod_metadata.npy"
)


SPLIT_PATH = (
    DATA_DIR
    / "eegmat_cbramod_split.npz"
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
    / "cbramod_history.csv"
)


SUMMARY_PATH = (
    RESULT_DIR
    / "cbramod_test_summary.csv"
)


SUBJECT_RESULT_PATH = (
    RESULT_DIR
    / "cbramod_subject_results.csv"
)


LOSS_CURVE_PATH = (
    RESULT_DIR
    / "cbramod_loss_curve.png"
)


ACCURACY_CURVE_PATH = (
    RESULT_DIR
    / "cbramod_accuracy_curve.png"
)


CONFUSION_PATH = (
    RESULT_DIR
    / "cbramod_confusion_matrix.png"
)


# ============================================================
# 5. CBraMod Config
# ============================================================

MODEL_ID = (
    "braindecode/cbramod-pretrained"
)


NUM_CHANNELS = 2

N_TIMES = 800

SFREQ = 200

NUM_CLASSES = 2


# ============================================================
# 6. Training Config
# ============================================================

BATCH_SIZE = 32

MAX_EPOCHS = 30


# Foundation Model 全参数微调，
# 学习率不要过高。

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4


PATIENCE = 6

MIN_DELTA = 1e-4


# ============================================================
# 7. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 75)

print("CBraMod Pretrained Foundation Model")

print("=" * 75)


print(
    "\nDevice:",
    device
)


# ============================================================
# 8. 文件检查
# ============================================================

for path in [

    X_PATH,
    Y_PATH,
    METADATA_PATH,
    SPLIT_PATH

]:

    if not path.exists():

        raise FileNotFoundError(
            f"找不到文件：\n{path}"
        )


# ============================================================
# 9. 加载 EEGMAT
# ============================================================

print(
    "\n正在加载 CBraMod EEGMAT Dataset ..."
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
    "\nX:",
    X.shape
)


print(
    "y:",
    y.shape
)


print(
    "metadata:",
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
# 10. Dataset 检查
# ============================================================

if X.shape[1:] != (
    NUM_CHANNELS,
    N_TIMES
):

    raise RuntimeError(

        f"CBraMod 输入 shape 错误："
        f"{X.shape}"

    )


if not np.isfinite(
    X
).all():

    raise RuntimeError(

        "CBraMod 数据存在 NaN / Inf"

    )


if not set(
    np.unique(y).tolist()
).issubset(
    {
        0,
        1
    }
):

    raise RuntimeError(
        "Label 存在非法值"
    )


# ============================================================
# 11. Dataset
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

        real_index = (
            self.indices[
                index
            ]
        )


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
# 12. Dataset
# ============================================================

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
# 13. DataLoader
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
# 14. 加载公开预训练 CBraMod
# ============================================================

print("\n")

print("=" * 75)

print("Downloading / Loading Pretrained CBraMod")

print("=" * 75)


print(
    "\nModel:",
    MODEL_ID
)


# ------------------------------------------------------------
# 关键：
#
# n_outputs = 2
# n_chans   = 2
# n_times   = 800
# sfreq     = 200
#
# Braindecode 会加载预训练模型，
# 并根据 downstream task 重建分类 head。
# ------------------------------------------------------------

try:

    model = CBraMod.from_pretrained(

        MODEL_ID,

        n_outputs=NUM_CLASSES,

        n_chans=NUM_CHANNELS,

        n_times=N_TIMES,

        sfreq=SFREQ,

        strict=False

    )

except Exception as error:

    print("\n")

    print("=" * 75)

    print("CBraMod 加载失败")

    print("=" * 75)

    print(error)

    print(
        "\n请确认已经安装："
    )

    print(
        'pip install "braindecode[hub]"'
    )

    raise


model = model.to(
    device
)


print(
    "\n✓ CBraMod pretrained weights "
    "加载成功"
)


# ============================================================
# 15. Forward Shape Test
#
# 正式训练前一定先测试。
# ============================================================

print("\n")

print("=" * 75)

print("Forward Check")

print("=" * 75)


example_eeg = torch.from_numpy(

    X[
        train_indices[:2]
    ]

).float().to(device)


model.eval()


with torch.no_grad():

    example_output = model(
        example_eeg
    )


# ============================================================
# 16. 兼容不同 Braindecode 输出形式
# ============================================================

def get_logits(
    output
):

    # 最常见：
    # Tensor [B,2]

    if torch.is_tensor(
        output
    ):

        return output


    # 某些统一 API 可能返回 dict

    if isinstance(
        output,
        dict
    ):

        if "logits" in output:

            return output[
                "logits"
            ]


        if "output" in output:

            return output[
                "output"
            ]


        # 寻找第一个 Tensor

        for value in output.values():

            if torch.is_tensor(
                value
            ):

                return value


    # tuple / list

    if isinstance(
        output,
        (
            tuple,
            list
        )
    ):

        for value in output:

            if torch.is_tensor(
                value
            ):

                return value


    raise RuntimeError(

        f"无法识别 CBraMod 输出类型："
        f"{type(output)}"

    )


example_logits = get_logits(
    example_output
)


print(
    "Input shape:",
    example_eeg.shape
)


print(
    "Output shape:",
    example_logits.shape
)


if example_logits.shape != (
    2,
    NUM_CLASSES
):

    raise RuntimeError(

        f"CBraMod 输出异常："
        f"{example_logits.shape}"

    )


print(
    "\n✓ Forward Check 通过"
)


# ============================================================
# 17. 参数统计
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

print("=" * 75)

print("Parameter Statistics")

print("=" * 75)


print(
    "\nTotal parameters:",
    f"{total_params:,}"
)


print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)


print(
    "Trainable ratio:",
    f"{100 * trainable_params / total_params:.2f}%"
)


# ============================================================
# 18. Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 19. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

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

        (tn + tp)

        / max(
            len(targets),
            1
        )

    )


    recall_rest = (

        tn

        / max(
            tn + fp,
            1
        )

    )


    recall_task = (

        tp

        / max(
            tp + fn,
            1
        )

    )


    precision_rest = (

        tn

        / max(
            tn + fn,
            1
        )

    )


    precision_task = (

        tp

        / max(
            tp + fp,
            1
        )

    )


    f1_rest = (

        2
        * precision_rest
        * recall_rest

        / max(

            precision_rest
            + recall_rest,

            1e-12

        )

    )


    f1_task = (

        2
        * precision_task
        * recall_task

        / max(

            precision_task
            + recall_task,

            1e-12

        )

    )


    balanced_accuracy = (

        recall_rest
        + recall_task

    ) / 2


    macro_f1 = (

        f1_rest
        + f1_task

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
# 21. Evaluation
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


            output = model(
                eeg
            )


            logits = get_logits(
                output
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


best_epoch = 0


epochs_without_improvement = 0


history = []


# ============================================================
# 23. Training
# ============================================================

print("\n")

print("=" * 75)

print("开始 CBraMod Fine-tuning")

print("=" * 75)


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
    # Train Batch
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


        output = model(
            eeg
        )


        logits = get_logits(
            output
        )


        loss = criterion(

            logits,

            labels

        )


        loss.backward()


        # Foundation Model
        # 加 Gradient Clipping

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

                "total_parameters":
                    total_params,

                "trainable_parameters":
                    trainable_params,

                "model_id":
                    MODEL_ID,

                "sfreq":
                    SFREQ,

                "n_times":
                    N_TIMES,

                "n_chans":
                    NUM_CHANNELS

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
        f"{val_metrics['macro_f1']:.4f} | "

        f"{epoch_seconds:.1f}s"

    )


    if improved:

        print(
            "  ✓ 保存新的 Best CBraMod"
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
# 25. Loss Curve
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
    "CBraMod Fine-tuning - Loss"
)


plt.grid(
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(

    LOSS_CURVE_PATH,

    dpi=180

)


plt.close()


# ============================================================
# 26. Accuracy Curve
# ============================================================

train_accuracy = [

    row["train_accuracy"]

    for row in history

]


val_accuracy = [

    row["val_accuracy"]

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
    "CBraMod Fine-tuning - Accuracy"
)


plt.grid(
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(

    ACCURACY_CURVE_PATH,

    dpi=180

)


plt.close()


# ============================================================
# 27. 加载 Best Model
# ============================================================

print("\n")

print("=" * 75)

print("加载 Best CBraMod")

print("=" * 75)


if not BEST_PATH.exists():

    raise RuntimeError(
        "没有生成 Best CBraMod"
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
    "\nBest Epoch:",
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
# 28. Test
# ============================================================

print("\n")

print("=" * 75)

print("CBraMod Test Evaluation")

print("=" * 75)


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
# 29. Per-Subject
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

print("=" * 75)

print("Per-Subject Test Results")

print("=" * 75)


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
# 30. 保存 Per-Subject
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
# 31. Summary
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
            "CBraMod Pretrained Full FT"
        ]
    )


    writer.writerow(
        [
            "model_id",
            MODEL_ID
        ]
    )


    writer.writerow(
        [
            "sfreq",
            SFREQ
        ]
    )


    writer.writerow(
        [
            "n_chans",
            NUM_CHANNELS
        ]
    )


    writer.writerow(
        [
            "n_times",
            N_TIMES
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
# 32. Confusion Matrix
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
    "CBraMod - Test Confusion Matrix"
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

    dpi=180

)


plt.close()


# ============================================================
# 33. Final
# ============================================================

print("\n")

print("=" * 75)

print("CBraMod Fine-tuning 完成")

print("=" * 75)


print(
    "\nTotal Parameters:",
    f"{total_params:,}"
)


print(
    "Trainable Parameters:",
    f"{trainable_params:,}"
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