from pathlib import Path
import csv
import random
import time
import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from masked_eeg_model import MaskedEEGTransformer


# ============================================================
# 0. 重要修复
#
# PyTorch TransformerEncoderLayer 在 eval + no_grad 时，
# 可能启用 Fast Path。
#
# Fast Path 会直接访问：
#
# self.linear1.weight
# self.linear2.weight
#
# 从而绕开自定义 LoRA forward。
#
# 所以这里主动关闭 Fast Path。
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

    print(
        "✓ PyTorch Transformer Fast Path 已关闭"
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
    / "lora"
)


RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "lora"
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
    / "lora_history.csv"
)


SUMMARY_PATH = (
    RESULT_DIR
    / "lora_test_summary.csv"
)


SUBJECT_RESULT_PATH = (
    RESULT_DIR
    / "lora_subject_results.csv"
)


LOSS_CURVE_PATH = (
    RESULT_DIR
    / "lora_loss_curve.png"
)


ACCURACY_CURVE_PATH = (
    RESULT_DIR
    / "lora_accuracy_curve.png"
)


CONFUSION_PATH = (
    RESULT_DIR
    / "lora_confusion_matrix.png"
)


# ============================================================
# 5. Training Config
# ============================================================

BATCH_SIZE = 64

MAX_EPOCHS = 50

LEARNING_RATE = 5e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 8

MIN_DELTA = 1e-4


# ============================================================
# 6. LoRA Config
# ============================================================

LORA_RANK = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.05


# ============================================================
# 7. Device
# ============================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


print("\n")

print("=" * 70)

print("EEGMAT LoRA Fine-tuning")

print("=" * 70)


print(
    "\nDevice:",
    device
)


# ============================================================
# 8. 文件检查
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

            f"找不到文件：\n{path}"

        )


# ============================================================
# 9. 加载 EEGMAT
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
# 10. 基本数据检查
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
        "EEG 存在 NaN / Inf"
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
        "Label 存在异常"
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
# 12. Dataset 实例
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
# 14. LoRA Linear
# ============================================================

class LoRALinear(
    nn.Module
):

    """
    原始 Linear：

        y = Wx + b

    加入 LoRA：

        y = Wx + b
            +
            scaling * B(Ax)

    W 和 b：
        来自预训练模型
        完全冻结

    A / B：
        新加入
        可以训练
    """


    def __init__(
        self,
        base_linear,
        rank=8,
        alpha=16,
        dropout=0.05
    ):

        super().__init__()


        if not isinstance(
            base_linear,
            nn.Linear
        ):

            raise TypeError(
                "base_linear 必须是 nn.Linear"
            )


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


        # ====================================================
        # 原始 Linear
        # ====================================================

        self.base_linear = (
            base_linear
        )


        # 原始参数冻结
        for parameter in (
            self.base_linear.parameters()
        ):

            parameter.requires_grad = False


        # ====================================================
        # LoRA Dropout
        # ====================================================

        self.lora_dropout = nn.Dropout(
            dropout
        )


        # ====================================================
        # A:
        #
        # input_dim → rank
        # ====================================================

        self.lora_A = nn.Linear(

            self.in_features,

            rank,

            bias=False

        )


        # ====================================================
        # B:
        #
        # rank → output_dim
        # ====================================================

        self.lora_B = nn.Linear(

            rank,

            self.out_features,

            bias=False

        )


        # ====================================================
        # Initialization
        #
        # A = random
        # B = 0
        #
        # 因此刚开始：
        #
        # LoRA contribution = 0
        #
        # 模型保持原预训练输出
        # ====================================================

        nn.init.kaiming_uniform_(

            self.lora_A.weight,

            a=math.sqrt(5)

        )


        nn.init.zeros_(

            self.lora_B.weight

        )


    # ========================================================
    # PyTorch Transformer compatibility
    #
    # TransformerEncoderLayer 某些代码会访问：
    #
    # linear1.weight
    # linear1.bias
    #
    # 所以我们提供兼容属性。
    # ========================================================

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
        x
    ):

        # ====================================================
        # Original pretrained Linear
        # ====================================================

        base_output = (
            self.base_linear(
                x
            )
        )


        # ====================================================
        # LoRA branch
        # ====================================================

        lora_input = (
            self.lora_dropout(
                x
            )
        )


        low_rank = (
            self.lora_A(
                lora_input
            )
        )


        lora_output = (
            self.lora_B(
                low_rank
            )
        )


        # ====================================================
        # 合并
        # ====================================================

        output = (

            base_output

            +

            (
                self.scaling

                * lora_output
            )

        )


        return output


# ============================================================
# 15. 加载 Pretrained Encoder
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
# 16. 创建 Backbone
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
# 17. 加载 Pretrained Parameters
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


expected_missing = {

    "reconstruction_head.weight",

    "reconstruction_head.bias"

}


actual_missing = set(
    load_result.missing_keys
)


if actual_missing != expected_missing:

    print(
        "\n⚠️ Missing keys 与预期不完全一致"
    )


if len(
    load_result.unexpected_keys
) != 0:

    raise RuntimeError(

        "Encoder checkpoint "
        "与模型结构不匹配"

    )


# ============================================================
# 18. 冻结整个 Backbone
# ============================================================

for parameter in backbone.parameters():

    parameter.requires_grad = False


# ============================================================
# 19. 注入 LoRA
# ============================================================

print("\n")

print("=" * 70)

print("Inject LoRA")

print("=" * 70)


for layer_index, layer in enumerate(

    backbone.encoder.layers

):


    # ========================================================
    # FFN Linear 1
    # ========================================================

    layer.linear1 = LoRALinear(

        base_linear=(
            layer.linear1
        ),

        rank=LORA_RANK,

        alpha=LORA_ALPHA,

        dropout=LORA_DROPOUT

    )


    # ========================================================
    # FFN Linear 2
    # ========================================================

    layer.linear2 = LoRALinear(

        base_linear=(
            layer.linear2
        ),

        rank=LORA_RANK,

        alpha=LORA_ALPHA,

        dropout=LORA_DROPOUT

    )


    print(

        f"✓ Transformer Layer "
        f"{layer_index}: "

        f"linear1 + linear2 "
        f"LoRA injected"

    )


# ============================================================
# 20. LoRA Classification Model
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
        # Transformer
        #
        # Base frozen
        # LoRA trainable
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
# 21. 创建模型
# ============================================================

model = EEGLoraClassifier(

    backbone=backbone,

    embed_dim=config[
        "embed_dim"
    ],

    num_classes=2

).to(device)


# ============================================================
# 22. 参数统计
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


lora_params = sum(

    parameter.numel()

    for name, parameter
    in model.named_parameters()

    if (

        parameter.requires_grad

        and (
            "lora_A"
            in name
            or
            "lora_B"
            in name
        )

    )

)


classifier_params = sum(

    parameter.numel()

    for parameter
    in model.classifier.parameters()

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
    "LoRA parameters:",
    lora_params
)


print(
    "Classifier parameters:",
    classifier_params
)


print(
    "Trainable ratio:",
    f"{100 * trainable_params / total_params:.4f}%"
)


# ============================================================
# 23. 参数验收
# ============================================================

EXPECTED_LORA_PARAMS = 24576

EXPECTED_CLASSIFIER_PARAMS = 258

EXPECTED_TRAINABLE_PARAMS = (

    EXPECTED_LORA_PARAMS

    + EXPECTED_CLASSIFIER_PARAMS

)


print(
    "\nExpected LoRA parameters:",
    EXPECTED_LORA_PARAMS
)


print(
    "Expected classifier parameters:",
    EXPECTED_CLASSIFIER_PARAMS
)


print(
    "Expected trainable parameters:",
    EXPECTED_TRAINABLE_PARAMS
)


if lora_params != EXPECTED_LORA_PARAMS:

    raise RuntimeError(

        f"LoRA 参数数量异常："
        f"{lora_params}"

    )


if classifier_params != 258:

    raise RuntimeError(

        f"Classifier 参数异常："
        f"{classifier_params}"

    )


if (
    trainable_params
    != EXPECTED_TRAINABLE_PARAMS
):

    raise RuntimeError(

        f"Trainable 参数异常："
        f"{trainable_params}"

    )


# ============================================================
# 24. 再检查所有 Base Linear
# ============================================================

for name, parameter in (
    model.named_parameters()
):

    if (

        "base_linear" in name

        and parameter.requires_grad

    ):

        raise RuntimeError(

            f"Base pretrained 参数未冻结："
            f"{name}"

        )


print(
    "\n✓ Base pretrained weights 已冻结"
)


print(
    "✓ LoRA adapters 可训练"
)


print(
    "✓ Classifier 可训练"
)


# ============================================================
# 25. Loss
# ============================================================

criterion = (
    nn.CrossEntropyLoss()
)


# ============================================================
# 26. Optimizer
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
# 27. Metrics
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
# 28. Evaluation
# ============================================================

def evaluate(
    model,
    loader
):

    # ========================================================
    # Eval mode
    #
    # Fast Path 已经在文件最前面关闭，
    # 所以这里不会绕过 LoRA。
    # ========================================================

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
# 29. Training State
# ============================================================

best_val_loss = float(
    "inf"
)


best_epoch = 0


epochs_without_improvement = 0


history = []


# ============================================================
# 30. Training
# ============================================================

print("\n")

print("=" * 70)

print("开始 LoRA Fine-tuning")

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


        logits, _ = model(
            eeg
        )


        loss = criterion(

            logits,

            labels

        )


        loss.backward()


        # ====================================================
        # Gradient Clipping
        # ====================================================

        torch.nn.utils.clip_grad_norm_(

            trainable_parameter_list,

            max_norm=1.0

        )


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

                "trainable_parameters":
                    trainable_params,

                "lora_parameters":
                    lora_params,

                "classifier_parameters":
                    classifier_params,

                "lora_rank":
                    LORA_RANK,

                "lora_alpha":
                    LORA_ALPHA,

                "lora_dropout":
                    LORA_DROPOUT,

                "config":
                    config

            },

            BEST_PATH

        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # Epoch 输出
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
            "  ✓ 保存新的 Best LoRA Model"
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
# 31. 保存 History
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
# 32. Loss Curve
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
    "EEGMAT LoRA - Loss"
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
# 33. Accuracy Curve
# ============================================================

train_acc = [

    row["train_accuracy"]

    for row in history

]


val_acc = [

    row["val_accuracy"]

    for row in history

]


plt.figure(
    figsize=(9, 6)
)


plt.plot(

    epochs,

    train_acc,

    marker="o",

    label="Train Accuracy"

)


plt.plot(

    epochs,

    val_acc,

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
    "EEGMAT LoRA - Accuracy"
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
# 34. Load Best Model
# ============================================================

print("\n")

print("=" * 70)

print("加载 Best LoRA Model")

print("=" * 70)


if not BEST_PATH.exists():

    raise RuntimeError(

        "未生成 Best LoRA Model"

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
# 35. Test
# ============================================================

print("\n")

print("=" * 70)

print("LoRA Test Evaluation")

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
# 36. Per-Subject Test
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


    subject_metrics = (
        calculate_metrics(

            test_targets[
                subject_mask
            ],

            test_predictions[
                subject_mask
            ]

        )
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
# 37. 保存 Subject Results
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
# 38. 保存 Summary
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
            "LoRA"
        ]
    )


    writer.writerow(
        [
            "lora_rank",
            LORA_RANK
        ]
    )


    writer.writerow(
        [
            "lora_alpha",
            LORA_ALPHA
        ]
    )


    writer.writerow(
        [
            "lora_dropout",
            LORA_DROPOUT
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
            "lora_parameters",
            lora_params
        ]
    )


    writer.writerow(
        [
            "classifier_parameters",
            classifier_params
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
# 39. Confusion Matrix
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
    "EEGMAT LoRA - Test Confusion Matrix"
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
# 40. Final
# ============================================================

print("\n")

print("=" * 70)

print("LoRA Fine-tuning 完成")

print("=" * 70)


print(
    "\nTrainable Parameters:",
    trainable_params
)


print(
    "LoRA Parameters:",
    lora_params
)


print(
    "Classifier Parameters:",
    classifier_params
)


print(
    "\nBest Model:"
)

print(
    BEST_PATH
)


print(
    "\nTraining History:"
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