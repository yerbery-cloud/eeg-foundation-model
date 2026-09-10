from pathlib import Path
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from masked_eeg_model import MaskedEEGTransformer


# ============================================================
# 36_mask_ratio_robustness.py
#
# 目的：
# 比较 20-subject 与 60-subject 预训练模型在不同 Mask Ratio 下
# 的未见被试重建鲁棒性。
#
# 公平性原则：
# 1. 固定同一测试集：S055-S060
# 2. 每个 Mask Ratio 下，两模型使用完全相同的 EEG windows
# 3. 每个 batch 只生成一次 mask，两模型共享该 mask
# 4. 只在 masked patch 上计算 MSE
# ============================================================


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 20260902

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. Paths
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
    / "eegmmidb_fp1_fp2_60"
)

CHECKPOINT_20 = (
    PROJECT_ROOT
    / "checkpoints"
    / "pretrain20"
    / "best.pt"
)

CHECKPOINT_60 = (
    PROJECT_ROOT
    / "checkpoints"
    / "pretrain60"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "mask_ratio_robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def find_existing(base_dir, candidates):

    for name in candidates:

        path = base_dir / name

        if path.exists():
            return path

    raise FileNotFoundError(
        "找不到文件。\n"
        f"目录：{base_dir}\n"
        f"尝试过：{candidates}"
    )


X_PATH = find_existing(
    DATA_DIR,
    [
        "pretrain_60subjects_windows.npy",
        "pretrain60_windows.npy",
        "60subjects_windows.npy",
    ]
)

SPLIT_PATH = find_existing(
    DATA_DIR,
    [
        "pretrain_60subjects_split.npz",
        "pretrain60_split.npz",
        "60subjects_split.npz",
    ]
)

METADATA_PATH = None

for candidate in [
    DATA_DIR / "pretrain_60subjects_metadata.npy",
    DATA_DIR / "pretrain60_metadata.npy",
    DATA_DIR / "60subjects_metadata.npy",
]:

    if candidate.exists():
        METADATA_PATH = candidate
        break


# ============================================================
# 3. Model config
# ============================================================

NUM_CHANNELS = 2
TIME_POINTS = 640

PATCH_SIZE = 40
EMBED_DIM = 128

NUM_HEADS = 4
NUM_LAYERS = 4

FEEDFORWARD_DIM = 256
DROPOUT = 0.1

BATCH_SIZE = 64

MASK_RATIOS = [
    0.25,
    0.50,
    0.75,
]


# ============================================================
# 4. Output
# ============================================================

SUMMARY_PATH = (
    OUTPUT_DIR
    / "mask_ratio_robustness_summary.csv"
)

FIGURE_PATH = (
    OUTPUT_DIR
    / "mask_ratio_robustness.png"
)


# ============================================================
# 5. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 80)
print("Mask Ratio Robustness | 20 vs 60 Subjects")
print("=" * 80)

print(
    "\nDevice:",
    device
)


# ============================================================
# 6. Load same-test data
# ============================================================

X = np.load(
    X_PATH,
    mmap_mode="r"
)

split = np.load(
    SPLIT_PATH
)

if "test_indices" not in split.files:
    raise RuntimeError(
        "split.npz 中没有 test_indices"
    )

test_indices = np.asarray(
    split["test_indices"],
    dtype=np.int64
)

print(
    "\nDataset:",
    X_PATH
)

print(
    "Test windows:",
    len(test_indices)
)


if METADATA_PATH is not None:

    metadata = np.load(
        METADATA_PATH,
        mmap_mode="r"
    )

    test_subjects = np.unique(
        metadata[
            test_indices,
            0
        ]
    ).astype(int)

    print(
        "Test subjects:",
        test_subjects.tolist()
    )


# ============================================================
# 7. Dataset
# ============================================================

class SameTestDataset(
    Dataset
):

    def __init__(
        self,
        X,
        indices
    ):

        self.X = X

        self.indices = np.asarray(
            indices,
            dtype=np.int64
        )


    def __len__(self):

        return len(
            self.indices
        )


    def __getitem__(
        self,
        idx
    ):

        real_idx = self.indices[
            idx
        ]

        return torch.from_numpy(
            self.X[
                real_idx
            ].copy()
        ).float()


dataset = SameTestDataset(
    X,
    test_indices
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 8. Patchify
# ============================================================

def patchify(
    x
):

    B, C, T = x.shape

    if T % PATCH_SIZE != 0:

        raise RuntimeError(
            "TIME_POINTS 无法被 PATCH_SIZE 整除"
        )

    patches_per_channel = (
        T
        // PATCH_SIZE
    )

    return (
        x.reshape(
            B,
            C,
            patches_per_channel,
            PATCH_SIZE
        )
        .reshape(
            B,
            C * patches_per_channel,
            PATCH_SIZE
        )
    )


# ============================================================
# 9. Fixed mask generator
# ============================================================

def make_masked_batch(
    x,
    mask_ratio,
    generator
):

    target_patches = patchify(
        x
    )

    B, total_tokens, _ = (
        target_patches.shape
    )

    num_mask = int(
        round(
            total_tokens
            * mask_ratio
        )
    )

    num_mask = max(
        1,
        min(
            total_tokens - 1,
            num_mask
        )
    )

    random_values = torch.rand(
        B,
        total_tokens,
        generator=generator
    )

    sorted_indices = torch.argsort(
        random_values,
        dim=1
    )

    masked_indices = sorted_indices[
        :,
        :num_mask
    ]

    mask = torch.zeros(
        B,
        total_tokens,
        dtype=torch.bool
    )

    mask.scatter_(
        1,
        masked_indices,
        True
    )

    masked_patches = (
        target_patches.clone()
    )

    masked_patches[
        mask
    ] = 0.0

    patches_per_channel = (
        TIME_POINTS
        // PATCH_SIZE
    )

    masked_eeg = (
        masked_patches
        .reshape(
            B,
            NUM_CHANNELS,
            patches_per_channel,
            PATCH_SIZE
        )
        .reshape(
            B,
            NUM_CHANNELS,
            TIME_POINTS
        )
    )

    return (
        masked_eeg,
        target_patches,
        mask
    )


# ============================================================
# 10. Build / load model
# ============================================================

def build_model():

    return MaskedEEGTransformer(
        num_channels=NUM_CHANNELS,
        time_points=TIME_POINTS,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout=DROPOUT
    )


def unwrap_state_dict(
    checkpoint
):

    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            "Checkpoint 不是 dict"
        )

    for key in [
        "model_state_dict",
        "state_dict",
    ]:

        if (
            key in checkpoint
            and isinstance(
                checkpoint[key],
                dict
            )
        ):

            return checkpoint[key]

    if all(
        torch.is_tensor(value)
        for value
        in checkpoint.values()
    ):

        return checkpoint

    raise RuntimeError(
        "无法找到 model_state_dict / state_dict"
    )


def normalize_state_dict(
    state_dict,
    model
):

    model_keys = set(
        model.state_dict().keys()
    )

    candidates = [
        state_dict
    ]

    for prefix in [
        "module.",
        "model.",
        "backbone.",
    ]:

        stripped = {}

        for key, value in state_dict.items():

            if key.startswith(
                prefix
            ):

                new_key = key[
                    len(prefix):
                ]

            else:

                new_key = key

            stripped[
                new_key
            ] = value

        candidates.append(
            stripped
        )

    def overlap(
        candidate
    ):

        return len(
            model_keys.intersection(
                candidate.keys()
            )
        )

    best = max(
        candidates,
        key=overlap
    )

    if overlap(best) == 0:

        raise RuntimeError(
            "Checkpoint keys 无法匹配当前模型。"
        )

    return best


def load_model(
    path,
    name
):

    if not path.exists():

        raise FileNotFoundError(
            f"找不到 checkpoint：\n{path}"
        )

    model = build_model()

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    state_dict = unwrap_state_dict(
        checkpoint
    )

    state_dict = normalize_state_dict(
        state_dict,
        model
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False
        )
    )

    reconstruction_missing = [
        key
        for key in missing_keys
        if key.startswith(
            "reconstruction_head"
        )
    ]

    if reconstruction_missing:

        raise RuntimeError(
            f"{name} checkpoint 缺少 reconstruction_head。\n"
            "请使用 best.pt，而不是 best_encoder.pt。"
        )

    if missing_keys:

        print(
            f"\n{name} missing keys:",
            missing_keys
        )

    if unexpected_keys:

        print(
            f"\n{name} unexpected keys:",
            unexpected_keys
        )

    model = model.to(
        device
    )

    model.eval()

    return model


def get_predictions(
    output
):

    if isinstance(
        output,
        (tuple, list)
    ):

        return output[0]

    if isinstance(
        output,
        dict
    ):

        if "predictions" in output:
            return output[
                "predictions"
            ]

        if "reconstruction" in output:
            return output[
                "reconstruction"
            ]

    if torch.is_tensor(
        output
    ):

        return output

    raise RuntimeError(
        f"未知模型输出类型：{type(output)}"
    )


# ============================================================
# 11. Load models
# ============================================================

print("\nLoading models...")

model20 = load_model(
    CHECKPOINT_20,
    "20-subject"
)

model60 = load_model(
    CHECKPOINT_60,
    "60-subject"
)

print(
    "✓ Both models loaded"
)


# ============================================================
# 12. Evaluate one mask ratio
# ============================================================

def evaluate_ratio(
    mask_ratio
):

    # 每个 ratio 使用固定独立随机种子。
    # 同一 ratio 下两模型共用相同 mask。

    generator = torch.Generator()

    ratio_seed = (
        SEED
        + int(
            mask_ratio
            * 1000
        )
    )

    generator.manual_seed(
        ratio_seed
    )

    sse20 = 0.0
    sse60 = 0.0
    sse_zero = 0.0

    element_count = 0


    with torch.no_grad():

        for x in loader:

            (
                masked_eeg,
                target_patches,
                mask

            ) = make_masked_batch(

                x,

                mask_ratio,

                generator

            )

            masked_eeg = masked_eeg.to(
                device
            )

            target_patches = target_patches.to(
                device
            )

            mask = mask.to(
                device
            )


            pred20 = get_predictions(
                model20(
                    masked_eeg
                )
            )

            pred60 = get_predictions(
                model60(
                    masked_eeg
                )
            )


            error20 = (
                pred20[
                    mask
                ]
                - target_patches[
                    mask
                ]
            )

            error60 = (
                pred60[
                    mask
                ]
                - target_patches[
                    mask
                ]
            )

            zero_error = (
                target_patches[
                    mask
                ]
            )


            sse20 += float(
                torch.sum(
                    error20 ** 2
                ).item()
            )

            sse60 += float(
                torch.sum(
                    error60 ** 2
                ).item()
            )

            sse_zero += float(
                torch.sum(
                    zero_error ** 2
                ).item()
            )

            element_count += int(
                error20.numel()
            )


    mse20 = (
        sse20
        / element_count
    )

    mse60 = (
        sse60
        / element_count
    )

    mse_zero = (
        sse_zero
        / element_count
    )

    improvement = (

        (
            mse20
            - mse60
        )

        / mse20

        * 100.0

    )

    return {
        "mask_ratio":
            float(
                mask_ratio
            ),

        "20_subject_mse":
            float(
                mse20
            ),

        "60_subject_mse":
            float(
                mse60
            ),

        "zero_baseline_mse":
            float(
                mse_zero
            ),

        "relative_improvement_percent":
            float(
                improvement
            ),
    }


# ============================================================
# 13. Run robustness test
# ============================================================

records = []

print("\n")
print("=" * 80)
print("Robustness Evaluation")
print("=" * 80)

for ratio in MASK_RATIOS:

    result = evaluate_ratio(
        ratio
    )

    records.append(
        result
    )

    print(
        f"\nMask Ratio = {ratio:.0%}"
    )

    print(
        "20-subject MSE:",
        f"{result['20_subject_mse']:.6f}"
    )

    print(
        "60-subject MSE:",
        f"{result['60_subject_mse']:.6f}"
    )

    print(
        "Zero Baseline MSE:",
        f"{result['zero_baseline_mse']:.6f}"
    )

    print(
        "60 vs 20 improvement:",
        f"{result['relative_improvement_percent']:+.2f}%"
    )


# ============================================================
# 14. Save CSV
# ============================================================

with open(
    SUMMARY_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "mask_ratio",
            "20_subject_mse",
            "60_subject_mse",
            "zero_baseline_mse",
            "relative_improvement_percent",
        ]
    )

    writer.writeheader()

    writer.writerows(
        records
    )


# ============================================================
# 15. Plot
# ============================================================

ratios_percent = [
    record[
        "mask_ratio"
    ] * 100.0
    for record in records
]

mse20_values = [
    record[
        "20_subject_mse"
    ]
    for record in records
]

mse60_values = [
    record[
        "60_subject_mse"
    ]
    for record in records
]

zero_values = [
    record[
        "zero_baseline_mse"
    ]
    for record in records
]


plt.figure(
    figsize=(8.5, 6)
)

plt.plot(
    ratios_percent,
    mse20_values,
    marker="o",
    label="20-subject pretraining"
)

plt.plot(
    ratios_percent,
    mse60_values,
    marker="o",
    label="60-subject pretraining"
)

plt.plot(
    ratios_percent,
    zero_values,
    marker="o",
    linestyle="--",
    label="Zero baseline"
)

plt.xlabel(
    "Mask Ratio (%)"
)

plt.ylabel(
    "Masked Reconstruction MSE"
)

plt.title(
    "Mask Ratio Robustness on Same Unseen Subjects"
)

plt.xticks(
    ratios_percent
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_PATH,
    dpi=180
)

plt.close()


# ============================================================
# 16. Final
# ============================================================

print("\n")
print("=" * 80)
print("Mask Ratio Robustness 完成")
print("=" * 80)

print(
    "\nSummary:"
)

print(
    SUMMARY_PATH
)

print(
    "\nFigure:"
)

print(
    FIGURE_PATH
)

print(
    "\n建议：将 25% / 50% / 75% 三个点用于 PPT/报告，"
    "不要只强调单个最佳点，而要讨论随着遮挡比例增加，"
    "模型重建误差如何变化，以及 60-subject 是否持续优于 20-subject。"
)
