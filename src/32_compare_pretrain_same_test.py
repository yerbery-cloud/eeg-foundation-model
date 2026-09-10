from pathlib import Path
import csv
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from masked_eeg_model import MaskedEEGTransformer


# ============================================================
# 0. 目的
# ============================================================
#
# 公平比较：
#
#   20-subject pretrained model
#   60-subject pretrained model
#
# 在完全相同的未见 EEGMMIDB Test Subjects 上，
# 使用完全相同的 EEG windows、mask、mask ratio，
# 比较 masked reconstruction MSE。
#
# 这里使用 60-subject 实验固定的 Test split：
# S055-S060
#
# 对两套模型而言，这些被试都没有进入预训练 Train。
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
    / "scaling_20_vs_60"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. 数据文件
#
# 兼容几种可能的命名方式
# ============================================================

def find_existing(base_dir, candidates):

    for name in candidates:

        path = (
            base_dir
            / name
        )

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


# metadata 仅用于验收，不是必须

METADATA_CANDIDATES = [

    DATA_DIR
    / "pretrain_60subjects_metadata.npy",

    DATA_DIR
    / "pretrain60_metadata.npy",

    DATA_DIR
    / "60subjects_metadata.npy",

]


METADATA_PATH = None

for path in METADATA_CANDIDATES:

    if path.exists():

        METADATA_PATH = path

        break


# ============================================================
# 4. 输出
# ============================================================

SUMMARY_PATH = (
    OUTPUT_DIR
    / "same_test_reconstruction_summary.csv"
)

FIGURE_PATH = (
    OUTPUT_DIR
    / "same_test_reconstruction.png"
)


# ============================================================
# 5. Model Config
#
# 和原 20 / 60-subject pretraining 保持一致
# ============================================================

NUM_CHANNELS = 2
TIME_POINTS = 640

PATCH_SIZE = 40
EMBED_DIM = 128

NUM_HEADS = 4
NUM_LAYERS = 4

FEEDFORWARD_DIM = 256
DROPOUT = 0.1

MASK_RATIO = 0.50

BATCH_SIZE = 64


# ============================================================
# 6. Device
# ============================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


print("=" * 78)

print("20 vs 60 Subjects | Same-Test Reconstruction")

print("=" * 78)

print(
    "\nDevice:",
    device
)


# ============================================================
# 7. 文件检查
# ============================================================

for path in [

    X_PATH,
    SPLIT_PATH,
    CHECKPOINT_20,
    CHECKPOINT_60,

]:

    if not path.exists():

        raise FileNotFoundError(
            f"找不到文件：\n{path}"
        )


print(
    "\nDataset:"
)

print(
    X_PATH
)

print(
    "\n20-subject checkpoint:"
)

print(
    CHECKPOINT_20
)

print(
    "\n60-subject checkpoint:"
)

print(
    CHECKPOINT_60
)


# ============================================================
# 8. 读取数据
#
# mmap_mode='r'：
# 不一次性把整个 60-subject 数据读入内存。
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

    split[
        "test_indices"
    ],

    dtype=np.int64

)


print(
    "\nFull X shape:",
    X.shape
)

print(
    "Test windows:",
    len(
        test_indices
    )
)


if X.shape[1:] != (
    NUM_CHANNELS,
    TIME_POINTS
):

    raise RuntimeError(

        f"数据 shape 异常："
        f"{X.shape}"

    )


# ============================================================
# 9. 如果 metadata 存在，确认 Test Subjects
# ============================================================

test_subjects = None


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

    ).astype(
        int
    )


    print(
        "Test subjects:",
        test_subjects.tolist()
    )


    # 当前 60-subject 方案预期：
    # S055-S060

    expected_subjects = np.arange(
        55,
        61
    )


    if not np.array_equal(

        test_subjects,
        expected_subjects

    ):

        print(
            "\n[WARNING]"
        )

        print(
            "当前 Test Subjects 并不是预期的 S055-S060。"
        )

        print(
            "脚本仍然会继续，并保证两套模型使用相同 test_indices。"
        )


# ============================================================
# 10. Dataset
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


    def __len__(
        self
    ):

        return len(
            self.indices
        )


    def __getitem__(
        self,
        idx
    ):

        real_idx = (
            self.indices[
                idx
            ]
        )


        # mmap 是 readonly，
        # copy 后再转 Tensor，避免 PyTorch warning。

        eeg = torch.from_numpy(

            self.X[
                real_idx
            ].copy()

        ).float()


        return eeg


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
# 11. Patchify
# ============================================================

def patchify(
    x,
    patch_size
):

    # x:
    # [B,C,T]

    B, C, T = x.shape


    if T % patch_size != 0:

        raise RuntimeError(
            "TIME_POINTS 无法被 PATCH_SIZE 整除"
        )


    num_patches = (
        T
        // patch_size
    )


    patches = x.reshape(

        B,
        C,
        num_patches,
        patch_size

    )


    patches = patches.reshape(

        B,
        C * num_patches,
        patch_size

    )


    return patches


# ============================================================
# 12. 固定 Mask
#
# 关键：
# 同一个 batch 只生成一次 mask，
# 然后同时交给 20-model 和 60-model。
#
# 因此两个模型看到：
#
# - 完全相同 EEG
# - 完全相同 masked patches
# - 完全相同 reconstruction targets
# ============================================================

def make_fixed_masked_batch(
    x,
    generator
):

    target_patches = patchify(

        x,

        PATCH_SIZE

    )


    B, total_tokens, _ = (
        target_patches.shape
    )


    num_mask = int(

        total_tokens
        * MASK_RATIO

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


    # [B,C,P,patch]
    num_patches_per_channel = (
        TIME_POINTS
        // PATCH_SIZE
    )


    masked_eeg = masked_patches.reshape(

        B,
        NUM_CHANNELS,
        num_patches_per_channel,
        PATCH_SIZE

    )


    masked_eeg = masked_eeg.reshape(

        B,
        NUM_CHANNELS,
        TIME_POINTS

    )


    return (

        masked_eeg,
        target_patches,
        mask

    )


# ============================================================
# 13. 创建模型
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


# ============================================================
# 14. Checkpoint state_dict 解包
# ============================================================

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

            return checkpoint[
                key
            ]


    # best.pt 也可能直接就是 state_dict

    if all(

        torch.is_tensor(value)

        for value
        in checkpoint.values()

    ):

        return checkpoint


    raise RuntimeError(

        "无法从 checkpoint 中找到 "
        "model_state_dict / state_dict"

    )


# ============================================================
# 15. 自动处理常见 prefix
# ============================================================

def normalize_state_dict(
    state_dict,
    model
):

    model_keys = set(
        model.state_dict().keys()
    )


    candidate_dicts = []


    candidate_dicts.append(
        state_dict
    )


    prefixes = [

        "module.",
        "model.",
        "backbone.",

    ]


    for prefix in prefixes:

        stripped = {}


        for key, value in (
            state_dict.items()
        ):

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


        candidate_dicts.append(
            stripped
        )


    def overlap(
        candidate
    ):

        return len(

            model_keys
            .intersection(
                candidate.keys()
            )

        )


    best = max(

        candidate_dicts,

        key=overlap

    )


    if overlap(best) == 0:

        raise RuntimeError(

            "Checkpoint keys 与当前模型完全无法匹配。\n"
            f"Checkpoint 示例 keys："
            f"{list(state_dict.keys())[:10]}"

        )


    return best


# ============================================================
# 16. 加载完整预训练模型
#
# 注意这里必须加载 best.pt，
# 不能使用 best_encoder.pt。
#
# 因为 reconstruction evaluation 需要：
# reconstruction_head。
# ============================================================

def load_pretrained_model(
    path,
    model_name
):

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


    # reconstruction head 如果没加载到，
    # 说明 checkpoint 不适合做重建评价。

    reconstruction_missing = [

        key

        for key in missing_keys

        if key.startswith(
            "reconstruction_head"
        )

    ]


    if len(
        reconstruction_missing
    ) > 0:

        raise RuntimeError(

            f"\n{model_name} checkpoint "
            "没有包含 reconstruction_head。\n\n"
            "当前必须使用：\n"
            f"{path.parent / 'best.pt'}\n"
            "而不能使用 best_encoder.pt。\n\n"
            f"Missing keys: {missing_keys}"

        )


    if len(
        missing_keys
    ) > 0:

        print(
            f"\n{model_name} Missing keys:"
        )

        print(
            missing_keys
        )


    if len(
        unexpected_keys
    ) > 0:

        print(
            f"\n{model_name} Unexpected keys:"
        )

        print(
            unexpected_keys
        )


    model = model.to(
        device
    )


    model.eval()


    return model


# ============================================================
# 17. 加载两套模型
# ============================================================

print("\n")

print("=" * 78)

print("Loading Models")

print("=" * 78)


model20 = load_pretrained_model(

    CHECKPOINT_20,

    "20-subject"

)


print(
    "\n✓ 20-subject model loaded"
)


model60 = load_pretrained_model(

    CHECKPOINT_60,

    "60-subject"

)


print(
    "✓ 60-subject model loaded"
)


# ============================================================
# 18. 从模型输出中拿 predictions
# ============================================================

def get_predictions(
    output
):

    # 当前 MaskedEEGTransformer：
    # return predictions, representations

    if isinstance(

        output,

        (
            tuple,
            list

        )

    ):

        predictions = output[
            0
        ]


    elif isinstance(
        output,
        dict
    ):

        if "predictions" in output:

            predictions = output[
                "predictions"
            ]

        elif "reconstruction" in output:

            predictions = output[
                "reconstruction"
            ]

        else:

            raise RuntimeError(
                f"无法解析模型 dict 输出：{output.keys()}"
            )


    elif torch.is_tensor(
        output
    ):

        predictions = output


    else:

        raise RuntimeError(

            f"未知模型输出类型："
            f"{type(output)}"

        )


    return predictions


# ============================================================
# 19. 公平评价
#
# 使用同一个 generator，
# 每个 batch 只生成一次 mask。
#
# 两个模型使用相同 masked_eeg。
# ============================================================

generator = torch.Generator()

generator.manual_seed(
    SEED
)


sum_squared_error_20 = 0.0
sum_squared_error_60 = 0.0

masked_element_count = 0


print("\n")

print("=" * 78)

print("Same-Test Evaluation")

print("=" * 78)


with torch.no_grad():


    for batch_index, x in enumerate(
        loader,
        start=1
    ):


        # ----------------------------------------------------
        # 固定相同 mask
        # ----------------------------------------------------

        (
            masked_eeg,
            target_patches,
            mask

        ) = make_fixed_masked_batch(

            x,

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


        # ----------------------------------------------------
        # 20-subject model
        # ----------------------------------------------------

        pred20 = get_predictions(

            model20(
                masked_eeg
            )

        )


        # ----------------------------------------------------
        # 60-subject model
        # ----------------------------------------------------

        pred60 = get_predictions(

            model60(
                masked_eeg
            )

        )


        if (
            pred20.shape
            != target_patches.shape
        ):

            raise RuntimeError(

                f"20-model prediction shape 异常："
                f"{pred20.shape} "
                f"vs target "
                f"{target_patches.shape}"

            )


        if (
            pred60.shape
            != target_patches.shape
        ):

            raise RuntimeError(

                f"60-model prediction shape 异常："
                f"{pred60.shape} "
                f"vs target "
                f"{target_patches.shape}"

            )


        # ----------------------------------------------------
        # 只在 masked tokens 上评价
        #
        # pred[mask]:
        # [masked_tokens, PATCH_SIZE]
        # ----------------------------------------------------

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


        sum_squared_error_20 += float(

            torch.sum(
                error20 ** 2
            ).item()

        )


        sum_squared_error_60 += float(

            torch.sum(
                error60 ** 2
            ).item()

        )


        masked_element_count += int(

            error20.numel()

        )


        if (
            batch_index % 10 == 0
            or batch_index == len(loader)
        ):

            running_mse20 = (

                sum_squared_error_20
                / masked_element_count

            )


            running_mse60 = (

                sum_squared_error_60
                / masked_element_count

            )


            print(

                f"Batch "
                f"{batch_index:03d}/"
                f"{len(loader):03d} | "

                f"20-subj MSE "
                f"{running_mse20:.6f} | "

                f"60-subj MSE "
                f"{running_mse60:.6f}"

            )


# ============================================================
# 20. 最终指标
# ============================================================

mse20 = (

    sum_squared_error_20
    / masked_element_count

)


mse60 = (

    sum_squared_error_60
    / masked_element_count

)


absolute_change = (

    mse60
    - mse20

)


relative_improvement = (

    (
        mse20
        - mse60
    )

    / mse20

    * 100.0

)


print("\n")

print("=" * 78)

print("FINAL SAME-TEST RESULT")

print("=" * 78)


print(
    "\nTest windows:",
    len(
        test_indices
    )
)


if test_subjects is not None:

    print(
        "Test subjects:",
        test_subjects.tolist()
    )


print(
    "\n20-subject MSE:",
    f"{mse20:.6f}"
)


print(
    "60-subject MSE:",
    f"{mse60:.6f}"
)


print(
    "\n60 - 20 MSE:",
    f"{absolute_change:+.6f}"
)


print(
    "Relative improvement:",
    f"{relative_improvement:+.2f}%"
)


if mse60 < mse20:

    conclusion = (
        "60-subject pretraining achieved lower "
        "same-test reconstruction MSE."
    )

else:

    conclusion = (
        "60-subject pretraining did not reduce "
        "same-test reconstruction MSE."
    )


print(
    "\nConclusion:"
)

print(
    conclusion
)


# ============================================================
# 21. 保存 CSV
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
            "20_subjects",
            "60_subjects"
        ]
    )


    writer.writerow(
        [
            "same_test_reconstruction_mse",
            mse20,
            mse60
        ]
    )


    writer.writerow(
        [
            "test_windows",
            len(test_indices),
            len(test_indices)
        ]
    )


    writer.writerow(
        [
            "mask_ratio",
            MASK_RATIO,
            MASK_RATIO
        ]
    )


    writer.writerow(
        [
            "relative_improvement_percent",
            "",
            relative_improvement
        ]
    )


# ============================================================
# 22. 画图
# ============================================================

labels = [
    "20-subject",
    "60-subject"
]

values = [
    mse20,
    mse60
]


plt.figure(
    figsize=(8, 6)
)


bars = plt.bar(

    labels,

    values

)


plt.ylabel(
    "Masked Reconstruction MSE"
)


plt.title(
    "Same-Test Reconstruction: "
    "20 vs 60 Subjects"
)


plt.grid(
    axis="y",
    alpha=0.25
)


top = max(
    values
)


plt.ylim(
    0,
    top * 1.25
)


for bar, value in zip(
    bars,
    values
):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        value
        + top * 0.03,

        f"{value:.4f}",

        ha="center",

        va="bottom",

        fontsize=12

    )


plt.tight_layout()


plt.savefig(

    FIGURE_PATH,

    dpi=180

)


plt.close()


# ============================================================
# 23. Final
# ============================================================

print("\n")

print("=" * 78)

print("Same-Test Scaling Evaluation 完成")

print("=" * 78)


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
    "\nNext:"
)

print(
    "python src/31_compare_pretraining_scale.py"
)
