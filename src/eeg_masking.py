import torch


# ============================================================
# EEG → Patch
# ============================================================

def patchify_eeg(
    x,
    patch_size=40
):
    """
    将连续 EEG 切成 Patch。

    输入：
        x:
            [B, C, T]

        例如：
            [32, 2, 640]

    输出：
        patches:
            [B, C, P, patch_size]

        例如：
            [32, 2, 16, 40]
    """

    if x.ndim != 3:
        raise ValueError(
            "输入 EEG 必须是 [B, C, T]"
        )


    batch_size, channels, time_points = (
        x.shape
    )


    if time_points % patch_size != 0:

        raise ValueError(
            "EEG 时间点数量必须能够被 patch_size 整除"
        )


    num_patches = (
        time_points // patch_size
    )


    patches = x.reshape(
        batch_size,
        channels,
        num_patches,
        patch_size
    )


    return patches


# ============================================================
# Patch → 连续 EEG
# ============================================================

def unpatchify_eeg(
    patches
):
    """
    将 Patch 重新拼回连续 EEG。

    输入：
        [B, C, P, patch_size]

    输出：
        [B, C, T]
    """

    if patches.ndim != 4:

        raise ValueError(
            "输入 Patch 必须是 [B, C, P, patch_size]"
        )


    batch_size = patches.shape[0]

    channels = patches.shape[1]

    num_patches = patches.shape[2]

    patch_size = patches.shape[3]


    eeg = patches.reshape(
        batch_size,
        channels,
        num_patches * patch_size
    )


    return eeg


# ============================================================
# 随机 Patch Mask
# ============================================================

def random_patch_mask(
    x,
    patch_size=40,
    mask_ratio=0.5
):
    """
    对 EEG Patch 进行随机遮挡。

    输入：
        x:
            [B, C, T]

        patch_size:
            一个 Patch 包含多少采样点

        mask_ratio:
            Mask 比例
            例如 0.5 = 50%


    返回
    ----
    masked_eeg:
        Mask 后重新拼接的 EEG
        [B, C, T]


    target_patches:
        原始 Patch
        [B, N, patch_size]

        N = C × 每通道 Patch 数


    mask:
        Bool Tensor

        [B, N]

        True：
            这个 Patch 被 Mask

        False：
            这个 Patch 保留
    """


    if not 0.0 < mask_ratio < 1.0:

        raise ValueError(
            "mask_ratio 必须位于 0 和 1 之间"
        )


    # ========================================================
    # 1. EEG → Patch
    # ========================================================

    patches = patchify_eeg(
        x,
        patch_size=patch_size
    )


    # 当前：
    #
    # [B, C, P, patch_size]

    batch_size = patches.shape[0]

    channels = patches.shape[1]

    num_patches_per_channel = (
        patches.shape[2]
    )


    total_tokens = (
        channels
        * num_patches_per_channel
    )


    # ========================================================
    # 2. 合并 Channel 和 Patch 维度
    # ========================================================

    flat_patches = patches.reshape(
        batch_size,
        total_tokens,
        patch_size
    )


    # 例如：
    #
    # [32, 32, 40]


    # 保存原始 Patch
    # 以后作为 Reconstruction Target

    target_patches = (
        flat_patches.clone()
    )


    # ========================================================
    # 3. 决定要 Mask 多少 Token
    # ========================================================

    num_mask = int(
        total_tokens
        * mask_ratio
    )


    # 当前：
    #
    # 32 × 0.5
    # =
    # 16


    # ========================================================
    # 4. 为每个 Sample 生成随机顺序
    # ========================================================

    random_values = torch.rand(
        batch_size,
        total_tokens,
        device=x.device
    )


    random_indices = torch.argsort(
        random_values,
        dim=1
    )


    # ========================================================
    # 5. 取前 num_mask 个作为 Mask
    # ========================================================

    mask_indices = random_indices[
        :,
        :num_mask
    ]


    # ========================================================
    # 6. 创建 Bool Mask
    # ========================================================

    mask = torch.zeros(
        batch_size,
        total_tokens,
        dtype=torch.bool,
        device=x.device
    )


    mask.scatter_(
        dim=1,
        index=mask_indices,
        value=True
    )


    # ========================================================
    # 7. 将被 Mask 的 Patch 设置为 0
    # ========================================================

    masked_flat_patches = (
        flat_patches.clone()
    )


    masked_flat_patches[
        mask
    ] = 0.0


    # ========================================================
    # 8. 重新恢复 [B,C,P,patch_size]
    # ========================================================

    masked_patches = (
        masked_flat_patches.reshape(
            batch_size,
            channels,
            num_patches_per_channel,
            patch_size
        )
    )


    # ========================================================
    # 9. Patch → EEG
    # ========================================================

    masked_eeg = unpatchify_eeg(
        masked_patches
    )


    return (
        masked_eeg,
        target_patches,
        mask
    )