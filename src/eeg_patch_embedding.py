import torch
import torch.nn as nn


# ============================================================
# EEG Patch Embedding
# ============================================================

class EEGPatchEmbedding(nn.Module):

    def __init__(
        self,
        num_channels=2,
        time_points=640,
        patch_size=40,
        embed_dim=128
    ):
        """
        EEG Patch Embedding

        输入：
            [B, C, T]

        例如：
            [32, 2, 640]

        输出：
            [B, N, D]

        例如：
            [32, 32, 128]


        参数解释
        --------
        num_channels:
            EEG 通道数
            当前为 Fp1 + Fp2 = 2

        time_points:
            一个 EEG Window 的时间点数量
            当前 4 秒 × 160 Hz = 640

        patch_size:
            一个 Patch 包含多少时间点
            当前为 40

        embed_dim:
            每个 EEG Token 最终变成多少维向量
            当前为 128
        """

        super().__init__()


        # ====================================================
        # 1. 保存基本参数
        # ====================================================

        self.num_channels = num_channels

        self.time_points = time_points

        self.patch_size = patch_size

        self.embed_dim = embed_dim


        # ====================================================
        # 2. 检查能否整除
        # ====================================================

        if time_points % patch_size != 0:

            raise ValueError(
                "time_points 必须能够被 patch_size 整除"
            )


        # ====================================================
        # 3. 每个 Channel 有多少 Patch
        # ====================================================

        self.num_patches_per_channel = (
            time_points // patch_size
        )


        # 640 / 40 = 16


        # ====================================================
        # 4. 总 Token 数量
        # ====================================================

        self.num_tokens = (
            num_channels
            * self.num_patches_per_channel
        )


        # 2 × 16 = 32


        # ====================================================
        # 5. Patch → Embedding
        # ====================================================

        # 一个 Patch：
        #
        # [40]
        #
        # 通过 Linear：
        #
        # [40]
        # ↓
        # [128]

        self.projection = nn.Linear(
            patch_size,
            embed_dim
        )


        # ====================================================
        # 6. Channel Embedding
        # ====================================================

        # Transformer 本身不知道：
        #
        # 这个 Token 来自 Fp1
        # 还是来自 Fp2
        #
        # 所以给不同 Channel 一个可学习的 Embedding

        self.channel_embedding = nn.Embedding(
            num_channels,
            embed_dim
        )


        # ====================================================
        # 7. Position Embedding
        # ====================================================

        # Transformer 本身也不知道：
        #
        # P1 在 P2 前面
        # P2 在 P3 前面
        #
        # 所以加入时间位置 Embedding

        self.position_embedding = nn.Embedding(
            self.num_patches_per_channel,
            embed_dim
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x
    ):

        """
        输入：

        x:
            [B, C, T]

        输出：

        tokens:
            [B, C * P, D]
        """


        # ====================================================
        # 1. 检查输入维度
        # ====================================================

        if x.ndim != 3:

            raise ValueError(
                "输入 EEG 必须是 [B, C, T]"
            )


        batch_size, channels, time_points = (
            x.shape
        )


        if channels != self.num_channels:

            raise ValueError(
                f"预期 {self.num_channels} 个通道，"
                f"实际得到 {channels}"
            )


        if time_points != self.time_points:

            raise ValueError(
                f"预期 {self.time_points} 个时间点，"
                f"实际得到 {time_points}"
            )


        # ====================================================
        # 2. EEG 切 Patch
        # ====================================================

        # 原来：
        #
        # [B, C, 640]
        #
        # reshape：
        #
        # [B, C, 16, 40]

        patches = x.reshape(
            batch_size,
            self.num_channels,
            self.num_patches_per_channel,
            self.patch_size
        )


        # ====================================================
        # 3. Patch → Embedding
        # ====================================================

        # Linear 自动作用在最后一个维度
        #
        # [B, C, 16, 40]
        #
        # ↓
        #
        # [B, C, 16, 128]

        tokens = self.projection(
            patches
        )


        # ====================================================
        # 4. 创建 Channel ID
        # ====================================================

        # 例如：
        #
        # Channel 0 = Fp1
        # Channel 1 = Fp2

        channel_ids = torch.arange(
            self.num_channels,
            device=x.device
        )


        # shape：
        # [2]

        channel_emb = self.channel_embedding(
            channel_ids
        )


        # shape：
        # [2, 128]


        # 为了能和 tokens 相加：
        #
        # [1, 2, 1, 128]

        channel_emb = channel_emb[
            None,
            :,
            None,
            :
        ]


        # ====================================================
        # 5. 创建 Position ID
        # ====================================================

        position_ids = torch.arange(
            self.num_patches_per_channel,
            device=x.device
        )


        # 0,1,2,...,15


        position_emb = self.position_embedding(
            position_ids
        )


        # shape：
        # [16, 128]


        # 改成：
        #
        # [1, 1, 16, 128]

        position_emb = position_emb[
            None,
            None,
            :,
            :
        ]


        # ====================================================
        # 6. 三种信息相加
        # ====================================================

        tokens = (
            tokens
            + channel_emb
            + position_emb
        )


        # 当前：
        #
        # [B, 2, 16, 128]


        # ====================================================
        # 7. Channel + Patch 合并为 Token 维
        # ====================================================

        tokens = tokens.reshape(
            batch_size,
            self.num_tokens,
            self.embed_dim
        )


        # 最终：
        #
        # [B, 32, 128]


        return tokens