import torch
import torch.nn as nn

from eeg_patch_embedding import EEGPatchEmbedding


# ============================================================
# Masked EEG Transformer
# ============================================================

class MaskedEEGTransformer(nn.Module):

    def __init__(
        self,
        num_channels=2,
        time_points=640,
        patch_size=40,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        feedforward_dim=256,
        dropout=0.1
    ):
        """
        Masked EEG Transformer

        输入：
            Mask 后的 EEG

            shape:
            [B, C, T]

            例如：
            [32, 2, 640]


        输出：
            每个 Patch 的重建结果

            shape:
            [B, N, patch_size]

            例如：
            [32, 32, 40]
        """

        super().__init__()


        # ====================================================
        # 1. 保存参数
        # ====================================================

        self.num_channels = num_channels

        self.time_points = time_points

        self.patch_size = patch_size

        self.embed_dim = embed_dim


        # ====================================================
        # 2. EEG Patch Embedding
        # ====================================================

        self.patch_embedding = EEGPatchEmbedding(
            num_channels=num_channels,
            time_points=time_points,
            patch_size=patch_size,
            embed_dim=embed_dim
        )


        # ====================================================
        # 3. Transformer Encoder Layer
        # ====================================================

        encoder_layer = nn.TransformerEncoderLayer(

            # 每个 Token 的维度
            d_model=embed_dim,

            # Multi-head Attention 的 head 数量
            nhead=num_heads,

            # Transformer 中间 FFN 的维度
            dim_feedforward=feedforward_dim,

            # Dropout
            dropout=dropout,

            # 激活函数
            activation="gelu",

            # 输入格式：
            # [Batch, Sequence, Embedding]
            batch_first=True,

            # 先 LayerNorm 再 Attention/FFN
            norm_first=True
        )


        # ====================================================
        # 4. 堆叠多个 Transformer Layer
        # ====================================================

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


        # ====================================================
        # 5. 最后的 LayerNorm
        # ====================================================

        self.norm = nn.LayerNorm(
            embed_dim
        )


        # ====================================================
        # 6. Reconstruction Head
        # ====================================================

        # Transformer 输出：
        #
        # 每个 Token = 128 维
        #
        # 但是原始 EEG Patch：
        #
        # 每个 Patch = 40 个采样点
        #
        # 所以：
        #
        # 128
        # ↓
        # Linear
        # ↓
        # 40

        self.reconstruction_head = nn.Linear(
            embed_dim,
            patch_size
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        masked_eeg
    ):

        """
        输入：

        masked_eeg:
            [B, 2, 640]


        输出：

        predictions:
            [B, 32, 40]


        representations:
            [B, 32, 128]
        """


        # ====================================================
        # Step 1：Patch Embedding
        # ====================================================

        tokens = self.patch_embedding(
            masked_eeg
        )


        # shape：
        #
        # [B, 32, 128]


        # ====================================================
        # Step 2：Transformer Encoder
        # ====================================================

        representations = self.encoder(
            tokens
        )


        # shape：
        #
        # [B, 32, 128]


        # ====================================================
        # Step 3：LayerNorm
        # ====================================================

        representations = self.norm(
            representations
        )


        # ====================================================
        # Step 4：重建 EEG Patch
        # ====================================================

        predictions = self.reconstruction_head(
            representations
        )


        # shape：
        #
        # [B, 32, 40]


        return (
            predictions,
            representations
        )


# ============================================================
# Masked Reconstruction Loss
# ============================================================

def masked_reconstruction_loss(
    predictions,
    targets,
    mask
):
    """
    只对被 Mask 的 EEG Patch 计算 MSE。


    predictions:
        [B, N, patch_size]


    targets:
        [B, N, patch_size]


    mask:
        [B, N]

        True:
            被 Mask

        False:
            未 Mask
    """


    # ========================================================
    # 1. Shape 检查
    # ========================================================

    if predictions.shape != targets.shape:

        raise ValueError(
            "Prediction 和 Target shape 不一致："
            f"{predictions.shape} vs {targets.shape}"
        )


    if mask.shape != predictions.shape[:2]:

        raise ValueError(
            "Mask shape 与 Token shape 不匹配"
        )


    # ========================================================
    # 2. 只取被 Mask 的 Patch
    # ========================================================

    masked_predictions = predictions[
        mask
    ]

    masked_targets = targets[
        mask
    ]


    # 原来：
    #
    # predictions
    # [32, 32, 40]
    #
    # Mask 50%
    #
    # ↓
    #
    # [32 × 16, 40]
    #
    # =
    #
    # [512, 40]


    # ========================================================
    # 3. MSE Loss
    # ========================================================

    loss = nn.functional.mse_loss(
        masked_predictions,
        masked_targets
    )


    return loss