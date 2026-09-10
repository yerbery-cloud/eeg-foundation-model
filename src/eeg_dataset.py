import numpy as np
import torch

from torch.utils.data import Dataset


# ============================================================
# EEG 自监督预训练 Dataset
# ============================================================

class EEGPretrainDataset(Dataset):

    def __init__(
        self,
        data,
        metadata,
        indices
    ):
        """
        参数
        ----
        data:
            EEG NumPy 数组

            shape:
            [N, 2, 640]


        metadata:
            Metadata NumPy 数组

            每行：
            [subject, run, window_index]


        indices:
            当前 Dataset 使用哪些样本编号
        """

        self.data = data

        self.metadata = metadata

        self.indices = np.asarray(
            indices,
            dtype=np.int64
        )


    # ========================================================
    # Dataset 中有多少个样本
    # ========================================================

    def __len__(self):

        return len(
            self.indices
        )


    # ========================================================
    # 获取第 index 个样本
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        # ----------------------------------------------------
        # Dataset 内部编号
        # →
        # 原始 EEG 数据编号
        # ----------------------------------------------------

        real_index = self.indices[
            index
        ]


        # ----------------------------------------------------
        # 获取 EEG
        # ----------------------------------------------------

        eeg = self.data[
            real_index
        ]


        # ----------------------------------------------------
        # NumPy → PyTorch Tensor
        # ----------------------------------------------------

        eeg = torch.from_numpy(
            eeg
        ).float()


        # ----------------------------------------------------
        # 获取 metadata
        # ----------------------------------------------------

        meta = self.metadata[
            real_index
        ]


        meta = torch.from_numpy(
            meta
        ).long()


        return eeg, meta


# ============================================================
# 根据 Subject 划分 Dataset
# ============================================================

def split_by_subject(
    metadata,
    train_subjects,
    val_subjects,
    test_subjects
):

    """
    根据 Subject ID 划分样本。

    返回：
        train_indices
        val_indices
        test_indices
    """


    subject_ids = metadata[
        :,
        0
    ]


    # ========================================================
    # Train
    # ========================================================

    train_mask = np.isin(
        subject_ids,
        train_subjects
    )

    train_indices = np.where(
        train_mask
    )[0]


    # ========================================================
    # Validation
    # ========================================================

    val_mask = np.isin(
        subject_ids,
        val_subjects
    )

    val_indices = np.where(
        val_mask
    )[0]


    # ========================================================
    # Test
    # ========================================================

    test_mask = np.isin(
        subject_ids,
        test_subjects
    )

    test_indices = np.where(
        test_mask
    )[0]


    return (
        train_indices,
        val_indices,
        test_indices
    )