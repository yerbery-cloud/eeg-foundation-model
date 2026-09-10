# 现有项目怎么放进来

## 最稳妥的做法

**不要直接移动你当前正在工作的实验目录。**
先复制一份完整项目，作为 GitHub 开源副本，再整理。

## 1. 你截图里的 Python 文件

第一版建议把所有 `.py` 文件**原样复制到 `src/`**，先不要进一步拆目录，以免 import 路径失效。

包括：
- `01_download.py` 到 `37_loongbrain_calibrated_recording_demo.py`
- `eeg_dataset.py`
- `eeg_masking.py`
- `eeg_patch_embedding.py`
- `masked_eeg_model.py`

不要复制：
- `__pycache__/`
- 原始 EEG
- 私人 calibration CSV
- 大模型 checkpoint

## 2. requirements.txt

把你当前的 `requirements.txt` 放到仓库根目录：

```text
eeg-foundation-model/requirements.txt
```

## 3. figures/

只放少量最能说明项目的图：
- Masked Reconstruction 示意图
- 20 vs 60 Scaling
- 下游方法比较
- Mask Ratio Robustness
- 真机 Demo 结果

## 4. results/

只放小体积、去隐私的汇总结果，例如：
- `pretrain_scaling_summary.csv`
- `downstream_summary.csv`
- `mask_ratio_summary.csv`
- `demo_summary.csv`

## 5. data/

不要上传数据本体。只保留 `data/README.md`，告诉别人去哪里下载。

## 6. checkpoints/

第一版建议不直接提交 `.pt`。
以后可放到 GitHub Releases / Hugging Face / Zenodo。

## 7. docs/

可以放：
- 最终报告 PDF
- 最终汇报 PDF
- 真机 Demo 使用说明

发布前删掉个人路径和不适合公开的课程材料。
