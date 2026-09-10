# Source Code

This folder contains the Python source code for the full workflow, from data preparation to real-time LoongBrain deployment.

## Main workflow

### Data preparation and checks
`01_download.py` → `09_model_check.py`

### Self-supervised pretraining
`10_pretrain_pilot.py`, `11_reconstruction_check.py`, the 20/60-subject preprocessing, merge/split, pretraining, and evaluation scripts.

### Downstream experiments
EEGMAT preprocessing plus Linear Probe, Random Probe, LoRA, Full Fine-tuning, EEGNet, CBraMod, and comparison scripts.

### Scaling and robustness
`31_compare_pretraining_scale.py`, `31_compare_pretraining_scale_final.py`, `32_compare_pretrain_same_test.py`, `36_mask_ratio_robustness.py`

### Real-time LoongBrain deployment
`28_realtime_pipeline_test.py`, `29_lsl_stream_check.py`, LoongBrain real-time/calibration scripts, `34_train_personal_head.py`, `35_loongbrain_final_test.py`, and `37_loongbrain_calibrated_recording_demo.py`

### Shared modules
`eeg_dataset.py`, `eeg_masking.py`, `eeg_patch_embedding.py`, `masked_eeg_model.py`

## Notes

The numbered scripts are intentionally kept close to the original experimental order for reproducibility and to avoid breaking local imports. A future refactor may reorganize them into `data/`, `models/`, `pretrain/`, `downstream/`, `evaluation/`, and `realtime/`.
