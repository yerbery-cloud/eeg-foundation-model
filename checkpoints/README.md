# Checkpoints

Model checkpoints are intentionally **not included in the public repository by default**.

The local project may contain checkpoints for:

```text
pilot/
pretrain20/
pretrain60/
linear_probe/
linear_probe60/
random_probe/
lora/
full_finetune/
eegnet/
cbramod/
personal_head/
recording_demo_adapter/
```

## Why they are excluded

- Some checkpoint files are relatively large.
- Personal-head and recording-demo adapters are derived from subject-specific calibration data.
- The public repository focuses on code, reproducibility, and de-identified results.

Common weight files such as `*.pt`, `*.pth`, and `*.ckpt` are ignored by Git.

Selected pretrained weights may be released later through GitHub Releases, Hugging Face, or Zenodo after checking third-party licenses.
