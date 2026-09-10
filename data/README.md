# Data

Raw and processed EEG datasets are **not included** in this repository.

The project uses:

- **EEGMMIDB / PhysioNet** for self-supervised pretraining
- **EEGMAT** for REST vs. mental-arithmetic downstream classification
- **LoongBrain Fp1/Fp2 recordings** for private real-device experiments

## Recommended local layout

```text
data/
├── raw/
│   ├── eegmmidb/
│   └── eegmat/
└── processed/
    ├── pretrain20/
    ├── pretrain60/
    └── eegmat/
```

Public datasets should be downloaded from their official sources and used under their original licenses.

Raw LoongBrain EEG and subject-specific calibration data are kept private and are not intended for public release.
