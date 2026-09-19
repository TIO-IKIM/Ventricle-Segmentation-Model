# Ventricle Segmentation Model

Automatic segmentation of the **ventricular system** and **brain stem** in brain MRI (T1 and T2), based on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet). Accepts a DICOM series or NIfTI volumes and returns either a DICOM RTSTRUCT or a NIfTI label map.

> **Research use only. Not a medical device and not intended for clinical decision making.**

---

## Citation

If you use this code or the trained models, please cite:

tbd

Please also cite [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) since the model is based on it.

---

## License

 - Code: Apache License 2.0, modified by the Commons Clause and attribution terms in the License file (see [LICENSE](LICENSE)). \
 - Model weights: Creative Commons Attribution Non Commercial Share Alike 4.0 (CC BY-NC-SA 4.0) (see [LICENSE](LICENSE)). 

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (the script asserts `torch.cuda.is_available()`; CPU is not supported)
- Trained model weights (see below)


## Model weights

Weights are **not** included in this repository. Download them from `<WEIGHTS URL / Zenodo DOI>` and unpack so the tree looks like this:

```
nnUNet_results/
├── Dataset501_VenSeg3DT1/
│   └── nnUNetTrainer__nnUNetPlans__2d/
│       ├── fold_0/checkpoint_best.pth
│       ├── ...
│       ├── dataset.json
│       └── plans.json
└── Dataset502_VenSeg3DT2/
    └── nnUNetTrainer__nnUNetPlans__2d/
        └── ...
```

`nnUNet_results` must sit next to `run_inference.py`; the script sets the environment variable itself.

---

## Quick start

```bash
python run_inference.py \
  --input_folder  /path/to/patient_001 \
  --output_folder /path/to/output \
  --mr_sequence   T1
```

| Argument | Description |
| --- | --- |
| `--input_folder` | Folder with either the DICOM slices of **one** series, or NIfTI volumes |
| `--output_folder` | Folder where predictions are written (must exist) |
| `--mr_sequence` | `T1` (Dataset501) or `T2` (Dataset502) |

### Input and output behaviour

The format is decided from the **first file** in the input folder, so do not mix DICOM and NIfTI in one folder.

- **DICOM in** (`*.dcm`): the series is converted to NIfTI in a temporary folder, segmented, and written back as an RTSTRUCT at `output_folder/prediction_<folder_name>.dcm` with the ROIs `Ventricle` and `Brain stem`.
- **NIfTI in** (`*.nii.gz`): all volumes in the folder are segmented and label maps are written to `output_folder`. Files must follow the nnU-Net naming convention with a modality suffix, i.e. `case_0000.nii.gz`.

Label values in the NIfTI output: `1 = ventricle`, `2 = brain stem`, `0 = background`.

---

## Repository layout

```
.
├── run_inference.py     # CLI entry point: preprocessing, inference, RTSTRUCT export
├── requirements.txt
├── nnUNet_results/      # trained models (downloaded separately, git-ignored)
├── LICENSE
└── README.md
```

## Training

Models were trained with standard nnU-Net v2 ([nnU-Net](https://github.com/MIC-DKFZ/nnUNet))  in the `2d` configuration (`nnUNetTrainer__nnUNetPlans__2d`), using `checkpoint_best.pth` for inference. 

## Known limitations

- GPU is mandatory; there is no CPU fallback.
- The DICOM branch handles one series per input folder.
- Check orientation; verify overlays in your viewer before downstream use.

## Contact
tbd

## Acknowledgements

Built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [rt-utils](https://github.com/qurit/rt-utils).
