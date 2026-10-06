# Spatial-MTA Swin UNETR for brain metastasis segmentation

3D Swin UNETR with Spatial Multi-Token Attention (MTA), deep supervision and an auxiliary vessel head
([arch/hybrid_swin_architecture.py](arch/hybrid_swin_architecture.py)), trained on the public
**Pretreat-MetsToBrain-Masks** dataset (200 patients, T1 / T1c / T2 / FLAIR, 1 mm isotropic, skull-stripped).

## Results (30 held-out test patients, 15 %)

| Metric | Value |
|---|---|
| Dice (mean / median / pooled) | 0.691 / 0.761 / 0.779 |
| Patient detection rate | 93.3 % |
| Lesion-wise sensitivity | 65.8 % (75 / 114 lesions) |
| Lesion-wise precision | 57.3 % (~1.9 false-positive lesions / patient) |
| Voxel ROC-AUC / average precision | 0.969 / 0.804 |
| Voxel accuracy | 99.69 % (dominated by background; not informative) |

Single seed, no baseline comparison yet, so these numbers say nothing about whether MTA helps.
Small lesions (< ~0.2 mL) are the main failure mode. Full numbers: [runs/main/results.json](runs/main/results.json),
per-patient: [runs/main/test_per_patient.json](runs/main/test_per_patient.json).
Figures (PNG + PDF): [runs/main/figures](runs/main/figures).

## Setup

- **Split:** patient-level, seed 42 → 153 train / 17 validation (checkpoint selection) / 30 test ([split.json](runs/main/split.json)).
  The test set is never used for model selection.
- **Target:** enhancing tumour + necrosis (BraTS labels 3 and 1); edema is excluded.
- **Vessel head:** the dataset has no vessel annotations, so the auxiliary head is trained on **Frangi-vesselness
  pseudo-labels** computed from T1c (outside lesions) in [prep.py](prep.py).
- **Training:** 96³ patches (70 % centred on lesions), batch 2, AdamW, cosine LR, bf16, 100 epochs × 100 iterations,
  loss = Dice+BCE (main) + 0.5 × deep-supervision + 0.3 × vessel. Inference: sliding window, 50 % overlap, threshold 0.5.
- [fast_mta.py](fast_mta.py) replaces `SpatialKQConv3D.forward` with an exactly equivalent pair of matmuls
  (the depth-wise 3D conv on tiny windows is slow in cuDNN). Outputs and gradients match the original to fp64 precision.

## Reproduce

```bash
pip install torch nibabel scipy scikit-learn matplotlib tqdm
# 1. download the dataset from TCIA (needs IBM Aspera) and place it under data/raw, then edit RAW in prep.py
python prep.py                  # normalise, crop, vessel pseudo-labels -> data/proc
python train.py                 # -> runs/main (best.pt chosen on validation Dice)
python evaluate.py              # test-set metrics
python make_figures.py          # paper figures
```

## Data citation (required by TCIA, CC BY 4.0)

Ramakrishnan, D. et al. (2023). *A Large Open Access Dataset of Brain Metastasis 3D Segmentations on MRI with
Clinical and Imaging Feature Information (Version 1)* [dataset]. The Cancer Imaging Archive.
https://doi.org/10.7937/6be1-r748 — see also Ramakrishnan et al., Scientific Data 2024,
https://doi.org/10.1038/s41597-024-03021-9
