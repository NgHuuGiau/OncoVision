# MONAI Model Zoo - Downloaded 3D Segmentation Models

## Requested Models

### 1. Prostate MRI Segmentation
- **Model**: prostate_mri_anatomy (MONAI Bundle)
- **File**: models/pretrained/cancers/prostate/volume.pt (152 MB)
- **Config**: models/pretrained/cancers/prostate/volume.json
- **Modality**: MRI
- **Input**: 1x96x96x96 patches
- **Output**: 3 channels (background, central_gland, peripheral_zone)
- **Dice**: Central gland 0.88, Peripheral zone 0.75
- **Source**: Prostate158 dataset (10.5281/zenodo.6481141)

### 2. Cervical/Uterus Segmentation
- **Model**: vista3d (MONAI Bundle)
- **File**: models/pretrained/cancers/cervical/volume.pt (872 MB)
- **Config**: models/pretrained/cancers/cervical/volume.json
- **Modality**: CT
- **Input**: 1x128x128x128 patches
- **Output**: 132 channels covering whole body
- **Cervical relevant**: Channel 16 (prostate_or_uterus), Channel 118 (prostate)
- **Source**: MONAI Consortium - VISTA3D

### 3. Stomach/Gastric Segmentation
- **Model**: multi_organ_segmentation (MONAI Bundle)
- **File**: models/pretrained/cancers/stomach/volume.pt (616 MB)
- **Config**: models/pretrained/cancers/stomach/volume.json
- **Modality**: CT
- **Input**: 1x96x96x96 patches
- **Output**: 8 channels (background, artery, portal_vein, liver, spleen, stomach, gallbladder, pancreas)
- **Stomach channel**: 5
- **Dice**: 0.88 mean
- **Source**: Aichi Cancer Center, Japan

## Additional 3D Models Downloaded

### 4. Whole Body CT Segmentation (104 structures)
- **Model**: wholeBody_ct_segmentation
- **File**: bundles/wholeBody_ct_segmentation/models/model.pt (75 MB)
- **Modality**: CT
- **Includes**: stomach (ch6), prostate/bladder area (ch104), liver, pancreas, kidneys, etc.

### 5. Spleen CT Segmentation
- **Model**: spleen_ct_segmentation
- **File**: models/pretrained/cancers/spleen/volume.pt (19 MB)
- **Dice**: 0.961

### 6. Pancreas CT Segmentation (DiNTS)
- **Model**: pancreas_ct_dints_segmentation
- **File**: models/pretrained/cancers/pancreas/volume.pt (554 MB)

### 7. Kidney/Renal Structures Segmentation (includes tumors)
- **Model**: renalStructures_CECT_segmentation
- **File**: models/pretrained/cancers/kidney/volume.pt (76 MB)
- **Includes**: tumor/neoplasm channel (ch4), artery, vein, ureter, parenchyma

## Notes
- All models are PyTorch .pt format
- Config files (.json) contain metadata: input/output shapes, class mappings, modality, references
- Models downloaded via MONAI bundle download command from Hugging Face hosting
- Not for diagnostic use - research/educational purposes only
