# ImageNet-C-100 Dataset

This directory contains 100 selected ImageNet classes with various corruption types for robustness testing.

## Structure
```
ImageNet-C-100/
├── blur/
│   ├── defocus_blur/
│   ├── glass_blur/
│   ├── motion_blur/
│   └── zoom_blur/
├── digital/
│   ├── contrast/
│   ├── elastic_transform/
│   ├── jpeg_compression/
│   └── pixelate/
└── weather/
    ├── brightness/
    ├── fog/
    ├── frost/
    └── snow/
```

Each corruption type has 5 severity levels (1-5).

## Download

Due to size constraints, this dataset is not included in the repository.

**Option 1: Generate from ImageNet-C**
If you have the full ImageNet-C dataset, extract the 100 classes using:
```bash
python scripts/extract_imagenet_c_100.py --source /path/to/ImageNet-C --output ImageNet-C-100
```

**Option 2: Download from cloud storage**
[Add your download link here - Google Drive, OneDrive, etc.]

## Classes

The 100 selected classes are defined in `selected_100_classes.py`:
- 100 diverse categories
- Balanced across superclasses
- All present in COCO-style annotations

## Usage

Once downloaded, the scripts will automatically detect this folder and use it for evaluation:
```bash
python run_model_comparison.py --quick-test
```
