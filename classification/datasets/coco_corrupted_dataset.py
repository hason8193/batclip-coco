"""
COCO Corrupted Dataset for zero-shot BATCLIP evaluation
Supports various image corruptions for robustness testing
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image

import torch
from torch.utils.data import Dataset


class COCOCorruptedDataset(Dataset):
    """
    COCO Corrupted Dataset for Test-Time Adaptation
    
    Supports loading COCO validation images with various corruptions
    and their associated captions for zero-shot classification.
    """
    
    CORRUPTION_TYPES = [
        'brightness',
        'contrast', 
        'defocus_blur',
        'elastic_transform',
        'fog',
        'frost',
        'gaussian_noise',
        'impulse_noise',
        'jpeg_compression',
        'motion_blur',
        'pixelate',
        'shot_noise',
        'snow',
        'zoom_blur'
    ]
    
    def __init__(
        self,
        data_dir: str,
        corruption: str = 'gaussian_noise',
        severity: int = 5,
        transform=None,
        captions_json: Optional[str] = None,
        multi_label: bool = True
    ):
        """
        Args:
            data_dir: Path to COCOC_Dataset directory
            corruption: Type of corruption to load
            severity: Corruption severity level (1-5)
            transform: Image transformations to apply
            captions_json: Path to captions JSON file
            multi_label: If True, use multi-label classification (default: True)
        """
        self.data_dir = Path(data_dir)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        self.multi_label = multi_label
        
        # Load images from corrupted folder
        corruption_dir = f"{corruption}_severity_{severity}"
        self.image_dir = self.data_dir / "corrupted_images" / corruption_dir
        
        if not self.image_dir.exists():
            raise ValueError(f"Corruption directory not found: {self.image_dir}")
        
        # Get all images
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        
        # Load captions if provided
        self.captions = None
        if captions_json:
            self.captions = self._load_captions(captions_json)
        
        # Load ground truth labels
        if multi_label:
            self.labels = self._load_multilabel_labels()
            print(f"✓ Loaded {len(self.image_files)} images from {corruption_dir} (multi-label mode)")
        else:
            self.labels = self._load_labels()
            print(f"✓ Loaded {len(self.image_files)} images from {corruption_dir} (single-label mode)")
    
    def _load_captions(self, captions_json: str):
        """Load captions mapping from JSON file"""
        with open(captions_json, 'r') as f:
            captions = json.load(f)
        return captions
    
    def _load_labels(self):
        """Load ground truth labels from image_labels.json (single-label)"""
        labels_file = self.data_dir / "image_labels.json"
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                labels = json.load(f)
            return labels
        else:
            print(f"⚠ Warning: image_labels.json not found at {labels_file}")
            return {}
    
    def _load_multilabel_labels(self):
        """Load ground truth labels from image_labels_multilabel.json (multi-label)"""
        labels_file = self.data_dir / "image_labels_multilabel.json"
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                labels = json.load(f)
            return labels
        else:
            print(f"⚠ Warning: image_labels_multilabel.json not found at {labels_file}")
            print(f"  Falling back to single-label mode")
            self.multi_label = False
            return self._load_labels()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get caption if available
        caption = None
        if self.captions:
            img_key = str(img_path)
            if img_key in self.captions:
                caption = self.captions[img_key]['captions']
        
        # Extract image ID from filename (e.g., "000000397133.jpg" -> "000000397133")
        img_filename = img_path.stem  # Gets filename without extension
        
        # Get ground truth label(s)
        if self.multi_label:
            # Multi-label: return binary vector
            label_data = self.labels.get(img_filename, None)
            if label_data is None:
                # If not found, return zero vector
                target = torch.zeros(80, dtype=torch.float32)
            else:
                # Convert binary vector to tensor
                target = torch.tensor(label_data['binary_vector'], dtype=torch.float32)
        else:
            # Single-label: return class index
            target = self.labels.get(img_filename, -1)  # -1 if label not found
        
        # Return image, target, and image path (for visualization)
        return image, target, str(img_path)


def create_coco_corrupted_dataset(
    dataset_name: str,
    severity: int,
    data_dir: str,
    corruption: str,
    corruptions_seq: List[str],
    transform,
    setting: str,
    multi_label: bool = True,
):
    """
    Create COCO corrupted dataset for TTA
    
    Args:
        dataset_name: Name of dataset (should be 'coco' or 'coco_c')
        severity: Corruption severity level (1-5)
        data_dir: Path to COCOC_Dataset directory
        corruption: Current corruption type
        corruptions_seq: Sequence of all corruptions to test
        transform: Image transformations
        setting: Testing setting (e.g., 'reset_each_shift', 'continual')
        multi_label: If True, use multi-label classification (default: True)
    
    Returns:
        Dataset object for the specified corruption
    """
    
    # Load captions if available
    captions_json = os.path.join(data_dir, "corrupted_captions.json")
    if not os.path.exists(captions_json):
        print(f"⚠ Warning: Captions file not found at {captions_json}")
        captions_json = None
    
    # Handle different test settings
    if setting in ["reset_each_shift", "continual", "gradual"]:
        # Load single corruption type
        dataset = COCOCorruptedDataset(
            data_dir=data_dir,
            corruption=corruption,
            severity=severity,
            transform=transform,
            captions_json=captions_json,
            multi_label=multi_label
        )
    
    elif "mixed_domains" in setting:
        # Load all corruptions and mix them
        datasets = []
        for corr in corruptions_seq:
            ds = COCOCorruptedDataset(
                data_dir=data_dir,
                corruption=corr,
                severity=severity,
                transform=transform,
                captions_json=captions_json,
                multi_label=multi_label
            )
            datasets.append(ds)
        
        # Concatenate all datasets
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(datasets)
        print(f"✓ Mixed {len(datasets)} corruption types, total {len(dataset)} images")
    
    else:
        raise ValueError(f"Unknown setting: {setting}")
    
    return dataset


# COCO 80 classes (object detection categories)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
