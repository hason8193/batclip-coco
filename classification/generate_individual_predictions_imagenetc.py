#!/usr/bin/env python3
"""
Generate individual prediction visualizations for ImageNet blur corruptions
Supports: defocus_blur, glass_blur, motion_blur, zoom_blur
Saves each image as: 0001_corruption_severity_N.png
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from conf import cfg
from models.model import get_model
from datasets.cls_names import get_class_names
from utils.registry import ADAPTATION_REGISTRY

# Import adaptation methods to register them
import methods

# Import selected classes
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from selected_100_classes import SELECTED_CLASSES


def visualize_single_image(img_batch, img_pil_list, label_batch, model, class_names, 
                           corruption, severity, start_idx, save_dir, device,
                           use_adaptation=False, valid_class_indices=None):
    """Create visualization for a batch of images (single-label)"""
    
    # Predict with or without adaptation
    if use_adaptation:
        # Use BATCLIP adaptation (model updates itself during forward pass)
        output = model(img_batch.to(device))
    else:
        # Zero-shot inference (no adaptation)
        with torch.no_grad():
            output = model(img_batch.to(device))
    
    # Process each image in the batch
    saved_paths = []
    batch_size = img_batch.size(0)
    
    for i in range(batch_size):
        img_pil = img_pil_list[i]
        label_tensor = label_batch[i]
        index = start_idx + i
        
        # Filter to only valid classes (100 selected)
        if valid_class_indices is not None:
            # Create mask for valid classes
            mask = torch.ones(output.size(1), device=output.device) * float('-inf')
            mask[valid_class_indices] = 0
            filtered_output = output[i:i+1] + mask
            probs = torch.softmax(filtered_output, dim=1)
        else:
            probs = torch.softmax(output[i:i+1], dim=1)
        
        top_probs, top_indices = torch.topk(probs[0], k=5)
        pred_classes = [class_names[idx] for idx in top_indices.cpu().numpy()]
        pred_scores = top_probs.cpu().numpy()
        
        # Ground truth
        gt_idx = label_tensor.item()
        gt_class = class_names[gt_idx]
        
        # Check if prediction is correct
        is_correct = (top_indices[0].item() == gt_idx)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Image with predictions
        axes[0].imshow(img_pil)
        axes[0].axis('off')
        pred_text = 'Top-5 Predictions:\n' + '\n'.join([f"{j+1}. {cls}: {score:.3f}" 
                                                        for j, (cls, score) in enumerate(zip(pred_classes, pred_scores))])
        axes[0].set_title(pred_text, fontsize=9, color='blue', loc='left', pad=10)
        
        # Right: Image with ground truth
        axes[1].imshow(img_pil)
        axes[1].axis('off')
        gt_text = f'Ground Truth:\n\n{gt_class}'
        axes[1].set_title(gt_text, fontsize=10, color='green', loc='left', pad=10, weight='bold')
        
        # Main title with correctness indicator
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        status_color = "green" if is_correct else "red"
        
        method_name = "BATCLIP" if use_adaptation else "Zero-shot CLIP"
        title = f"{method_name} - {corruption.replace('_', ' ').title()} (Severity {severity}) - Image #{index+1:04d}"
        fig.suptitle(f"{title}\n{status}", 
                     fontsize=12, fontweight='bold', color=status_color)
        
        plt.tight_layout()
        
        # Save
        filename = f"{index+1:04d}_{corruption}_severity_{severity}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        saved_paths.append(save_path)
    
    return saved_paths


def load_imagenetc_dataset(data_dir, corruption, severity, transform):
    """Load ImageNet-C corruption dataset"""
    from torch.utils.data import Dataset
    
    class ImageNetCDataset(Dataset):
        def __init__(self, data_dir, corruption, severity, transform):
            self.transform = transform
            
            # Determine category subfolder
            blur_types = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']
            digital_types = ['contrast', 'elastic_transform', 'jpeg_compression', 'pixelate']
            weather_types = ['brightness', 'fog', 'frost', 'snow']
            
            if corruption in blur_types:
                category = 'blur'
            elif corruption in digital_types:
                category = 'digital'
            elif corruption in weather_types:
                category = 'weather'
            else:
                raise ValueError(f"Unknown corruption type: {corruption}")
            
            # Structure: {category}/{corruption}/{severity}/{class_id}/image.JPEG
            corruption_dir = os.path.join(data_dir, category, corruption, str(severity))
            
            if not os.path.exists(corruption_dir):
                raise ValueError(f"Corruption directory does not exist: {corruption_dir}")
            
            # Collect all images and their labels
            self.images = []
            self.labels = []
            
            # Get ImageNet class mapping (class name to index)
            imagenet_classes = get_class_names('imagenet')
            
            # Build class name to index mapping (normalize underscores to spaces)
            if isinstance(imagenet_classes, dict):
                # Dict format: {synset_id: class_name}
                class_name_to_idx = {class_name.replace(' ', '_'): idx for idx, class_name in enumerate(imagenet_classes.values())}
            else:
                # List format: [class_name1, class_name2, ...]
                class_name_to_idx = {class_name.replace(' ', '_'): idx for idx, class_name in enumerate(imagenet_classes)}
            
            print(f"[+] Scanning {corruption_dir}...")
            
            for class_folder in sorted(os.listdir(corruption_dir), key=str.lower):
                class_path = os.path.join(corruption_dir, class_folder)
                if not os.path.isdir(class_path):
                    continue
                
                # Get class index by matching folder name to class name
                if class_folder in class_name_to_idx:
                    class_idx = class_name_to_idx[class_folder]
                else:
                    print(f"  ⚠ Warning: Unknown class folder '{class_folder}', skipping")
                    continue
                
                # Collect all images in this class
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.JPEG', '.jpg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(class_idx)
            
            print(f"[+] Loaded {len(self.images)} images from {corruption}_severity_{severity}")
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img_path = self.images[idx]
            label = self.labels[idx]
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_path
    
    return ImageNetCDataset(data_dir, corruption, severity, transform)


def generate_individual_predictions(data_dir='ImageNet-C', corruption='defocus_blur', severity=1, 
                                   max_images=None, adaptation_method='ours', batch_size=8):
    """Generate individual prediction images for ImageNet-C corruptions"""
    
    use_adaptation = (adaptation_method != 'source')
    
    print("\n" + "="*80)
    print(f"Dataset: ImageNet-C Corruptions")
    print(f"Generating Individual Predictions: {corruption} (severity {severity})")
    print(f"Method: {'BATCLIP (' + adaptation_method + ')' if use_adaptation else 'Zero-shot CLIP'}")
    print("="*80)
    
    # Setup config
    cfg.CORRUPTION.DATASET = "imagenet_c"
    cfg.MODEL.ARCH = "ViT-B-16"
    cfg.MODEL.USE_CLIP = True
    cfg.MODEL.WEIGHTS = "openai"
    cfg.MODEL.ADAPTATION = adaptation_method
    cfg.TEST.BATCH_SIZE = batch_size
    cfg.CUDNN.BENCHMARK = True
    cfg.OPTIM.STEPS = 1
    cfg.OPTIM.LR = 0.001
    cfg.MODEL.EPISODIC = False
    
    # Use CuPL prompts (file gốc với đầy đủ 1000 classes)
    cfg.CLIP.PROMPT_MODE = "cupl"
    cfg.CLIP.PROMPT_PATH = "datasets/cupl_prompts/CuPL_ImageNet_prompts.json"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[+] Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load base model
    print(f"\n[+] Loading model...")
    num_classes = 1000  # ImageNet has 1000 classes
    base_model, preprocess_fn = get_model(cfg, num_classes=num_classes, device=device)
    
    # Setup test-time adaptation method
    if use_adaptation:
        print(f"[+] Setting up BATCLIP adaptation: {adaptation_method}")
        available_adaptations = ADAPTATION_REGISTRY.registered_names()
        assert adaptation_method in available_adaptations, \
            f"The adaptation '{adaptation_method}' is not supported! Choose from: {available_adaptations}"
        model = ADAPTATION_REGISTRY.get(adaptation_method)(cfg=cfg, model=base_model, num_classes=num_classes)
    else:
        print("[+] Using zero-shot CLIP (no adaptation)")
        model = base_model
        model.eval()
    
    # Load class names
    class_names_dict = get_class_names('imagenet')
    if isinstance(class_names_dict, dict):
        class_names = list(class_names_dict.values())
    else:
        class_names = class_names_dict
    print(f"[+] Loaded {len(class_names)} class names")
    
    # Load dataset
    print(f"\n[+] Loading dataset: {corruption} severity {severity}...")
    print(f"[+] Data directory: {data_dir}")
    dataset = load_imagenetc_dataset(data_dir, corruption, severity, preprocess_fn)
    
    total_images = len(dataset)
    if max_images is not None:
        total_images = min(total_images, max_images)
    
    print(f"[+] Found {len(dataset)} images (processing {total_images})")
    
    # Build class name to index mapping for filtering predictions
    if isinstance(class_names_dict, dict):
        class_name_to_idx = {class_name.replace(' ', '_'): idx for idx, class_name in enumerate(class_names_dict.values())}
    else:
        class_name_to_idx = {class_name.replace(' ', '_'): idx for idx, class_name in enumerate(class_names_dict)}
    
    # Create output directory with severity level
    output_dir = Path(f"visualizations/individual_imagenetc/{corruption}_severity_{severity}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Output directory: {output_dir}")
    
    # Process images in batches
    print(f"\n[+] Processing images (batch size: {batch_size})...")
    successful = 0
    failed = 0
    correct = 0
    
    # Create batches
    num_batches = (total_images + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
        try:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_images)
            actual_batch_size = end_idx - start_idx
            
            # Collect batch data
            img_batch = []
            label_batch = []
            img_pil_list = []
            
            for idx in range(start_idx, end_idx):
                img_tensor, label, img_path = dataset[idx]
                img_batch.append(img_tensor)
                label_batch.append(torch.tensor(label))
                
                # Load PIL image for visualization
                img_pil = Image.open(img_path).convert('RGB')
                img_pil_list.append(img_pil)
            
            # Stack into batch tensors
            img_batch = torch.stack(img_batch)
            label_batch = torch.stack(label_batch)
            
            # Get valid class indices for the 100 selected classes
            valid_class_indices = [class_name_to_idx[cls.replace(' ', '_')] 
                                  for cls in SELECTED_CLASSES 
                                  if cls.replace(' ', '_') in class_name_to_idx]
            valid_class_indices = torch.tensor(valid_class_indices, device=device)
            
            # Generate visualizations for batch
            saved_paths = visualize_single_image(
                img_batch, img_pil_list, label_batch, model, class_names,
                corruption, severity, start_idx, output_dir, device, 
                use_adaptation=use_adaptation,
                valid_class_indices=valid_class_indices
            )
            
            successful += len(saved_paths)
            
            # Count correct predictions (top-1 accuracy) - filter to 100 classes
            with torch.no_grad():
                if use_adaptation:
                    output = model(img_batch.to(device))
                else:
                    output = model(img_batch.to(device))
                
                # Mask output to only include the 100 selected classes
                mask = torch.ones_like(output) * float('-inf')
                mask[:, valid_class_indices] = 0
                filtered_output = output + mask
                
                preds = filtered_output.argmax(dim=1)
                correct += (preds.cpu() == label_batch).sum().item()
            
        except Exception as e:
            print(f"\n  ⚠ Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            failed += actual_batch_size
            continue
    
    # Calculate accuracy
    accuracy = (correct / successful * 100) if successful > 0 else 0
    
    # Summary
    print("\n" + "="*80)
    print(f"[+] Generation Complete!")
    print(f"  Dataset: ImageNet-C Corruptions")
    print(f"  Corruption: {corruption} (severity {severity})")
    print(f"  Method: {'BATCLIP' if use_adaptation else 'Zero-shot CLIP'}")
    print(f"  Successful: {successful}/{total_images}")
    print(f"  Top-1 Accuracy: {accuracy:.2f}%")
    if failed > 0:
        print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")
    print("="*80 + "\n")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate individual predictions for ImageNet-C corruptions')
    parser.add_argument('--data-dir', type=str, default='ImageNet-C',
                       help='Path to ImageNet-C corruptions folder')
    parser.add_argument('--corruption', type=str, default='defocus_blur',
                       choices=['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                               'contrast', 'elastic_transform', 'jpeg_compression', 'pixelate',
                               'brightness', 'fog', 'frost', 'snow'],
                       help='Corruption type (blur: defocus_blur, glass_blur, motion_blur, zoom_blur | digital: contrast, elastic_transform, jpeg_compression, pixelate | weather: brightness, fog, frost, snow)')
    parser.add_argument('--severity', type=int, default=1, choices=[1, 2, 3, 4, 5],
                       help='Corruption severity level (1-5)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process (default: all)')
    parser.add_argument('--adaptation', type=str, default='ours',
                       choices=['source', 'ours', 'tent', 'cotta', 'eata', 'sar', 'rpl'],
                       help='Adaptation method: source=zero-shot, ours=BATCLIP, others=TTA methods')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing (required for TTA)')
    
    args = parser.parse_args()
    
    generate_individual_predictions(
        data_dir=args.data_dir,
        corruption=args.corruption,
        severity=args.severity,
        max_images=args.max_images,
        adaptation_method=args.adaptation,
        batch_size=args.batch_size
    )
