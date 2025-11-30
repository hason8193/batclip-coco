#!/usr/bin/env python3
"""
Generate individual prediction visualizations for each image in a corruption set
Saves each image as a separate file: 0001_corruption_severity_N.png
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from conf import cfg
from models.model import get_model
from datasets.cls_names import get_class_names
from datasets.coco_corrupted_dataset import COCOCorruptedDataset
from utils.registry import ADAPTATION_REGISTRY

# Import adaptation methods to register them
import methods


def visualize_single_image(img_batch, img_pil_list, label_batch, model, class_names, 
                           corruption, severity, start_idx, save_dir, device, multi_label=True,
                           use_adaptation=False):
    """Create visualization for a batch of images"""
    
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
        
        if multi_label:
            # Multi-label: sigmoid activation
            probs = torch.sigmoid(output[i] / 100.0)
            # Get top predictions
            top_probs, top_indices = torch.topk(probs, k=5)
            pred_classes = [class_names[idx] for idx in top_indices.cpu().numpy()]
            pred_scores = top_probs.cpu().numpy()
        else:
            # Single-label: softmax
            probs = torch.softmax(output[i:i+1], dim=1)
            top_probs, top_indices = torch.topk(probs[0], k=5)
            pred_classes = [class_names[idx] for idx in top_indices.cpu().numpy()]
            pred_scores = top_probs.cpu().numpy()
        
        # Ground truth
        if multi_label:
            gt_indices = torch.where(label_tensor > 0.5)[0].cpu().numpy()
            gt_classes = [class_names[idx] for idx in gt_indices]
        else:
            gt_idx = label_tensor.item()
            gt_classes = [class_names[gt_idx]]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Left: Image with predictions
        axes[0].imshow(img_pil)
        axes[0].axis('off')
        pred_text = 'Top Predictions:\n' + '\n'.join([f"{j+1}. {cls}: {score:.3f}" 
                                                        for j, (cls, score) in enumerate(zip(pred_classes, pred_scores))])
        axes[0].set_title(pred_text, fontsize=10, color='blue', loc='left')
        
        # Right: Image with ground truth
        axes[1].imshow(img_pil)
        axes[1].axis('off')
        gt_text = 'Ground Truth:\n' + '\n'.join([f"• {cls}" for cls in gt_classes[:10]])
        if len(gt_classes) > 10:
            gt_text += f"\n  +{len(gt_classes)-10} more"
        axes[1].set_title(gt_text, fontsize=10, color='green', loc='left')
        
        # Main title
        method_name = "BATCLIP" if use_adaptation else "Zero-shot CLIP"
        fig.suptitle(f"{method_name} - {corruption.replace('_', ' ').title()} - Severity {severity} - Image #{index+1:04d}", 
                     fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        filename = f"{index+1:04d}_{corruption}_severity_{severity}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        saved_paths.append(save_path)
    
    return saved_paths


def generate_individual_predictions(corruption='defocus_blur', severity=1, max_images=None, 
                                   adaptation_method='ours', batch_size=8):
    """Generate individual prediction images for all samples in a corruption set"""
    
    use_adaptation = (adaptation_method != 'source')
    
    print("\n" + "="*80)
    print(f"Generating Individual Predictions: {corruption} (severity {severity})")
    print(f"Method: {'BATCLIP (' + adaptation_method + ')' if use_adaptation else 'Zero-shot CLIP'}")
    print("="*80)
    
    # Setup config
    cfg.CORRUPTION.DATASET = "coco_c"
    cfg.MODEL.ARCH = "ViT-B-16"
    cfg.MODEL.USE_CLIP = True
    cfg.MODEL.WEIGHTS = "openai"
    cfg.MODEL.ADAPTATION = adaptation_method
    cfg.TEST.BATCH_SIZE = batch_size
    cfg.CUDNN.BENCHMARK = True
    cfg.OPTIM.STEPS = 1
    cfg.OPTIM.LR = 0.001
    cfg.MODEL.EPISODIC = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load base model
    print(f"\n✓ Loading model...")
    num_classes = 80
    base_model, preprocess_fn = get_model(cfg, num_classes=num_classes, device=device)
    
    # Setup test-time adaptation method
    if use_adaptation:
        print(f"✓ Setting up BATCLIP adaptation: {adaptation_method}")
        available_adaptations = ADAPTATION_REGISTRY.registered_names()
        assert adaptation_method in available_adaptations, \
            f"The adaptation '{adaptation_method}' is not supported! Choose from: {available_adaptations}"
        model = ADAPTATION_REGISTRY.get(adaptation_method)(cfg=cfg, model=base_model, num_classes=num_classes)
    else:
        print("✓ Using zero-shot CLIP (no adaptation)")
        model = base_model
        model.eval()
    
    # Load class names
    class_names = get_class_names("coco_c")
    print(f"✓ Loaded {len(class_names)} class names")
    
    # Load dataset
    print(f"\n✓ Loading dataset: {corruption} severity {severity}...")
    dataset = COCOCorruptedDataset(
        data_dir='COCOC_Dataset',
        corruption=corruption,
        severity=severity,
        transform=preprocess_fn,
        multi_label=True
    )
    
    total_images = len(dataset)
    if max_images is not None:
        total_images = min(total_images, max_images)
    
    print(f"✓ Found {len(dataset)} images (processing {total_images})")
    
    # Create output directory with severity level
    output_dir = Path(f"visualizations/individual/{corruption}_severity_{severity}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Process images in batches
    print(f"\n✓ Processing images (batch size: {batch_size})...")
    successful = 0
    failed = 0
    
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
                img_tensor, label_tensor, img_path = dataset[idx]
                img_batch.append(img_tensor)
                label_batch.append(label_tensor)
                img_pil_list.append(Image.open(img_path).convert('RGB'))
            
            # Stack into batch tensors
            img_batch = torch.stack(img_batch)
            label_batch = torch.stack(label_batch)
            
            # Generate visualizations for batch
            saved_paths = visualize_single_image(
                img_batch, img_pil_list, label_batch, model, class_names,
                corruption, severity, start_idx, output_dir, device, 
                multi_label=True, use_adaptation=use_adaptation
            )
            
            successful += len(saved_paths)
            
        except Exception as e:
            print(f"\n  ⚠ Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            failed += actual_batch_size
            continue
    
    # Summary
    print("\n" + "="*80)
    print(f"✓ Generation Complete!")
    print(f"  Successful: {successful}/{total_images}")
    if failed > 0:
        print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")
    print("="*80 + "\n")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate individual prediction visualizations for each image'
    )
    parser.add_argument('--corruption', type=str, required=True,
                       help='Corruption type (e.g., defocus_blur, gaussian_noise)')
    parser.add_argument('--severity', type=int, required=True,
                       help='Severity level (1-5)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process (default: all)')
    parser.add_argument('--adaptation', type=str, default='ours',
                       choices=['source', 'ours', 'tent', 'cotta', 'eata', 'sar', 'rpl'],
                       help='Adaptation method: source=zero-shot, ours=BATCLIP (default: ours)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for adaptation (default: 8)')
    
    args = parser.parse_args()
    
    generate_individual_predictions(
        corruption=args.corruption,
        severity=args.severity,
        max_images=args.max_images,
        adaptation_method=args.adaptation,
        batch_size=args.batch_size
    )
