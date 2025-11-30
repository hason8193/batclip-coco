#!/usr/bin/env python3
"""
BATCLIP Visualization - Uses the actual BATCLIP model with test-time adaptation.
This is the adapted/robust version, not vanilla CLIP.
"""

import argparse
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from conf import cfg
from models.model import get_model
from datasets.cls_names import get_class_names


def load_batclip_model():
    """Load BATCLIP model with configuration."""
    print("\n" + "="*80)
    print("BATCLIP Visualization - Loading adapted model...")
    print("="*80)
    
    # Setup config for COCO
    cfg.CORRUPTION.DATASET = "coco_c"
    cfg.MODEL.ARCH = "ViT-B-16"
    cfg.MODEL.USE_CLIP = True
    cfg.MODEL.WEIGHTS = "openai"
    cfg.TEST.BATCH_SIZE = 1
    cfg.CUDNN.BENCHMARK = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[OK] Device: {device}")
    if torch.cuda.is_available():
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
    
    # Build BATCLIP model
    print("\n     Loading BATCLIP model...")
    num_classes = 80  # COCO has 80 classes
    model, preprocess_fn = get_model(cfg, num_classes=num_classes, device=device)
    model.eval()
    
    print("[OK] BATCLIP model ready!")
    print(f"     Architecture: {cfg.MODEL.ARCH}")
    print(f"     Dataset: {cfg.CORRUPTION.DATASET}")
    print()
    
    return model, preprocess_fn, device


def visualize_batclip_predictions(model, preprocess_fn, class_names, device,
                                   corruption_type, severity, num_samples, save_dir):
    """Visualize BATCLIP predictions."""
    
    # Load labels
    labels_path = Path("COCOC_Dataset/image_labels.json")
    if not labels_path.exists():
        print(f"[X] Labels not found: {labels_path}")
        return
    
    with open(labels_path, 'r') as f:
        image_labels = json.load(f)
    
    # Get images
    img_dir = Path(f"COCOC_Dataset/corrupted_images/{corruption_type}_severity_{severity}")
    if not img_dir.exists():
        print(f"[X] Not found: {img_dir}")
        return
    
    image_files = sorted(list(img_dir.glob("*.jpg")))[:num_samples]
    if len(image_files) == 0:
        print(f"[X] No images in {img_dir}")
        return
    
    print(f"[{corruption_type}] Processing {len(image_files)} samples with BATCLIP...")
    
    # Process images
    images = []
    predictions = []
    ground_truths = []
    confidences = []
    
    model.eval()
    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            
            # Preprocess with BATCLIP's preprocessing
            img_tensor = preprocess_fn(img).unsqueeze(0).to(device)
            
            # BATCLIP prediction
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
            
            # Ground truth
            img_id = img_path.stem
            gt_idx = image_labels.get(img_id, -1)
            
            images.append(np.array(img))
            predictions.append(pred_idx)
            ground_truths.append(gt_idx)
            confidences.append(confidence)
    
    # Accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g and g != -1)
    total = sum(1 for g in ground_truths if g != -1)
    accuracy = 100 * correct / total if total > 0 else 0
    
    # Debug: show some predictions
    if len(predictions) > 0:
        print(f"\n  Sample predictions:")
        for i in range(min(3, len(predictions))):
            print(f"    Image {i}: Pred={class_names[predictions[i]]}, GT={class_names[ground_truths[i]] if ground_truths[i] != -1 else 'Unknown'}")
    
    # Visualize
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4.5))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    for idx, (img, pred, gt, conf) in enumerate(zip(images, predictions, ground_truths, confidences)):
        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')
        
        is_correct = (pred == gt and gt != -1)
        color = 'green' if is_correct else 'red'
        
        title = f"BATCLIP: {class_names[pred]} ({conf:.1%})\nGT: {class_names[gt] if gt != -1 else 'Unknown'}"
        ax.set_title(title, color=color, fontsize=9, weight='bold', pad=10)
    
    for idx in range(n, len(axes)):
        axes[idx].axis('off')
    
    # Legend
    correct_patch = mpatches.Patch(color='green', label='Correct')
    incorrect_patch = mpatches.Patch(color='red', label='Incorrect')
    fig.legend(handles=[correct_patch, incorrect_patch], 
              loc='upper right', fontsize=10)
    
    title = f"BATCLIP - {corruption_type.replace('_', ' ').title()} (Severity {severity}) - Accuracy: {accuracy:.1f}%"
    plt.suptitle(title, fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    save_path = Path(save_dir) / f"batclip_{corruption_type}_s{severity}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"            Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"            Saved: {save_path.name}")


def main():
    parser = argparse.ArgumentParser(description="BATCLIP COCO-C Visualization")
    parser.add_argument("--corruption", type=str, default="gaussian_noise",
                       help="Corruption type or 'all' (default: gaussian_noise)")
    parser.add_argument("--severity", type=int, default=5,
                       help="Severity 1-5 (default: 5)")
    parser.add_argument("--num-samples", type=int, default=8,
                       help="Number of samples (default: 8)")
    parser.add_argument("--save-dir", type=str, default="visualizations/batclip",
                       help="Output directory (default: visualizations/batclip)")
    args = parser.parse_args()
    
    all_corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness', 'contrast',
        'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    if args.corruption == "all":
        corruptions = all_corruptions
    else:
        if args.corruption not in all_corruptions:
            print(f"[X] Unknown corruption: {args.corruption}")
            print(f"    Available: {', '.join(all_corruptions)}")
            return
        corruptions = [args.corruption]
    
    # Load BATCLIP model once
    model, preprocess_fn, device = load_batclip_model()
    class_names = get_class_names(cfg.CORRUPTION.DATASET)
    
    # Process
    print(f"Visualizing {len(corruptions)} corruption(s) x {args.num_samples} samples...\n")
    for corruption in corruptions:
        visualize_batclip_predictions(
            model, preprocess_fn, class_names, device,
            corruption, args.severity, args.num_samples, args.save_dir
        )
    
    print(f"\n{'='*80}")
    print(f"[OK] Done! Saved to: {args.save_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
