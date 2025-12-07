#!/usr/bin/env python3
"""
Generate predictions using standard pre-trained models (MobileNet, DenseNet, ResNet, etc.)
For comparison with BATCLIP on corrupted ImageNet
"""

import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from datasets.cls_names import get_class_names
from selected_100_classes import SELECTED_CLASSES


def get_standard_model(model_name, device):
    """Load pre-trained standard model"""
    
    print(f"[+] Loading {model_name}...")
    
    if model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=True)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from: mobilenet_v3_large, densenet201, resnet101")
    
    model = model.to(device)
    model.eval()
    
    # Standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return model, preprocess


def visualize_single_image(img_batch, img_pil_list, label_batch, model, class_names, 
                           model_name, corruption, severity, start_idx, save_dir, device,
                           valid_class_indices=None):
    """Create visualization for a batch of images"""
    
    # Predict (no adaptation - standard inference)
    with torch.no_grad():
        output = model(img_batch.to(device))
    
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
        
        title = f"{model_name.upper()} - {corruption.replace('_', ' ').title()} (Severity {severity}) - Image #{index+1:04d}"
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
            
            corruption_dir = os.path.join(data_dir, category, corruption, str(severity))
            
            if not os.path.exists(corruption_dir):
                raise ValueError(f"Corruption directory does not exist: {corruption_dir}")
            
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
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_path
    
    return ImageNetCDataset(data_dir, corruption, severity, transform)


def generate_individual_predictions(data_dir='ImageNet-C', corruption='defocus_blur', severity=1, 
                                   max_images=None, model_name='resnet50', batch_size=32):
    """Generate individual prediction images using standard models"""
    
    print("\n" + "="*80)
    print(f"Dataset: ImageNet-C Corruptions")
    print(f"Generating Individual Predictions: {corruption} (severity {severity})")
    print(f"Model: {model_name.upper()}")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[+] Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model, preprocess_fn = get_standard_model(model_name, device)
    
    # Load class names
    class_names = get_class_names('imagenet')
    print(f"[+] Loaded {len(class_names)} class names")
    
    # Build class name to index mapping for filtering predictions
    if isinstance(class_names, dict):
        class_name_to_idx = {class_name.replace(' ', '_'): idx for idx, class_name in enumerate(class_names.values())}
    else:
        class_name_to_idx = {class_name.replace(' ', '_'): idx for idx, class_name in enumerate(class_names)}
    
    # Get valid class indices for the 100 selected classes
    valid_class_indices = [class_name_to_idx[cls.replace(' ', '_')] 
                          for cls in SELECTED_CLASSES 
                          if cls.replace(' ', '_') in class_name_to_idx]
    valid_class_indices = torch.tensor(valid_class_indices, device=device)
    print(f"[+] Filtering predictions to {len(valid_class_indices)} selected classes")
    
    # Load dataset
    print(f"\n[+] Loading dataset: {corruption} severity {severity}...")
    dataset = load_imagenetc_dataset(data_dir, corruption, severity, preprocess_fn)
    
    total_images = len(dataset)
    if max_images is not None:
        total_images = min(total_images, max_images)
    
    print(f"[+] Found {len(dataset)} images (processing {total_images})")
    
    # Create output directory
    output_dir = Path(f"visualizations/individual_{model_name}/{corruption}_severity_{severity}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Output directory: {output_dir}")
    
    # Process images in batches
    print(f"\n[+] Processing images (batch size: {batch_size})...")
    successful = 0
    failed = 0
    correct = 0
    
    num_batches = (total_images + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
        try:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_images)
            
            # Collect batch data
            img_batch = []
            label_batch = []
            img_pil_list = []
            
            for idx in range(start_idx, end_idx):
                img_tensor, label, img_path = dataset[idx]
                img_batch.append(img_tensor)
                label_batch.append(torch.tensor(label))
                img_pil_list.append(Image.open(img_path).convert('RGB'))
            
            img_batch = torch.stack(img_batch)
            label_batch = torch.stack(label_batch)
            
            # Generate visualizations
            saved_paths = visualize_single_image(
                img_batch, img_pil_list, label_batch, model, class_names,
                model_name, corruption, severity, start_idx, output_dir, device,
                valid_class_indices=valid_class_indices
            )
            
            successful += len(saved_paths)
            
            # Count correct predictions (filtered to 100 classes)
            with torch.no_grad():
                output = model(img_batch.to(device))
                
                # Mask output to only include the 100 selected classes
                mask = torch.ones_like(output) * float('-inf')
                mask[:, valid_class_indices] = 0
                filtered_output = output + mask
                
                preds = filtered_output.argmax(dim=1)
                correct += (preds.cpu() == label_batch).sum().item()
            
        except Exception as e:
            print(f"\n  ⚠ Error processing batch {batch_idx}: {e}")
            failed += (end_idx - start_idx)
            continue
    
    # Calculate accuracy
    accuracy = (correct / successful * 100) if successful > 0 else 0
    
    # Summary
    print("\n" + "="*80)
    print(f"[+] Generation Complete!")
    print(f"  Model: {model_name.upper()}")
    print(f"  Corruption: {corruption} (severity {severity})")
    print(f"  Successful: {successful}/{total_images}")
    print(f"  Top-1 Accuracy: {accuracy:.2f}%")
    if failed > 0:
        print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")
    print("="*80 + "\n")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions using standard models')
    parser.add_argument('--data-dir', type=str, default='ImageNet-C',
                       help='Path to ImageNet-C folder')
    parser.add_argument('--corruption', type=str, default='defocus_blur',
                       choices=['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                               'contrast', 'elastic_transform', 'jpeg_compression', 'pixelate',
                               'brightness', 'fog', 'frost', 'snow'],
                       help='Corruption type')
    parser.add_argument('--severity', type=int, default=1, choices=[1, 2, 3, 4, 5],
                       help='Severity level (1-5)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum images to process')
    parser.add_argument('--model', type=str, default='resnet101',
                       choices=['mobilenet_v3_large', 'densenet201', 'resnet101'],
                       help='Model architecture (best from each family)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    
    generate_individual_predictions(
        data_dir=args.data_dir,
        corruption=args.corruption,
        severity=args.severity,
        max_images=args.max_images,
        model_name=args.model,
        batch_size=args.batch_size
    )
