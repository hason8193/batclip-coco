"""
Visualization utilities for COCO Zero-Shot predictions
Shows corrupted images with predictions and captions
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import textwrap
import os


def visualize_batch_predictions(images, predictions, labels, class_names, 
                                corruption_type="", num_samples=4, save_path=None,
                                captions=None, image_ids=None, multi_label=False,
                                image_paths=None):
    """
    Visualize a batch of images with their predictions and ground truth
    
    Args:
        images: Tensor of images [B, C, H, W]
        predictions: Tensor of predicted class indices [B] for single-label, or probabilities [B, C] for multi-label
        labels: Tensor of ground truth class indices [B] for single-label, or binary vectors [B, C] for multi-label
        class_names: List of class names (80 COCO object classes)
        corruption_type: Name of the corruption type
        num_samples: Number of samples to display
        save_path: Optional path to save the visualization
        captions: Optional list of COCO captions for each image
        image_ids: Optional list of image IDs
        multi_label: If True, treat as multi-label classification
    """
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(6*num_samples, 7))
    if num_samples == 1:
        axes = [axes]
    
    for idx in range(num_samples):
        # Load image directly from disk if path is available (better quality)
        if image_paths and idx < len(image_paths):
            from PIL import Image
            img = Image.open(image_paths[idx]).convert('RGB')
            img = np.array(img) / 255.0  # Normalize to [0, 1]
        else:
            # Fallback: Convert tensor to numpy image
            img = images[idx].cpu().numpy()
            
            # Handle different image formats
            if img.shape[0] == 3:  # CHW format
                img = np.transpose(img, (1, 2, 0))
            
            # Denormalize if needed (assuming CLIP normalization)
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img = img * std + mean
            img = np.clip(img, 0, 1)
        
        # Display image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Create caption with prediction
        title_parts = []
        if corruption_type:
            title_parts.append(f"{corruption_type}")
        
        if multi_label:
            # Multi-label: show top-k predictions and all ground truth labels
            pred_probs = predictions[idx].cpu().numpy()
            label_vec = labels[idx].cpu().numpy()
            
            # Get top-3 predictions
            top_k_indices = np.argsort(pred_probs)[-3:][::-1]
            top_k_probs = pred_probs[top_k_indices]
            pred_names = [f"{class_names[i]} ({top_k_probs[j]:.2f})" 
                         for j, i in enumerate(top_k_indices)]
            title_parts.append(f"Pred: {', '.join(pred_names)}")
            
            # Get ground truth labels
            gt_indices = np.where(label_vec > 0.5)[0]
            if len(gt_indices) > 0:
                gt_names = [class_names[i] for i in gt_indices]
                # Show up to 5 GT labels
                if len(gt_names) > 5:
                    gt_names = gt_names[:5] + [f"... +{len(gt_names)-5} more"]
                title_parts.append(f"GT: {', '.join(gt_names)}")
            else:
                title_parts.append("GT: None")
        else:
            # Single-label: show prediction and ground truth
            pred_idx = predictions[idx].item()
            pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"
            title_parts.append(f"Predicted: {pred_name}")
            
            # Ground truth
            if labels is not None and labels[idx].item() >= 0:
                gt_idx = labels[idx].item()
                gt_name = class_names[gt_idx] if gt_idx < len(class_names) else f"Class_{gt_idx}"
                title_parts.append(f"GT: {gt_name}")
        
        # Show actual COCO caption if available
        if captions and idx < len(captions):
            caption_text = captions[idx]
            if isinstance(caption_text, list):
                caption_text = caption_text[0] if caption_text else "No caption"
            # Wrap long captions
            wrapped = textwrap.fill(caption_text, width=40)
            title_parts.append(f"\nCaption: {wrapped}")
        
        title = '\n'.join(title_parts)
        axes[idx].set_title(title, fontsize=8, weight='bold', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_corruption_gallery(dataloader, model, class_names, corruption_type, 
                              device='cuda', num_images=8, save_dir='visualizations'):
    """
    Create a gallery showing various corrupted images and their predictions
    
    Args:
        dataloader: DataLoader for corrupted images
        model: CLIP or classification model
        class_names: List of class names
        corruption_type: Name of the corruption
        device: 'cuda' or 'cpu'
        num_images: Number of images to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    images_collected = []
    preds_collected = []
    labels_collected = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Handle different data formats
            if isinstance(batch_data, (list, tuple)):
                images = batch_data[0]
                labels = batch_data[1]
            else:
                images = batch_data
                labels = torch.zeros(len(images), dtype=torch.long)
            
            # Handle list of images (for augmentation)
            if isinstance(images, list):
                images = images[0]
            
            images = images.to(device)
            
            # Get predictions
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            
            images_collected.append(images.cpu())
            preds_collected.append(predictions.cpu())
            labels_collected.append(labels)
            
            if sum(len(x) for x in images_collected) >= num_images:
                break
    
    if not images_collected:
        print(f"Warning: No images collected for {corruption_type}")
        return None
    
    # Concatenate collected batches
    all_images = torch.cat(images_collected)[:num_images]
    all_preds = torch.cat(preds_collected)[:num_images]
    all_labels = torch.cat(labels_collected)[:num_images]
    
    # Save visualization
    save_path = os.path.join(save_dir, f'{corruption_type}_predictions.png')
    visualize_batch_predictions(
        all_images, all_preds, all_labels, class_names,
        corruption_type=corruption_type, num_samples=min(num_images, 8),
        save_path=save_path
    )
    
    return save_path
