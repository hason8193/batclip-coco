import torch
import logging
import numpy as np
from typing import Union
from tqdm import tqdm
from datasets.imagenet_subsets import IMAGENET_D_MAPPING
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


logger = logging.getLogger(__name__)


def compute_multilabel_metrics(predictions: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    """
    Compute multi-label classification metrics
    
    Args:
        predictions: Predicted probabilities (N, C) - after sigmoid
        labels: Ground truth binary labels (N, C)
        threshold: Threshold for converting probabilities to binary predictions
    
    Returns:
        dict: Dictionary containing mAP, precision, recall, F1 scores
    """
    # Convert to numpy for sklearn
    preds_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Binary predictions using threshold
    preds_binary = (preds_np >= threshold).astype(int)
    
    # Compute metrics
    try:
        # mAP (mean Average Precision) - most important for multi-label
        mAP = average_precision_score(labels_np, preds_np, average='micro')
        mAP_macro = average_precision_score(labels_np, preds_np, average='macro')
        
        # Precision, Recall, F1 (micro and macro averaging)
        precision_micro = precision_score(labels_np, preds_binary, average='micro', zero_division=0)
        recall_micro = recall_score(labels_np, preds_binary, average='micro', zero_division=0)
        f1_micro = f1_score(labels_np, preds_binary, average='micro', zero_division=0)
        
        precision_macro = precision_score(labels_np, preds_binary, average='macro', zero_division=0)
        recall_macro = recall_score(labels_np, preds_binary, average='macro', zero_division=0)
        f1_macro = f1_score(labels_np, preds_binary, average='macro', zero_division=0)
        
        # Per-class AP (useful for analysis)
        per_class_ap = average_precision_score(labels_np, preds_np, average=None)
        
        return {
            'mAP_micro': mAP,
            'mAP_macro': mAP_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'per_class_ap': per_class_ap
        }
    except Exception as e:
        logger.warning(f"Error computing multi-label metrics: {e}")
        return {
            'mAP_micro': 0.0,
            'mAP_macro': 0.0,
            'precision_micro': 0.0,
            'recall_micro': 0.0,
            'f1_micro': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'per_class_ap': np.zeros(labels_np.shape[1])
        }


def split_results_by_domain(domain_dict: dict, data: list, predictions: torch.tensor):
    """
    Separates the label prediction pairs by domain
    Input:
        domain_dict: Dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
        data: List containing [images, labels, domains, ...]
        predictions: Tensor containing the predictions of the model
    Returns:
        domain_dict: Updated dictionary containing the domain seperated label prediction pairs
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict: dict, domain_seq: list):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    Input:
        domain_dict: Dictionary containing the labels and predictions for each domain
        domain_seq: Order to print the results (if all domains are contained in the domain dict)
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    domain_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting the results by domain...")
    for key in domain_names:
        label_prediction_arr = np.array(domain_dict[key])  # rows: samples, cols: (label, prediction)
        correct.append((label_prediction_arr[:, 0] == label_prediction_arr[:, 1]).sum())
        num_samples.append(label_prediction_arr.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 print_every: int,
                 device: Union[str, torch.device],
                 visualize: bool = False,
                 vis_dir: str = 'visualizations',
                 vis_samples: int = 8,
                 class_names: list = None,
                 severity: int = None,
                 multi_label: bool = False):

    num_correct = 0.
    num_samples = 0
    
    # Multi-label tracking
    all_predictions = []
    all_labels = []
    
    # Visualization setup
    vis_images = []
    vis_preds = []
    vis_labels = []
    vis_paths = []  # Store image paths for high-quality visualization
    should_visualize = visualize and num_samples == 0  # Only on first batch
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, desc=f"Batches [{domain_name}]", leave=False, unit="batch")):
            imgs, labels = data[0], data[1]
            output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            
            if multi_label:
                # Multi-label classification
                # CLIP outputs scaled cosine similarity: logit_scale * cos_sim
                # These scores are typically in range [0, 100] with logit_scale ~= 100
                # For multi-label, we can't use softmax (that's for mutually exclusive classes)
                # Instead, normalize scores to [0, 1] range and use as probabilities
                # Sigmoid works better than raw scores for independent probabilities
                probs = torch.sigmoid(output / 100.0)  # Scale down before sigmoid
                
                # For predictions, use top-k approach (avg 2.93 labels/image in COCO)
                # This is more appropriate than fixed threshold for CLIP scores
                k = 3  # Average number of labels per image
                topk_values, topk_indices = torch.topk(output, k=k, dim=1)
                predictions = torch.zeros_like(output)
                predictions.scatter_(1, topk_indices, 1.0)
                
                # Store for metric computation
                all_predictions.append(probs.cpu())
                all_labels.append(labels.cpu())
                
                # For visualization, store top-k predictions
                if should_visualize and len(vis_images) < vis_samples:
                    vis_imgs = imgs[0] if isinstance(imgs, list) else imgs
                    num_to_collect = min(vis_samples - len(vis_images), len(vis_imgs))
                    vis_images.append(vis_imgs[:num_to_collect].cpu())
                    vis_preds.append(probs[:num_to_collect].cpu())  # Store probabilities
                    vis_labels.append(labels[:num_to_collect].cpu())
                    # Try to get image paths from data loader
                    if len(data) > 2:
                        paths = data[2] if not isinstance(data[2], torch.Tensor) else None
                        if paths:
                            vis_paths.extend(paths[:num_to_collect])
                    should_visualize = len(vis_images) * vis_images[0].shape[0] < vis_samples
                
                # Compute subset accuracy (all labels correct)
                num_correct += ((predictions == labels.to(device)).all(dim=1)).float().sum()
            else:
                # Single-label classification
                predictions = output.argmax(1)
                
                # Collect samples for visualization (first batch only)
                if should_visualize and len(vis_images) < vis_samples:
                    # Get actual images (not augmented views)
                    vis_imgs = imgs[0] if isinstance(imgs, list) else imgs
                    num_to_collect = min(vis_samples - len(vis_images), len(vis_imgs))
                    vis_images.append(vis_imgs[:num_to_collect].cpu())
                    vis_preds.append(predictions[:num_to_collect].cpu())
                    vis_labels.append(labels[:num_to_collect].cpu())
                    # Try to get image paths from data loader
                    if len(data) > 2:
                        paths = data[2] if not isinstance(data[2], torch.Tensor) else None
                        if paths:
                            vis_paths.extend(paths[:num_to_collect])
                    should_visualize = len(vis_images) * vis_images[0].shape[0] < vis_samples

                if dataset_name == "imagenet_d" and domain_name != "none":
                    mapping_vector = list(IMAGENET_D_MAPPING.values())
                    predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

                num_correct += (predictions == labels.to(device)).float().sum()

                if "mixed_domains" in setting and len(data) >= 3:
                    domain_dict = split_results_by_domain(domain_dict, data, predictions)

            # track progress
            num_samples += imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
            if print_every > 0 and (i+1) % print_every == 0:
                if multi_label:
                    # Compute running mAP for progress tracking
                    temp_preds = torch.cat(all_predictions)
                    temp_labels = torch.cat(all_labels)
                    metrics = compute_multilabel_metrics(temp_preds, temp_labels)
                    logger.info(f"#batches={i+1:<6} #samples={num_samples:<9} mAP = {metrics['mAP_micro']:.2%}")
                else:
                    logger.info(f"#batches={i+1:<6} #samples={num_samples:<9} error = {1 - num_correct / num_samples:.2%}")

            if dataset_name == "ccc" and num_samples >= 7500000:
                break
    
    # Generate visualization if requested
    if visualize and vis_images and class_names:
        try:
            from utils.visualization import visualize_batch_predictions
            import os
            os.makedirs(vis_dir, exist_ok=True)
            
            all_vis_images = torch.cat(vis_images) if vis_images else None
            all_vis_preds = torch.cat(vis_preds) if vis_preds else None
            all_vis_labels = torch.cat(vis_labels) if vis_labels else None
            
            if all_vis_images is not None:
                # Try to load COCO captions for this domain
                captions = None
                try:
                    import json
                    caption_file = os.path.join('COCOC_Dataset', 'captions_val2017.json')
                    if os.path.exists(caption_file):
                        with open(caption_file, 'r') as f:
                            caption_data = json.load(f)
                        # Map image_id to captions
                        caption_map = {}
                        for ann in caption_data.get('annotations', []):
                            img_id = ann['image_id']
                            caption = ann['caption']
                            if img_id not in caption_map:
                                caption_map[img_id] = []
                            caption_map[img_id].append(caption)
                        # For visualization, we'll use the first caption for each image
                        # Note: We don't have image_ids in current implementation,
                        # so captions will be None for now
                except Exception as e:
                    logger.debug(f"Could not load captions: {e}")
                
                # Include severity in filename if provided
                filename = f'{domain_name}_severity_{severity}_predictions.png' if severity is not None else f'{domain_name}_predictions.png'
                save_path = os.path.join(vis_dir, filename)
                visualize_batch_predictions(
                    all_vis_images, all_vis_preds, all_vis_labels, class_names,
                    corruption_type=domain_name, num_samples=min(vis_samples, len(all_vis_images)),
                    save_path=save_path,
                    captions=captions,
                    multi_label=multi_label,
                    image_paths=vis_paths if vis_paths else None
                )
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    # Compute final metrics
    if multi_label:
        # Compute multi-label metrics
        all_predictions_tensor = torch.cat(all_predictions)
        all_labels_tensor = torch.cat(all_labels)
        metrics = compute_multilabel_metrics(all_predictions_tensor, all_labels_tensor)
        
        # Log detailed metrics
        logger.info(f"Multi-label metrics for {domain_name}:")
        logger.info(f"  mAP (micro): {metrics['mAP_micro']:.2%}")
        logger.info(f"  mAP (macro): {metrics['mAP_macro']:.2%}")
        logger.info(f"  Precision (micro): {metrics['precision_micro']:.2%}")
        logger.info(f"  Recall (micro): {metrics['recall_micro']:.2%}")
        logger.info(f"  F1 (micro): {metrics['f1_micro']:.2%}")
        
        # Return mAP as primary metric (higher is better, so we return it as "accuracy")
        accuracy = metrics['mAP_micro']
        return accuracy, domain_dict, num_samples, metrics
    else:
        accuracy = num_correct.item() / num_samples
        return accuracy, domain_dict, num_samples
