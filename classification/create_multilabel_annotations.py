"""
Script to create multi-label annotations for COCO dataset from instances_val2017.json
Converts single-label annotations to multi-label format (binary vectors)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def create_multilabel_annotations():
    """
    Create multi-label annotations from COCO instances
    Each image can have multiple object categories
    """
    
    # Paths
    coco_dir = Path("COCOC_Dataset")
    instances_path = coco_dir / "instances_val2017.json"
    output_path = coco_dir / "image_labels_multilabel.json"
    
    # Load instances
    print(f"Loading annotations from {instances_path}...")
    with open(instances_path, 'r') as f:
        coco_data = json.load(f)
    
    # Build category ID to index mapping (COCO has 80 classes with non-contiguous IDs)
    categories = coco_data['categories']
    # COCO category IDs are not contiguous (e.g., 1, 2, 3, 4, 5, ..., 90)
    # We need to map them to 0-79 for our 80-class system
    category_id_to_idx = {}
    for idx, cat in enumerate(sorted(categories, key=lambda x: x['id'])):
        category_id_to_idx[cat['id']] = idx
    
    print(f"Found {len(categories)} categories")
    print(f"Category ID range: {min(category_id_to_idx.keys())} to {max(category_id_to_idx.keys())}")
    
    # Build image_id to annotations mapping
    image_to_categories = defaultdict(set)
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        if category_id in category_id_to_idx:
            category_idx = category_id_to_idx[category_id]
            image_to_categories[image_id].add(category_idx)
    
    # Create image filename to category mapping
    image_labels_multilabel = {}
    for img in coco_data['images']:
        image_id = img['id']
        # Extract filename without extension (e.g., "000000397133")
        filename = Path(img['file_name']).stem
        
        # Get all categories for this image
        categories_set = image_to_categories.get(image_id, set())
        # Convert to sorted list for consistency
        categories_list = sorted(list(categories_set))
        
        # Create binary vector (80 classes)
        binary_vector = [0] * 80
        for cat_idx in categories_list:
            binary_vector[cat_idx] = 1
        
        image_labels_multilabel[filename] = {
            'categories': categories_list,
            'binary_vector': binary_vector,
            'num_objects': len(categories_list)
        }
    
    # Statistics
    num_images = len(image_labels_multilabel)
    num_single_label = sum(1 for v in image_labels_multilabel.values() if v['num_objects'] == 1)
    num_multi_label = sum(1 for v in image_labels_multilabel.values() if v['num_objects'] > 1)
    avg_labels = sum(v['num_objects'] for v in image_labels_multilabel.values()) / num_images
    
    print(f"\nStatistics:")
    print(f"  Total images: {num_images}")
    print(f"  Single-label images: {num_single_label} ({100*num_single_label/num_images:.1f}%)")
    print(f"  Multi-label images: {num_multi_label} ({100*num_multi_label/num_images:.1f}%)")
    print(f"  Average labels per image: {avg_labels:.2f}")
    print(f"  Max labels per image: {max(v['num_objects'] for v in image_labels_multilabel.values())}")
    
    # Save to JSON
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(image_labels_multilabel, f, indent=2)
    
    print(f"✓ Created multi-label annotations for {num_images} images")
    print(f"✓ Saved to: {output_path}")
    
    # Print example
    example_key = list(image_labels_multilabel.keys())[0]
    example = image_labels_multilabel[example_key]
    print(f"\nExample (image {example_key}):")
    print(f"  Categories: {example['categories']}")
    print(f"  Num objects: {example['num_objects']}")
    
    return output_path


if __name__ == "__main__":
    create_multilabel_annotations()
