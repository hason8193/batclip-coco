#!/usr/bin/env python3
"""
Run comparison between BATCLIP and 3 standard models on ImageNet-C-100
Models: BATCLIP, ResNet101, DenseNet201, MobileNetV3-Large

Tests on 100 selected ImageNet classes from selected_100_classes.py

NOTE: This script runs each model sequentially and shows live progress.
You'll need to manually note the accuracies from the output.
"""

import subprocess
import sys
from pathlib import Path
import time
from selected_100_classes import SELECTED_CLASSES

# Configuration
print(f"[+] Testing on {len(SELECTED_CLASSES)} selected ImageNet classes")
print(f"    Examples: {', '.join(SELECTED_CLASSES[:5])}...\n")

CORRUPTIONS = [
    # Blur
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    # Digital
    'contrast', 'elastic_transform', 'jpeg_compression', 'pixelate',
    # Weather
    'brightness', 'fog', 'frost', 'snow'
]

SEVERITIES = [1, 2, 3, 4, 5]

MODELS = [
    {'name': 'batclip', 'script': 'generate_individual_predictions_imagenetc.py', 
     'args': ['--adaptation', 'ours', '--batch-size', '8']},
    {'name': 'resnet101', 'script': 'generate_individual_predictions_standard_models.py',
     'args': ['--model', 'resnet101', '--batch-size', '32']},
    {'name': 'densenet201', 'script': 'generate_individual_predictions_standard_models.py',
     'args': ['--model', 'densenet201', '--batch-size', '32']},
    {'name': 'mobilenet_v3', 'script': 'generate_individual_predictions_standard_models.py',
     'args': ['--model', 'mobilenet_v3_large', '--batch-size', '32']},
]

def run_model_comparison(corruption, severity, max_images=None):
    """Run all 4 models on a specific corruption and severity"""
    
    print("\n" + "="*80)
    print(f"COMPARISON: {corruption.upper()} - Severity {severity}")
    print("="*80)
    
    results = {}
    
    for model_config in MODELS:
        model_name = model_config['name']
        script = model_config['script']
        model_args = model_config['args']
        
        print(f"\n[{model_name.upper()}]")
        print("-" * 80)
        
        # Build command
        cmd = [
            'python', script,
            '--data-dir', 'ImageNet-C-100',
            '--corruption', corruption,
            '--severity', str(severity)
        ]
        
        if max_images:
            cmd.extend(['--max-images', str(max_images)])
        
        cmd.extend(model_args)
        
        try:
            # Run the model and capture output
            print(f"Command: {' '.join(cmd)}\n")
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            
            # Print output
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            # Parse accuracy from output
            accuracy = None
            for line in result.stdout.split('\n'):
                if 'Top-1 Accuracy:' in line:
                    # Extract accuracy like "Top-1 Accuracy: 74.40%"
                    try:
                        accuracy = float(line.split('Top-1 Accuracy:')[1].strip().replace('%', ''))
                        break
                    except:
                        pass
            
            if accuracy is not None:
                print(f"\n✓ {model_name} completed. Accuracy: {accuracy:.2f}%")
                results[model_name] = accuracy
            else:
                print(f"\n✓ {model_name} completed but could not parse accuracy.")
                results[model_name] = None
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running {model_name}: {e}")
            if e.output:
                print(e.output)
            results[model_name] = None
    
    # Summary
    print("\n" + "="*80)
    print(f"SUMMARY: {corruption} - Severity {severity}")
    print("="*80)
    for model_name, acc in results.items():
        if acc is not None:
            print(f"  {model_name:15s}: {acc:6.2f}%")
        else:
            print(f"  {model_name:15s}: FAILED")
    print("="*80)
    
    return results


def run_full_comparison(selected_corruptions=None, selected_severities=None, max_images=None):
    """
    Run full comparison across multiple corruptions and severities
    
    Args:
        selected_corruptions: List of corruptions to test (default: all)
        selected_severities: List of severities to test (default: all)
        max_images: Limit number of images per corruption/severity
    """
    
    corruptions = selected_corruptions or CORRUPTIONS
    severities = selected_severities or SEVERITIES
    
    print("\n" + "="*80)
    print("BATCLIP vs Standard Models - Full Comparison")
    print("="*80)
    print(f"Dataset: ImageNet-C-100")
    print(f"Models: {len(MODELS)}")
    for model in MODELS:
        print(f"  - {model['name']}")
    print(f"Corruptions: {len(corruptions)}")
    print(f"Severities: {severities}")
    if max_images:
        print(f"Max images per test: {max_images}")
    print("="*80)
    
    all_results = {}
    
    for corruption in corruptions:
        for severity in severities:
            key = f"{corruption}_s{severity}"
            all_results[key] = run_model_comparison(corruption, severity, max_images)
    
    # Final summary
    print("\n\n" + "="*80)
    print("FINAL RESULTS - All Tests")
    print("="*80)
    
    # Calculate averages
    model_totals = {model['name']: {'sum': 0, 'count': 0} for model in MODELS}
    
    for key, results in all_results.items():
        print(f"\n{key}:")
        for model_name, acc in results.items():
            if acc is not None:
                print(f"  {model_name:15s}: {acc:6.2f}%")
                model_totals[model_name]['sum'] += acc
                model_totals[model_name]['count'] += 1
            else:
                print(f"  {model_name:15s}: FAILED")
    
    # Print averages
    print("\n" + "="*80)
    print("AVERAGE ACCURACIES")
    print("="*80)
    for model_name, stats in model_totals.items():
        if stats['count'] > 0:
            avg = stats['sum'] / stats['count']
            print(f"  {model_name:15s}: {avg:6.2f}% (across {stats['count']} tests)")
        else:
            print(f"  {model_name:15s}: No successful tests")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare BATCLIP with standard models on ImageNet-C-100')
    parser.add_argument('--corruption', type=str, default=None,
                       choices=CORRUPTIONS,
                       help='Single corruption to test (default: all)')
    parser.add_argument('--severity', type=int, default=None, choices=[1, 2, 3, 4, 5],
                       help='Single severity to test (default: all)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Limit images per test (useful for quick testing)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test: 1 corruption, severity 3, 100 images')
    
    args = parser.parse_args()
    
    if args.quick_test:
        # Quick test mode
        print("Running QUICK TEST mode")
        run_model_comparison('defocus_blur', 3, max_images=100)
    elif args.corruption and args.severity:
        # Single test
        run_model_comparison(args.corruption, args.severity, args.max_images)
    elif args.corruption:
        # Single corruption, all severities
        run_full_comparison([args.corruption], SEVERITIES, args.max_images)
    else:
        # Full comparison
        run_full_comparison(max_images=args.max_images)
