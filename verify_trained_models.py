"""
Utility script to verify that models are trained on actual code data
Checks model checkpoints to ensure they were trained properly
"""

import torch
import argparse
from pathlib import Path
import json


def check_checkpoint(checkpoint_path):
    """
    Check if a checkpoint was trained on actual code data
    
    Args:
        checkpoint_path: Path to the model checkpoint
    
    Returns:
        Dictionary with verification results
    """
    path = Path(checkpoint_path)
    
    if not path.exists():
        return {
            'status': 'error',
            'message': f'Checkpoint not found: {checkpoint_path}'
        }
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check metadata
        info = {
            'checkpoint_path': str(checkpoint_path),
            'status': 'verified',
        }
        
        # Check for training metadata
        if 'trained_on' in checkpoint:
            dataset = checkpoint['trained_on']
            info['dataset'] = dataset
            
            # Check if it's a real code dataset
            known_code_datasets = [
                'bigcode/the-stack',
                'bigcode/the-stack-smol',
                'codeparrot/github-code',
            ]
            
            is_code_dataset = any(ds in dataset for ds in known_code_datasets)
            info['is_code_dataset'] = is_code_dataset
            
            if not is_code_dataset:
                info['warning'] = 'Dataset does not appear to be a known code dataset'
        else:
            info['dataset'] = 'unknown'
            info['warning'] = 'No training metadata found in checkpoint'
        
        # Check for variant info
        if 'variant_name' in checkpoint:
            info['variant'] = checkpoint['variant_name']
        
        # Check for training metrics
        if 'best_val_loss' in checkpoint:
            info['best_val_loss'] = checkpoint['best_val_loss']
        elif 'val_loss' in checkpoint:
            info['val_loss'] = checkpoint['val_loss']
        
        if 'epoch' in checkpoint:
            info['epochs_trained'] = checkpoint['epoch'] + 1
        elif 'total_epochs' in checkpoint:
            info['epochs_trained'] = checkpoint['total_epochs']
        
        return info
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error loading checkpoint: {e}'
        }


def verify_trained_models(base_dir='./trained_models'):
    """
    Verify all trained models in the directory structure
    
    Args:
        base_dir: Base directory containing trained models
    
    Returns:
        Dictionary with verification results for all models
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return {
            'status': 'error',
            'message': f'Directory not found: {base_dir}'
        }
    
    results = {}
    
    # Expected variants
    from model_variants import MODEL_VARIANTS
    
    for variant_name in MODEL_VARIANTS:
        variant_dir = base_path / variant_name
        
        if not variant_dir.exists():
            results[variant_name] = {
                'status': 'not_found',
                'message': f'No training directory found for {variant_name}'
            }
            continue
        
        # Check for best model
        best_model_path = variant_dir / 'best_model.pt'
        final_model_path = variant_dir / f'{variant_name}_final.pt'
        
        checkpoint_to_check = None
        if best_model_path.exists():
            checkpoint_to_check = best_model_path
        elif final_model_path.exists():
            checkpoint_to_check = final_model_path
        
        if checkpoint_to_check:
            results[variant_name] = check_checkpoint(checkpoint_to_check)
        else:
            results[variant_name] = {
                'status': 'no_checkpoint',
                'message': f'No checkpoint found in {variant_dir}'
            }
    
    return results


def print_verification_report(results):
    """Print a formatted verification report"""
    print("\n" + "="*100)
    print(" " * 30 + "SmallCoder Model Verification Report")
    print("="*100)
    print()
    
    if 'status' in results and results['status'] == 'error':
        print(f"❌ Error: {results['message']}")
        return
    
    # Count statuses
    verified = []
    warnings = []
    not_found = []
    errors = []
    
    for variant, info in results.items():
        status = info.get('status', 'unknown')
        
        if status == 'verified':
            if 'is_code_dataset' in info and info['is_code_dataset']:
                verified.append(variant)
            else:
                warnings.append(variant)
        elif status == 'not_found' or status == 'no_checkpoint':
            not_found.append(variant)
        else:
            errors.append(variant)
    
    # Summary
    print("SUMMARY:")
    print(f"  ✓ Verified (trained on code): {len(verified)}")
    print(f"  ⚠  Warnings: {len(warnings)}")
    print(f"  ✗ Not found: {len(not_found)}")
    print(f"  ❌ Errors: {len(errors)}")
    print()
    
    # Details
    if verified:
        print("-"*100)
        print("VERIFIED MODELS (Trained on actual code):")
        print("-"*100)
        for variant in verified:
            info = results[variant]
            print(f"\n✓ {variant}")
            print(f"  Dataset: {info.get('dataset', 'unknown')}")
            if 'best_val_loss' in info:
                print(f"  Best val loss: {info['best_val_loss']:.4f}")
            if 'epochs_trained' in info:
                print(f"  Epochs trained: {info['epochs_trained']}")
            print(f"  Checkpoint: {info.get('checkpoint_path', 'N/A')}")
    
    if warnings:
        print("\n" + "-"*100)
        print("WARNINGS:")
        print("-"*100)
        for variant in warnings:
            info = results[variant]
            print(f"\n⚠  {variant}")
            print(f"  Warning: {info.get('warning', 'Unknown warning')}")
            print(f"  Dataset: {info.get('dataset', 'unknown')}")
    
    if not_found:
        print("\n" + "-"*100)
        print("NOT TRAINED:")
        print("-"*100)
        for variant in not_found:
            info = results[variant]
            print(f"\n✗ {variant}")
            print(f"  {info.get('message', 'No information available')}")
    
    if errors:
        print("\n" + "-"*100)
        print("ERRORS:")
        print("-"*100)
        for variant in errors:
            info = results[variant]
            print(f"\n❌ {variant}")
            print(f"  {info.get('message', 'Unknown error')}")
    
    print("\n" + "="*100)
    
    # Recommendations
    if not_found or errors or warnings:
        print("\nRECOMMENDATIONS:")
        print("-"*100)
        if not_found or errors:
            print("• Train missing variants using:")
            print("  python train_all_variants.py")
            print()
        if warnings:
            print("• Review models with warnings to ensure they were trained on appropriate data")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='Verify that SmallCoder models are trained on actual code data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify all models in default directory
  python verify_trained_models.py

  # Verify models in custom directory
  python verify_trained_models.py --model_dir ./my_models

  # Check a specific checkpoint
  python verify_trained_models.py --checkpoint ./trained_models/SmallCoder-Tiny/best_model.pt
        """
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./trained_models',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Specific checkpoint to verify'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # Verify single checkpoint
        result = check_checkpoint(args.checkpoint)
        
        if args.json:
            import json
            print(json.dumps(result, indent=2))
        else:
            print_verification_report({Path(args.checkpoint).stem: result})
    else:
        # Verify all models
        results = verify_trained_models(args.model_dir)
        
        if args.json:
            import json
            print(json.dumps(results, indent=2))
        else:
            print_verification_report(results)


if __name__ == "__main__":
    main()
