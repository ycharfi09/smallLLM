"""
Train all SmallCoder model variants on actual code data
This script systematically trains all 6 model variants (Tiny, Small, Medium + LC versions)
on real code datasets to make them usable for coding tasks.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
from pathlib import Path
from datetime import datetime

from model import SmallCoderForCausalLM, count_parameters
from model_variants import get_variant_config, MODEL_VARIANTS
from train import CodeDataset, train_epoch, evaluate


def load_code_dataset(dataset_name, max_samples=100000, streaming=True):
    """
    Load actual code dataset from HuggingFace
    
    Args:
        dataset_name: Name of the dataset (e.g., 'bigcode/the-stack-smol')
        max_samples: Maximum number of samples to load
        streaming: Whether to use streaming mode
    
    Returns:
        List of dictionaries with 'text' key containing code samples
    """
    print(f"\nLoading dataset: {dataset_name}")
    print(f"Maximum samples: {max_samples:,}")
    print("This may take a few minutes depending on your internet connection...")
    
    try:
        dataset = load_dataset(dataset_name, split="train", streaming=streaming)
        
        train_data = []
        print("Processing dataset samples...")
        for i, item in enumerate(tqdm(dataset, desc="Loading", total=max_samples)):
            if i >= max_samples:
                break
            
            # Extract text from different possible keys
            text = None
            if 'content' in item:
                text = item['content']
            elif 'text' in item:
                text = item['text']
            elif 'code' in item:
                text = item['code']
            
            if text and len(text.strip()) > 50:  # Filter out very short samples
                train_data.append({'text': text})
        
        if len(train_data) == 0:
            raise ValueError("No valid code samples found in dataset")
        
        print(f"✓ Successfully loaded {len(train_data):,} code samples")
        return train_data
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error loading dataset: {e}")
        
        # Check for specific error about deprecated dataset scripts
        if "Dataset scripts are no longer supported" in error_msg:
            print("\n⚠️  DATASET SCRIPT ERROR DETECTED!")
            print(f"The dataset '{dataset_name}' uses a custom loading script which is no longer supported")
            print("in newer versions of the HuggingFace datasets library (v2.14.0+).")
            print("\n📋 RECOMMENDED ALTERNATIVE DATASETS:")
            print("  • bigcode/the-stack-smol (default, recommended)")
            print("  • bigcode/the-stack-dedup")
            print("  • codeparrot/github-code-clean (if available)")
            print("\n✅ TO FIX: Run with a supported dataset:")
            print(f"   python {os.path.basename(__file__)} --dataset bigcode/the-stack-smol")
            print("\nFor more info: https://huggingface.co/docs/datasets/about_dataset_load")
        else:
            print("\n⚠️  CRITICAL: Unable to load actual code dataset!")
            print("This script requires real code data to train the models.")
            print("\nPossible solutions:")
            print("1. Check your internet connection")
            print("2. Verify the dataset name is correct")
            print("3. Try a different dataset with --dataset flag")
            print("   Example: --dataset bigcode/the-stack-smol")
        
        print("\nRefusing to train on dummy data.")
        raise RuntimeError("Cannot proceed without actual code dataset")


def train_variant(
    variant_name,
    train_data,
    tokenizer,
    output_base_dir,
    args,
    device
):
    """
    Train a single model variant
    
    Args:
        variant_name: Name of the variant (e.g., 'SmallCoder-Tiny')
        train_data: List of code samples
        tokenizer: Tokenizer instance
        output_base_dir: Base output directory
        args: Training arguments
        device: Device to train on
    
    Returns:
        Path to the saved model checkpoint
    """
    print("\n" + "="*100)
    print(f"Training {variant_name}")
    print("="*100)
    
    # Get variant config
    config = get_variant_config(variant_name)
    config.vocab_size = len(tokenizer)
    
    # Create model
    print(f"\nCreating model...")
    model = SmallCoderForCausalLM(config)
    params = count_parameters(model)
    print(f"Model parameters: {params:,} (~{params/1e6:.1f}M)")
    print(f"Architecture: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
    print(f"Context length: {config.max_position_embeddings} tokens")
    
    model = model.to(device)
    
    # Prepare dataset
    print(f"\nPreparing training and validation datasets...")
    split_idx = int(0.95 * len(train_data))
    train_dataset = CodeDataset(train_data[:split_idx], tokenizer, args.max_length)
    val_dataset = CodeDataset(train_data[split_idx:], tokenizer, args.max_length)
    
    print(f"Train dataset size: {len(train_dataset):,}")
    print(f"Val dataset size: {len(val_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # Create output directory for this variant
    variant_dir = output_base_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\n{'-'*100}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'-'*100}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.gradient_accumulation_steps
        )
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        print(f"Val loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = variant_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__,
                'variant_name': variant_name,
                'trained_on': args.dataset,
            }, checkpoint_path)
            print(f"✓ Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_epochs == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = variant_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__,
                'variant_name': variant_name,
                'trained_on': args.dataset,
            }, checkpoint_path)
            print(f"✓ Saved checkpoint to {checkpoint_path}")
    
    # Save final model with clear naming
    final_path = variant_dir / f"{variant_name}_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'variant_name': variant_name,
        'trained_on': args.dataset,
        'best_val_loss': best_val_loss,
        'total_epochs': args.num_epochs,
    }, final_path)
    print(f"\n✓ Training complete for {variant_name}!")
    print(f"✓ Final model saved to {final_path}")
    print(f"✓ Best validation loss: {best_val_loss:.4f}")
    
    # Save training info
    info = {
        'variant_name': variant_name,
        'parameters': params,
        'parameters_m': round(params / 1e6, 1),
        'dataset': args.dataset,
        'num_epochs': args.num_epochs,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_loss,
        'config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_heads': config.num_attention_heads,
            'num_kv_heads': config.num_key_value_heads,
            'max_context': config.max_position_embeddings,
        },
        'trained_at': datetime.now().isoformat(),
    }
    
    with open(variant_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    # Clean up
    del model
    del optimizer
    del scheduler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return final_path, best_val_loss


def main():
    parser = argparse.ArgumentParser(
        description='Train all SmallCoder model variants on actual code data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all variants with default settings
  python train_all_variants.py

  # Train specific variants only
  python train_all_variants.py --variants SmallCoder-Tiny SmallCoder-Small

  # Use a different dataset (bigcode/the-stack-dedup recommended)
  python train_all_variants.py --dataset bigcode/the-stack-dedup

  # Quick training for testing (fewer samples, fewer epochs)
  python train_all_variants.py --max-samples 10000 --num-epochs 1

  # Resume training for a specific variant
  python train_all_variants.py --variants SmallCoder-Medium --resume
        """
    )
    
    parser.add_argument(
        '--variants',
        nargs='+',
        default=None,
        help='Specific variants to train (default: all variants). '
             'Example: --variants SmallCoder-Tiny SmallCoder-Small'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./trained_models',
        help='Base output directory for all models'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='codellama/CodeLlama-7b-hf',
        help='Tokenizer to use'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='bigcode/the-stack-smol',
        help='Dataset to use (must be a real code dataset)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100000,
        help='Maximum number of code samples to load (default: 100,000)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size per device'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=16,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of epochs per variant'
    )
    parser.add_argument(
        '--save_epochs',
        type=int,
        default=5,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--use_fp16',
        action='store_true',
        help='Use mixed precision training'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from existing checkpoints'
    )
    parser.add_argument(
        '--list-variants',
        action='store_true',
        help='List all available variants and exit'
    )
    parser.add_argument(
        '--skip-val',
        action='store_true',
        help='Skip validation during training (faster but no progress tracking)'
    )
    
    args = parser.parse_args()
    
    # List variants
    if args.list_variants:
        from model_variants import list_variants
        list_variants()
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*100)
    print(" " * 30 + "SmallCoder Multi-Variant Training")
    print("="*100)
    print(f"\nUsing device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  WARNING: Training on CPU will be very slow!")
        print("Consider using a GPU for better performance.")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer):,})")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("Make sure you have internet connection for first-time download.")
        return
    
    # Load code dataset - THIS IS CRITICAL
    # We MUST use actual code data, not dummy data
    train_data = load_code_dataset(
        args.dataset,
        max_samples=args.max_samples,
        streaming=True
    )
    
    # Get variants to train
    if args.variants:
        variants_to_train = args.variants
        # Validate variants
        for v in variants_to_train:
            if v not in MODEL_VARIANTS:
                print(f"❌ Error: Unknown variant '{v}'")
                print(f"Available variants: {', '.join(MODEL_VARIANTS.keys())}")
                return
    else:
        # Train all variants by default
        variants_to_train = list(MODEL_VARIANTS.keys())
    
    print(f"\n{'='*100}")
    print(f"Will train {len(variants_to_train)} variant(s):")
    for v in variants_to_train:
        variant_info = MODEL_VARIANTS[v]
        print(f"  • {v}: {variant_info['estimated_params_m']}M params, "
              f"{variant_info['max_position_embeddings']} context")
    print(f"{'='*100}")
    
    # Train each variant
    results = []
    for i, variant_name in enumerate(variants_to_train, 1):
        print(f"\n\n{'#'*100}")
        print(f"# Training variant {i}/{len(variants_to_train)}: {variant_name}")
        print(f"{'#'*100}")
        
        try:
            checkpoint_path, best_val_loss = train_variant(
                variant_name=variant_name,
                train_data=train_data,
                tokenizer=tokenizer,
                output_base_dir=output_dir,
                args=args,
                device=device
            )
            
            results.append({
                'variant': variant_name,
                'status': 'success',
                'checkpoint': str(checkpoint_path),
                'best_val_loss': best_val_loss,
            })
            
        except Exception as e:
            print(f"\n❌ Error training {variant_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'variant': variant_name,
                'status': 'failed',
                'error': str(e),
            })
            
            # Ask if user wants to continue
            if i < len(variants_to_train):
                print(f"\nContinuing with next variant...")
    
    # Print summary
    print("\n\n" + "="*100)
    print(" " * 35 + "TRAINING SUMMARY")
    print("="*100)
    print()
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Total variants: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()
    
    if successful:
        print("Successfully trained variants:")
        print("-" * 100)
        for r in successful:
            print(f"  ✓ {r['variant']}")
            print(f"    Checkpoint: {r['checkpoint']}")
            print(f"    Best val loss: {r['best_val_loss']:.4f}")
    
    if failed:
        print("\nFailed variants:")
        print("-" * 100)
        for r in failed:
            print(f"  ✗ {r['variant']}: {r['error']}")
    
    # Save overall summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'results': results,
            'dataset': args.dataset,
            'num_samples': len(train_data),
            'num_epochs': args.num_epochs,
            'timestamp': datetime.now().isoformat(),
            'args': vars(args),
        }, f, indent=2)
    
    print(f"\n✓ Training summary saved to {summary_path}")
    print("\n" + "="*100)
    print("Training complete!")
    print("="*100)
    
    # Print usage instructions
    if successful:
        print("\nYou can now use the trained models:")
        print("-" * 100)
        for r in successful:
            variant_name = r['variant']
            checkpoint = r['checkpoint']
            print(f"\n# Use {variant_name}:")
            print(f"python inference.py --checkpoint {checkpoint} --tokenizer {args.tokenizer} --interactive")
            print(f"# Or with CLI agent:")
            print(f"python cli_agent.py --variant {variant_name} --checkpoint {checkpoint} --interactive")


if __name__ == "__main__":
    main()
