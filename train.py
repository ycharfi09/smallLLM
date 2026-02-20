"""
Training script for SmallCoder model
Optimized for limited hardware (8GB RAM, 2GB VRAM)
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

from model import SmallCoderConfig, SmallCoderForCausalLM, count_parameters


class CodeDataset(Dataset):
    """Dataset for code training"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=4,
                epoch=0, global_step=0, checkpoint_every=1000, drive_checkpoint_dir='/content/drive/MyDrive'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    accumulated_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss'] / gradient_accumulation_steps
        loss.backward()
        accumulated_loss += loss.item() * gradient_accumulation_steps
        
        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Auto-save checkpoint every N optimizer steps to Google Drive (or configured dir)
            if global_step % checkpoint_every == 0:
                avg_loss = accumulated_loss / gradient_accumulation_steps
                ckpt_path = os.path.join(drive_checkpoint_dir, f"smallcoder_ckpt_{global_step}.pt")
                try:
                    torch.save({
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, ckpt_path)
                    print(f"\n✓ Checkpoint saved at step {global_step} | loss: {avg_loss:.4f} → {ckpt_path}")
                except Exception as e:
                    print(f"\n⚠️  Failed to save checkpoint at step {global_step}: {e}")
                    print("Ensure the checkpoint directory is accessible (e.g. mount Google Drive on Colab).")
            accumulated_loss = 0.0
        
        total_loss += loss.item() * gradient_accumulation_steps
        pbar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
    
    return total_loss / len(dataloader), global_step


def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs['loss'].item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train SmallCoder model')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--tokenizer', type=str, default='codellama/CodeLlama-7b-hf', 
                        help='Tokenizer to use')
    parser.add_argument('--dataset', type=str, default='bigcode/the-stack-smol',
                        help='Dataset to use')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, 
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--drive_checkpoint_dir', type=str, default='/content/drive/MyDrive',
                        help='Directory for auto-save checkpoints (e.g. Google Drive on Colab)')
    parser.add_argument('--use_fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--allow_dummy_data', action='store_true', 
                        help='Allow fallback to dummy data if dataset loading fails (not recommended for production)')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    print("Creating model...")
    config = SmallCoderConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.max_length,
    )
    model = SmallCoderForCausalLM(config)
    print(f"Model parameters: {count_parameters(model):,} (~{count_parameters(model)/1e6:.1f}M)")
    
    model = model.to(device)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    try:
        dataset = load_dataset(args.dataset, split="train", streaming=True)
        # Take a subset for training (adjust as needed)
        train_data = []
        print("Processing code samples...")
        for i, item in enumerate(tqdm(dataset, desc="Loading dataset")):
            if i >= 20000:  # Limit to 20k examples for initial training
                break
            if 'content' in item:
                train_data.append({'text': item['content']})
            elif 'text' in item:
                train_data.append({'text': item['text']})
        
        if len(train_data) == 0:
            raise ValueError("No valid code samples found in dataset")
        
        print(f"✓ Successfully loaded {len(train_data):,} code samples from actual dataset")
        
        # Split train/val
        split_idx = int(0.95 * len(train_data))
        train_dataset = CodeDataset(train_data[:split_idx], tokenizer, args.max_length)
        val_dataset = CodeDataset(train_data[split_idx:], tokenizer, args.max_length)
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error loading dataset: {e}")
        
        if not args.allow_dummy_data:
            # Check for specific error about deprecated dataset scripts
            if "Dataset scripts are no longer supported" in error_msg:
                print("\n⚠️  DATASET SCRIPT ERROR DETECTED!")
                print(f"The dataset '{args.dataset}' uses a custom loading script which is no longer supported")
                print("in newer versions of the HuggingFace datasets library (v2.14.0+).")
                print("\n📋 RECOMMENDED ALTERNATIVE DATASETS:")
                print("  • bigcode/the-stack-smol (default, recommended)")
                print("  • bigcode/the-stack-dedup")
                print("\n✅ TO FIX: Run with a supported dataset:")
                print(f"   python train.py --dataset bigcode/the-stack-smol")
                print("\nFor more info: https://huggingface.co/docs/datasets/about_dataset_load")
            else:
                print("\n⚠️  CRITICAL: Unable to load actual code dataset!")
                print("This script requires real code data to train a usable coding model.")
                print("\nPossible solutions:")
                print("1. Check your internet connection")
                print("2. Verify the dataset name is correct")
                print("3. Try a different dataset: --dataset bigcode/the-stack-smol")
                print("4. If you want to use dummy data for testing, add --allow_dummy_data flag")
            
            print("\nExiting to prevent training on insufficient data.")
            raise SystemExit(1)
        
        print("\n⚠️  WARNING: Using dummy dataset for demonstration...")
        print("This is NOT suitable for training a usable coding model!")
        print("The model will not produce meaningful code completions.")
        print("Use --dataset with actual code data for production training.")
        
        # Create dummy dataset
        dummy_data = [
            {'text': 'def hello_world():\n    print("Hello, World!")'},
            {'text': 'class MyClass:\n    def __init__(self):\n        self.value = 0'},
            {'text': 'for i in range(10):\n    print(i)'},
        ] * 100
        train_dataset = CodeDataset(dummy_data[:80], tokenizer, args.max_length)
        val_dataset = CodeDataset(dummy_data[80:], tokenizer, args.max_length)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for limited memory
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
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_fp16 and device.type == 'cuda' else None
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.gradient_accumulation_steps,
            epoch=epoch, global_step=global_step, checkpoint_every=args.save_steps,
            drive_checkpoint_dir=args.drive_checkpoint_dir
        )
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        print(f"Val loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2)


if __name__ == "__main__":
    main()
