"""
Knowledge Distillation script for SmallCoder
Train SmallCoder by distilling knowledge from a larger teacher model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
from pathlib import Path

from model import SmallCoderConfig, SmallCoderForCausalLM, count_parameters
from train import CodeDataset


class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation"""
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        # Knowledge distillation loss
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        kd_loss = self.kl_div(student_probs, teacher_probs) * (self.temperature ** 2)
        
        # Combine losses
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        
        return total_loss, ce_loss, kd_loss


def distill_epoch(student, teacher, dataloader, optimizer, loss_fn, device):
    """Distillation training for one epoch"""
    student.train()
    teacher.eval()
    
    total_loss = 0
    total_ce_loss = 0
    total_kd_loss = 0
    
    pbar = tqdm(dataloader, desc="Distilling")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits
        
        # Get student outputs
        student_outputs = student(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        student_logits = student_outputs['logits']
        
        # Compute distillation loss
        loss, ce_loss, kd_loss = loss_fn(
            student_logits[:, :-1, :].contiguous(),
            teacher_logits[:, :-1, :].contiguous(),
            labels[:, 1:].contiguous()
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_kd_loss += kd_loss.item()
        
        pbar.set_postfix({
            'loss': loss.item(),
            'ce': ce_loss.item(),
            'kd': kd_loss.item()
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_kd_loss = total_kd_loss / len(dataloader)
    
    return avg_loss, avg_ce_loss, avg_kd_loss


def main():
    parser = argparse.ArgumentParser(description='Distill SmallCoder from larger model')
    parser.add_argument('--teacher_model', type=str, default='codellama/CodeLlama-7b-hf',
                        help='Teacher model name or path')
    parser.add_argument('--output_dir', type=str, default='./distilled_model',
                        help='Output directory')
    parser.add_argument('--tokenizer', type=str, default='codellama/CodeLlama-7b-hf',
                        help='Tokenizer to use')
    parser.add_argument('--dataset', type=str, default='bigcode/the-stack-smol',
                        help='Dataset to use')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for CE loss vs KD loss')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Distillation temperature')
    parser.add_argument('--allow_dummy_data', action='store_true',
                        help='Allow fallback to dummy data if dataset loading fails (not recommended)')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='Maximum number of samples to load from dataset')
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
    
    # Load teacher model
    print(f"\nLoading teacher model: {args.teacher_model}")
    print("This may take a while for large models...")
    try:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.float16,
            device_map='auto',
            load_in_8bit=True,  # Use 8-bit to save memory
        )
        teacher.eval()
        print("Teacher model loaded successfully!")
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        print("Note: This requires sufficient memory and internet connection.")
        print("For demonstration, continuing without teacher model...")
        return
    
    # Create student model
    print("\nCreating student model (SmallCoder)...")
    config = SmallCoderConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.max_length,
    )
    student = SmallCoderForCausalLM(config)
    print(f"Student parameters: {count_parameters(student):,} (~{count_parameters(student)/1e6:.1f}M)")
    student = student.to(device)
    
    # Load actual code dataset
    print("\nPreparing dataset...")
    try:
        from datasets import load_dataset as hf_load_dataset
        print(f"Loading code dataset: {args.dataset}")
        hf_dataset = hf_load_dataset(args.dataset, split="train", streaming=True)
        
        code_data = []
        for i, item in enumerate(tqdm(hf_dataset, desc="Loading code samples", total=args.max_samples)):
            if i >= args.max_samples:
                break
            
            text = None
            if 'content' in item:
                text = item['content']
            elif 'text' in item:
                text = item['text']
            
            if text and len(text.strip()) > 50:
                code_data.append({'text': text})
        
        if len(code_data) == 0:
            raise ValueError("No valid code samples found")
        
        print(f"✓ Loaded {len(code_data):,} code samples")
        dataset = CodeDataset(code_data, tokenizer, args.max_length)
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error loading code dataset: {e}")
        
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
                print(f"   python distill.py --dataset bigcode/the-stack-smol")
                print("\nFor more info: https://huggingface.co/docs/datasets/about_dataset_load")
            else:
                print("\n⚠️  CRITICAL: Unable to load actual code dataset!")
                print("Knowledge distillation requires real code data for effective learning.")
                print("\nPossible solutions:")
                print("1. Check your internet connection")
                print("2. Verify the dataset name is correct")
                print("3. Use --allow_dummy_data flag for testing (not recommended)")
            raise SystemExit(1)
        
        print("\n⚠️  WARNING: Using dummy dataset for demonstration...")
        print("This is NOT suitable for production distillation!")
        dummy_data = [
            {'text': 'def hello_world():\n    print("Hello, World!")'},
            {'text': 'class MyClass:\n    def __init__(self):\n        self.value = 0'},
        ] * 50
        dataset = CodeDataset(dummy_data, tokenizer, args.max_length)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Setup training
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    loss_fn = DistillationLoss(alpha=args.alpha, temperature=args.temperature)
    
    # Training loop
    print("\nStarting distillation training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        avg_loss, avg_ce, avg_kd = distill_epoch(
            student, teacher, dataloader, optimizer, loss_fn, device
        )
        
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"  CE Loss: {avg_ce:.4f}")
        print(f"  KD Loss: {avg_kd:.4f}")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"distilled_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config.__dict__,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\nDistillation complete!")


if __name__ == "__main__":
    main()
