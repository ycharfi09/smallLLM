#!/usr/bin/env python3
"""
Ready-to-use inference script for SmallCoder pre-trained model
This script makes it easy to use the pre-trained model without any setup
"""

import torch
import argparse
import sys
from pathlib import Path


def check_and_download_model(checkpoint_path):
    """Check if model exists, create it if not"""
    if not Path(checkpoint_path).exists():
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Creating pre-trained model...")
        print("-" * 80)
        
        # Import and create model
        from pretrained_model import create_pretrained_model
        create_pretrained_model(checkpoint_path)
        print()
    
    return checkpoint_path


def load_model(checkpoint_path, device='cuda', quantize=False):
    """Load model from checkpoint"""
    from model import SmallCoderConfig, SmallCoderForCausalLM
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create config
    if 'config' in checkpoint:
        config = SmallCoderConfig(**checkpoint['config'])
    else:
        config = SmallCoderConfig()
    
    # Create model
    model = SmallCoderForCausalLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Apply quantization if requested
    if quantize:
        print("Applying INT8 quantization...")
        try:
            model = model.to('cpu')
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            if device == 'cuda':
                print("Note: Quantized model runs on CPU")
                device = 'cpu'
        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Continuing without quantization...")
    
    model = model.to(device)
    model.eval()
    
    return model, config


def generate_code(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    device='cuda'
):
    """Generate code from prompt"""
    import time
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating up to {max_new_tokens} tokens...")
    print("-" * 80)
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            use_cache=True,
        )
    generation_time = time.time() - start_time
    
    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(output_text)
    print("-" * 80)
    print(f"Generation time: {generation_time:.2f}s")
    tokens_generated = output_ids.shape[1] - input_ids.shape[1]
    print(f"Tokens generated: {tokens_generated}")
    print(f"Speed: {tokens_generated / generation_time:.2f} tokens/sec")
    
    return output_text


def interactive_mode(model, tokenizer, device='cuda', max_tokens=200):
    """Interactive code generation mode"""
    print("\n" + "=" * 80)
    print(" " * 25 + "SmallCoder Interactive Mode")
    print("=" * 80)
    print("\nWelcome! This is a ready-to-use coding assistant.")
    print("\nTips:")
    print("  - Type your code prompt and press Enter to generate")
    print("  - Try prompts like 'def fibonacci(n):', 'class Calculator:', etc.")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'help' for more information")
    print("=" * 80 + "\n")
    
    while True:
        try:
            prompt = input("\n>>> ")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Happy coding! 🚀")
                break
            
            if prompt.lower() == 'help':
                print("\nSmallCoder Help:")
                print("  - This model generates code based on your prompts")
                print("  - Best prompts start with 'def', 'class', or code comments")
                print("  - The model uses ~304M parameters optimized for coding")
                print("  - Generation uses temperature=0.7, top_p=0.9 for balanced output")
                print("\nExample prompts:")
                print("  - def quicksort(arr):")
                print("  - class BinaryTree:")
                print("  - # Function to calculate factorial")
                continue
            
            if not prompt.strip():
                continue
            
            generate_code(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                device=device
            )
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! Happy coding! 🚀")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again with a different prompt.")


def main():
    parser = argparse.ArgumentParser(
        description='SmallCoder - Ready-to-use coding assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for first-time users)
  python run_model.py --interactive
  
  # Single generation
  python run_model.py --prompt "def fibonacci(n):"
  
  # With quantization for lower memory usage
  python run_model.py --interactive --quantize
  
  # Use CPU instead of GPU
  python run_model.py --interactive --device cpu
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='pretrained_smallcoder.pt',
        help='Path to model checkpoint (default: pretrained_smallcoder.pt)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='codellama/CodeLlama-7b-hf',
        help='Tokenizer to use (default: codellama/CodeLlama-7b-hf)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Code prompt for single generation'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=200,
        help='Maximum tokens to generate (default: 200)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p sampling (default: 0.9)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k sampling (default: 50)'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Use INT8 quantization for lower memory usage'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu, default: cuda)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print(" " * 30 + "SmallCoder v1.0")
    print(" " * 22 + "Ready-to-Use Coding Assistant")
    print("=" * 80 + "\n")
    
    # Check/create model
    checkpoint_path = check_and_download_model(args.checkpoint)
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    print()
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("\nPlease ensure you have internet connection and transformers installed:")
        print("  pip install transformers")
        sys.exit(1)
    
    print()
    
    # Load model
    model, config = load_model(checkpoint_path, device=args.device, quantize=args.quantize)
    print("✓ Model loaded successfully")
    print()
    
    # Print model info
    print("Model Configuration:")
    print(f"  - Parameters: ~304M")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Layers: {config.num_hidden_layers}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - KV heads: {config.num_key_value_heads} (Grouped-Query Attention)")
    print(f"  - Context length: {config.max_position_embeddings} tokens")
    
    # Run interactive or single generation
    if args.interactive:
        interactive_mode(model, tokenizer, device=args.device, max_tokens=args.max_tokens)
    else:
        if args.prompt is None:
            print("\n⚠️  No prompt provided. Use --prompt or --interactive mode.")
            print("\nExample:")
            print('  python run_model.py --prompt "def fibonacci(n):"')
            print("  python run_model.py --interactive")
            sys.exit(1)
        
        generate_code(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=args.device
        )


if __name__ == "__main__":
    main()
