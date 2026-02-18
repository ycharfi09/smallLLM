"""
Inference script for SmallCoder model
Supports quantization for efficient inference on limited hardware
"""

import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer
import time

from model import SmallCoderConfig, SmallCoderForCausalLM


def load_model(checkpoint_path, device='cuda', quantize=False):
    """Load model from checkpoint"""
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
    model = model.to(device)
    model.eval()
    
    # Apply quantization if requested
    if quantize and device == 'cuda':
        print("Applying INT8 quantization...")
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Continuing without quantization...")
    
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
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens...\n")
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
    print(f"\nGeneration time: {generation_time:.2f}s")
    print(f"Tokens per second: {max_new_tokens / generation_time:.2f}")
    
    return output_text


def interactive_mode(model, tokenizer, device='cuda'):
    """Interactive code generation mode"""
    print("\n" + "=" * 80)
    print("SmallCoder Interactive Mode")
    print("=" * 80)
    print("Type your code prompt and press Enter to generate.")
    print("Type 'quit' or 'exit' to exit.")
    print("=" * 80 + "\n")
    
    while True:
        try:
            prompt = input("\n>>> ")
            if prompt.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if not prompt.strip():
                continue
            
            generate_code(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                device=device
            )
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='SmallCoder Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='codellama/CodeLlama-7b-hf',
                        help='Tokenizer to use')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=200, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--quantize', action='store_true', help='Use INT8 quantization')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model, config = load_model(args.checkpoint, device=args.device, quantize=args.quantize)
    
    # Print model info
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Max sequence length: {config.max_position_embeddings}")
    
    # Interactive or single generation
    if args.interactive:
        interactive_mode(model, tokenizer, device=args.device)
    else:
        if args.prompt is None:
            args.prompt = "def fibonacci(n):"
        
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
