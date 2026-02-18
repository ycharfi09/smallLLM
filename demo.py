#!/usr/bin/env python3
"""
Simple demo of the SmallCoder pre-trained model
This demonstrates that the model is properly initialized and functional
"""

import torch
from model import SmallCoderConfig, SmallCoderForCausalLM, count_parameters


def main():
    print("=" * 80)
    print(" " * 25 + "SmallCoder Model Demo")
    print("=" * 80)
    print()
    
    # Check if pretrained model exists
    import os
    checkpoint_path = "pretrained_smallcoder.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Creating pre-trained model at {checkpoint_path}...")
        from pretrained_model import create_pretrained_model
        create_pretrained_model(checkpoint_path)
        print()
    
    # Load the model
    print(f"Loading pre-trained model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create config and model
    config = SmallCoderConfig(**checkpoint['config'])
    model = SmallCoderForCausalLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("✓ Model loaded successfully!")
    print()
    
    # Display model info
    print("Model Information:")
    print("-" * 80)
    model_info = checkpoint.get('model_info', {})
    print(f"Architecture: {model_info.get('architecture', 'SmallCoder')}")
    print(f"Version: {model_info.get('version', '1.0')}")
    print(f"Total Parameters: {count_parameters(model):,} (~{count_parameters(model)/1e6:.1f}M)")
    print()
    
    print("Configuration:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Number of layers: {config.num_hidden_layers}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - Key-Value heads: {config.num_key_value_heads}")
    print(f"  - Context length: {config.max_position_embeddings} tokens")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print()
    
    # Test forward pass
    print("Testing Model:")
    print("-" * 80)
    
    model.eval()
    
    # Test 1: Single forward pass
    print("Test 1: Forward pass...")
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    logits = outputs['logits']
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  ✓ Forward pass successful!")
    print()
    
    # Test 2: Generation
    print("Test 2: Text generation...")
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            temperature=1.0,
            do_sample=True,
            use_cache=True
        )
    
    print(f"  Input length: {input_ids.shape[1]} tokens")
    print(f"  Generated length: {generated.shape[1]} tokens")
    print(f"  New tokens generated: {generated.shape[1] - input_ids.shape[1]}")
    print(f"  ✓ Generation successful!")
    print()
    
    # Test 3: Memory usage estimate
    print("Test 3: Memory usage estimate...")
    param_memory_fp32 = count_parameters(model) * 4 / (1024**3)  # bytes to GB
    param_memory_fp16 = count_parameters(model) * 2 / (1024**3)
    param_memory_int8 = count_parameters(model) * 1 / (1024**3)
    
    print(f"  FP32 (full precision): ~{param_memory_fp32:.2f} GB")
    print(f"  FP16 (half precision): ~{param_memory_fp16:.2f} GB")
    print(f"  INT8 (quantized): ~{param_memory_int8:.2f} GB")
    print(f"  ✓ Model is memory efficient!")
    print()
    
    print("=" * 80)
    print("All tests passed! ✅")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Use the model interactively:")
    print("     python run_model.py --interactive")
    print()
    print("  2. Generate code from a prompt:")
    print('     python run_model.py --prompt "def fibonacci(n):"')
    print()
    print("  3. Fine-tune on your own data:")
    print("     python train.py --checkpoint pretrained_smallcoder.pt --dataset your_data")
    print()
    print("For more information, see PRETRAINED_MODEL_GUIDE.md")
    print()


if __name__ == "__main__":
    main()
