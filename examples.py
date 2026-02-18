"""
Example usage of SmallCoder model
Demonstrates basic functionality and API
"""

import torch
from transformers import AutoTokenizer
from model import SmallCoderConfig, SmallCoderForCausalLM, count_parameters


def example_1_create_model():
    """Example 1: Create and inspect model"""
    print("\n" + "="*80)
    print("Example 1: Creating SmallCoder Model")
    print("="*80 + "\n")
    
    # Create configuration
    config = SmallCoderConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4096,
    )
    
    # Create model
    model = SmallCoderForCausalLM(config)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: ~{count_parameters(model) / 1e6:.1f}M parameters")
    print(f"\nArchitecture details:")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Number of layers: {config.num_hidden_layers}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - Key-Value heads: {config.num_key_value_heads}")
    print(f"  - Max context length: {config.max_position_embeddings}")
    
    return model, config


def example_2_forward_pass(model):
    """Example 2: Forward pass"""
    print("\n" + "="*80)
    print("Example 2: Forward Pass")
    print("="*80 + "\n")
    
    # Create dummy input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 32000, (batch_size, seq_length))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Expected shape: [batch_size={batch_size}, seq_length={seq_length}, vocab_size=32000]")
    print("✓ Forward pass successful!")


def example_3_generation():
    """Example 3: Text generation"""
    print("\n" + "="*80)
    print("Example 3: Code Generation")
    print("="*80 + "\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        print("Note: Tokenizer download requires internet connection.")
        print("Using mock tokenizer for demonstration...")
        return
    
    # Create small model for quick demo
    config = SmallCoderConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = SmallCoderForCausalLM(config)
    model.eval()
    
    # Generate code
    prompt = "def hello_world():"
    print(f"Prompt: {prompt}")
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    print("Generating (this will produce random output since model is not trained)...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated:\n{generated_text}")
    print("\nNote: Output is random since model is untrained. Train the model to get meaningful results!")


def example_4_memory_efficient_config():
    """Example 4: Memory-efficient configurations"""
    print("\n" + "="*80)
    print("Example 4: Memory-Efficient Configurations")
    print("="*80 + "\n")
    
    configs = [
        {
            "name": "Tiny (100M params)",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "kv_heads": 3,
        },
        {
            "name": "Small (304M params) - Default",
            "hidden_size": 1152,
            "num_layers": 18,
            "num_heads": 16,
            "kv_heads": 4,
        },
        {
            "name": "Medium (700M params)",
            "hidden_size": 1280,
            "num_layers": 24,
            "num_heads": 20,
            "kv_heads": 5,
        },
    ]
    
    for cfg in configs:
        config = SmallCoderConfig(
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_layers"],
            num_attention_heads=cfg["num_heads"],
            num_key_value_heads=cfg["kv_heads"],
            intermediate_size=int(cfg["hidden_size"] * 2.89),
        )
        model = SmallCoderForCausalLM(config)
        params = count_parameters(model)
        
        print(f"{cfg['name']}:")
        print(f"  Parameters: {params:,} (~{params/1e6:.0f}M)")
        print(f"  Hidden: {cfg['hidden_size']}, Layers: {cfg['num_layers']}")
        print(f"  Attention heads: {cfg['num_heads']}, KV heads: {cfg['kv_heads']}")
        print(f"  Estimated VRAM (FP16): ~{params * 2 / 1e9:.2f}GB")
        print()


def example_5_training_tips():
    """Example 5: Training tips"""
    print("\n" + "="*80)
    print("Example 5: Training Tips for Limited Hardware")
    print("="*80 + "\n")
    
    tips = """
    For 2GB VRAM (e.g., NVIDIA MX450):
    
    1. Reduce batch size to 1 or 2
    2. Use gradient accumulation (16-32 steps) for effective larger batches
    3. Enable mixed precision training (FP16)
    4. Use gradient checkpointing to trade computation for memory
    5. Reduce sequence length to 256 or 512 tokens
    6. Consider using CPU offloading for optimizer states
    
    Example training command:
    ```bash
    python train.py \\
        --batch_size 1 \\
        --gradient_accumulation_steps 32 \\
        --max_length 256 \\
        --use_fp16 \\
        --gradient_checkpointing \\
        --num_epochs 3
    ```
    
    For inference optimization:
    
    1. Use INT8 quantization to reduce memory by 4x
    2. Enable KV caching for faster generation
    3. Reduce max_new_tokens for faster responses
    4. Consider batch size 1 for memory efficiency
    
    Example inference command:
    ```bash
    python inference.py \\
        --checkpoint model.pt \\
        --quantize \\
        --max_tokens 150 \\
        --interactive
    ```
    """
    
    print(tips)


def main():
    """Run all examples"""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*20 + "SmallCoder Usage Examples" + " "*33 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    # Run examples
    model, config = example_1_create_model()
    example_2_forward_pass(model)
    example_3_generation()
    example_4_memory_efficient_config()
    example_5_training_tips()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
