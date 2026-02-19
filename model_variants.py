"""
SmallCoder Model Variants Configuration
Provides different model sizes for various use cases and hardware constraints
"""

from dataclasses import dataclass
from typing import Optional
from model import SmallCoderConfig


# Model variant configurations
MODEL_VARIANTS = {
    "SmallCoder-Tiny": {
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "max_position_embeddings": 2048,
        "description": "Ultra-compact model (~85M params) for minimal hardware",
        "estimated_params_m": 85,
        "recommended_vram_gb": 0.5,
        "recommended_ram_gb": 4,
    },
    "SmallCoder-Small": {
        "hidden_size": 960,
        "intermediate_size": 2688,
        "num_hidden_layers": 16,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
        "description": "Compact model (~180M params) for resource-constrained environments",
        "estimated_params_m": 180,
        "recommended_vram_gb": 1.0,
        "recommended_ram_gb": 6,
    },
    "SmallCoder-Medium": {
        "hidden_size": 1152,
        "intermediate_size": 3328,
        "num_hidden_layers": 18,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
        "description": "Balanced model (~304M params) - original SmallCoder",
        "estimated_params_m": 304,
        "recommended_vram_gb": 2.0,
        "recommended_ram_gb": 8,
    },
    "SmallCoder-Tiny-LC": {
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "max_position_embeddings": 8192,
        "description": "Long context variant of Tiny (~85M params) with 8K context",
        "estimated_params_m": 85,
        "recommended_vram_gb": 0.8,
        "recommended_ram_gb": 4,
    },
    "SmallCoder-Small-LC": {
        "hidden_size": 960,
        "intermediate_size": 2688,
        "num_hidden_layers": 16,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "max_position_embeddings": 8192,
        "description": "Long context variant of Small (~180M params) with 8K context",
        "estimated_params_m": 180,
        "recommended_vram_gb": 1.5,
        "recommended_ram_gb": 6,
    },
    "SmallCoder-Medium-LC": {
        "hidden_size": 1152,
        "intermediate_size": 3328,
        "num_hidden_layers": 18,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 8192,
        "description": "Long context variant of Medium (~304M params) with 8K context",
        "estimated_params_m": 304,
        "recommended_vram_gb": 2.5,
        "recommended_ram_gb": 8,
    },
}


def get_variant_config(variant_name: str) -> SmallCoderConfig:
    """
    Get a SmallCoderConfig for a specific model variant
    
    Args:
        variant_name: Name of the variant (e.g., "SmallCoder-Tiny", "SmallCoder-Small-LC")
    
    Returns:
        SmallCoderConfig configured for the specified variant
    
    Raises:
        ValueError: If variant_name is not recognized
    """
    if variant_name not in MODEL_VARIANTS:
        available = ", ".join(MODEL_VARIANTS.keys())
        raise ValueError(f"Unknown variant '{variant_name}'. Available: {available}")
    
    variant = MODEL_VARIANTS[variant_name]
    
    # Create config with variant-specific parameters
    config = SmallCoderConfig(
        hidden_size=variant["hidden_size"],
        intermediate_size=variant["intermediate_size"],
        num_hidden_layers=variant["num_hidden_layers"],
        num_attention_heads=variant["num_attention_heads"],
        num_key_value_heads=variant["num_key_value_heads"],
        max_position_embeddings=variant["max_position_embeddings"],
    )
    
    return config


def list_variants():
    """Print all available model variants with their specifications"""
    print("\n" + "="*100)
    print(" " * 35 + "SmallCoder Model Variants")
    print("="*100)
    print()
    
    # Standard variants
    print("STANDARD VARIANTS (4K context):")
    print("-" * 100)
    for name in ["SmallCoder-Tiny", "SmallCoder-Small", "SmallCoder-Medium"]:
        variant = MODEL_VARIANTS[name]
        print(f"\n{name}:")
        print(f"  Description: {variant['description']}")
        print(f"  Parameters: ~{variant['estimated_params_m']}M")
        print(f"  Context: {variant['max_position_embeddings']} tokens")
        print(f"  Recommended VRAM: {variant['recommended_vram_gb']}GB")
        print(f"  Recommended RAM: {variant['recommended_ram_gb']}GB")
        print(f"  Hidden Size: {variant['hidden_size']}, Layers: {variant['num_hidden_layers']}")
    
    # Long context variants
    print("\n" + "-" * 100)
    print("\nLONG CONTEXT VARIANTS (8K context):")
    print("-" * 100)
    for name in ["SmallCoder-Tiny-LC", "SmallCoder-Small-LC", "SmallCoder-Medium-LC"]:
        variant = MODEL_VARIANTS[name]
        print(f"\n{name}:")
        print(f"  Description: {variant['description']}")
        print(f"  Parameters: ~{variant['estimated_params_m']}M")
        print(f"  Context: {variant['max_position_embeddings']} tokens")
        print(f"  Recommended VRAM: {variant['recommended_vram_gb']}GB")
        print(f"  Recommended RAM: {variant['recommended_ram_gb']}GB")
        print(f"  Hidden Size: {variant['hidden_size']}, Layers: {variant['num_hidden_layers']}")
    
    print("\n" + "="*100)
    print()


def get_variant_info(variant_name: str) -> dict:
    """Get detailed information about a specific variant"""
    if variant_name not in MODEL_VARIANTS:
        available = ", ".join(MODEL_VARIANTS.keys())
        raise ValueError(f"Unknown variant '{variant_name}'. Available: {available}")
    
    return MODEL_VARIANTS[variant_name].copy()


if __name__ == "__main__":
    # Demo: List all variants
    list_variants()
    
    # Demo: Create configs for each variant
    print("\nCreating model configurations:")
    print("-" * 100)
    for variant_name in MODEL_VARIANTS.keys():
        config = get_variant_config(variant_name)
        print(f"{variant_name}: {config.num_hidden_layers} layers, "
              f"{config.hidden_size} hidden size, "
              f"{config.max_position_embeddings} max tokens")
