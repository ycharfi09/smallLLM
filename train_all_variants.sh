#!/bin/bash
# Example script showing how to train all SmallCoder variants on actual code
# This is a reference example - adjust parameters based on your hardware

echo "========================================"
echo "SmallCoder: Train All Variants on Code"
echo "========================================"
echo ""
echo "This script will train all 6 model variants on actual code data"
echo "from the bigcode/the-stack-smol dataset."
echo ""
echo "Requirements:"
echo "  - Internet connection (to download dataset and tokenizer)"
echo "  - GPU with 2GB+ VRAM (recommended) or CPU (slower)"
echo "  - 8GB+ RAM"
echo "  - ~10-20 hours training time (varies by hardware)"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import torch; import transformers; import datasets" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required packages not installed."
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi
echo "✓ Dependencies installed"
echo ""

# Option 1: Train all variants (recommended for production)
echo "Option 1: Train ALL variants (full training)"
echo "-------------------------------------------"
echo "Command:"
echo "  python train_all_variants.py \\"
echo "    --dataset bigcode/the-stack-smol \\"
echo "    --max-samples 100000 \\"
echo "    --num-epochs 3 \\"
echo "    --batch_size 2 \\"
echo "    --gradient_accumulation_steps 16 \\"
echo "    --use_fp16"
echo ""
echo "This will train all 6 variants and save them in ./trained_models/"
echo "Estimated time: 10-20 hours on GPU, longer on CPU"
echo ""

# Option 2: Quick test training
echo "Option 2: Quick test training (for validation)"
echo "----------------------------------------------"
echo "Command:"
echo "  python train_all_variants.py \\"
echo "    --variants SmallCoder-Tiny \\"
echo "    --max-samples 10000 \\"
echo "    --num-epochs 1"
echo ""
echo "This will quickly train just the Tiny variant for testing"
echo "Estimated time: 30-60 minutes"
echo ""

# Option 3: Train specific variants
echo "Option 3: Train specific variants"
echo "---------------------------------"
echo "Command:"
echo "  python train_all_variants.py \\"
echo "    --variants SmallCoder-Small SmallCoder-Medium \\"
echo "    --max-samples 50000 \\"
echo "    --num-epochs 2"
echo ""
echo "This trains only selected variants"
echo ""

# Prompt user to choose
echo "========================================"
echo "Ready to start training?"
echo "========================================"
echo ""
echo "To start training, run one of the commands above."
echo ""
echo "For full documentation, see TRAINING_GUIDE.md"
echo ""

# Note: Uncomment the line below to actually run training
# Adjust the command based on your needs

# Example: Quick test training (uncomment to run)
# python train_all_variants.py --variants SmallCoder-Tiny --max-samples 10000 --num-epochs 1

# Example: Full training (uncomment to run)
# python train_all_variants.py --use_fp16
