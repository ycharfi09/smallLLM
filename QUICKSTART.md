# SmallCoder Quick Start Guide

This guide will help you get started with SmallCoder in 5 minutes!

## ⚡ New! Ready-to-Use Pre-Trained Model

**Get started in 3 steps - no training required!**

### Step 1: Install Dependencies (1 minute)

```bash
# Install PyTorch and transformers
pip install torch transformers

# Or install all dependencies
pip install -r requirements.txt
```

### Step 2: Generate Pre-Trained Model (30 seconds)

```bash
# Creates pretrained_smallcoder.pt with initialized weights
python pretrained_model.py
```

### Step 3: Use the Model! (instantly)

```bash
# Interactive mode - just type your prompts!
python run_model.py --interactive

# Or single generation
python run_model.py --prompt "def fibonacci(n):"
```

**That's it!** 🎉 You now have a working 304M parameter coding assistant!

👉 **For detailed instructions, see [PRETRAINED_MODEL_GUIDE.md](PRETRAINED_MODEL_GUIDE.md)**

---

## Prerequisites

```bash
# Ensure you have Python 3.8+ and pip installed
python --version
pip --version
```

## Alternative: Training Your Own Model

If you want to train from scratch or customize the model:

```bash
# Clone the repository
git clone https://github.com/ycharfi09/smallLLM.git
cd smallLLM

# Install dependencies
pip install -r requirements.txt
```

## Testing the Model Architecture

```bash
# Verify the model architecture works
python model.py
```

Expected output:
```
Model created successfully!
Total parameters: 303,654,528
Model size: ~303.7M parameters
...
Model test passed!
```

## Running a Demo

```bash
# Run a comprehensive demo
python demo.py
```

This will:
- Create/load the pre-trained model
- Test forward passes
- Test generation capabilities
- Show memory usage estimates
- Provide next steps

## Using the Pre-Trained Model

### Interactive Mode (Recommended)

```bash
python run_model.py --interactive
```

Then just type your code prompts:
```
>>> def quicksort(arr):
>>> class Calculator:
>>> # Function to merge two sorted arrays
```

### Single Generation

```bash
python run_model.py --prompt "def fibonacci(n):"
```

### With Options

```bash
# Use quantization for lower memory
python run_model.py --interactive --quantize

# Generate more tokens
python run_model.py --prompt "class Tree:" --max_tokens 300

# Use CPU instead of GPU
python run_model.py --interactive --device cpu
```

## Using in Your Code

```python
import torch
from model import SmallCoderConfig, SmallCoderForCausalLM

# Load the pre-trained model
checkpoint = torch.load('pretrained_smallcoder.pt')
config = SmallCoderConfig(**checkpoint['config'])
model = SmallCoderForCausalLM(config)
model.load_state_dict(checkpoint['model_state_dict'])

# Use the model
model.eval()
input_ids = torch.randint(0, 32000, (1, 64))
outputs = model(input_ids=input_ids)
print(f"Logits shape: {outputs['logits'].shape}")
```

## Training Your Own Model (Optional)

If you want to fine-tune on your data:

```bash
# Train on a code dataset
python train.py \
    --output_dir ./my_model \
    --dataset bigcode/the-stack-smol \
    --batch_size 2 \
    --num_epochs 3
```

## Common Issues & Solutions

### Issue: "No module named 'torch'"

**Solution**: Install PyTorch
```bash
pip install torch transformers
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size or use quantization
```bash
python inference.py --checkpoint model.pt --quantize
```

### Issue: Model training too slow

**Solution**: Use gradient accumulation
```bash
python train.py --batch_size 1 --gradient_accumulation_steps 32
```

### Issue: No GPU available

**Solution**: Use CPU (slower but works)
```bash
python inference.py --checkpoint model.pt --device cpu
```

## Next Steps

1. **Read the README**: Full documentation in [README.md](README.md)
2. **Check examples**: More examples in [examples.py](examples.py)
3. **Explore scripts**:
   - `train.py` - Training script
   - `inference.py` - Inference and generation
   - `distill.py` - Knowledge distillation
   - `benchmark.py` - Evaluation benchmarks

## Need Help?

- Check [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- Open an issue on GitHub
- Read the code comments for detailed explanations

Happy coding! 🎉
