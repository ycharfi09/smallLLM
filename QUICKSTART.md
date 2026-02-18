# SmallCoder Quick Start Guide

This guide will help you get started with SmallCoder in 5 minutes!

## Prerequisites

```bash
# Ensure you have Python 3.8+ and pip installed
python --version
pip --version
```

## Step 1: Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/ycharfi09/smallLLM.git
cd smallLLM

# Install core dependencies
pip install torch transformers tokenizers

# Or install all dependencies
pip install -r requirements.txt
```

## Step 2: Test the Model (1 minute)

```bash
# Verify the model works
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

## Step 3: Run Examples (2 minutes)

```bash
# See usage examples
python examples.py
```

This will show you:
- How to create a model
- How to run forward passes
- Different model configurations
- Training and inference tips

## Step 4: Start Coding! 🚀

### Option A: Use Pre-trained Model (Recommended)

```bash
# Download or use a pre-trained checkpoint
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --tokenizer codellama/CodeLlama-7b-hf \
    --interactive
```

### Option B: Train Your Own Model

```bash
# Train on a code dataset
python train.py \
    --output_dir ./my_model \
    --dataset bigcode/the-stack-smol \
    --batch_size 2 \
    --num_epochs 3
```

### Option C: Use in Your Code

```python
import torch
from model import SmallCoderConfig, SmallCoderForCausalLM

# Create model
config = SmallCoderConfig()
model = SmallCoderForCausalLM(config)

# Use the model
input_ids = torch.randint(0, 32000, (1, 64))
outputs = model(input_ids=input_ids)
print(f"Logits shape: {outputs['logits'].shape}")
```

## Common Issues & Solutions

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
