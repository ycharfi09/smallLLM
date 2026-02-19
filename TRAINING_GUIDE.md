# SmallCoder Training Guide 🎓

This guide explains how to train SmallCoder models on actual code datasets to make them usable for coding tasks.

## 🎯 Overview

SmallCoder provides **6 model variants** that need to be trained on real code data:

### Standard Context Variants (2K-4K tokens)
- **SmallCoder-Tiny**: ~100M params, 2K context
- **SmallCoder-Small**: ~194M params, 4K context  
- **SmallCoder-Medium**: ~304M params, 4K context

### Long Context Variants (8K tokens)
- **SmallCoder-Tiny-LC**: ~100M params, 8K context
- **SmallCoder-Small-LC**: ~194M params, 8K context
- **SmallCoder-Medium-LC**: ~304M params, 8K context

## 🚀 Quick Start: Train All Variants

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify CUDA is available (recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Train All Variants (Recommended)

The easiest way to train all model variants on actual code:

```bash
# Train all 6 variants on the default code dataset
python train_all_variants.py

# This will:
# - Load 100,000 code samples from bigcode/the-stack-smol
# - Train each variant for 3 epochs
# - Save checkpoints in ./trained_models/
# - Generate a training summary
```

**Expected time**: 
- With GPU: 6-12 hours total (1-2 hours per variant)
- With CPU: 24-48 hours (much slower)

### Train Specific Variants

If you only need certain variants:

```bash
# Train only the Tiny and Small variants
python train_all_variants.py --variants SmallCoder-Tiny SmallCoder-Small

# Train only Long Context variants
python train_all_variants.py --variants SmallCoder-Tiny-LC SmallCoder-Small-LC SmallCoder-Medium-LC
```

### Quick Test Training

For testing the training pipeline (faster, fewer samples):

```bash
# Quick training with 10k samples and 1 epoch
python train_all_variants.py --max-samples 10000 --num-epochs 1
```

## 📊 Using Different Datasets

### Default: The Stack (Smol)

The default dataset is `bigcode/the-stack-smol`, which contains diverse code from multiple languages:

```bash
python train_all_variants.py --dataset bigcode/the-stack-smol
```

### Alternative Datasets

```bash
# Use The Stack (full version - very large, requires more time)
python train_all_variants.py --dataset bigcode/the-stack

# Use GitHub code dataset
python train_all_variants.py --dataset codeparrot/github-code

# Filter by language (Python only example)
python train_all_variants.py --dataset bigcode/the-stack-smol --subset python
```

### Custom Dataset

To use your own code dataset:

```python
# Your dataset should be in HuggingFace datasets format
# with a 'text' or 'content' field containing code samples

# Option 1: Upload to HuggingFace Hub and use the name
python train_all_variants.py --dataset your-username/your-dataset

# Option 2: Modify train_all_variants.py to load local data
```

## 🔧 Training Individual Variants

If you prefer to train variants one at a time with custom settings:

### Example 1: Train Tiny Variant

```bash
python train.py \
    --output_dir ./checkpoints/SmallCoder-Tiny \
    --dataset bigcode/the-stack-smol \
    --max_length 512 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --use_fp16
```

### Example 2: Train Medium-LC Variant (Long Context)

```bash
python train.py \
    --output_dir ./checkpoints/SmallCoder-Medium-LC \
    --dataset bigcode/the-stack-smol \
    --max_length 1024 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_epochs 3 \
    --use_fp16
```

**Note**: The standard `train.py` script trains using the SmallCoder-Medium config by default. To train other variants with `train.py`, you would need to modify the config creation in the script.

## ⚙️ Training Parameters

### Memory Optimization

**For 2GB VRAM (e.g., MX450):**
```bash
python train_all_variants.py \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_length 256 \
    --use_fp16
```

**For 4GB+ VRAM:**
```bash
python train_all_variants.py \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_length 512 \
    --use_fp16
```

**For 8GB+ VRAM:**
```bash
python train_all_variants.py \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_length 1024 \
    --use_fp16
```

### Quality vs Speed Trade-offs

**Fast training (lower quality):**
```bash
python train_all_variants.py \
    --max-samples 50000 \
    --num-epochs 2 \
    --learning_rate 5e-4
```

**Best quality (slower):**
```bash
python train_all_variants.py \
    --max-samples 200000 \
    --num-epochs 5 \
    --learning_rate 2e-4
```

## 📁 Output Structure

After training, your directory structure will look like:

```
trained_models/
├── SmallCoder-Tiny/
│   ├── best_model.pt
│   ├── SmallCoder-Tiny_final.pt
│   ├── training_info.json
│   └── checkpoint_epoch_*.pt
├── SmallCoder-Small/
│   └── ...
├── SmallCoder-Medium/
│   └── ...
├── SmallCoder-Tiny-LC/
│   └── ...
├── SmallCoder-Small-LC/
│   └── ...
├── SmallCoder-Medium-LC/
│   └── ...
└── training_summary.json
```

## 🎯 Using Trained Models

After training, use your models:

### With inference.py

```bash
# Use the best checkpoint
python inference.py \
    --checkpoint trained_models/SmallCoder-Medium/best_model.pt \
    --tokenizer codellama/CodeLlama-7b-hf \
    --interactive
```

### With CLI Agent

```bash
# Use trained model with CLI agent
python cli_agent.py \
    --variant SmallCoder-Medium \
    --checkpoint trained_models/SmallCoder-Medium/best_model.pt \
    --interactive
```

### Programmatically

```python
import torch
from model import SmallCoderConfig, SmallCoderForCausalLM

# Load trained model
checkpoint = torch.load('trained_models/SmallCoder-Small/best_model.pt')
config = SmallCoderConfig(**checkpoint['config'])
model = SmallCoderForCausalLM(config)
model.load_state_dict(checkpoint['model_state_dict'])

# Use the model
model.eval()
# ... your generation code ...
```

## 🔍 Monitoring Training

### Check Training Progress

The training script provides detailed progress:
- Loss values per epoch
- Validation loss for quality tracking
- Checkpoint saves at regular intervals

### Resume Training

If training is interrupted, you can resume:

```bash
python train_all_variants.py --resume
```

**Note**: The current implementation saves checkpoints but full resume functionality would need to be implemented.

## 🐛 Troubleshooting

### Issue: "Unable to load actual code dataset"

**Cause**: No internet connection or dataset not accessible

**Solution**:
```bash
# 1. Check internet connection
ping huggingface.co

# 2. Try downloading dataset separately
python -c "from datasets import load_dataset; load_dataset('bigcode/the-stack-smol', split='train[:100]')"

# 3. Use a local dataset if available
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size and sequence length
```bash
python train_all_variants.py \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_length 256
```

### Issue: Training is very slow

**Solutions**:
- Use GPU instead of CPU
- Reduce `--max-samples` for faster iteration
- Use `--use_fp16` for mixed precision training
- Train fewer variants at once

### Issue: "No module named 'datasets'"

**Solution**:
```bash
pip install datasets
# or
pip install -r requirements.txt
```

## 📚 Best Practices

### 1. Start with One Variant

Before training all variants, test with one:

```bash
# Test training Tiny variant first (fastest)
python train_all_variants.py --variants SmallCoder-Tiny --num-epochs 1
```

### 2. Monitor First Epoch

Watch the first epoch closely:
- Loss should decrease
- No OOM errors
- Reasonable speed (check tokens/sec)

### 3. Adjust Hyperparameters

Based on first epoch results:
- If loss is too high: increase `--num-epochs` or `--max-samples`
- If OOM: reduce `--batch_size` or `--max_length`
- If too slow: reduce `--max-samples` or use smaller variants

### 4. Validate Trained Models

After training, test the models:

```bash
# Quick test
python inference.py \
    --checkpoint trained_models/SmallCoder-Tiny/best_model.pt \
    --tokenizer codellama/CodeLlama-7b-hf \
    --prompt "def fibonacci(n):"

# Interactive test
python run_model.py \
    --checkpoint trained_models/SmallCoder-Small/best_model.pt \
    --interactive
```

## 🎓 Advanced Training

### Knowledge Distillation

Train from a larger teacher model:

```bash
python distill.py \
    --teacher_model codellama/CodeLlama-7b-hf \
    --output_dir ./distilled_model \
    --dataset bigcode/the-stack-smol \
    --max_samples 50000 \
    --num_epochs 3
```

### Fine-tuning Pre-trained Model

If you have a pre-trained checkpoint, fine-tune it:

```bash
# Fine-tune on your specific code
python train.py \
    --checkpoint pretrained_smallcoder.pt \
    --dataset your_dataset \
    --num_epochs 2 \
    --learning_rate 1e-4
```

**Note**: The `--checkpoint` flag would need to be added to `train.py` for this to work.

## 📊 Expected Results

After training on 100k code samples for 3 epochs:

| Variant | Training Time (GPU) | Final Loss | Quality |
|---------|-------------------|------------|---------|
| Tiny | ~1-2 hours | ~2.5-3.0 | Good |
| Small | ~2-3 hours | ~2.2-2.7 | Better |
| Medium | ~3-4 hours | ~2.0-2.5 | Best |
| *-LC variants | +20-30% time | Similar | Similar |

**Quality improvements with more training:**
- More epochs: Better code understanding
- More samples: Better generalization
- Larger context: Better long-range dependencies

## 🔗 Related Documentation

- **[README.md](README.md)** - Main project documentation
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[PRETRAINED_MODEL_GUIDE.md](PRETRAINED_MODEL_GUIDE.md)** - Using pre-initialized models
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contributing guidelines

## 💡 Tips for Success

1. **Use actual code data** - Never train on dummy data for production
2. **Start small** - Test with one variant first
3. **Monitor closely** - Watch the first epoch for issues
4. **Save checkpoints** - Training can be interrupted
5. **Validate results** - Test models after training
6. **Be patient** - Quality training takes time

## 🆘 Need Help?

- Check the error messages - they often contain solutions
- Review the examples in this guide
- Open an issue on GitHub with:
  - Your hardware specs
  - Command you ran
  - Error messages
  - Training logs

---

**Happy Training! 🚀**

*Remember: Quality models come from quality data and sufficient training time.*
