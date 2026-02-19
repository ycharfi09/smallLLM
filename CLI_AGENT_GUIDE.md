# SmallCoder CLI Agent - Quick Reference

The SmallCoder CLI Agent is an interactive coding assistant similar to Claude Code, providing file operations, code review, and multi-turn conversations.

## 🚀 Quick Start

```bash
# Start with default Medium variant
python cli_agent.py --interactive

# Use a smaller variant for limited hardware
python cli_agent.py --variant SmallCoder-Tiny --interactive

# Use with quantization
python cli_agent.py --variant SmallCoder-Small --quantize --interactive

# List all available variants
python cli_agent.py --list-variants
```

## 📋 Available Commands

### Code Generation
- `<your prompt>` - Generate code or get answers to coding questions
- Example: `def quicksort(arr):`
- Example: `class BinaryTree:`

### File Operations
- `/read <filename>` - Read a file from the workspace
- `/write <filename>` - Write the last generated code to a file
- `/ls [pattern]` - List files in workspace (optional glob pattern like `*.py`)
- `/pwd` - Show current workspace directory
- `/cd <path>` - Change workspace directory

### Code Analysis
- `/review <code|filename>` - Review code for bugs, quality, and improvements
- `/explain <code|filename>` - Explain what code does
- `/fix <issue>` - Get suggestions to fix code issues

### Conversation Management
- `/history` - Show conversation history
- `/clear` - Clear conversation history

### Help & Exit
- `/help` - Show help message
- `/quit` or `/exit` - Exit the agent

## 💡 Example Workflows

### 1. Generate and Save Code
```bash
>>> def fibonacci(n):
# Model generates fibonacci implementation

>>> /write fibonacci.py
✓ File written: fibonacci.py
```

### 2. Read and Review Existing Code
```bash
>>> /read mycode.py
--- Content of mycode.py ---
...

>>> /review mycode.py
Reviewing code...
- Suggestion 1: Add error handling for edge cases
- Suggestion 2: Consider using type hints
...
```

### 3. Interactive Code Development
```bash
>>> class Stack:
# Model generates Stack class

>>> /explain class Stack: ...
Explaining code...
This is a Stack data structure implementation...

>>> /review
# Reviews the generated code
```

### 4. Work with Files in Directory
```bash
>>> /pwd
Current workspace: /home/user/project

>>> /ls *.py
Files:
  main.py
  utils.py
  tests.py

>>> /read utils.py
--- Content of utils.py ---
...
```

## 🎯 Model Variant Selection

Choose the right variant for your hardware:

| Variant | Parameters | VRAM | Best For | Command |
|---------|-----------|------|----------|---------|
| Tiny | 100M | 0.5GB | Minimal hardware | `--variant SmallCoder-Tiny` |
| Small | 194M | 1.0GB | Budget GPUs | `--variant SmallCoder-Small` |
| Medium | 304M | 2.0GB | Standard laptops | `--variant SmallCoder-Medium` (default) |
| Tiny-LC | 100M | 0.8GB | Long context, minimal | `--variant SmallCoder-Tiny-LC` |
| Small-LC | 194M | 1.5GB | Long context, budget | `--variant SmallCoder-Small-LC` |
| Medium-LC | 304M | 2.5GB | Long context, full | `--variant SmallCoder-Medium-LC` |

## ⚙️ Configuration Options

```bash
python cli_agent.py \
    --variant SmallCoder-Small \      # Model variant
    --interactive \                    # Interactive mode
    --device cuda \                    # Device (cuda/cpu)
    --quantize \                       # Enable INT8 quantization
    --max-tokens 500 \                 # Max tokens to generate
    --temperature 0.7 \                # Sampling temperature (0.0-1.0)
    --workspace /path/to/project       # Workspace directory
```

## 🔧 Tips & Tricks

### For Better Code Generation
1. **Be specific**: Include function signatures, class names
2. **Add context**: Mention requirements, constraints
3. **Use comments**: Start with `# Function to...`
4. **Iterate**: Use conversation history for refinements

### For Limited Hardware
1. Use `SmallCoder-Tiny` with `--quantize`
2. Set `--device cpu` if no GPU
3. Reduce `--max-tokens` for faster generation
4. Clear history periodically with `/clear`

### For Large Codebases
1. Use LC (Long Context) variants
2. Read files selectively, don't load everything
3. Review specific sections at a time
4. Use `/ls` to navigate structure

## 🐛 Troubleshooting

### Out of Memory
- Use a smaller variant (e.g., `SmallCoder-Tiny`)
- Add `--quantize` flag
- Use `--device cpu`
- Reduce `--max-tokens`

### Slow Generation
- Normal on CPU (2-10 tokens/sec)
- Use GPU with `--device cuda`
- Use smaller variant for faster speed
- Reduce context with `/clear`

### Model Not Found
- The agent creates initialized models if no checkpoint exists
- For pre-trained models, run `python pretrained_model.py` first
- Check checkpoint naming: `pretrained_smallcoder_<variant>.pt`

## 📚 Advanced Usage

### Custom Workspace
```bash
python cli_agent.py --workspace /path/to/my/project --interactive
```

### CPU-Only Mode
```bash
python cli_agent.py --device cpu --variant SmallCoder-Tiny --interactive
```

### High-Quality Generation
```bash
python cli_agent.py \
    --variant SmallCoder-Medium-LC \
    --temperature 0.5 \
    --max-tokens 1000 \
    --interactive
```

### Quick Prototyping
```bash
python cli_agent.py \
    --variant SmallCoder-Tiny \
    --temperature 0.9 \
    --max-tokens 200 \
    --interactive
```

## 🆚 Comparison with Other Tools

| Feature | SmallCoder CLI | Claude Code | GitHub Copilot |
|---------|---------------|-------------|----------------|
| Fully Local | ✅ | ❌ | ❌ |
| File Operations | ✅ | ✅ | Limited |
| Code Review | ✅ | ✅ | ❌ |
| Multiple Variants | ✅ | ❌ | ❌ |
| Conversation History | ✅ | ✅ | Limited |
| Works Offline | ✅ | ❌ | ❌ |
| Cost | Free | Paid | Paid |
| Min Hardware | 4GB RAM | Cloud | Cloud |

---

**Happy Coding! 🚀**

For more information, see:
- [Main README](README.md)
- [Model Variants Guide](model_variants.py)
- [Benchmarking Guide](benchmark_variants.py)
