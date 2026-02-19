# Quick Reference Guide - New Features

This guide provides a quick reference for the new features in SmallCoder v2.0.

## Extended Context Length

**What changed:** Context length increased from 4096 to 8192 tokens (2x larger)

**Benefits:**
- Process longer code files in a single pass
- Maintain more context in conversations
- Handle complex multi-file projects better

**No action required:** This is enabled automatically in the model configuration.

---

## Coding Task Test Suite

**Purpose:** Comprehensive benchmarking across 11 coding categories

**Usage:**

```bash
# Run all 15+ tasks
python code_tasks.py

# Run specific category
python code_tasks.py --category algorithms
python code_tasks.py --category data_structures

# Save results
python code_tasks.py --output results.json

# Adjust generation length
python code_tasks.py --max_tokens 500
```

**Available Categories:**
- `algorithms` - Sorting, searching, graph algorithms
- `data_structures` - Trees, lists, hash tables
- `debugging` - Bug detection and fixing
- `refactoring` - Code improvement
- `explanation` - Code analysis
- `web_dev` - REST APIs
- `async` - Asynchronous programming
- `error_handling` - Exception handling
- `testing` - Unit test creation
- `database` - SQL queries
- `design_patterns` - Software patterns

**Output:** Detailed JSON with scores, timing, and generated code

---

## Coding Agent

**Purpose:** Intelligent multi-step problem solving with automatic refinement

### Interactive Mode

```bash
python coding_agent.py --interactive
```

**Commands:**
- `solve <problem>` - Solve a coding problem
- `analyze` - Analyze code (multi-line input)
- `execute` - Execute code (multi-line input)
- `help` - Show help
- `quit` - Exit

### Problem Solving Mode

```bash
# Solve a single problem
python coding_agent.py --problem "Write a function to reverse a string"

# With custom test cases
python coding_agent.py --problem "Implement bubble sort" --tests tests.json

# Adjust iterations
python coding_agent.py --problem "Complex task" --max_iterations 5

# Save results
python coding_agent.py --problem "Task" --output solution.json
```

### Agent Capabilities

**Multi-step Refinement:**
1. Generates initial solution
2. Analyzes code for issues
3. Executes and tests code
4. Refines based on feedback
5. Repeats until success

**Code Analysis:**
- Missing docstrings detection
- Error handling checks
- Function complexity analysis
- Best practice suggestions

**Safe Execution:**
- Isolated temporary environment
- 5-second timeout protection
- Detailed error reporting
- Test case validation

### Test Case Format

Create a JSON file with test cases:

```json
[
  {
    "input": "print(fibonacci(5))",
    "expected": "5"
  },
  {
    "input": "print(fibonacci(10))",
    "expected": "55"
  }
]
```

---

## Quick Start Examples

### 1. Try the Extended Context

```python
from model import SmallCoderConfig

config = SmallCoderConfig()
print(f"Context length: {config.max_position_embeddings}")  # 8192
```

### 2. Run Benchmarks

```bash
# Quick test on algorithms
python code_tasks.py --category algorithms --max_tokens 200

# Full comprehensive test
python code_tasks.py --output full_results.json
```

### 3. Use the Agent

```bash
# Start interactive agent
python coding_agent.py --interactive

# Then try:
Agent> solve Write a function to find prime numbers
Agent> analyze
# (paste your code)
END
```

### 4. Demo All Features

```bash
python demo_new_features.py
```

---

## Performance Tips

**For Limited Hardware (2GB VRAM):**
- Use `--quantize` flag for lower memory
- Reduce `--max_tokens` for faster testing
- Run categories separately rather than all at once

**For Best Results:**
- Use `--max_iterations 5` for complex problems
- Provide test cases when possible for agent mode
- Use category filtering to focus on specific areas

**Memory Usage:**
- Extended context adds minimal overhead
- Agent mode uses same memory as normal inference
- Test suite can be run incrementally by category

---

## Integration Examples

### In Your Python Code

```python
from coding_agent import CodingAgent
from model import SmallCoderForCausalLM, SmallCoderConfig
from transformers import AutoTokenizer
import torch

# Load model
config = SmallCoderConfig()
model = SmallCoderForCausalLM(config)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

# Create agent
agent = CodingAgent(model, tokenizer, device='cuda')

# Solve a problem
result = agent.solve_problem(
    "Write a function to check if a number is prime",
    test_cases=[
        ("print(is_prime(7))", "True"),
        ("print(is_prime(4))", "False")
    ]
)

if result['solved']:
    print("Success:", result['final_solution']['code'])
```

### Running Tests Programmatically

```python
from code_tasks import run_coding_tasks
from inference import load_model
from transformers import AutoTokenizer

# Load model and tokenizer
model, config = load_model('pretrained_smallcoder.pt', device='cuda')
tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-hf')

# Run tests
results = run_coding_tasks(
    model=model,
    tokenizer=tokenizer,
    device='cuda',
    filter_category='algorithms',
    max_tokens=300
)

print(f"Average score: {results['summary']['avg_keyword_score']:.2%}")
```

---

## Troubleshooting

**Issue:** Out of memory during testing
- **Solution:** Use `--category` to run fewer tasks at once
- **Solution:** Reduce `--max_tokens` to generate less

**Issue:** Agent not finding solution
- **Solution:** Increase `--max_iterations`
- **Solution:** Simplify problem description
- **Solution:** Provide test cases for guidance

**Issue:** Code execution fails
- **Solution:** Check for syntax errors in generated code
- **Solution:** Ensure test cases are properly formatted
- **Solution:** Try with more iterations

**Issue:** Slow performance
- **Solution:** Use `--device cpu` if GPU is busy
- **Solution:** Use `--quantize` for faster inference
- **Solution:** Reduce generation length

---

## Additional Resources

- Full documentation: [README.md](README.md)
- Implementation details: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Pre-trained model guide: [PRETRAINED_MODEL_GUIDE.md](PRETRAINED_MODEL_GUIDE.md)

## Feedback

Found an issue or have a suggestion? Please open an issue on GitHub!
