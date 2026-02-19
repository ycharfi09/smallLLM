#!/usr/bin/env python3
"""
Integration test for SmallCoder v2.0 new features
Tests the basic functionality without requiring a trained model
"""

import sys
import traceback

def test_model_config():
    """Test 1: Verify extended context length in model configuration"""
    print("Test 1: Model Configuration")
    print("-" * 60)
    
    try:
        from model import SmallCoderConfig, SmallCoderForCausalLM, count_parameters
        
        config = SmallCoderConfig()
        
        # Verify context length
        assert config.max_position_embeddings == 8192, \
            f"Expected context length 8192, got {config.max_position_embeddings}"
        
        # Verify other parameters unchanged
        assert config.hidden_size == 1152, "Hidden size changed unexpectedly"
        assert config.num_hidden_layers == 18, "Number of layers changed unexpectedly"
        
        # Create model to verify it works
        model = SmallCoderForCausalLM(config)
        params = count_parameters(model)
        
        assert 300_000_000 < params < 310_000_000, \
            f"Parameter count unexpected: {params}"
        
        print(f"✓ Context length: {config.max_position_embeddings} tokens")
        print(f"✓ Parameters: {params:,} (~{params/1e6:.0f}M)")
        print(f"✓ Model creation successful")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


def test_coding_tasks_module():
    """Test 2: Verify coding tasks module"""
    print("\nTest 2: Coding Tasks Module")
    print("-" * 60)
    
    try:
        import code_tasks
        
        # Verify tasks exist
        assert len(code_tasks.CODING_TASKS) >= 15, \
            f"Expected at least 15 tasks, got {len(code_tasks.CODING_TASKS)}"
        
        # Verify categories
        categories = set(t['category'] for t in code_tasks.CODING_TASKS)
        expected_categories = {
            'algorithms', 'data_structures', 'debugging', 'refactoring',
            'explanation', 'web_dev', 'async', 'error_handling',
            'testing', 'database', 'design_patterns'
        }
        
        assert expected_categories.issubset(categories), \
            f"Missing expected categories: {expected_categories - categories}"
        
        # Verify task structure
        for task in code_tasks.CODING_TASKS:
            assert 'name' in task, "Task missing 'name'"
            assert 'prompt' in task, "Task missing 'prompt'"
            assert 'category' in task, "Task missing 'category'"
            assert 'expected_keywords' in task, "Task missing 'expected_keywords'"
            assert 'difficulty' in task, "Task missing 'difficulty'"
        
        print(f"✓ Total tasks: {len(code_tasks.CODING_TASKS)}")
        print(f"✓ Categories: {len(categories)}")
        print(f"✓ All tasks properly structured")
        
        # Test category counts
        category_counts = {}
        for task in code_tasks.CODING_TASKS:
            cat = task['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("\nTask distribution:")
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat:20s} : {count} tasks")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


def test_coding_agent_module():
    """Test 3: Verify coding agent module"""
    print("\nTest 3: Coding Agent Module")
    print("-" * 60)
    
    try:
        from coding_agent import CodingAgent
        
        # Verify class exists and can be imported
        assert CodingAgent is not None, "CodingAgent class not found"
        
        # Verify key methods exist
        required_methods = [
            'generate', 'extract_code', 'execute_code',
            'analyze_code', 'solve_problem', 'interactive_coding'
        ]
        
        for method in required_methods:
            assert hasattr(CodingAgent, method), f"Missing method: {method}"
        
        print(f"✓ CodingAgent class available")
        print(f"✓ All required methods present:")
        for method in required_methods:
            print(f"  - {method}")
        
        # Test code extraction
        agent_methods = CodingAgent.__dict__
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


def test_code_analysis():
    """Test 4: Test code analysis functionality"""
    print("\nTest 4: Code Analysis Functionality")
    print("-" * 60)
    
    try:
        from coding_agent import CodingAgent
        from model import SmallCoderConfig, SmallCoderForCausalLM
        import torch
        
        # Create a dummy agent (won't use model for this test)
        config = SmallCoderConfig()
        model = SmallCoderForCausalLM(config)
        
        # Mock tokenizer for testing
        class MockTokenizer:
            def encode(self, text, return_tensors=None):
                return torch.zeros((1, 10), dtype=torch.long)
            
            def decode(self, ids, skip_special_tokens=False):
                return "mock output"
        
        agent = CodingAgent(model, MockTokenizer(), device='cpu')
        
        # Test code analysis
        test_code = '''
def calculate(x):
    return x * 2

def process_file(filename):
    with open(filename) as f:
        return f.read()
'''
        
        analysis = agent.analyze_code(test_code)
        
        assert 'issues' in analysis, "Analysis missing 'issues'"
        assert 'suggestions' in analysis, "Analysis missing 'suggestions'"
        assert 'lines_of_code' in analysis, "Analysis missing 'lines_of_code'"
        
        print(f"✓ Code analysis working")
        print(f"  Lines of code: {analysis['lines_of_code']}")
        print(f"  Issues found: {len(analysis['issues'])}")
        print(f"  Suggestions: {len(analysis['suggestions'])}")
        
        if analysis['issues']:
            print("\n  Issues detected:")
            for issue in analysis['issues']:
                print(f"    - {issue}")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


def test_code_extraction():
    """Test 5: Test code extraction from text"""
    print("\nTest 5: Code Extraction")
    print("-" * 60)
    
    try:
        from coding_agent import CodingAgent
        from model import SmallCoderConfig, SmallCoderForCausalLM
        
        config = SmallCoderConfig()
        model = SmallCoderForCausalLM(config)
        
        class MockTokenizer:
            def encode(self, text, return_tensors=None):
                import torch
                return torch.zeros((1, 10), dtype=torch.long)
        
        agent = CodingAgent(model, MockTokenizer(), device='cpu')
        
        # Test extraction from markdown code blocks
        text1 = '''Here's the solution:
```python
def hello():
    print("Hello World")
```
'''
        code1 = agent.extract_code(text1)
        assert code1 is not None, "Failed to extract code from markdown"
        assert 'def hello' in code1, "Extracted code incomplete"
        
        # Test extraction from plain text
        text2 = '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)'''
        
        code2 = agent.extract_code(text2)
        assert code2 is not None, "Failed to extract code from plain text"
        assert 'def fibonacci' in code2, "Extracted code incomplete"
        
        print("✓ Code extraction from markdown blocks working")
        print("✓ Code extraction from plain text working")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


def test_documentation():
    """Test 6: Verify documentation files exist"""
    print("\nTest 6: Documentation")
    print("-" * 60)
    
    try:
        from pathlib import Path
        
        required_files = [
            'README.md',
            'CHANGELOG.md',
            'QUICK_REFERENCE.md',
            'code_tasks.py',
            'coding_agent.py',
            'demo_new_features.py'
        ]
        
        for filename in required_files:
            filepath = Path(filename)
            assert filepath.exists(), f"Missing file: {filename}"
            assert filepath.stat().st_size > 0, f"Empty file: {filename}"
        
        print("✓ All required files exist:")
        for filename in required_files:
            size = Path(filename).stat().st_size
            print(f"  {filename:30s} ({size:,} bytes)")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print(" "*10 + "SmallCoder v2.0 Integration Tests")
    print("="*60 + "\n")
    
    tests = [
        test_model_config,
        test_coding_tasks_module,
        test_coding_agent_module,
        test_code_analysis,
        test_code_extraction,
        test_documentation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} {test_name}")
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    if passed == total:
        print("🎉 All tests passed! SmallCoder v2.0 is ready to use.")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
