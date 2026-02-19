"""
Lightweight Coding Agent for SmallCoder
Provides agent capabilities for multi-step problem solving, code execution, and iterative refinement
"""

import torch
from transformers import AutoTokenizer
import subprocess
import tempfile
import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CodingAgent:
    """A lightweight coding agent that can solve problems iteratively"""
    
    def __init__(self, model, tokenizer, device='cuda', max_iterations=3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_iterations = max_iterations
        self.conversation_history = []
        
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate code from a prompt"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Truncate if too long (leave room for generation)
        max_input_length = 6000  # Leave room in 8192 context
        if input_ids.shape[1] > max_input_length:
            input_ids = input_ids[:, -max_input_length:]
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                do_sample=True,
            )
        
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
    
    def extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from generated text"""
        # Try to find code between triple backticks
        code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        code_blocks = re.findall(r'```\n(.*?)```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, try to extract code starting with def or class
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        return None
    
    def execute_code(self, code: str, test_input: Optional[str] = None) -> Dict[str, any]:
        """Execute Python code in a safe temporary environment"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                if test_input:
                    f.write('\n\n' + test_input)
                temp_file = f.name
            
            # Execute with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            os.unlink(temp_file)
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Execution timed out (>5 seconds)',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }
    
    def analyze_code(self, code: str) -> Dict[str, any]:
        """Analyze code for common issues and provide feedback"""
        issues = []
        suggestions = []
        
        # Check for common issues
        lines = code.split('\n')
        
        # Check for missing docstrings
        if 'def ' in code and '"""' not in code and "'''" not in code:
            issues.append("Missing docstrings for functions")
            suggestions.append("Add docstrings to describe what your functions do")
        
        # Check for error handling
        if 'open(' in code and 'try:' not in code:
            issues.append("File operations without error handling")
            suggestions.append("Use try-except blocks when working with files")
        
        # Check for type hints
        if 'def ' in code and '->' not in code:
            suggestions.append("Consider adding type hints for better code clarity")
        
        # Check for overly long functions
        in_function = False
        function_length = 0
        max_length = 0
        
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
                function_length = 0
            elif in_function:
                function_length += 1
                if line.strip() and not line.strip().startswith('#'):
                    max_length = max(max_length, function_length)
                if line.strip().startswith('def ') or (not line.strip() and function_length > 5):
                    in_function = False
        
        if max_length > 50:
            issues.append("Very long function detected")
            suggestions.append("Consider breaking down long functions into smaller ones")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'lines_of_code': len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        }
    
    def solve_problem(self, problem: str, test_cases: Optional[List[Tuple[str, str]]] = None) -> Dict[str, any]:
        """
        Solve a coding problem with iterative refinement
        
        Args:
            problem: Description of the problem to solve
            test_cases: Optional list of (input, expected_output) tuples
        
        Returns:
            Dictionary with solution, execution results, and metadata
        """
        print(f"\n{'='*80}")
        print("CODING AGENT - PROBLEM SOLVING")
        print(f"{'='*80}\n")
        print(f"Problem: {problem}\n")
        
        iterations = []
        current_prompt = f"# Problem: {problem}\n# Write a Python solution:\n\n"
        
        for iteration in range(self.max_iterations):
            print(f"--- Iteration {iteration + 1}/{self.max_iterations} ---\n")
            
            # Generate code
            print("Generating code...")
            start_time = time.time()
            generated_text = self.generate(current_prompt, max_tokens=500)
            generation_time = time.time() - start_time
            
            # Extract code
            code = self.extract_code(generated_text)
            if not code:
                print("⚠️  Could not extract code from generation")
                iterations.append({
                    'iteration': iteration + 1,
                    'generated_text': generated_text,
                    'code': None,
                    'execution_result': None,
                    'analysis': None,
                    'success': False
                })
                continue
            
            print(f"✓ Generated code ({generation_time:.2f}s):")
            print("-" * 40)
            print(code[:500] + ("..." if len(code) > 500 else ""))
            print("-" * 40)
            
            # Analyze code
            print("\nAnalyzing code...")
            analysis = self.analyze_code(code)
            if analysis['issues']:
                print("⚠️  Issues found:")
                for issue in analysis['issues']:
                    print(f"  - {issue}")
            if analysis['suggestions']:
                print("💡 Suggestions:")
                for suggestion in analysis['suggestions']:
                    print(f"  - {suggestion}")
            
            # Execute code
            execution_result = None
            all_tests_passed = True
            
            if test_cases:
                print("\nRunning test cases...")
                test_results = []
                
                for i, (test_input, expected_output) in enumerate(test_cases, 1):
                    print(f"  Test {i}/{len(test_cases)}: ", end="")
                    result = self.execute_code(code, test_input)
                    
                    if result['success']:
                        actual_output = result['stdout'].strip()
                        expected = expected_output.strip()
                        passed = actual_output == expected
                        
                        if passed:
                            print("✓ PASSED")
                        else:
                            print(f"✗ FAILED")
                            print(f"    Expected: {expected}")
                            print(f"    Got: {actual_output}")
                            all_tests_passed = False
                        
                        test_results.append({
                            'test_number': i,
                            'passed': passed,
                            'expected': expected,
                            'actual': actual_output
                        })
                    else:
                        print(f"✗ ERROR: {result['stderr']}")
                        all_tests_passed = False
                        test_results.append({
                            'test_number': i,
                            'passed': False,
                            'error': result['stderr']
                        })
                
                execution_result = {
                    'test_results': test_results,
                    'all_passed': all_tests_passed
                }
            else:
                # Just try to execute the code
                print("\nExecuting code...")
                result = self.execute_code(code)
                if result['success']:
                    print("✓ Code executed successfully")
                    if result['stdout']:
                        print(f"Output: {result['stdout']}")
                else:
                    print(f"✗ Execution failed: {result['stderr']}")
                    all_tests_passed = False
                
                execution_result = result
            
            # Store iteration results
            iteration_data = {
                'iteration': iteration + 1,
                'generated_text': generated_text,
                'code': code,
                'execution_result': execution_result,
                'analysis': analysis,
                'success': all_tests_passed,
                'generation_time': generation_time
            }
            iterations.append(iteration_data)
            
            # Check if we're done
            if all_tests_passed:
                print(f"\n✓ Solution found in {iteration + 1} iteration(s)!")
                break
            
            # Prepare prompt for next iteration
            if iteration < self.max_iterations - 1:
                print("\nRefining solution for next iteration...")
                feedback = []
                
                if not all_tests_passed and test_cases:
                    feedback.append("Some test cases failed. Please fix the code.")
                
                if analysis['issues']:
                    feedback.append("Address these issues: " + ", ".join(analysis['issues']))
                
                current_prompt = (
                    f"# Problem: {problem}\n"
                    f"# Previous attempt had issues. Feedback:\n"
                    f"# {' '.join(feedback)}\n\n"
                    f"# Previous code:\n{code}\n\n"
                    f"# Improved solution:\n\n"
                )
            
            print()
        
        # Determine final result
        successful_iterations = [it for it in iterations if it['success']]
        final_solution = successful_iterations[0] if successful_iterations else iterations[-1]
        
        print(f"{'='*80}")
        print("FINAL RESULT")
        print(f"{'='*80}")
        
        if successful_iterations:
            print(f"✓ Successfully solved in {final_solution['iteration']} iteration(s)")
        else:
            print("✗ Could not find a working solution within iteration limit")
        
        print(f"{'='*80}\n")
        
        return {
            'problem': problem,
            'iterations': iterations,
            'final_solution': final_solution,
            'solved': len(successful_iterations) > 0,
            'iterations_used': len(iterations)
        }
    
    def interactive_coding(self):
        """Interactive coding mode with agent capabilities"""
        print("\n" + "="*80)
        print(" "*25 + "SmallCoder Coding Agent")
        print("="*80)
        print("\nWelcome to the interactive coding agent!")
        print("\nCommands:")
        print("  - solve <problem>: Solve a coding problem")
        print("  - analyze <code>: Analyze code for issues")
        print("  - execute <code>: Execute Python code")
        print("  - help: Show this help message")
        print("  - quit/exit: Exit the agent")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("\nAgent> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! Happy coding! 🚀")
                    break
                
                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  - solve <problem>: Solve a coding problem")
                    print("  - analyze: Enter multi-line code analysis mode")
                    print("  - execute: Enter multi-line code execution mode")
                    print("  - help: Show this help message")
                    print("  - quit/exit: Exit the agent")
                    continue
                
                if user_input.lower().startswith('solve '):
                    problem = user_input[6:].strip()
                    if problem:
                        result = self.solve_problem(problem)
                        if result['solved']:
                            print(f"\n✓ Solution:\n{result['final_solution']['code']}")
                        else:
                            print(f"\n✗ Could not solve completely. Last attempt:\n{result['final_solution']['code']}")
                    else:
                        print("Please provide a problem description.")
                    continue
                
                if user_input.lower() == 'analyze':
                    print("Enter code to analyze (type 'END' on a new line when done):")
                    code_lines = []
                    while True:
                        line = input()
                        if line.strip() == 'END':
                            break
                        code_lines.append(line)
                    
                    code = '\n'.join(code_lines)
                    analysis = self.analyze_code(code)
                    
                    print("\nAnalysis Results:")
                    print(f"Lines of code: {analysis['lines_of_code']}")
                    
                    if analysis['issues']:
                        print("\n⚠️  Issues:")
                        for issue in analysis['issues']:
                            print(f"  - {issue}")
                    
                    if analysis['suggestions']:
                        print("\n💡 Suggestions:")
                        for suggestion in analysis['suggestions']:
                            print(f"  - {suggestion}")
                    
                    if not analysis['issues'] and not analysis['suggestions']:
                        print("\n✓ No issues found!")
                    
                    continue
                
                if user_input.lower() == 'execute':
                    print("Enter code to execute (type 'END' on a new line when done):")
                    code_lines = []
                    while True:
                        line = input()
                        if line.strip() == 'END':
                            break
                        code_lines.append(line)
                    
                    code = '\n'.join(code_lines)
                    result = self.execute_code(code)
                    
                    if result['success']:
                        print("\n✓ Execution successful!")
                        if result['stdout']:
                            print(f"Output:\n{result['stdout']}")
                    else:
                        print(f"\n✗ Execution failed!")
                        if result['stderr']:
                            print(f"Error:\n{result['stderr']}")
                    
                    continue
                
                # Default: treat as problem to solve
                print(f"Treating as problem to solve: {user_input}")
                result = self.solve_problem(user_input)
                if result['solved']:
                    print(f"\n✓ Solution:\n{result['final_solution']['code']}")
                else:
                    print(f"\n✗ Could not solve completely. Last attempt:\n{result['final_solution']['code']}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Happy coding! 🚀")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SmallCoder Lightweight Coding Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive agent mode
  python coding_agent.py --interactive
  
  # Solve a specific problem
  python coding_agent.py --problem "Write a function to reverse a string"
  
  # Solve with test cases
  python coding_agent.py --problem "Write fibonacci function" --tests test_cases.json
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='pretrained_smallcoder.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='codellama/CodeLlama-7b-hf',
        help='Tokenizer to use'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--problem',
        type=str,
        help='Problem to solve'
    )
    parser.add_argument(
        '--tests',
        type=str,
        help='JSON file with test cases'
    )
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=3,
        help='Maximum iterations for problem solving'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (JSON)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" "*25 + "SmallCoder Coding Agent")
    print("="*80 + "\n")
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}\n")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded\n")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return 1
    
    # Check/load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Model not found at {args.checkpoint}")
        print("Creating pre-trained model...")
        from pretrained_model import create_pretrained_model
        create_pretrained_model(str(checkpoint_path))
        print()
    
    print(f"Loading model from {args.checkpoint}")
    from inference import load_model
    model, config = load_model(args.checkpoint, device=args.device, quantize=False)
    print("✓ Model loaded\n")
    
    # Create agent
    agent = CodingAgent(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_iterations=args.max_iterations
    )
    
    # Run based on mode
    if args.interactive:
        agent.interactive_coding()
    elif args.problem:
        # Load test cases if provided
        test_cases = None
        if args.tests:
            with open(args.tests, 'r') as f:
                test_data = json.load(f)
                test_cases = [(tc['input'], tc['expected']) for tc in test_data]
        
        # Solve problem
        result = agent.solve_problem(args.problem, test_cases)
        
        # Save results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.output}")
        
        return 0 if result['solved'] else 1
    else:
        print("No mode specified. Use --interactive or --problem")
        print("Run with --help for more information")
        return 1


if __name__ == "__main__":
    exit(main())
