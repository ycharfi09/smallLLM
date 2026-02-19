#!/usr/bin/env python3
"""
SmallCoder CLI Agent Interface
An interactive coding assistant similar to Claude Code with file editing,
code review, and multi-turn conversation capabilities
"""

import torch
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List
from transformers import AutoTokenizer

from model import SmallCoderConfig, SmallCoderForCausalLM
from model_variants import get_variant_config, list_variants, MODEL_VARIANTS


class SmallCoderAgent:
    """Interactive coding agent with file operations and conversation history"""
    
    def __init__(
        self,
        model,
        tokenizer,
        config,
        device='cuda',
        max_tokens=500,
        temperature=0.7,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.conversation_history = []
        self.workspace_path = Path.cwd()
        
    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✓ Conversation history cleared")
    
    def generate_response(self, prompt: str, use_history: bool = True) -> str:
        """Generate a response from the model"""
        # Build context from history
        if use_history and self.conversation_history:
            context_parts = []
            for msg in self.conversation_history[-5:]:  # Last 5 messages
                prefix = "Human: " if msg["role"] == "user" else "Assistant: "
                context_parts.append(f"{prefix}{msg['content']}")
            context = "\n".join(context_parts) + f"\nHuman: {prompt}\nAssistant:"
        else:
            context = prompt
        
        # Encode
        input_ids = self.tokenizer.encode(context, return_tensors='pt')
        input_ids = input_ids[:, :self.config.max_position_embeddings]  # Truncate if needed
        input_ids = input_ids.to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                use_cache=True,
            )
        
        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract only the new part
        if use_history:
            generated = output_text[len(context):]
        else:
            generated = output_text[len(prompt):]
        
        return generated.strip()
    
    def read_file(self, filepath: str) -> str:
        """Read a file from the workspace"""
        try:
            file_path = self.workspace_path / filepath
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file: {e}"
    
    def write_file(self, filepath: str, content: str) -> str:
        """Write content to a file in the workspace"""
        try:
            file_path = self.workspace_path / filepath
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"✓ File written: {filepath}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    def list_files(self, pattern: str = "*") -> List[str]:
        """List files in the workspace matching a pattern"""
        try:
            files = list(self.workspace_path.glob(pattern))
            return [str(f.relative_to(self.workspace_path)) for f in files if f.is_file()]
        except Exception as e:
            return [f"Error listing files: {e}"]
    
    def code_review(self, code: str) -> str:
        """Perform a code review"""
        prompt = f"""Review the following code and provide feedback on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Suggestions for improvement

Code:
```
{code}
```

Review:"""
        return self.generate_response(prompt, use_history=False)
    
    def explain_code(self, code: str) -> str:
        """Explain what a piece of code does"""
        prompt = f"""Explain what this code does in clear terms:

Code:
```
{code}
```

Explanation:"""
        return self.generate_response(prompt, use_history=False)
    
    def fix_code(self, code: str, issue: str) -> str:
        """Suggest fixes for code issues"""
        prompt = f"""Fix the following issue in the code:

Issue: {issue}

Code:
```
{code}
```

Fixed code:"""
        return self.generate_response(prompt, use_history=False)


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*90)
    print(" " * 30 + "SmallCoder CLI Agent")
    print(" " * 25 + "Interactive Coding Assistant")
    print("="*90)
    print("\nAn AI-powered coding assistant with file operations and code review capabilities")
    print()


def print_help():
    """Print help information"""
    help_text = """
Available Commands:
  
  Code Generation:
    <your prompt>           - Generate code or get answers
    
  File Operations:
    /read <file>            - Read a file from workspace
    /write <file>           - Write generated code to file (use after generation)
    /ls [pattern]           - List files in workspace (pattern like *.py)
    
  Code Analysis:
    /review <code|file>     - Review code or file
    /explain <code|file>    - Explain code or file
    /fix <issue> <code>     - Suggest fixes for code issues
    
  Conversation:
    /history                - Show conversation history
    /clear                  - Clear conversation history
    
  Workspace:
    /pwd                    - Show current workspace directory
    /cd <path>              - Change workspace directory
    
  Utilities:
    /help                   - Show this help message
    /quit or /exit          - Exit the agent
    
Examples:
  >>> def quicksort(arr):
  >>> /read utils.py
  >>> /review utils.py
  >>> /explain def bubble_sort(arr): ...
"""
    print(help_text)


def load_model(variant_name: str, device: str, quantize: bool):
    """Load a model variant"""
    print(f"Loading {variant_name}...")
    
    # Get configuration
    config = get_variant_config(variant_name)
    
    # Create model
    model = SmallCoderForCausalLM(config)
    
    # Check if checkpoint exists
    checkpoint_path = f"pretrained_{variant_name.lower().replace('-', '_')}.pt"
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Checkpoint loaded")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_path}")
        print("✓ Using freshly initialized model")
    
    # Quantization
    if quantize:
        print("Applying INT8 quantization...")
        try:
            model = model.to('cpu')
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            if device == 'cuda':
                print("Note: Quantized model runs on CPU")
                device = 'cpu'
        except Exception as e:
            print(f"Quantization failed: {e}, continuing without it")
    
    model = model.to(device)
    model.eval()
    
    return model, config, device


def interactive_mode(agent: SmallCoderAgent):
    """Run the agent in interactive mode"""
    print_banner()
    print("Type /help for available commands, /quit to exit")
    print("="*90 + "\n")
    
    last_generated_code = None
    
    while True:
        try:
            # Get user input
            user_input = input("\n>>> ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command in ['/quit', '/exit', '/q']:
                    print("\nGoodbye! Happy coding! 🚀")
                    break
                
                elif command == '/help':
                    print_help()
                
                elif command == '/clear':
                    agent.clear_history()
                
                elif command == '/history':
                    print("\nConversation History:")
                    print("-" * 90)
                    for i, msg in enumerate(agent.conversation_history, 1):
                        role = "You" if msg["role"] == "user" else "Agent"
                        print(f"{i}. {role}: {msg['content'][:100]}...")
                    if not agent.conversation_history:
                        print("(empty)")
                
                elif command == '/read':
                    if not args:
                        print("Usage: /read <filename>")
                    else:
                        content = agent.read_file(args)
                        print(f"\n--- Content of {args} ---")
                        print(content)
                        print("-" * 90)
                
                elif command == '/write':
                    if not args:
                        print("Usage: /write <filename>")
                    elif last_generated_code:
                        result = agent.write_file(args, last_generated_code)
                        print(result)
                    else:
                        print("No code to write. Generate code first.")
                
                elif command == '/ls':
                    pattern = args if args else "*"
                    files = agent.list_files(pattern)
                    print("\nFiles:")
                    for f in files[:50]:  # Limit to 50 files
                        print(f"  {f}")
                    if len(files) > 50:
                        print(f"  ... and {len(files) - 50} more")
                
                elif command == '/review':
                    if not args:
                        print("Usage: /review <code or filename>")
                    else:
                        # Check if it's a file
                        if Path(agent.workspace_path / args).exists():
                            code = agent.read_file(args)
                        else:
                            code = args
                        
                        print("\nReviewing code...")
                        print("-" * 90)
                        review = agent.code_review(code)
                        print(review)
                        print("-" * 90)
                
                elif command == '/explain':
                    if not args:
                        print("Usage: /explain <code or filename>")
                    else:
                        # Check if it's a file
                        if Path(agent.workspace_path / args).exists():
                            code = agent.read_file(args)
                        else:
                            code = args
                        
                        print("\nExplaining code...")
                        print("-" * 90)
                        explanation = agent.explain_code(code)
                        print(explanation)
                        print("-" * 90)
                
                elif command == '/fix':
                    print("Usage: /fix <issue description>")
                    print("Then provide code in next message")
                
                elif command == '/pwd':
                    print(f"Current workspace: {agent.workspace_path}")
                
                elif command == '/cd':
                    if not args:
                        print("Usage: /cd <directory>")
                    else:
                        new_path = Path(args).resolve()
                        if new_path.exists() and new_path.is_dir():
                            agent.workspace_path = new_path
                            print(f"✓ Changed workspace to: {agent.workspace_path}")
                        else:
                            print(f"Error: Directory not found: {args}")
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type /help for available commands")
            
            else:
                # Regular prompt - generate response
                agent.add_to_history("user", user_input)
                
                print("\nGenerating response...")
                print("-" * 90)
                
                response = agent.generate_response(user_input)
                print(response)
                print("-" * 90)
                
                agent.add_to_history("assistant", response)
                last_generated_code = response  # Save for /write command
        
        except KeyboardInterrupt:
            print("\n\nUse /quit to exit")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description='SmallCoder CLI Agent - Interactive Coding Assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default medium model
  python cli_agent.py --interactive
  
  # Use a specific variant
  python cli_agent.py --variant SmallCoder-Small --interactive
  
  # List available variants
  python cli_agent.py --list-variants
  
  # Use with quantization for lower memory
  python cli_agent.py --variant SmallCoder-Tiny --quantize --interactive
        """
    )
    
    parser.add_argument(
        '--variant',
        type=str,
        default='SmallCoder-Medium',
        help='Model variant to use (default: SmallCoder-Medium)'
    )
    parser.add_argument(
        '--list-variants',
        action='store_true',
        help='List all available model variants and exit'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='codellama/CodeLlama-7b-hf',
        help='Tokenizer to use (default: codellama/CodeLlama-7b-hf)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu, default: cuda)'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Use INT8 quantization for lower memory usage'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Maximum tokens to generate (default: 500)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        default='.',
        help='Workspace directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # List variants and exit
    if args.list_variants:
        list_variants()
        return
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure transformers is installed: pip install transformers")
        sys.exit(1)
    
    # Load model
    try:
        model, config, device = load_model(args.variant, args.device, args.quantize)
        print("✓ Model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create agent
    agent = SmallCoderAgent(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    agent.workspace_path = Path(args.workspace).resolve()
    
    # Run interactive mode
    if args.interactive:
        interactive_mode(agent)
    else:
        print("\n⚠️  No mode specified. Use --interactive for interactive mode")
        print("Example: python cli_agent.py --interactive")


if __name__ == "__main__":
    main()
