import re
import asyncio
import json
import os
import sys
import random
import csv
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import tempfile
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.llm import LLM
from modules.prompt_template import (
    VARIANT_GENERATION_PROMPT, 
    OPTIMIZATION_PROMPT, 
    TRICKS_PROMPT,
    FIX_FAILED_VARIANT_PROMPT,
    KNOWLEDGE_BASE_TRICKS_PROMPT,
    PROVIDED_TRICKS_PROMPT
)

# Add Google Code Golf utils to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print("Project root:", PROJECT_ROOT)
# print("Adding code golf utils path:", os.path.join(PROJECT_ROOT, "google-code-golf-2025", "code_golf_utils"))
sys.path.append(os.path.join(PROJECT_ROOT, "google-code-golf-2025", "code_golf_utils"))
from code_golf_utils import load_examples, verify_program

@dataclass
class Attempt:
    code: str
    passed: bool
    length: int = 0
    minified_code: str = ""
    minified_length: int = 0
    summary: str = ""
    strategy: str = ""
    
    def __post_init__(self):
        self.length = len(self.code.encode())
        if self.minified_code:
            self.minified_length = len(self.minified_code.encode())
    
    def min_length(self) -> int:
        """Return the minified code length for consistent comparison"""
        if not self.passed:
            return 2**32 - 1
        # Always use minified length for comparison if available
        if self.minified_code and self.minified_length > 0:
            return self.minified_length
        return self.length
    
    def update_minified(self, minified_code: str):
        """Update minified code and its length"""
        self.minified_code = minified_code
        self.minified_length = len(minified_code.encode())

class CodeJudge:
    def __init__(self, task_num: int, examples: Dict, timeout: int = 2):
        self.task_num = task_num
        self.examples = examples
        self.timeout = timeout
    
    async def execute(self, code: str) -> tuple[bool, str]:
        """Execute code against Google Code Golf test cases"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                import io
                from contextlib import redirect_stdout
                
                captured_output = io.StringIO()
                
                with redirect_stdout(captured_output):
                    try:
                        verify_program(self.task_num, self.examples, temp_file)
                        verification_output = captured_output.getvalue()
                        
                        if "Your code IS READY for submission!".lower() in verification_output.lower():
                            return True, "All tests passed"
                        else:
                            return False, verification_output.split("The expected result is shown in green; your actual result is shown in red.")[0]
                            
                    except Exception as e:
                        return False, f"Verification error: {str(e)}"
                        
            finally:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            return False, f"Runtime error: {str(e)}"

async def minify_code(code: str) -> str:
    """Advanced code minification for Google Code Golf"""
    import re
    
    lines = []
    for line in code.split('\n'):
        line = re.sub(r'#.*', '', line)
        if line.strip():
            lines.append(line)
    
    minified = '\n'.join(lines)
    
    replacements = [
        (r'\s*=\s*', '='),
        (r'\s*\+\s*', '+'),
        (r'\s*-\s*', '-'),
        (r'\s*\*\s*', '*'),
        (r'\s*/\s*', '/'),
        (r'\s*==\s*', '=='),
        (r'\s*!=\s*', '!='),
        (r'\s*<=\s*', '<='),
        (r'\s*>=\s*', '>='),
        (r'\s*<\s*', '<'),
        (r'\s*>\s*', '>'),
        (r'\s*\[\s*', '['),
        (r'\s*\]\s*', ']'),
        (r'\s*\(\s*', '('),
        (r'\s*\)\s*', ')'),
        (r'\s*,\s*', ','),
        (r'\s*:\s*', ':'),
        (r'\n    ', '\n '),
    ]
    
    for old, new in replacements:
        minified = re.sub(old, new, minified)
    
    return minified

class GolfAgent:
    def __init__(
        self, 
        llm: LLM, 
        task_num: int,
        initial_solution: str,
        n_variants: int = 10,
        generation_factor: float = 2.0,
        max_steps: int = 16,
        early_stop_steps: int = 5,
        max_iterations: int = 5,
        use_shortest_hint: bool = False,
        logger: logging.Logger = None,
        record_prompts: bool = True,
        record_details: bool = True,
    ):
        self.project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.llm = llm
        self.task_num = task_num
        self.examples = load_examples(task_num)
        self.judge = CodeJudge(task_num, self.examples)
        self.initial_solution = initial_solution
        
        # Configuration for advanced strategy
        self.n_variants = n_variants
        self.generation_factor = generation_factor  # Generate generation_factor * n variants, then select n
        self.use_shortest_hint = use_shortest_hint
        self.shortest_known = self._load_shortest_length() if use_shortest_hint else None
        
        # Existing configuration
        self.max_steps = max_steps
        self.early_stop_steps = early_stop_steps
        self.max_iterations = max_iterations
        # set logger path
        logs_dir = os.path.join(self.project_path, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{logs_dir}/task_{task_num:03d}_{timestamp}.log"
        
        # Print to console for immediate feedback
        print(f"Initializing GolfAgent for task {task_num}")
        print(f"Log file will be: {log_file}")
        
        self.logger = logger or self._setup_logger()
        
        # Immediate log to verify logger is working
        self.logger.info(f"GolfAgent initialized for task {task_num}")
        self.logger.info(f"Project path: {self.project_path}")
        self.logger.info(f"Log file: {log_file}")
        
        # Load generator code and tricks
        self.generator_code = self._load_generator_code()
        self.tricks = self._load_tricks()
        
        # State tracking
        self.step = 0
        self.total_llm_calls = 0
        self.all_attempts: List[List[Attempt]] = []  # Attempts for each variant path
        self.best_attempt: Optional[Attempt] = None
        
        # Prompt logging tracking
        self.logged_prompts = set()  # Track which prompt types have been logged

        self.record_prompts = record_prompts
        self.record_details = record_details
        
    def _setup_logger(self) -> logging.Logger:
        """Setup a basic logger if none provided"""
        # Create a directory the task logs if it doesn't exist
        logs_dir = os.path.join(self.project_path, 'logs', f"task{self.task_num:03d}")  
        os.makedirs(logs_dir, exist_ok=True)
        # Generate a timestamped log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"{timestamp}.log")
        # Use a unique logger name to avoid conflicts
        logger_name = f'GolfAgent_task{self.task_num}_{datetime.now().strftime("%H%M%S")}'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Always add console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Add file handler if log_file is provided
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logger initialized for task {self.task_num}, log file: {log_file}")
        
        return logger
    
    def _log_prompt_once(self, prompt_type: str, prompt: str):
        """Log a prompt only if it hasn't been logged before"""
        if self.record_prompts and prompt_type not in self.logged_prompts:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"FIRST {prompt_type.upper()} PROMPT:\n")
            self.logger.info(prompt)
            self.logger.info(f"{'='*60}\n")
            self.logged_prompts.add(prompt_type)
    
    def _log_llm_conversation(self, conversation_type: str, messages: List[Dict], response: str, variant_id: int = -1, step: int = -1):
        """Log detailed LLM conversation history"""
        if not self.record_details:
            return
            
        prefix = f"[PATH_{variant_id:02d}]" if variant_id >= 0 else "[GENERAL]"
        step_info = f" Step {step}" if step >= 0 else ""
        
        self.logger.info(f"\n{'-'*50}")
        self.logger.info(f"{prefix} LLM CONVERSATION - {conversation_type.upper()}{step_info}")
        self.logger.info(f"Total LLM calls so far: {self.total_llm_calls}")
        self.logger.info(f"{'-'*50}")
        
        # Log conversation messages
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            self.logger.info(f"[{i+1}] {role}:")
            # Truncate very long content for readability
            if len(content) > 2000:
                self.logger.info(f"{content[:1000]}...\n[TRUNCATED - {len(content)} total chars]...\n{content[-500:]}")
            else:
                self.logger.info(content)
            self.logger.info("")
        
        # Log response
        self.logger.info(f"LLM RESPONSE:")
        if len(response) > 2000:
            self.logger.info(f"{response[:1000]}...\n[TRUNCATED - {len(response)} total chars]...\n{response[-500:]}")
        else:
            self.logger.info(response)
        self.logger.info(f"{'-'*50}\n")
    
    def _log_code_execution(self, code: str, passed: bool, info: str, context: str = "", variant_id: int = -1):
        """Log code execution details"""
        prefix = f"[PATH_{variant_id:02d}]" if variant_id >= 0 else "[EXECUTION]"
        status = "✓ PASSED" if passed else "✗ FAILED"
        
        self.logger.info(f"\n{prefix} CODE EXECUTION - {status}")
        if context:
            self.logger.info(f"Context: {context}")
        self.logger.info(f"Code length: {len(code.encode())} bytes")
        self.logger.info(f"Code:\n{code}")
        self.logger.info(f"Execution result: {info}")
        self.logger.info("")
    
    def _load_shortest_length(self) -> Optional[int]:
        """Load shortest known solution length from CSV"""
        try:
            with open(os.path.join(self.project_path, "files", "shortest_solutions_len.csv"), 'r') as f:
                reader = csv.reader(f)
                lengths = list(reader)
                if self.task_num <= len(lengths):
                    return int(lengths[self.task_num - 1][0])
        except Exception as e:
            self.logger.warning(f"Could not load shortest length: {e}")
        return None
    
    def _load_generator_code(self) -> str:
        """Load generator code for the task"""
        try:
            with open(os.path.join(self.project_path, "generate", f"task{self.task_num:03d}.py"), 'r') as f:
                code = f.read()
                return "def " + code.split("def")[1].strip()
        except Exception as e:
            self.logger.warning(f"Could not load generator code: {e}")
            return ""
    
    def _load_tricks(self) -> List[Dict]:
        """Load code golf tricks from JSON"""
        try:
            with open(os.path.join(self.project_path, "files", "tricks.json"), 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load tricks: {e}")
            return []
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from response, considering answer_begin/answer_end format"""
        # First try to extract content from <answer_begin>...</answer_end> blocks
        answer_block_pattern = r'<answer_begin>(.*?)<answer_end>'
        answer_match = re.search(answer_block_pattern, response, re.DOTALL)
        
        if answer_match:
            answer_content = answer_match.group(1).strip()
        else:
            # Fallback to full response if no answer block found
            answer_content = response
        
        # Now extract code from markdown blocks within the answer content
        patterns = [
            r"```(?:python|py|python3)\s+(.*?)\s*```",
            r"```\s*(.*?)\s*```"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, answer_content, re.DOTALL)
            if matches:
                code = matches[-1].strip()
                if 'def p(' not in code and 'def solve(' in code:
                    code = code.replace('def solve(', 'def p(')
                elif 'def ' not in code:
                    lines = code.split('\n')
                    indented_lines = [' ' + line if line.strip() else line for line in lines]
                    code = 'def p(g):\n' + '\n'.join(indented_lines)
                return code
        return None
    
    def _extract_variants(self, response: str) -> List[tuple[str, str]]:
        """Extract multiple code variants from LLM response"""
        variants = []
        
        # First extract content from <answer_begin>...</answer_end> block
        answer_block_pattern = r'<answer_begin>(.*?)<answer_end>'
        answer_match = re.search(answer_block_pattern, response, re.DOTALL)
        
        if answer_match:
            answer_content = answer_match.group(1).strip()
        else:
            # Fallback to full response if no answer block found
            answer_content = response
        
        # Look for sections like "### Variant 1:", "### Variant 2:", etc.
        sections = re.split(r'(?:##|###)\s*(?:Strategy|Variant|Approach)\s*\d+[:\s]*', answer_content)
        
        for section in sections[1:]:  # Skip first empty section
            # Extract strategy description from **Strategy:** field
            strategy_match = re.search(r'\*\*(?:Strategy|Core Strategy):\*\*\s*(.*?)(?:\*\*|```|$)', section, re.DOTALL)
            if strategy_match:
                strategy = strategy_match.group(1).strip()
                # Clean up the strategy text
                strategy = re.sub(r'\s+', ' ', strategy)
                strategy = strategy.split('\n')[0]  # Take first line only
            else:
                # Fallback: extract first meaningful line
                lines = section.strip().split('\n')
                strategy = ""
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('**') and not line.startswith('```'):
                        strategy = line
                        break
                if not strategy:
                    strategy = "Unknown Strategy"
            
            # Extract code
            code = self._extract_code(section)
            if code and strategy:
                variants.append((code, strategy))
                
            # Also check for variant end marker
            if "**End of Variant" in section:
                continue
        
        return variants
    
    def _create_variant_generation_prompt(self) -> str:
        """Create prompt for generating multiple code variants"""
        examples_str = self._format_examples()
        
        shortest_hint = ""
        if self.shortest_known:
            shortest_hint = f"\nNote: The shortest known solution for this task is {self.shortest_known} bytes. Try to approach or beat this length."
        
        return VARIANT_GENERATION_PROMPT.format(
            n_variants=int(self.n_variants * self.generation_factor),
            examples_str=examples_str,
            generator_code=self.generator_code,
            initial_solution=self.initial_solution,
        )
    
    def _create_optimization_prompt(self, code: str, attempt_history: List[Attempt], iteration: int) -> str:
        """Create prompt for optimizing a specific variant"""
        examples_str = self._format_examples()
        
        # Build history context
        history_str = ""
        if attempt_history:
            history_str = "\nOptimization History:\n"
            for i, attempt in enumerate(attempt_history[-3:]):
                status = "✓ PASSED" if attempt.passed else "✗ FAILED"
                length = attempt.min_length() if attempt.passed else "N/A"
                history_str += f"Attempt {i+1}: {status}, {length} bytes\n"
                if attempt.summary:
                    history_str += f"Notes: {attempt.summary}\n"
        
        shortest_hint = ""
        if self.shortest_known:
            shortest_hint = f"\nNote: The shortest known solution is {self.shortest_known} bytes. Try to approach or beat this length."
        
        return OPTIMIZATION_PROMPT.format(
            examples_str=examples_str,
            generator_code=self.generator_code,
            code=code,
            history_str=history_str,
            shortest_hint=shortest_hint if self.shortest_known else "",
        )
    
    def _create_tricks_prompt(self, code: str) -> str:
        """Create prompt for applying specific golf tricks"""
        tricks_str = ""
        for i, trick in enumerate(self.tricks):
            tricks_str += f"\n{i+1}. {trick['trick']}\n"
            tricks_str += f"   Before: {trick['input']}\n"
            tricks_str += f"   After: {trick['output']}\n"
            tricks_str += f"   Note: {trick['note']}\n"
        
        if self.shortest_known:
            shortest_hint = f"\nNote: The shortest known solution is {self.shortest_known} bytes. Try to approach or beat this length.\n"

        return TRICKS_PROMPT.format(
            code=code,
            tricks_str=tricks_str,
            shortest_hint=shortest_hint if self.shortest_known else "",
        )
    
    def _format_examples(self) -> str:
        """Format examples for display in prompt, combining train, test, and arc-gen."""
        result = "Examples:\n"
        total_length = 0
        max_length = 8000  # Maximum total length of examples
        max_examples = 10  # Maximum number of examples to include

        # Combine examples from train, test, and arc-gen
        all_examples = (
            self.examples.get('train', []) +
            self.examples.get('test', []) +
            self.examples.get('arc-gen', [])
        )

        for i, example in enumerate(all_examples[:max_examples]):
            example_str = (
                f"Example {i+1}:\n"
                f"Input: {example['input']}\n"
                f"Output: {example['output']}\n\n"
            )
            if total_length + len(example_str) > max_length:
                break

            result += example_str
            total_length += len(example_str)

        return result
    
    async def _generate_variants(self) -> List[Attempt]:
        """Generate multiple code variants using different strategies"""
        self.logger.info(f"Generating {int(self.n_variants * self.generation_factor)} code variants...")
        
        prompt = self._create_variant_generation_prompt()
        self._log_prompt_once("variant_generation", prompt)
        
        self.total_llm_calls += 1
        messages = [{"role": "user", "content": prompt}]
        
        # set temperature higher for more diversity
        self.logger.info(f"[VARIANT_GEN] Call LLM to generate variants...")
        response = await self.llm.chat(
            messages,
            temperature=0.8,
            reasoning_effort="high",
            response_format={"type": "text"}  # Use text format to parse multiple code blocks
        )
        
        # Log the conversation
        self._log_llm_conversation("VARIANT_GENERATION", messages, response)
        generate_history = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        
        # Extract variants
        variants_data = self._extract_variants(response)
        variants = []
        
        self.logger.info(f"[VARIANT_GEN] Extracted {len(variants_data)} variants from LLM response")
        
        for i, (code, strategy) in enumerate(variants_data):
            if code:
                self.logger.info(f"[VARIANT_GEN] Testing variant {i+1}: {strategy}")
                
                # Test the variant
                passed, info = await self.judge.execute(code)
                attempt = Attempt(code=code, passed=passed, summary=info, strategy=strategy)
                
                # Log execution details
                self._log_code_execution(code, passed, info, f"Variant {i+1}: {strategy}")
                
                # Always try to minify the code
                try:
                    minified = await minify_code(code)
                    attempt.update_minified(minified)
                    self.logger.info(f"[VARIANT_GEN] Minified from {len(code.encode())} to {len(minified.encode())} bytes")
                except Exception as e:
                    self.logger.warning(f"Failed to minify code: {e}")
                
                variants.append(attempt)
                self.logger.info(f"[VARIANT_GEN] {strategy} - {'PASSED' if passed else 'FAILED'} - {attempt.min_length()} bytes")
        
        # Select best variants (prioritize working ones, then by length)
        working_variants = [v for v in variants if v.passed]
        failed_variants = [v for v in variants if not v.passed]
        
        self.logger.info(f"[VARIANT_GEN] Generated {len(variants)} total variants: {len(working_variants)} working, {len(failed_variants)} failed")
        
        # Sort working variants by length, take best ones
        working_variants.sort(key=lambda x: x.min_length())
        selected = working_variants[:self.n_variants]
        
        # If we don't have enough working variants, try to fix some failed ones
        if len(selected) < self.n_variants:
            remaining = self.n_variants - len(selected)
            self.logger.info(f"[VARIANT_GEN] Need {remaining} more variants, attempting to fix failed ones...")
            fixed_variants = await self._fix_failed_variants(failed_variants, generate_history, remaining)  # Try to fix more than needed
            selected.extend(fixed_variants[:remaining])
        
        self.logger.info(f"[VARIANT_GEN] Selected {len(selected)} variants for optimization ({len([v for v in selected if v.passed])} working)")
        
        # Log selected variants summary
        for i, variant in enumerate(selected):
            status = "✓" if variant.passed else "✗"
            self.logger.info(f"[VARIANT_GEN] Selected {i+1}: {status} {variant.min_length()} bytes - {variant.strategy}")
        
        return selected

    async def _fix_failed_variants(self, failed_variants: List[Attempt], generate_history, fix_num) -> List[Attempt]:
        """Try to fix failed variants by addressing their specific issues with multiple iterations and parallel processing"""
        fixed_variants = []
        
        self.logger.info(f"[VARIANT_FIX] Attempting to fix {len(failed_variants)} failed variants, need {fix_num} successful fixes")
        
        # Create tasks for parallel processing
        fix_tasks = []
        for i, variant in enumerate(failed_variants):
            task = asyncio.create_task(self._fix_single_variant(variant, generate_history, i))  # Wrap coroutine as Task
            fix_tasks.append(task)
        
        # Process variants in parallel and collect results as they complete
        completed_count = 0
        for completed_task in asyncio.as_completed(fix_tasks):
            try:
                fixed_attempt = await completed_task
                if fixed_attempt and fixed_attempt.passed:
                    fixed_variants.append(fixed_attempt)
                    self.logger.info(f"[VARIANT_FIX] ✓ Successfully fixed variant ({len(fixed_variants)}/{fix_num}): {fixed_attempt.strategy} - {fixed_attempt.min_length()} bytes")
                    
                    # Stop when we have enough successful fixes
                    if len(fixed_variants) >= fix_num:
                        self.logger.info(f"[VARIANT_FIX] Reached target of {fix_num} successful fixes, stopping")
                        break
                
                completed_count += 1
                if completed_count >= len(failed_variants):
                    self.logger.info(f"[VARIANT_FIX] All variants processed")
                    break
                    
            except Exception as e:
                self.logger.warning(f"[VARIANT_FIX] ✗ Failed to process variant task: {e}")
                completed_count += 1
        
        # Cancel remaining tasks if we reached the target
        for task in fix_tasks:
            if not task.done():
                task.cancel()
        
        self.logger.info(f"[VARIANT_FIX] Successfully fixed {len(fixed_variants)}/{len(failed_variants)} variants")
        return fixed_variants[:fix_num]
    
    async def _fix_single_variant(self, variant: Attempt, generate_history: List[Dict], variant_index: int) -> Optional[Attempt]:
        """Fix a single variant with multiple iterations and conversation history"""
        variant_prefix = f"[FIX_{variant_index:02d}]"
        self.logger.info(f"{variant_prefix} Starting fix for variant: {variant.strategy}")
        self.logger.info(f"{variant_prefix} Original code:\n{variant.code}")
        self.logger.info(f"{variant_prefix} Error: {variant.summary}")
        
        # Initialize conversation history with generation context
        history = generate_history[:]  # Copy the generation history
        
        # Add initial fix request
        initial_fix_message = f"""The following code variant failed during testing:

**Strategy:** {variant.strategy}

**Failed Code:**
```python
{variant.code}
```

**Error:** {variant.summary}

Please fix this code. Provide the corrected code in a Python code block, which means using triple backticks (```python ... ```). The fixed code should also follows the previous requirements."""

        history.append({"role": "user", "content": initial_fix_message})
        
        for iteration in range(self.max_iterations):
            try:
                self.total_llm_calls += 1
                self.logger.info(f"{variant_prefix} Fix attempt {iteration+1}/{self.max_iterations}")
                
                # Call LLM with conversation history
                self.logger.info(f"{variant_prefix} Call LLM to fix variant...")
                response = await self.llm.chat(
                    history,
                    temperature=0.4,
                    reasoning_effort="high",
                    response_format={"type": "text"}
                )
                
                # Log the conversation for first few iterations
                self._log_llm_conversation("VARIANT_FIX", history[-1:], response)
                
                # Add LLM response to history
                history.append({"role": "assistant", "content": response})
                
                # Extract and test the fixed code
                fixed_code = self._extract_code(response)
                if not fixed_code:
                    self.logger.warning(f"{variant_prefix} No code extracted from response on iteration {iteration+1}")
                    if iteration < self.max_iterations - 1:
                        history.append({
                            "role": "user", 
                            "content": "No code extracted from your response. Please provide the fixed code in a Python code block, using triple backticks (```python ... ```). The fixed code should also follows the previous requirements."
                        })
                    continue
                
                # Test the fixed code
                passed, info = await self.judge.execute(fixed_code)
                fixed_attempt = Attempt(code=fixed_code, passed=passed, summary=info, strategy=f"Fixed: {variant.strategy}")
                
                # Log execution details
                self._log_code_execution(fixed_code, passed, info, f"Fixed variant iteration {iteration+1}", variant_index)
                
                if passed:
                    # Minify the fixed code
                    try:
                        minified = await minify_code(fixed_code)
                        fixed_attempt.update_minified(minified)
                        self.logger.info(f"{variant_prefix} Minified from {len(fixed_code.encode())} to {len(minified.encode())} bytes")
                    except Exception as e:
                        self.logger.warning(f"Failed to minify code: {e}")
                    
                    self.logger.info(f"{variant_prefix} ✓ Successfully fixed on iteration {iteration+1}")
                    return fixed_attempt
                else:
                    self.logger.warning(f"{variant_prefix} ✗ Fix attempt {iteration+1} failed: {info}")
                    if iteration < self.max_iterations - 1:
                        # Add failure feedback to conversation
                        history.append({
                            "role": "user",
                            "content": f"The code still failed with error:\n{info}. Please fix this issue and provide the corrected code in a Python code block, using triple backticks (```python ... ```). The fixed code should also follows the previous requirements."
                        })
                    
            except Exception as e:
                self.logger.warning(f"{variant_prefix} ✗ Exception during fix attempt {iteration+1}: {e}")
                if iteration < self.max_iterations - 1:
                    history.append({
                        "role": "user",
                        "content": "There was an error processing your response. Please provide the fixed code again in the required format, using triple backticks (```python ... ```), and ensure it follows the previous requirements."
                    })
        
        self.logger.warning(f"{variant_prefix} ✗ Failed to fix variant after {self.max_iterations} attempts")
        return None
    
    async def _optimize_variant(self, initial_attempt: Attempt, variant_id: int) -> List[Attempt]:
        """Optimize a single variant through sequential improvement"""
        path_prefix = f"[PATH_{variant_id:02d}]"
        self.logger.info(f"{path_prefix} Starting optimization: {initial_attempt.strategy}")
        
        attempts = [initial_attempt]
        current_best = initial_attempt
        improvements = []
        
        # Regular optimization steps
        for step in range(self.max_steps):
            self.logger.info(f"{path_prefix} Step {step+1}/{self.max_steps}")
            
            # Create conversation history for this optimization
            history = []
            best_before = current_best.min_length()
            
            # Generate optimization prompt
            prompt = self._create_optimization_prompt(current_best.code, attempts[-3:], step)
            if variant_id == 0:  # Only log the first variant's optimization prompt
                self._log_prompt_once("optimization", prompt)
            
            history = [{"role": "user", "content": prompt}]
            
            # Try to get improved code
            for iteration in range(self.max_iterations):
                self.total_llm_calls += 1
                self.logger.info(f"{path_prefix} Step {step+1}, Iteration {iteration+1}/{self.max_iterations}")
                
                # 代码优化：低温度以获得精确性
                self.logger.info(f"{path_prefix} Call LLM to optimize code...")
                response = await self.llm.chat(
                    history,
                    temperature=0.3,
                    reasoning_effort="high",
                    response_format={"type": "text"}
                )
                
                # Log conversation for first path or on important steps
                self._log_llm_conversation("OPTIMIZATION", history[-1:], response, variant_id, step)
                
                history.append({"role": "assistant", "content": response})
                
                code = self._extract_code(response)
                if not code:
                    self.logger.warning(f"{path_prefix} No code extracted from response")
                    history.append({
                        "role": "user", 
                        "content": "Please provide your optimized code in a Python code block, using triple backticks (```python ... ```). The optimized code should also follows the previous requirements."
                    })
                    continue
                
                # Test the code
                passed, info = await self.judge.execute(code)
                attempt = Attempt(code=code, passed=passed, summary=info, strategy=current_best.strategy)
                
                # Log execution details
                self._log_code_execution(code, passed, info, f"Optimization step {step+1}, iteration {iteration+1}", variant_id)
                
                # Always try to minify the code
                try:
                    minified = await minify_code(code)
                    attempt.update_minified(minified)
                    if passed:
                        self.logger.info(f"{path_prefix} Minified from {len(code.encode())} to {len(minified.encode())} bytes")
                except Exception as e:
                    self.logger.warning(f"Failed to minify code: {e}")
                
                if passed:
                    self.logger.info(f"{path_prefix} ✓ Generated working code ({attempt.min_length()} bytes)")
                    break
                else:
                    self.logger.warning(f"{path_prefix} ✗ Code failed: {info}")
                    if iteration < self.max_iterations - 1:
                        history.append({
                            "role": "user",
                            "content": f"Code failed:\n{info}.\n Please fix and try again. Provide the corrected code in a Python code block, using triple backticks (```python ... ```). The fixed code should also follows the previous requirements."
                        })
                    else:
                        self.logger.warning(f"{path_prefix} All iterations failed for this step")
            
            attempts.append(attempt)
            
            # Check for improvement
            if attempt.passed and attempt.min_length() < best_before:
                improvement = best_before - attempt.min_length()
                current_best = attempt
                improvements.append(True)
                self.logger.info(f"{path_prefix} ✓ Improved by {improvement} bytes to {attempt.min_length()}")
            else:
                improvements.append(False)
                self.logger.info(f"{path_prefix} ⚬ No improvement, best remains {current_best.min_length()}")
            
            # Early stopping
            if len(improvements) >= self.early_stop_steps:
                recent = improvements[-self.early_stop_steps:]
                if not any(recent):
                    self.logger.info(f"{path_prefix} Early stopping - no improvements in {self.early_stop_steps} steps")
                    break
        
        # Apply tricks if optimization stalled
        if current_best.passed and len(improvements) >= 2 and not any(improvements[-2:]):
            self.logger.info(f"{path_prefix} Applying golf tricks...")
            tricks_result = await self._apply_tricks_iteratively(current_best, variant_id)
            if tricks_result and tricks_result.passed and tricks_result.min_length() < current_best.min_length():
                attempts.append(tricks_result)
                current_best = tricks_result
                self.logger.info(f"{path_prefix} ✓ Tricks improved to {tricks_result.min_length()} bytes")
        
        self.logger.info(f"{path_prefix} Final result: {current_best.min_length()} bytes ({'PASSED' if current_best.passed else 'FAILED'})")
        return attempts
    
    async def _apply_tricks_iteratively(self, current_best: Attempt, variant_id: int = -1) -> Optional[Attempt]:
        """Apply golf tricks iteratively until no more improvements"""
        path_prefix = f"[PATH_{variant_id:02d}]" if variant_id >= 0 else "[TRICKS]"
        self.logger.info(f"{path_prefix} Starting iterative tricks application...")
        
        best_so_far = current_best
        iteration = 0
        max_trick_iterations = 5  # Prevent infinite loops
        
        while iteration < max_trick_iterations:
            iteration += 1
            self.logger.info(f"{path_prefix} Tricks iteration {iteration}")
            
            # First, scan knowledge base for applicable tricks
            knowledge_tricks = await self._scan_knowledge_base_tricks(best_so_far, variant_id)
            
            # Then apply provided tricks
            provided_tricks = await self._apply_provided_tricks(best_so_far, variant_id)
            
            # Try both approaches and pick the best
            candidates = [best_so_far]
            if knowledge_tricks and knowledge_tricks.passed:
                candidates.append(knowledge_tricks)
            if provided_tricks and provided_tricks.passed:
                candidates.append(provided_tricks)
            
            # Find the best candidate
            new_best = min(candidates, key=lambda x: x.min_length())
            
            # Check if we made progress
            if new_best.min_length() < best_so_far.min_length():
                improvement = best_so_far.min_length() - new_best.min_length()
                self.logger.info(f"{path_prefix} Tricks iteration {iteration}: ✓ Improved by {improvement} bytes to {new_best.min_length()}")
                best_so_far = new_best
            else:
                self.logger.info(f"{path_prefix} Tricks iteration {iteration}: ⚬ No improvement, stopping")
                break
        
        return best_so_far if best_so_far != current_best else None
    
    async def _scan_knowledge_base_tricks(self, current_best: Attempt, variant_id: int = -1) -> Optional[Attempt]:
        """Scan LLM's knowledge base for applicable golf tricks"""
        path_prefix = f"[PATH_{variant_id:02d}]" if variant_id >= 0 else "[TRICKS]"
        self.logger.info(f"{path_prefix} Scanning knowledge base for applicable tricks...")
        
        shortest_hint = ""
        if self.shortest_known:
            shortest_hint = f"\nNote: The shortest known solution is {self.shortest_known} bytes. Try to approach or beat this length."
        
        scan_prompt = KNOWLEDGE_BASE_TRICKS_PROMPT.format(
            code=current_best.code,
            shortest_hint=shortest_hint if self.shortest_known else "",
        )
        if variant_id == 0:  # Only log the first variant's knowledge base tricks prompt
            self._log_prompt_once("knowledge_base_tricks", scan_prompt)

        try:
            self.total_llm_calls += 1
            messages = [{"role": "user", "content": scan_prompt}]
            
            # 知识库扫描：中等温度，专注于技巧应用
            self.logger.info(f"{path_prefix} Call LLM to scan knowledge base for tricks...")
            response = await self.llm.chat(
                messages,
                temperature=0.5,
                reasoning_effort="medium",
                response_format={"type": "text"}
            )
            
            # Log conversation for first variant
            self._log_llm_conversation("KNOWLEDGE_BASE_TRICKS", messages, response, variant_id)
            
            if "NO_APPLICABLE_TRICKS".lower() in response.lower():
                self.logger.info(f"{path_prefix} LLM found no applicable knowledge base tricks")
                return None
            
            code = self._extract_code(response)
            if not code or code == current_best.code:
                self.logger.info(f"{path_prefix} No new code from knowledge base tricks")
                return None
            
            passed, info = await self.judge.execute(code)
            attempt = Attempt(code=code, passed=passed, summary=f"Knowledge base tricks: {info}", strategy=current_best.strategy)
            
            # Log execution details
            self._log_code_execution(code, passed, info, "Knowledge base tricks application", variant_id)
            
            if passed:
                try:
                    minified = await minify_code(code)
                    attempt.update_minified(minified)
                    improvement = current_best.min_length() - attempt.min_length()
                    self.logger.info(f"{path_prefix} ✓ Knowledge base tricks improved by {improvement} bytes ({attempt.min_length()} bytes)")
                except:
                    pass
            else:
                self.logger.warning(f"{path_prefix} ✗ Knowledge base tricks produced failing code")
            
            return attempt
        except Exception as e:
            self.logger.warning(f"{path_prefix} Knowledge base tricks failed: {e}")
            return None
    
    async def _apply_provided_tricks(self, current_best: Attempt, variant_id: int = -1) -> Optional[Attempt]:
        """Apply tricks from the provided tricks database"""
        path_prefix = f"[PATH_{variant_id:02d}]" if variant_id >= 0 else "[TRICKS]"
        self.logger.info(f"{path_prefix} Applying provided tricks database...")
        
        tricks_str = ""
        for i, trick in enumerate(self.tricks):
            tricks_str += f"\n{i+1}. {trick['trick']}\n"
            tricks_str += f"   Before: {trick['input']}\n"
            tricks_str += f"   After: {trick['output']}\n"
            tricks_str += f"   Note: {trick['note']}\n"
        
        self.logger.info(f"{path_prefix} Using {len(self.tricks)} tricks from database")
        
        prompt = PROVIDED_TRICKS_PROMPT.format(
            code=current_best.code,
            tricks_str=tricks_str
        )
        if variant_id == 0:  # Only log the first variant's provided tricks prompt
            self._log_prompt_once("provided_tricks", prompt)

        try:
            self.total_llm_calls += 1
            messages = [{"role": "user", "content": prompt}]
            
            # 提供技巧应用：低温度，高精确性
            self.logger.info(f"{path_prefix} Call LLM to apply provided tricks...")
            response = await self.llm.chat(
                messages,
                temperature=0.2,
                reasoning_effort="high",
                response_format={"type": "text"}
            )
            
            # Log conversation for first variant
            self._log_llm_conversation("PROVIDED_TRICKS", messages, response, variant_id)
            
            if "NO_APPLICABLE_TRICKS".lower() in response.lower():
                self.logger.info(f"{path_prefix} LLM found no applicable provided tricks")
                return None
            
            code = self._extract_code(response)
            if not code or code == current_best.code:
                self.logger.info(f"{path_prefix} No new code from provided tricks")
                return None
            
            passed, info = await self.judge.execute(code)
            attempt = Attempt(code=code, passed=passed, summary=f"Provided tricks: {info}", strategy=current_best.strategy)
            
            # Log execution details
            self._log_code_execution(code, passed, info, "Provided tricks application", variant_id)
            
            if passed:
                try:
                    minified = await minify_code(code)
                    attempt.update_minified(minified)
                    improvement = current_best.min_length() - attempt.min_length()
                    self.logger.info(f"{path_prefix} ✓ Provided tricks improved by {improvement} bytes ({attempt.min_length()} bytes)")
                except:
                    pass
            else:
                self.logger.warning(f"{path_prefix} ✗ Provided tricks produced failing code")
            
            return attempt
        except Exception as e:
            self.logger.warning(f"{path_prefix} Provided tricks application failed: {e}")
            return None
    
    async def _apply_tricks(self, current_best: Attempt) -> Optional[Attempt]:
        """Legacy single tricks application - kept for compatibility"""
        return await self._apply_provided_tricks(current_best)
    
    async def run(self) -> Attempt:
        """Run the advanced golf optimization process"""
        self.logger.info(f"Starting advanced golf optimization for task {self.task_num}...")
        
        # Verify initial solution
        initial_passed, info = await self.judge.execute(self.initial_solution)
        if not initial_passed:
            self.logger.error("Initial solution doesn't pass verification!")
            self.logger.error(info)
            return None
        
        # Phase 1: Generate diverse variants
        variants = await self._generate_variants()
        
        if not variants:
            self.logger.error("No variants generated!")
            return None
        
        # Phase 2: Optimize each variant in parallel
        self.logger.info(f"Starting parallel optimization of {len(variants)} variants...")
        self.logger.info("=" * 80)
        
        # create tasks for each variant
        optimization_tasks = []
        for i, variant in enumerate(variants):
            task = self._optimize_variant(variant, i)
            optimization_tasks.append(task)
        
        # run all optimizations concurrently
        all_paths_attempts = await asyncio.gather(*optimization_tasks)
        self.all_attempts = all_paths_attempts
        
        self.logger.info("=" * 80)
        self.logger.info("All parallel optimizations completed")
        
        # Phase 3: Select best result from all paths
        best_attempts = []
        for i, path_attempts in enumerate(all_paths_attempts):
            # Find best attempt in this path
            valid_attempts = [a for a in path_attempts if a.passed]
            if valid_attempts:
                path_best = min(valid_attempts, key=lambda x: x.min_length())
                best_attempts.append(path_best)
                self.logger.info(f"[PATH_{i:02d}] Best result: {path_best.min_length()} bytes ({path_best.strategy})")
            else:
                self.logger.info(f"[PATH_{i:02d}] No valid solutions found")
        
        if not best_attempts:
            self.logger.error("No valid solutions found!")
            return None
        
        # Select overall best
        self.best_attempt = min(best_attempts, key=lambda x: x.min_length())
        
        # Final summary
        initial_length = len(self.initial_solution.encode())
        final_length = self.best_attempt.min_length()
        total_improvement = initial_length - final_length
        
        self.logger.info("=== Advanced Golf Optimization Complete ===")
        self.logger.info(f"Task: {self.task_num}")
        self.logger.info(f"Variants generated: {len(variants)}")
        self.logger.info(f"Total LLM calls: {self.total_llm_calls}")
        self.logger.info(f"Initial solution: {initial_length} bytes")
        self.logger.info(f"Final solution: {final_length} bytes")
        self.logger.info(f"Total improvement: {total_improvement} bytes ({100*total_improvement/initial_length:.1f}%)")
        self.logger.info(f"Best strategy: {self.best_attempt.strategy}")
        
        # Log detailed comparison
        self.logger.info("\n=== SOLUTION COMPARISON ===")
        self.logger.info("Initial solution:")
        self.logger.info(self.initial_solution)
        self.logger.info(f"\nFinal solution ({final_length} bytes):")
        self.logger.info(self.best_attempt.code)
        if self.best_attempt.minified_code:
            self.logger.info(f"\nMinified solution ({len(self.best_attempt.minified_code.encode())} bytes):")
            self.logger.info(self.best_attempt.minified_code)
        
        # Log optimization statistics
        self.logger.info("\n=== OPTIMIZATION STATISTICS ===")
        for i, path_attempts in enumerate(self.all_attempts):
            valid_attempts = [a for a in path_attempts if a.passed]
            if valid_attempts:
                path_best = min(valid_attempts, key=lambda x: x.min_length())
                initial_len = path_attempts[0].min_length() if path_attempts[0].passed else "N/A"
                final_len = path_best.min_length()
                path_improvement = (path_attempts[0].min_length() - final_len) if path_attempts[0].passed else 0
                self.logger.info(f"Path {i:02d}: {initial_len} → {final_len} bytes ({path_improvement:+d}), {len(path_attempts)} attempts")
                self.logger.info(f"         Strategy: {path_best.strategy}")
            else:
                self.logger.info(f"Path {i:02d}: No valid solutions found")
        
        # Save final code
        if self.best_attempt:
            final_code = self.best_attempt.minified_code or self.best_attempt.code
            output_file = f"task{self.task_num:03d}_advanced_optimized.py"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_code)
            self.logger.info(f"\nFinal optimized code saved to {output_file}")
        
        return self.best_attempt

# Example usage
if __name__ == "__main__":
    import asyncio
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=4, help="Task number to optimize")
    parser.add_argument("--model", type=str, default="deepseek-reasoner", help="LLM model ID")
    parser.add_argument("--n_variants", type=int, default=20, help="Number of variants to generate")
    parser.add_argument("--generation_factor", type=float, default=1.5, help="Generation factor for variants")
    parser.add_argument("--max_steps", type=int, default=32, help="Max optimization steps per variant")
    parser.add_argument("--early_stop_steps", type=int, default=5, help="Early stopping steps")
    parser.add_argument("--max_iterations", type=int, default=5, help="Max iterations per optimization step")
    parser.add_argument("--use_shortest_hint", type=str, default="True", help="Use shortest hint")
    parser.add_argument("--record_prompts", type=str, default="True", help="Whether to record prompts in logs")
    parser.add_argument("--record_details", type=str, default="True", help="Whether to record detailed information in logs")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout for LLM calls in seconds")

    args = parser.parse_args()

    args.use_shortest_hint = args.use_shortest_hint.lower() == "true"
    args.record_prompts = args.record_prompts.lower() == "true"
    args.record_details = args.record_details.lower() == "true"

    async def main():
        task_num = args.task
        
        # Load initial solution
        try:
            with open(os.path.join(PROJECT_ROOT, "solutions", "latest", f"task{task_num:03d}.py"), 'r') as f:
                initial_solution = f.read()
        except:
            initial_solution = """def p(g):
    return g  # placeholder
"""
        
        # Initialize LLM and agent
        model_id = args.model
        print(f"Initializing LLM with model: {model_id}")
        llm = LLM(model_id=model_id, timeout=args.timeout)
        
        print(f"Creating GolfAgent for task {task_num}...")
        agent = GolfAgent(
            llm=llm, 
            task_num=task_num,
            initial_solution=initial_solution,
            n_variants=args.n_variants,
            generation_factor=args.generation_factor,
            max_steps=args.max_steps,
            max_iterations=args.max_iterations,
            early_stop_steps=args.early_stop_steps,
            use_shortest_hint=args.use_shortest_hint,
            record_prompts=args.record_prompts,
            record_details=args.record_details,
        )
        
        print(f"Starting optimization for task {task_num}...")
        # Run optimization
        best_solution = await agent.run()
        
        if best_solution:
            print(f"\nBest solution for task {task_num} ({best_solution.min_length()} bytes):")
            print(best_solution.code)
            if best_solution.minified_code:
                print(f"\nMinified ({len(best_solution.minified_code.encode())} bytes):")
                print(best_solution.minified_code)
    
    asyncio.run(main())