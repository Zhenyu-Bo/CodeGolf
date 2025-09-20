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
    step: int = 0  # Which step this attempt was created
    iteration: int = 0  # Which iteration within the step
    timestamp: str = ""  # When this attempt was created
    variant_index: int = -1  # Original variant index for conversation history tracking
    
    def __post_init__(self):
        self.length = len(self.code.encode())
        if self.minified_code:
            self.minified_length = len(self.minified_code.encode())
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S")
    
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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'code': self.code,
            'passed': self.passed,
            'length': self.length,
            'minified_code': self.minified_code,
            'minified_length': self.minified_length,
            'summary': self.summary,
            'strategy': self.strategy,
            'step': self.step,
            'iteration': self.iteration,
            'timestamp': self.timestamp,
            'variant_index': self.variant_index,
            'min_length': self.min_length() if self.passed else None
        }

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
    """Simple minification: remove comments and blank lines"""
    lines = code.split('\n')
    new_lines = []
    for line in lines:
        line = re.sub(r'#.*', '', line)  # Remove comments
        if line.strip() != '':
            new_lines.append(line.rstrip())
    return '\n'.join(new_lines)

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
        resume_from_state: bool = False,  # New parameter to enable state resumption
        reasoning_effort: str = "high",  # New parameter for reasoning effort
    ):
        self.project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.llm = llm
        self.reasoning_effort = reasoning_effort
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
        
        # Create proper log directory structure with timestamp folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_timestamp = timestamp  # Store for consistent use across methods
        
        # logs/task001/20250918_143022/
        task_logs_dir = os.path.join(self.project_path, 'logs', f"task{task_num:03d}")
        self.session_log_dir = os.path.join(task_logs_dir, timestamp)
        os.makedirs(self.session_log_dir, exist_ok=True)
        
        # Main agent log file: output.log
        self.main_log_file = os.path.join(self.session_log_dir, "output.log")
        
        # Print to console for immediate feedback
        print(f"Initializing GolfAgent for task {task_num}")
        print(f"Session log directory: {self.session_log_dir}")
        print(f"Main log file: {self.main_log_file}")
        
        self.logger = logger or self._setup_logger()
        
        # Immediate log to verify logger is working
        self.logger.info(f"GolfAgent initialized for task {task_num}")
        self.logger.info(f"Project path: {self.project_path}")
        self.logger.info(f"Session log directory: {self.session_log_dir}")
        self.logger.info(f"Main log file: {self.main_log_file}")
        
        # Load generator code and tricks
        self.generator_code = self._load_generator_code()
        self.tricks = self._load_tricks()
        
        # State tracking
        self.step = 0
        self.total_llm_calls = 0
        self.all_attempts: List[List[Attempt]] = []  # Attempts for each variant path
        self.best_attempt: Optional[Attempt] = None
        
        # Optimization path tracking
        self.optimization_paths: List[Dict] = []  # Track optimization state changes for each variant
        
        # Conversation tracking - unified per-variant system
        self.variant_histories: List[List[Dict]] = []  # Conversation history for each variant
        
        # State persistence
        self.save_file = os.path.join(self.project_path, 'outputs', f'task{task_num:03d}_state.json')
        os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
        
        # Prompt logging tracking
        self.logged_prompts = set()  # Track which prompt types have been logged

        self.record_prompts = record_prompts
        self.record_details = record_details
        
        # Try to load previous state if requested
        if resume_from_state:
            loaded = self._load_state_from_file()
            if loaded:
                self.logger.info("Successfully resumed from previous state")
            else:
                self.logger.info("No previous state found, starting fresh")
        
    def _setup_logger(self) -> logging.Logger:
        """Setup a logger that writes to output.log in the session timestamp directory"""
        # Use a unique logger name to avoid conflicts
        logger_name = f'GolfAgent_task{self.task_num}_{self.session_timestamp}'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Always add console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Add file handler for main log file (output.log)
        file_handler = logging.FileHandler(self.main_log_file, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logger initialized for task {self.task_num}, main log: {self.main_log_file}")
        
        return logger
    
    def _save_variant_conversations(self):
        """Variant conversations are now saved in real-time, this function kept for compatibility"""
        # Conversations are now saved in real-time via _append_conversation_realtime
        # This function is kept for compatibility but no longer does bulk saving
        self.logger.info("Variant conversations already saved in real-time to session directory")
    
    def _log_prompt_once(self, prompt_type: str, prompt: str):
        """Log a prompt only if it hasn't been logged before"""
        if self.record_prompts and prompt_type not in self.logged_prompts:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"FIRST {prompt_type.upper()} PROMPT:\n")
            self.logger.info(prompt)
            self.logger.info(f"{'='*60}\n")
            self.logged_prompts.add(prompt_type)
    
    def _log_llm_conversation(self, conversation_type: str, messages: List[Dict], response: str, variant_id: int = -1, step: int = -1):
        """Log LLM conversation summary to main log, full conversation is saved separately for variants"""
        if not self.record_details:
            return
            
        prefix = f"[PATH_{variant_id:02d}]" if variant_id >= 0 else "[GENERAL]"
        step_info = f" Step {step}" if step >= 0 else ""
        
        # Always log only summary info to main log (output.log) regardless of variant_id
        # Calculate conversation stats
        total_input_chars = sum(len(msg.get('content', '')) for msg in messages)
        response_chars = len(response)
        
        self.logger.info(f"{prefix} LLM CONVERSATION SUMMARY - {conversation_type.upper()}{step_info}")
        self.logger.info(f"Total LLM calls so far: {self.total_llm_calls}")
        self.logger.info(f"Input: {len(messages)} message(s), {total_input_chars} chars | Response: {response_chars} chars")
        
        if variant_id >= 0:
            self.logger.info(f"Full conversation saved to variant_{variant_id:02d}_conversations.json")
        else:
            self.logger.info(f"Full conversation saved to general_conversations.json")

    def _append_conversation_realtime(self, messages: List[Dict], response: str, conversation_type: str, variant_id: int = -1):
        """Append conversation to the appropriate JSON file in real-time"""
        try:
            # Determine the conversation file path
            if variant_id >= 0:
                conv_file = os.path.join(self.session_log_dir, f"variant_{variant_id:02d}_conversations.json")
            else:
                conv_file = os.path.join(self.session_log_dir, "general_conversations.json")
            
            # Create the conversation entry
            conversation_entry = {
                'timestamp': datetime.now().isoformat(),
                'conversation_type': conversation_type,
                'messages': messages,
                'response': response,
                'total_llm_calls': self.total_llm_calls
            }
            
            # Load existing data or create new structure
            if os.path.exists(conv_file):
                with open(conv_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                # Create initial structure
                data = {
                    'metadata': {
                        'task_num': self.task_num,
                        'variant_id': variant_id,
                        'session_timestamp': self.session_timestamp,
                        'created_at': datetime.now().isoformat(),
                        'total_turns': 0
                    },
                    'conversations': []
                }
            
            # Append the new conversation
            data['conversations'].append(conversation_entry)
            data['metadata']['total_turns'] = len(data['conversations'])
            data['metadata']['last_updated'] = datetime.now().isoformat()
            
            # Save back to file
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"Failed to append conversation to {conv_file}: {e}")
    
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
    
    def _record_optimization_step(self, variant_idx: int, step_type: str, 
                                  attempt: Optional[Attempt] = None, 
                                  details: Optional[Dict] = None):
        """Record an optimization step for tracking variant progress."""
        if variant_idx >= len(self.optimization_paths):
            # Initialize paths for new variants
            while len(self.optimization_paths) <= variant_idx:
                self.optimization_paths.append({
                    'variant_idx': len(self.optimization_paths),
                    'steps': [],
                    'current_status': 'initialized',
                    'best_length': None,
                    'test_status': None
                })
        
        step_data = {
            'step': self.step,
            'type': step_type,
            'timestamp': datetime.now().isoformat(),
        }
        
        if attempt:
            step_data.update({
                'code_length': len(attempt.code),
                'test_passed': attempt.passed,
                'strategy': attempt.strategy
            })
            
            # Update path summary
            path = self.optimization_paths[variant_idx]
            path['current_status'] = 'passed' if attempt.passed else 'failed'
            path['test_status'] = 'passed' if attempt.passed else 'failed'
            
            if attempt.passed:
                if path['best_length'] is None or len(attempt.code) < path['best_length']:
                    path['best_length'] = len(attempt.code)
        
        if details:
            step_data['details'] = details
            
        self.optimization_paths[variant_idx]['steps'].append(step_data)
        
        if self.record_details:
            self.logger.info(f"Optimization step recorded: variant {variant_idx}, "
                       f"type {step_type}, step {self.step}")

    def _save_state_to_file(self, include_analysis: bool = True):
        """Save the current state of the agent to file with optional analysis."""
        if not self.save_file:
            return

        self.logger.info(f'Step {self.step}: Saving agent state to {self.save_file}...')

        # Convert Attempt objects to dictionaries for JSON serialization
        all_attempts_serialized = []
        for path_attempts in self.all_attempts:
            path_serialized = [attempt.to_dict() for attempt in path_attempts]
            all_attempts_serialized.append(path_serialized)

        # Prepare optimization analysis if requested
        optimization_analysis = None
        if include_analysis and self.optimization_paths:
            optimization_analysis = {
                'summary': {
                    'total_variants': len(self.optimization_paths),
                    'variants_with_improvements': sum(1 for path in self.optimization_paths 
                                                    if path.get('best_length') is not None),
                    'total_steps_recorded': sum(len(path['steps']) for path in self.optimization_paths),
                    'successful_variants': sum(1 for path in self.optimization_paths 
                                             if path.get('test_status') == 'passed')
                },
                'paths': self.optimization_paths
            }

        # Gather all critical state variables into a dictionary
        state = {
            'task_num': self.task_num,
            'step': self.step,
            'total_llm_calls': self.total_llm_calls,
            'timestamp': datetime.now().isoformat(),
            # Convert attempts to dictionaries for JSON serialization
            'all_attempts': all_attempts_serialized,
            'best_attempt': self.best_attempt.to_dict() if self.best_attempt else None,
            'variant_histories': self.variant_histories,  # Unified conversation tracking
            'optimization_analysis': optimization_analysis,  # Combined optimization data
            'configuration': {
                'n_variants': self.n_variants,
                'generation_factor': self.generation_factor,
                'max_steps': self.max_steps,
                'early_stop_steps': self.early_stop_steps,
                'max_iterations': self.max_iterations,
                'use_shortest_hint': self.use_shortest_hint,
            },
            'initial_solution': self.initial_solution,
            'shortest_known': self.shortest_known,
        }

        try:
            with open(self.save_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            self.logger.info(f'Step {self.step}: Agent state successfully saved to {self.save_file}')
            
            # Save variant conversations to separate files
            self._save_variant_conversations()
            
            # Also save a separate optimization paths file if analysis is included
            if optimization_analysis:
                paths_file = f"task{self.task_num:03d}_optimization_paths.json"
                with open(paths_file, 'w', encoding='utf-8') as f:
                    json.dump(optimization_analysis, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Optimization analysis also saved to {paths_file}")
                
        except Exception as e:
            self.logger.error(f'Failed to save state to {self.save_file}: {e}')

    def _load_state_from_file(self) -> bool:
        """Load agent state from file if it exists."""
        if not os.path.exists(self.save_file):
            return False

        self.logger.info(f'Loading agent state from {self.save_file}...')
        
        try:
            with open(self.save_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Restore basic state
            self.step = state.get('step', 0)
            self.total_llm_calls = state.get('total_llm_calls', 0)
            self.variant_histories = state.get('variant_histories', [])  # Unified conversation tracking
            
            # Handle both old and new optimization data formats
            optimization_analysis = state.get('optimization_analysis')
            if optimization_analysis:
                self.optimization_paths = optimization_analysis.get('paths', [])
            else:
                # Fallback for old format
                self.optimization_paths = state.get('optimization_paths', [])
            
            # Restore attempts
            all_attempts_data = state.get('all_attempts', [])
            self.all_attempts = []
            for path_attempts_data in all_attempts_data:
                path_attempts = []
                for attempt_data in path_attempts_data:
                    attempt = Attempt(
                        code=attempt_data.get('code', ''),
                        passed=attempt_data.get('passed', False),
                        length=attempt_data.get('length', 0),
                        minified_code=attempt_data.get('minified_code', ''),
                        minified_length=attempt_data.get('minified_length', 0),
                        summary=attempt_data.get('summary', ''),
                        strategy=attempt_data.get('strategy', ''),
                        step=attempt_data.get('step', 0),
                        iteration=attempt_data.get('iteration', 0),
                        timestamp=attempt_data.get('timestamp', ''),
                        variant_index=attempt_data.get('variant_index', -1)
                    )
                    path_attempts.append(attempt)
                self.all_attempts.append(path_attempts)
            
            # Restore best attempt
            best_attempt_data = state.get('best_attempt')
            if best_attempt_data:
                self.best_attempt = Attempt(
                    code=best_attempt_data.get('code', ''),
                    passed=best_attempt_data.get('passed', False),
                    length=best_attempt_data.get('length', 0),
                    minified_code=best_attempt_data.get('minified_code', ''),
                    minified_length=best_attempt_data.get('minified_length', 0),
                    summary=best_attempt_data.get('summary', ''),
                    strategy=best_attempt_data.get('strategy', ''),
                    step=best_attempt_data.get('step', 0),
                    iteration=best_attempt_data.get('iteration', 0),
                    timestamp=best_attempt_data.get('timestamp', ''),
                    variant_index=best_attempt_data.get('variant_index', -1)
                )
            
            total_conversations = sum(len(vh) for vh in self.variant_histories)
            self.logger.info(f'Successfully loaded state: step {self.step}, {self.total_llm_calls} LLM calls, {total_conversations} conversations across {len(self.variant_histories)} variants')
            return True
            
        except Exception as e:
            self.logger.error(f'Failed to load state from {self.save_file}: {e}')
            return False

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
    
    def _create_optimization_prompt(self, code: str, step: int, variant_strategy: str = "") -> str:
        """Create prompt for optimizing a specific variant (simplified for conversation history)"""
        shortest_hint = ""
        if self.shortest_known:
            shortest_hint = f"\nNote: The shortest known solution is {self.shortest_known} bytes. Try to approach or beat this length."
        
        return OPTIMIZATION_PROMPT.format(
            variant_strategy=variant_strategy,
            code=code,
            generator_code=self.generator_code,
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
        max_length = 4000  # Maximum total length of examples
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
            reasoning_effort=self.reasoning_effort,
            response_format={"type": "text"}  # Use text format to parse multiple code blocks
        )
        
        # Log the conversation
        self._log_llm_conversation("VARIANT_GENERATION", messages, response)
        
        # Save conversation in real-time (general conversation, not variant-specific)
        self._append_conversation_realtime(messages, response, "VARIANT_GENERATION", variant_id=-1)
        
        generate_history = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        
        # Extract variants
        variants_data = self._extract_variants(response)
        variants = []
        
        self.logger.info(f"[VARIANT_GEN] Extracted {len(variants_data)} variants from LLM response")
        
        # Initialize variant_histories for each variant with the shared generation context
        self.variant_histories = []
        
        for i, (code, strategy) in enumerate(variants_data):
            if code:
                self.logger.info(f"[VARIANT_GEN] Testing variant {i+1}: {strategy}")
                
                # Create conversation history for this variant starting with generation context
                variant_history = generate_history[:]  # Copy the shared generation context
                self.variant_histories.append(variant_history)
                
                # Test the variant
                passed, info = await self.judge.execute(code)
                attempt = Attempt(code=code, passed=passed, summary=info, strategy=strategy)
                # Store the original variant index for later reference
                attempt.variant_index = i
                
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
            fixed_variants = await self._fix_failed_variants(failed_variants, remaining)
            selected.extend(fixed_variants[:remaining])
        
        self.logger.info(f"[VARIANT_GEN] Selected {len(selected)} variants for optimization ({len([v for v in selected if v.passed])} working)")
        
        # Log selected variants summary
        for i, variant in enumerate(selected):
            status = "✓" if variant.passed else "✗"
            self.logger.info(f"[VARIANT_GEN] Selected {i+1}: {status} {variant.min_length()} bytes - {variant.strategy}")
        
        return selected

    async def _fix_failed_variants(self, failed_variants: List[Attempt], fix_num) -> List[Attempt]:
        """Try to fix failed variants by addressing their specific issues with multiple iterations and parallel processing"""
        fixed_variants = []
        
        self.logger.info(f"[VARIANT_FIX] Attempting to fix {len(failed_variants)} failed variants, need {fix_num} successful fixes")
        
        # Create tasks for parallel processing
        fix_tasks = []
        for i, variant in enumerate(failed_variants):
            task = asyncio.create_task(self._fix_single_variant(variant, i))  # Wrap coroutine as Task
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
    
    async def _fix_single_variant(self, variant: Attempt, variant_index: int) -> Optional[Attempt]:
        """Fix a single variant with multiple iterations and conversation history"""
        # Get the original variant index to access the correct conversation history
        original_variant_id = getattr(variant, 'variant_index', variant_index)
        variant_prefix = f"[FIX_{variant_index:02d}]"
        self.logger.info(f"{variant_prefix} Starting fix for variant: {variant.strategy}")
        self.logger.info(f"{variant_prefix} Original code:\n{variant.code}")
        self.logger.info(f"{variant_prefix} Error: {variant.summary}")
        
        # Use the existing conversation history for this variant
        if original_variant_id < len(self.variant_histories):
            history = self.variant_histories[original_variant_id][:]  # Copy the variant's history
        else:
            self.logger.warning(f"{variant_prefix} No conversation history found for variant {original_variant_id}, creating new one")
            history = []
        
        # Add initial fix request
        initial_fix_message = f"""The following code variant failed during testing:

**Strategy:** {variant.strategy}

**Failed Code:**
```python
{variant.code}
```

**Error:** {variant.summary}

Please fix this code. You can refer to our previous discussions which include input-output examples, testcases generator code, and initial solution.

Note that the generator code is very important to understand the task requirements.

Provide the corrected code in a Python code block, which means using triple backticks (```python ... ```). The fixed code should also follows the previous requirements."""

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
                    reasoning_effort=self.reasoning_effort,
                    response_format={"type": "text"}
                )
                
                # Log the conversation for first few iterations
                self._log_llm_conversation("VARIANT_FIX", history[-1:], response, variant_id=original_variant_id)
                
                # Save conversation in real-time to variant file
                self._append_conversation_realtime(history[-1:], response, "VARIANT_FIX", variant_id=original_variant_id)
                
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
                    
                    # Update the variant's conversation history with the successful fix
                    if original_variant_id < len(self.variant_histories):
                        self.variant_histories[original_variant_id] = history[:]
                        self.logger.info(f"{variant_prefix} Updated conversation history for variant {original_variant_id}")
                    
                    # Store the original variant index for later reference
                    fixed_attempt.variant_index = original_variant_id
                    
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
        
        # Record optimization start
        self._record_optimization_step(variant_id, 'optimization_started', initial_attempt)
        
        attempts = [initial_attempt]
        current_best = initial_attempt
        improvements = []
        
        # Get the variant's existing conversation history (should contain generation and possibly fix context)
        original_variant_id = getattr(initial_attempt, 'variant_index', variant_id)
        if original_variant_id < len(self.variant_histories):
            variant_history = self.variant_histories[original_variant_id]
            self.logger.info(f"{path_prefix} Using existing conversation history with {len(variant_history)} turns")
        else:
            # Fallback: create new history if none exists
            self.logger.warning(f"{path_prefix} No conversation history found for variant {original_variant_id}, creating new one")
            while len(self.variant_histories) <= original_variant_id:
                self.variant_histories.append([])
            variant_history = self.variant_histories[original_variant_id]
        
        # Add initial optimization context to conversation history
        initial_prompt = self._create_optimization_prompt(current_best.code, 0, current_best.strategy)
        if variant_id == 0:  # Only log the first variant's optimization prompt
            self._log_prompt_once("optimization", initial_prompt)
        
        variant_history.append({"role": "user", "content": initial_prompt})
        
        # Regular optimization steps
        for step in range(self.max_steps):
            self.logger.info(f"{path_prefix} Step {step+1}/{self.max_steps}")
            
            best_before = current_best.min_length()
            
            # Try to get improved code
            for iteration in range(self.max_iterations):
                self.total_llm_calls += 1
                self.logger.info(f"{path_prefix} Step {step+1}, Iteration {iteration+1}/{self.max_iterations}")
                
                # Use accumulated conversation history
                self.logger.info(f"{path_prefix} Call LLM to optimize code with {len(variant_history)} conversation turns...")
                response = await self.llm.chat(
                    variant_history,
                    temperature=0.3,
                    reasoning_effort=self.reasoning_effort,
                    response_format={"type": "text"}
                )
                
                # Log conversation for first path or on important steps
                self._log_llm_conversation("OPTIMIZATION", variant_history[-1:], response, variant_id, step)
                
                # Save conversation in real-time to variant file
                self._append_conversation_realtime(variant_history[-1:], response, "OPTIMIZATION", variant_id=original_variant_id)
                
                variant_history.append({"role": "assistant", "content": response})
                
                code = self._extract_code(response)
                if not code:
                    self.logger.warning(f"{path_prefix} No code extracted from response")
                    follow_up = "Please provide your optimized code in a Python code block, using triple backticks (```python ... ```). The optimized code should also follows the previous requirements."
                    variant_history.append({"role": "user", "content": follow_up})
                    continue
                
                # Test the code
                passed, info = await self.judge.execute(code)
                attempt = Attempt(code=code, passed=passed, summary=info, strategy=current_best.strategy)
                # Inherit variant_index from the initial attempt
                attempt.variant_index = getattr(initial_attempt, 'variant_index', original_variant_id)
                
                # Record attempt
                self._record_optimization_step(variant_id, 'attempt_tested', attempt, 
                                             {'step': step, 'iteration': iteration})
                
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
                        fixed_prompt = f"The code you provided still failed with the following error:\n{info}.\nPlease fix this issue and provide the corrected code in a Python code block, using triple backticks (```python ... ```). The fixed code should also follows the previous requirements."
                        fixed_prompt += "\nYou can refer to our previous discussions which include input-output examples, testcases generator code, and initial solution."
                        fixed_prompt += "\nNote that the generator code is very important to understand the task requirements."
                        variant_history.append({"role": "user", "content": fixed_prompt})
                    else:
                        self.logger.warning(f"{path_prefix} All iterations failed for this step")
            
            attempts.append(attempt)
            
            # Check for improvement
            if attempt.passed and attempt.min_length() < best_before:
                improvement = best_before - attempt.min_length()
                current_best = attempt
                improvements.append(True)
                self.logger.info(f"{path_prefix} ✓ Improved by {improvement} bytes to {attempt.min_length()}")
                
                # Record improvement
                self._record_optimization_step(variant_id, 'improvement_found', attempt, 
                                             {'improvement_bytes': improvement, 'step': step})
                
                # Add improvement feedback to conversation history for next step
                if step < self.max_steps - 1:  # Don't add if this is the last step
                    improvement_feedback = f"Great! The code improved by {improvement} bytes to {attempt.min_length()} bytes. Let's continue optimizing to make it even shorter:"
                    variant_history.append({"role": "user", "content": improvement_feedback})
            else:
                improvements.append(False)
                self.logger.info(f"{path_prefix} ⚬ No improvement, best remains {current_best.min_length()}")
                
                # Record no improvement
                self._record_optimization_step(variant_id, 'no_improvement', attempt, 
                                             {'step': step, 'current_best_length': current_best.min_length()})
                
                # Add feedback for no improvement
                if step < self.max_steps - 1:  # Don't add if this is the last step
                    no_improvement_feedback = f"The current attempt didn't improve the length (still {current_best.min_length()} bytes). Let's try a different optimization approach:"
                    variant_history.append({"role": "user", "content": no_improvement_feedback})
            
            # Early stopping
            if len(improvements) >= self.early_stop_steps:
                recent = improvements[-self.early_stop_steps:]
                if not any(recent):
                    self.logger.info(f"{path_prefix} Early stopping - no improvements in {self.early_stop_steps} steps")
                    break
        
        # Apply tricks if optimization stalled
        if current_best.passed and len(improvements) >= 2 and not any(improvements[-2:]):
            self.logger.info(f"{path_prefix} Applying golf tricks...")
            self._record_optimization_step(variant_id, 'tricks_started', current_best)
            
            tricks_result = await self._apply_tricks_iteratively(current_best, variant_id)
            if tricks_result and tricks_result.passed and tricks_result.min_length() < current_best.min_length():
                attempts.append(tricks_result)
                current_best = tricks_result
                improvement = len(attempts[0].code) - tricks_result.min_length()
                self.logger.info(f"{path_prefix} ✓ Tricks improved to {tricks_result.min_length()} bytes")
                
                # Record tricks improvement
                self._record_optimization_step(variant_id, 'tricks_improved', tricks_result, 
                                             {'tricks_improvement_bytes': improvement})
            else:
                self._record_optimization_step(variant_id, 'tricks_no_improvement', current_best)
        
        # Record optimization completion
        self._record_optimization_step(variant_id, 'optimization_completed', current_best, 
                                     {'total_attempts': len(attempts), 'total_improvements': sum(improvements)})
        
        self.logger.info(f"{path_prefix} Final result: {current_best.min_length()} bytes ({'PASSED' if current_best.passed else 'FAILED'})")
        self.logger.info(f"{path_prefix} Conversation history accumulated: {len(variant_history)} turns")
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
                reasoning_effort=self.reasoning_effort,
                response_format={"type": "text"}
            )
            
            # Log conversation for first variant
            self._log_llm_conversation("KNOWLEDGE_BASE_TRICKS", messages, response, variant_id)
            
            # Save conversation in real-time to variant file
            self._append_conversation_realtime(messages, response, "KNOWLEDGE_BASE_TRICKS", variant_id=variant_id)
            
            if "NO_APPLICABLE_TRICKS".lower() in response.lower():
                self.logger.info(f"{path_prefix} LLM found no applicable knowledge base tricks")
                return None
            
            code = self._extract_code(response)
            if not code or code == current_best.code:
                self.logger.info(f"{path_prefix} No new code from knowledge base tricks")
                return None
            
            passed, info = await self.judge.execute(code)
            attempt = Attempt(code=code, passed=passed, summary=f"Knowledge base tricks: {info}", strategy=current_best.strategy)
            # Inherit variant_index from current_best
            attempt.variant_index = getattr(current_best, 'variant_index', -1)
            
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
                reasoning_effort=self.reasoning_effort,
                response_format={"type": "text"}
            )
            
            # Log conversation for first variant
            self._log_llm_conversation("PROVIDED_TRICKS", messages, response, variant_id)
            
            # Save conversation in real-time to variant file
            self._append_conversation_realtime(messages, response, "PROVIDED_TRICKS", variant_id=variant_id)
            
            if "NO_APPLICABLE_TRICKS".lower() in response.lower():
                self.logger.info(f"{path_prefix} LLM found no applicable provided tricks")
                return None
            
            code = self._extract_code(response)
            if not code or code == current_best.code:
                self.logger.info(f"{path_prefix} No new code from provided tricks")
                return None
            
            passed, info = await self.judge.execute(code)
            attempt = Attempt(code=code, passed=passed, summary=f"Provided tricks: {info}", strategy=current_best.strategy)
            # Inherit variant_index from current_best
            attempt.variant_index = getattr(current_best, 'variant_index', -1)
            
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
        
        # Record initial variant generation
        for i, variant in enumerate(variants):
            # Use original variant index for recording
            original_idx = getattr(variant, 'variant_index', i)
            self._record_optimization_step(original_idx, 'variant_generated', variant, 
                                         {'strategy': variant.strategy})
        
        # Phase 2: Optimize each variant in parallel
        self.logger.info(f"Starting parallel optimization of {len(variants)} variants...")
        self.logger.info("=" * 80)
        
        # create tasks for each variant
        optimization_tasks = []
        for i, variant in enumerate(variants):
            # Use the original variant index to ensure correct conversation history access
            original_variant_id = getattr(variant, 'variant_index', i)
            task = self._optimize_variant(variant, original_variant_id)
            optimization_tasks.append(task)
        
        # run all optimizations concurrently
        all_paths_attempts = await asyncio.gather(*optimization_tasks)
        self.all_attempts = all_paths_attempts
        
        # Save state after parallel optimization completes (similar to agent_v1)
        self._save_state_to_file()
        
        self.logger.info("=" * 80)
        self.logger.info("All parallel optimizations completed")
        
        # Phase 3: Select best result from all paths
        best_attempts = []
        for i, path_attempts in enumerate(all_paths_attempts):
            # Get the original variant for better logging
            variant = variants[i]
            original_variant_id = getattr(variant, 'variant_index', i)
            
            # Find best attempt in this path
            valid_attempts = [a for a in path_attempts if a.passed]
            if valid_attempts:
                path_best = min(valid_attempts, key=lambda x: x.min_length())
                best_attempts.append(path_best)
                self.logger.info(f"[PATH_{i:02d}/VAR_{original_variant_id:02d}] Best result: {path_best.min_length()} bytes ({path_best.strategy})")
            else:
                self.logger.info(f"[PATH_{i:02d}/VAR_{original_variant_id:02d}] No valid solutions found")
        
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
        
        # Save final state with optimization analysis
        self._save_state_to_file(include_analysis=True)
        
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
    parser.add_argument("--record_prompts", type=str, default="False", help="Whether to record prompts in logs")
    parser.add_argument("--record_details", type=str, default="True", help="Whether to record detailed information in logs")
    parser.add_argument("--resume_from_state", type=str, default="False", help="Resume from previous saved state")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout for LLM calls in seconds")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="Reasoning effort level for LLM calls")

    args = parser.parse_args()

    args.use_shortest_hint = args.use_shortest_hint.lower() == "true"
    args.record_prompts = args.record_prompts.lower() == "true"
    args.record_details = args.record_details.lower() == "true"
    args.resume_from_state = args.resume_from_state.lower() == "true"

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
            resume_from_state=args.resume_from_state,
            reasoning_effort=args.reasoning_effort,
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