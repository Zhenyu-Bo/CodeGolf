import re
import asyncio
import json
import os
import sys
import random
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# Import modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from modules.llm import LLM
from modules.common import CodeJudge, format_examples, minify_code, load_examples, load_generator_code, load_task_solution, extract_code, extract_answer_content, setup_logger
from modules.trick_pool import Trick, TrickPoolManager
from modules.prompt_v2 import (
    PROBLEM_UNDERSTANDING_PROMPT,
    TRICK_APPLICATION_PROMPT, 
    SELF_GOLF_PROMPT,
    KNOWLEDGE_SCAN_PROMPT
)


class SessionManager:
    """Simple session management for conversation history and logging"""
    
    def __init__(self, task_num: int, session_id: Optional[str] = None):
        self.task_num = task_num
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path("sessions") / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
    def get_log_file(self) -> str:
        return str(self.session_dir / f'task{self.task_num:03d}_agent.log')
    
    def save_history(self, conversation_history: List[Dict], current_code: str, initial_solution: str) -> None:
        """Save conversation history"""
        try:
            history_file = self.session_dir / f'task{self.task_num:03d}_history.json'
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'task_num': self.task_num,
                    'conversation_history': conversation_history,
                    'current_best_code': current_code,
                    'initial_solution': initial_solution,
                    'saved_at': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # Ignore save errors
    
    def load_latest_history(self) -> Optional[Dict]:
        """Load history from latest session"""
        try:
            sessions_dir = Path("sessions")
            if not sessions_dir.exists():
                return None
            
            # Find latest session with this task
            for session_dir in sorted(sessions_dir.iterdir(), reverse=True):
                if not session_dir.is_dir():
                    continue
                
                history_file = session_dir / f'task{self.task_num:03d}_history.json'
                if history_file.exists():
                    with open(history_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
            
            return None
        except Exception:
            return None


class TrickBasedGolfAgent:
    """Trick pool-based code golf optimization agent"""
    
    def __init__(
        self,
        llm: LLM,
        task_num: int,
        initial_solution: str,
        trick_pool: TrickPoolManager,
        max_iterations: int = 10,
        tricks_per_sample: int = 5,
        require_human_confirmation: bool = False,
        session_id: Optional[str] = None,
        restore_from_latest: bool = False
    ):
        self.llm = llm
        self.task_num = task_num
        self.initial_solution = initial_solution
        self.trick_pool = trick_pool
        self.max_iterations = max_iterations
        self.tricks_per_sample = tricks_per_sample
        self.require_human_confirmation = require_human_confirmation
        
        # Session management
        self.session_manager = SessionManager(task_num, session_id)
        self.logger = setup_logger(f"Agent_T{task_num}", self.session_manager.get_log_file())
        
        # Load task data
        self.examples = load_examples(task_num)
        self.generator_code = load_generator_code(task_num)
        self.judge = CodeJudge(task_num, self.examples)
        
        # Optimization state
        self.current_best_code = initial_solution
        self.current_best_length = len(initial_solution)
        self.problem_understood = False
        self.human_confirmed = False
        self.conversation_history: List[Dict[str, str]] = []
        
        # Restore from latest session if requested
        if restore_from_latest:
            self._restore_from_latest()
        
    def _restore_from_latest(self) -> None:
        """Restore from latest session if available"""
        history_data = self.session_manager.load_latest_history()
        if history_data:
            self.conversation_history = history_data.get('conversation_history', [])
            self.current_best_code = history_data.get('current_best_code', self.initial_solution)
            self.current_best_length = len(self.current_best_code)
            self.logger.info(f"Restored {len(self.conversation_history)} messages from previous session")
    
    async def _call_llm_with_history(self, prompt: str, phase: str = "optimization") -> str:
        """Call LLM with conversation history"""
        self.conversation_history.append({"role": "user", "content": prompt})
        response = await self.llm.call(self.conversation_history)
        self.conversation_history.append({"role": "assistant", "content": response})
        self.logger.info(f"[{phase}] LLM call completed")
        return response
    
    async def understand_problem(self) -> bool:
        """Problem understanding phase"""
        self.logger.info("Phase 1: Understanding the problem...")
        
        # Format examples
        examples_str = format_examples(examples=self.examples, max_examples=10, max_length=4000)
        
        # Build problem understanding prompt
        prompt = PROBLEM_UNDERSTANDING_PROMPT.format(
            examples_str=examples_str,
            generator_code=self.generator_code,
            current_solution=self.initial_solution
        )
        
        try:
            response = await self._call_llm_with_history(prompt, "problem_understanding")
            self.logger.info("Problem understanding response received")
            self.logger.info(f"Response: {response}")
            
            # If human confirmation is required
            if self.require_human_confirmation:
                print("\n" + "="*80)
                print("PROBLEM UNDERSTANDING ANALYSIS:")
                print("="*80)
                print(response)
                print("="*80)
                
                user_input = input("\nIs the problem understanding correct? (y/n): ").strip().lower()
                self.human_confirmed = user_input in ['y', 'yes']
                
                if not self.human_confirmed:
                    self.logger.info("Human rejected problem understanding")
                    return False
            else:
                self.human_confirmed = True
            
            return self.problem_understood and self.human_confirmed
            
        except Exception as e:
            self.logger.error(f"Failed to understand problem: {e}")
            return False
    
    async def apply_tricks_phase(self) -> bool:
        """Trick application phase"""
        self.logger.info("Phase 2: Applying tricks from pool...")
        
        # Add context about entering trick application phase
        if self.conversation_history:  # Only if we have previous conversation
            context_prompt = f"Now we'll move to Phase 2: Trick Application. I'll present you with proven optimization tricks from our knowledge base which have been effective in other similar tasks. Current code length: {self.current_best_length} chars."
            await self._call_llm_with_history(context_prompt, "phase_transition")
        
        # Reset sampling state
        self.trick_pool.reset_sampling()
        
        improved = False
        tested_tricks = set()
        
        while True:
            # Sample tricks
            available_tricks = self.trick_pool.sample_tricks(
                self.tricks_per_sample, 
                exclude_tasks={self.task_num}
            )
            
            if not available_tricks:
                self.logger.info("No more tricks available to test")
                break
            
            # Filter already tested tricks
            new_tricks = [t for t in available_tricks if t not in tested_tricks]
            if not new_tricks:
                break
            
            self.logger.info(f"Testing {len(new_tricks)} tricks...")
            
            # Build trick information
            tricks_info = ""
            for i, trick in enumerate(new_tricks, 1):
                tricks_info += f"Trick {i}: {trick.description}\n"
                tricks_info += f"Sub-problem: {trick.sub_problem}\n"
                tricks_info += f"Example transformation:\n"
                tricks_info += f"Before: {trick.input_code}\n"
                tricks_info += f"After: {trick.output_code}\n"
                tricks_info += f"Length reduction: {trick.length_reduction} chars\n\n"
            
            # Build trick application prompt
            prompt = TRICK_APPLICATION_PROMPT.format(
                current_code=self.current_best_code,
                tricks_info=tricks_info
            )
            
            try:
                response = await self._call_llm_with_history(prompt, "trick_application")
                
                # Parse response and extract applied tricks
                applied_any = await self._process_trick_applications(response, new_tricks)
                if applied_any:
                    improved = True
                
                # Mark tricks as tested
                tested_tricks.update(new_tricks)
                
            except Exception as e:
                self.logger.error(f"Failed to apply tricks: {e}")
                tested_tricks.update(new_tricks)
        
        return improved
    
    async def _process_trick_applications(self, response: str, tricks: List[Trick]) -> bool:
        """Process trick application response"""
        improved = False
        
        # Use regex to parse results for each trick
        trick_sections = re.split(r'\*\*Trick \d+:', response)[1:]  # Remove first empty split
        
        for i, section in enumerate(trick_sections):
            if i >= len(tricks):
                break
                
            trick = tricks[i]
            
            # Check if marked as applicable
            applicable_match = re.search(r'Applicable.*?:\s*(Yes|No)', section, re.IGNORECASE)
            if not applicable_match or applicable_match.group(1).lower() != 'yes':
                continue
            
            # Extract modified code
            modified_code = extract_code(section)
            if not modified_code:
                continue
            
            # Verify code
            passed, info = await self.judge.execute(modified_code)
            if not passed:
                self.logger.warning(f"Trick application failed verification: {info}")
                continue
            
            # Check length improvement
            new_length = len(modified_code)
            if new_length < self.current_best_length:
                improvement = self.current_best_length - new_length
                self.logger.info(f"Trick '{trick.description}' improved code by {improvement} chars")
                
                # Update current best code
                self.current_best_code = modified_code
                self.current_best_length = new_length
                
                # Update trick success usage
                self.trick_pool.update_trick_success(trick, self.task_num)
                improved = True
            else:
                self.logger.info(f"Trick '{trick.description}' did not improve length")
                # Unsuccessful tricks don't need updates since we only record success counts
        
        return improved
    
    async def self_golf_phase(self) -> bool:
        """Self-directed golf optimization phase"""
        self.logger.info("Phase 3: Self-directed golf optimization...")
        
        # Add context about entering self-golf phase
        if self.conversation_history:  # Only if we have previous conversation
            context_prompt = f"Now we'll move to Phase 3: Self-directed Golf Optimization. Please use your expertise to find structural improvements beyond the tricks we've already tried. Current code length: {self.current_best_length} chars."
            await self._call_llm_with_history(context_prompt, "phase_transition")
        
        improved = False
        
        for iteration in range(3):  # Perform 3 rounds of self-optimization
            self.logger.info(f"Self-golf iteration {iteration + 1}/3")
            
            # Add context about current optimization state if this is not the first Self Golf iteration
            if iteration > 0:
                context_prompt = f"We are now in self-golf iteration {iteration + 1}/3. Current best code length: {self.current_best_length} chars. Please continue optimizing based on our previous discussions."
                await self._call_llm_with_history(context_prompt, "self_golf_context")
            
            # Build self-golf prompt
            prompt = SELF_GOLF_PROMPT.format(
                current_code=self.current_best_code,
                generator_code=self.generator_code
            )
            
            try:
                response = await self._call_llm_with_history(prompt, "self_golf")
                
                # Extract optimized code
                optimized_code = extract_code(response)
                if not optimized_code:
                    self.logger.warning("No code found in self-golf response")
                    continue
                
                # Verify code
                passed, info = await self.judge.execute(optimized_code)
                if not passed:
                    self.logger.warning(f"Self-golf optimization failed verification: {info}")
                    continue
                
                # Check improvement
                new_length = len(optimized_code)
                if new_length < self.current_best_length:
                    improvement = self.current_best_length - new_length
                    self.logger.info(f"Self-golf improved code by {improvement} chars")
                    
                    # Extract new tricks from response
                    await self._extract_tricks_from_response(
                        self.current_best_code, 
                        optimized_code, 
                        response
                    )
                    
                    # Update current best code
                    self.current_best_code = optimized_code
                    self.current_best_length = new_length
                    

                    improved = True
                else:
                    self.logger.info("Self-golf did not improve length")
            
            except Exception as e:
                self.logger.error(f"Self-golf iteration {iteration + 1} failed: {e}")
        
        return improved
    
    async def knowledge_scan_phase(self) -> bool:
        """Knowledge base scanning phase"""
        self.logger.info("Phase 4: Knowledge base scanning...")
        
        # Add context about entering knowledge scan phase
        if self.conversation_history:  # Only if we have previous conversation
            context_prompt = f"Now we'll move to Phase 4: Knowledge Base Scanning. Please scan your extensive knowledge of Python code golf techniques for any additional optimizations. Current code length: {self.current_best_length} chars."
            await self._call_llm_with_history(context_prompt, "phase_transition")
        
        improved = False
        
        for iteration in range(2):  # Perform 2 rounds of knowledge scanning
            self.logger.info(f"Knowledge scan iteration {iteration + 1}/2")
            
            # Add context about optimization progress if this is not the first iteration
            if iteration > 0:
                context_prompt = f"We are now in knowledge scan iteration {iteration + 1}/2. Current best code length: {self.current_best_length} chars. Please scan your knowledge base for additional optimization techniques based on our previous discussions."
                await self._call_llm_with_history(context_prompt, "knowledge_scan_context")
            
            # Build problem context
            problem_context = f"Task {self.task_num}: Grid transformation problem"
            
            # Build knowledge scan prompt
            prompt = KNOWLEDGE_SCAN_PROMPT.format(
                current_code=self.current_best_code,
                problem_context=problem_context
            )
            
            try:
                response = await self._call_llm_with_history(prompt, "knowledge_scan")
                
                # Extract final optimized code
                # First try to find code after "Final Optimized Code:" label
                if "Final Optimized Code:" in response:
                    final_section = response.split("Final Optimized Code:")[1]
                    optimized_code = extract_code(final_section)
                else:
                    # Fallback to extracting any code from the response
                    optimized_code = extract_code(response)
                
                if not optimized_code:
                    self.logger.warning("No final code found in knowledge scan response")
                    continue
                
                # Verify code
                passed, info = await self.judge.execute(optimized_code)
                if not passed:
                    self.logger.warning(f"Knowledge scan optimization failed verification: {info}")
                    continue
                
                # Check improvement
                new_length = len(optimized_code)
                if new_length < self.current_best_length:
                    improvement = self.current_best_length - new_length
                    self.logger.info(f"Knowledge scan improved code by {improvement} chars")
                    
                    # Extract new tricks from response
                    await self._extract_tricks_from_response(
                        self.current_best_code, 
                        optimized_code, 
                        response
                    )
                    
                    # Update current best code
                    self.current_best_code = optimized_code
                    self.current_best_length = new_length
                    
                    improved = True
                else:
                    self.logger.info("Knowledge scan did not improve length")
            
            except Exception as e:
                self.logger.error(f"Knowledge scan iteration {iteration + 1} failed: {e}")
        
        return improved
    
    async def _extract_tricks_from_response(self, old_code: str, new_code: str, response: str) -> None:
        """Extract new tricks from LLM response"""
        try:
            # Try to extract trick descriptions from response
            techniques = re.findall(r'\*\*Technique \d+: (.*?)\*\*\s*- \*\*Description:\*\* (.*?)(?=\*\*|$)', response, re.DOTALL)
            
            for name, description in techniques:
                # Create new trick
                trick = Trick(
                    description=f"{name.strip()}: {description.strip()[:200]}",
                    sub_problem=f"Extracted from task {self.task_num}",
                    input_code=old_code,
                    output_code=new_code,
                    success_count=1,
                    tasks_applied={self.task_num}
                )
                
                self.trick_pool.add_trick(trick)
                self.logger.info(f"Extracted new trick: {trick.description[:50]}...")
        
        except Exception as e:
            self.logger.error(f"Failed to extract tricks from response: {e}")
    
    async def run(self) -> Dict[str, Any]:
        """Run complete golf optimization workflow"""
        self.logger.info(f"Starting trick-based golf optimization for task {self.task_num}")
        
        results = {
            'task_num': self.task_num,
            'initial_length': len(self.initial_solution),
            'phases_completed': [],
            'total_improvement': 0,
            'final_code': self.current_best_code,
            'final_length': self.current_best_length
        }
        
        try:
            # Phase 1: Problem understanding
            if await self.understand_problem():
                results['phases_completed'].append('problem_understanding')
                self.logger.info("✓ Problem understanding phase completed")
            else:
                self.logger.error("✗ Problem understanding phase failed")
                return results
            
            # Phase 2: Apply tricks from pool
            if await self.apply_tricks_phase():
                results['phases_completed'].append('trick_application')
                self.logger.info("✓ Trick application phase completed with improvements")
            else:
                self.logger.info("○ Trick application phase completed without improvements")
            
            # Phase 3: Self-directed golf optimization
            if await self.self_golf_phase():
                results['phases_completed'].append('self_golf')
                self.logger.info("✓ Self-golf phase completed with improvements")
            else:
                self.logger.info("○ Self-golf phase completed without improvements")
            
            # Phase 4: Knowledge base scanning
            if await self.knowledge_scan_phase():
                results['phases_completed'].append('knowledge_scan')
                self.logger.info("✓ Knowledge scan phase completed with improvements")
            else:
                self.logger.info("○ Knowledge scan phase completed without improvements")
            
            # Calculate total improvement
            results['total_improvement'] = len(self.initial_solution) - self.current_best_length
            results['final_code'] = self.current_best_code
            results['final_length'] = self.current_best_length
            self.logger.info(f"Golf optimization completed. Total improvement: {results['total_improvement']} chars")
            self.logger.info(f"Final code length: {results['final_length']} chars")
            
            # Save conversation history
            self.session_manager.save_history(self.conversation_history, self.current_best_code, self.initial_solution)
            
        except Exception as e:
            self.logger.error(f"Golf optimization failed: {e}")
            results['error'] = str(e)
        
        return results

class MultiTaskGolfAgent:
    """Multi-task code golf optimization agent"""
    
    def __init__(
        self,
        llm: LLM,
        trick_pool_path: str = "shared_trick_pool.json",
        max_rounds: int = 3,
        tasks_per_round: int = 50,
        require_human_confirmation: bool = False,
        session_id: Optional[str] = None,
        restore_from_latest: bool = False
    ):
        self.llm = llm
        self.trick_pool = TrickPoolManager(trick_pool_path)
        self.max_rounds = max_rounds
        self.tasks_per_round = tasks_per_round
        self.require_human_confirmation = require_human_confirmation
        self.restore_from_latest = restore_from_latest
        
        # Simple session management
        self.session_manager = SessionManager(0, session_id)  # task_num=0 for multi-task
        self.logger = setup_logger("MultiTaskAgent", self.session_manager.get_log_file())
    
    async def run_single_task(self, task_num: int) -> Dict[str, Any]:
        """Run optimization for a single task"""
        self.logger.info(f"Processing task {task_num}...")
        
        # Load initial solution
        initial_solution = load_task_solution(task_num)
        if not initial_solution:
            self.logger.error(f"No initial solution found for task {task_num}")
            return {
                'task_num': task_num,
                'error': 'No initial solution found',
                'success': False
            }
        
        try:
            # Create single-task agent
            agent = TrickBasedGolfAgent(
                llm=self.llm,
                task_num=task_num,
                initial_solution=initial_solution,
                trick_pool=self.trick_pool,
                require_human_confirmation=self.require_human_confirmation,
                session_id=self.session_manager.session_id,
                restore_from_latest=self.restore_from_latest
            )
            
            # Run optimization
            result = await agent.run()
            result['success'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process task {task_num}: {e}")
            return {
                'task_num': task_num,
                'error': str(e),
                'success': False
            }
    
    async def run_round(self, round_num: int, task_list: List[int]) -> Dict[str, Any]:
        """Run one round of multi-task optimization"""
        self.logger.info(f"Starting round {round_num} with {len(task_list)} tasks...")
        
        round_start_time = time.time()
        round_results = {
            'round_num': round_num,
            'tasks_processed': 0,
            'tasks_improved': 0,
            'total_improvement': 0,
            'trick_pool_size_start': len(self.trick_pool.tricks),
            'task_results': {}
        }
        
        for task_num in task_list:
            task_result = await self.run_single_task(task_num)
            round_results['task_results'][task_num] = task_result
            
            if task_result.get('success', False):
                round_results['tasks_processed'] += 1
                improvement = task_result.get('total_improvement', 0)
                if improvement > 0:
                    round_results['tasks_improved'] += 1
                    round_results['total_improvement'] += improvement
                    
                self.logger.info(f"Task {task_num}: {improvement:+d} chars improvement")
            else:
                self.logger.error(f"Task {task_num}: Failed - {task_result.get('error', 'Unknown error')}")
        
        round_results['trick_pool_size_end'] = len(self.trick_pool.tricks)
        round_results['new_tricks_added'] = round_results['trick_pool_size_end'] - round_results['trick_pool_size_start']
        round_results['duration'] = time.time() - round_start_time
        
        self.logger.info(f"Round {round_num} completed:")
        self.logger.info(f"  - Tasks processed: {round_results['tasks_processed']}/{len(task_list)}")
        self.logger.info(f"  - Tasks improved: {round_results['tasks_improved']}")
        self.logger.info(f"  - Total improvement: {round_results['total_improvement']} chars")
        self.logger.info(f"  - New tricks added: {round_results['new_tricks_added']}")
        self.logger.info(f"  - Duration: {round_results['duration']:.1f}s")
        
        return round_results
    
    async def run(self, task_list: Optional[List[int]] = None) -> Dict[str, Any]:
        """Run complete multi-round multi-task optimization"""
        self.logger.info("Starting multi-task golf optimization...")
        
        # Get task list - all tasks 1-400 are available
        if task_list is None:
            # Generate task list 1-400, limited by tasks_per_round
            task_list = list(range(1, min(401, self.tasks_per_round + 1)))
        
        self.logger.info(f"Processing {len(task_list)} tasks over {self.max_rounds} rounds")
        
        overall_results = {
            'total_rounds': self.max_rounds,
            'tasks_per_round': len(task_list),
            'round_results': [],
            'overall_stats': {
                'total_tasks_processed': 0,
                'total_tasks_improved': 0,
                'total_improvement': 0,
                'final_trick_pool_size': 0
            }
        }
        
        start_time = time.time()
        
        try:
            for round_num in range(1, self.max_rounds + 1):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"ROUND {round_num}/{self.max_rounds}")
                self.logger.info(f"{'='*60}")
                
                # Randomly shuffle task order to avoid order bias
                shuffled_tasks = task_list.copy()
                random.shuffle(shuffled_tasks)
                
                round_result = await self.run_round(round_num, shuffled_tasks)
                overall_results['round_results'].append(round_result)
                
                # Update overall statistics
                overall_results['overall_stats']['total_tasks_processed'] += round_result['tasks_processed']
                overall_results['overall_stats']['total_tasks_improved'] += round_result['tasks_improved']
                overall_results['overall_stats']['total_improvement'] += round_result['total_improvement']
                
                # Output trick pool statistics
                trick_stats = self.trick_pool.get_stats()
                self.logger.info(f"Trick pool stats: {trick_stats}")
                
                # Pause between rounds
                if round_num < self.max_rounds:
                    self.logger.info(f"Completed round {round_num}, preparing for next round...")
                    await asyncio.sleep(1)  # Brief pause
            
            # Final statistics
            overall_results['overall_stats']['final_trick_pool_size'] = len(self.trick_pool.tricks)
            overall_results['total_duration'] = time.time() - start_time
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info("MULTI-TASK OPTIMIZATION COMPLETED")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Total duration: {overall_results['total_duration']:.1f}s")
            self.logger.info(f"Tasks processed: {overall_results['overall_stats']['total_tasks_processed']}")
            self.logger.info(f"Tasks improved: {overall_results['overall_stats']['total_tasks_improved']}")
            self.logger.info(f"Total improvement: {overall_results['overall_stats']['total_improvement']} chars")
            self.logger.info(f"Final trick pool size: {overall_results['overall_stats']['final_trick_pool_size']}")
            
            # Save complete results
            self._save_results(overall_results)
            
        except Exception as e:
            self.logger.error(f"Multi-task optimization failed: {e}")
            overall_results['error'] = str(e)
        
        return overall_results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to session directory"""
        try:
            results_file = self.session_manager.session_dir / "multi_task_results.json"
            results['session_id'] = self.session_manager.session_id
            results['saved_at'] = datetime.now().isoformat()
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Results saved to {results_file}")
        except Exception:
            pass  # Ignore save errors

# Main program entry point
async def main():
    """Main program entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trick-based Code Golf Agent")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi", 
                       help="Run mode: single task or multi-task")
    parser.add_argument("--task", type=int, help="Task number for single mode")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds for multi mode")
    parser.add_argument("--tasks-per-round", type=int, default=50, help="Tasks per round")
    parser.add_argument("--human-confirm", action="store_true", 
                       help="Require human confirmation for problem understanding")
    parser.add_argument("--trick-pool", type=str, default="shared_trick_pool.json",
                       help="Path to trick pool file")
    parser.add_argument("--session-id", type=str, help="Specific session ID to use")
    parser.add_argument("--restore-latest", action="store_true",
                       help="Restore from latest session")
    parser.add_argument("--list-sessions", action="store_true",
                       help="List all available sessions")
    # LLM arguments
    parser.add_argument("--model", type=str, default="o4-mini", help="LLM model to use")
    parser.add_argument("--timeout", type=int, default=600, help="LLM call timeout in seconds")
    parser.add_argument("--stream", type=str, default="false", help="Stream LLM responses (true/false)")
    
    args = parser.parse_args()

    args.stream = args.stream.lower() == "true"
    
    # Handle list sessions command
    if args.list_sessions:
        sessions_dir = Path("sessions")
        if sessions_dir.exists():
            sessions = sorted([d.name for d in sessions_dir.iterdir() if d.is_dir()], reverse=True)
            print(f"Found {len(sessions)} sessions:")
            for session in sessions[:10]:  # Show latest 10
                print(f"  - {session}")
        else:
            print("No sessions found.")
        return
    
    # Initialize LLM
    llm = LLM(model_id=args.model, timeout=args.timeout, stream=args.stream)
    
    if args.mode == "single":
        if not args.task:
            print("Error: Task number required for single mode")
            return
        
        # Load initial solution
        initial_solution = load_task_solution(args.task)
        
        if not initial_solution:
            print(f"Error: Solution not found for task {args.task}")
            return
        
        # Create trick pool
        trick_pool = TrickPoolManager(args.trick_pool)
        
        # Create single-task agent
        agent = TrickBasedGolfAgent(
            llm=llm,
            task_num=args.task,
            initial_solution=initial_solution,
            trick_pool=trick_pool,
            require_human_confirmation=args.human_confirm,
            session_id=args.session_id,
            restore_from_latest=args.restore_latest
        )
        
        # Run optimization
        result = await agent.run()
        
        print(f"\nOptimization completed for task {args.task}:")
        print(f"Initial length: {result['initial_length']} chars")
        print(f"Final length: {result['final_length']} chars")
        print(f"Improvement: {result['total_improvement']} chars")
        print(f"Phases completed: {', '.join(result['phases_completed'])}")
        
    else:  # multi mode
        # Create multi-task agent
        multi_agent = MultiTaskGolfAgent(
            llm=llm,
            trick_pool_path=args.trick_pool,
            max_rounds=args.rounds,
            tasks_per_round=args.tasks_per_round,
            require_human_confirmation=args.human_confirm,
            session_id=args.session_id,
            restore_from_latest=args.restore_latest
        )
        
        # Run multi-task optimization
        results = await multi_agent.run()
        
        if 'error' not in results:
            print(f"\nMulti-task optimization completed:")
            print(f"Total tasks processed: {results['overall_stats']['total_tasks_processed']}")
            print(f"Total tasks improved: {results['overall_stats']['total_tasks_improved']}")
            print(f"Total improvement: {results['overall_stats']['total_improvement']} chars")
            print(f"Final trick pool size: {results['overall_stats']['final_trick_pool_size']}")
        else:
            print(f"Multi-task optimization failed: {results['error']}")

if __name__ == "__main__":
    asyncio.run(main())
