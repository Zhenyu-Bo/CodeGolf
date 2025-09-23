import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from modules.llm import LLM
from modules.agent_v1 import GolfAgent

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
parser.add_argument("--stream", type=str, default="False", help="Enable streaming for LLM responses")

args = parser.parse_args()

args.use_shortest_hint = args.use_shortest_hint.lower() == "true"
args.record_prompts = args.record_prompts.lower() == "true"
args.record_details = args.record_details.lower() == "true"
args.resume_from_state = args.resume_from_state.lower() == "true"
args.stream = args.stream.lower() == "true"

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
    llm = LLM(model_id=model_id, timeout=args.timeout, stream=args.stream)
    
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