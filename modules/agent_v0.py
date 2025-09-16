import os
import re
import sys
import json
import time
import hydra
import inspect
import traceback

from copy import deepcopy
from threading import Semaphore
from loguru import logger as global_logger
from typing import Callable, Optional, List
from omegaconf import OmegaConf, DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from tenacity import retry, RetryCallState, wait_random_exponential, stop_after_attempt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from utils.model import get_qwen_agent_model
from utils.miscs import dumps_matrix, setup_file_logger
from utils.submit_program import submit_program, solution_manager
from modules.prompt_v0 import system_prompt, system_prompt_solve, user_prompt_template


class Agent:
    """
    The Agent class is responsible for interacting with a large language model (LLM)
    to solve coding tasks. It manages the conversation, uses tools, and iterates
    to find an optimal solution for a given task.
    """

    def __init__(self, config: DictConfig):
        """
        Initializes the Agent instance.

        Args:
            config (DictConfig): A Hydra/OmegaConf configuration object containing
                                 settings for the agent, model, and logging.
        """
        self.config = config.agent
        assert self.config.mode in ["solve", "golf"]
        self.agent_name: str = self.config.name
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Set up a dedicated logger for this agent instance
        log_file = os.path.join(self.config.save_dir, f"{self.agent_name}.log")
        self.logger = setup_file_logger(self.agent_name, log_file, self.config.log_level)

        # Initialize the language model
        self.model = get_qwen_agent_model(config.model)

        # Conversation history
        self.messages: list[dict] = []

        # Tools available to the agent
        self.tools: list[dict] = []
        self.tool_name_to_func: dict[str, Callable] = {}

        # Iteration counter as a class member, initialized to 0
        self.iteration_count: int = 0

        # For tracking performance during a run
        self.initial_best_solution: Optional[str] = None
        self.initial_best_solution_length: int | float = float('inf')
        self.produced_best_solution_length: int | float = float('inf')

        self.logger.info(f"Agent initialized in '{self.config.mode}' mode.")

    def save_messages(self):
        """
        Saves the current conversation history to a JSON file.
        This is useful for debugging and resuming.
        """
        save_dir = self.config.save_dir
        save_file = os.path.join(save_dir, f"{self.agent_name}.json")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, "w") as f:
            json.dump(self.messages, f, indent=4, ensure_ascii=False)
        self.logger.debug(f"Messages saved to {save_file}")

    def register_function_tool(self, tool: Callable, name: str = "", desc: str = "", alias: str = ""):
        """
        Registers a Python function as a tool that the LLM can call.

        Args:
            tool (Callable): The Python function to be registered.
            name (str, optional): The name of the tool. Defaults to the function's name.
            desc (str, optional): A description of what the tool does. Defaults to the function's docstring.
            alias (str, optional): An alternative name for the tool.
        """
        if not callable(tool):
            raise ValueError("tool must be a callable object")

        if not name:
            name = tool.__name__
        if not desc:
            desc = str(tool.__doc__)

        self.tool_name_to_func[name] = tool
        if alias:
            self.tool_name_to_func[alias] = tool

        properties = {}
        required = []
        params = inspect.signature(tool).parameters
        for param_name, param in params.items():
            properties[param_name] = {
                "type": param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "Not specified",
                "description": f"Refer to the description of parameter {param_name}.",
            }
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        self.tools.append({
            "name": name,
            "description": desc,
            "parameters": parameters,
        })
        self.logger.info(f"Registered tool '{name}' (alias: '{alias}')")

    def call_tool(self, function_call: dict[str]):
        """
        Executes a tool function based on the LLM's request.
        """
        try:
            function_name = function_call.get("name", "None")
            if function_name not in self.tool_name_to_func:
                self.logger.error(f"Iter {self.iteration_count}: Attempted to call unknown function '{function_name}'.")
                return f"Function '{function_name}' not found. Please use one of the available tools."

            func = self.tool_name_to_func[function_name]
            args = json.loads(function_call["arguments"])
            result = func(**args)

            if func.__name__ == "submit_program" and result[0] > 0:  # sorry for the ugly code
                self.produced_best_solution_length = min(self.produced_best_solution_length,
                                                         len(args["code"].encode("utf-8")))
            return result
        except Exception:
            return traceback.format_exc()

    def get_completion_error_logging(self, retry_state: RetryCallState):
        """
        A callback for `tenacity` retry decorator to log API call errors.
        """
        self.logger.error(
            f"Iter {self.iteration_count}: LLM API call failed. Attempt {retry_state.attempt_number}. Error: {retry_state.outcome.exception()}"
        )

    @retry(wait=wait_random_exponential(min=1, max=60),
           stop=stop_after_attempt(6),
           before_sleep=lambda rs: rs.args[0].get_completion_error_logging)
    def get_completion(self) -> list[dict]:
        """
        Calls the LLM API, with an exponential backoff retry mechanism.
        """
        responses = self.model.chat(messages=self.messages, functions=self.tools, stream=False)
        self.logger.info(f"Iter {self.iteration_count}: Received completion: {repr(responses)}")
        return responses
    
    def _load_generation_code(self, task_id: str) -> str:
        """
        Loads the generation code for a specific task.
        """
        generation_file = os.path.join(self.project_root, "google-code-golf-2025", "generate", f"task{task_id}.py")
        if os.path.exists(generation_file):
            with open(generation_file, "r") as f:
                code = f.read()
            self.logger.info(f"Loaded generation code for task{task_id} from {generation_file}")
            return code
        else:
            self.logger.warning(f"Generation code for task{task_id} not found.")
            return ""

    def _prepare_initial_prompt(self, task_id: str, task: dict) -> str:
        """
        Constructs the initial user prompt with examples and context.
        """
        self.logger.info(f"Preparing initial prompt for task {task_id}.")
        examples = []
        for testcase in task["train"] + task["test"]:
            examples.append(testcase)
            if len(examples) >= 6:
                break

        examples_str = ""
        for idx, testcase in enumerate(examples):
            examples_str += f"\nExample {idx} Input:\n{dumps_matrix(testcase['input'])}\nExample {idx} Output:\n{dumps_matrix(testcase['output'])}\n"

        generation_code = self._load_generation_code(task_id)

        user_prompt = user_prompt_template.format(task_id=task_id, examples_str=examples_str, generation_code=generation_code)

        if self.initial_best_solution is not None and self.config.add_existing:
            self.logger.info(
                f"Found existing best solution with length {self.initial_best_solution_length}. Adding to prompt.")
            user_prompt += f"\nThe current best solution is as follows. This code provides correct logic, but you do not necessarily need to start optimizing length from it. \n```python3\n{self.initial_best_solution}\n```\n"
        self.logger.debug(f"Initial user prompt for task {task_id}:\n{user_prompt}")
        return user_prompt

    def _process_model_response(self, responses: list[dict], task_id: str) -> list[dict]:
        """
        Processes model's response to find function calls or mock one if code is present.
        
        Args:
            responses (list[dict]): The list of messages from the model's response.
            task_id (str): The current task ID.

        Returns:
            list[dict]: A list of function calls to be executed.
        """
        function_calls = []
        for message in responses:
            if function_call := message.get("function_call", None):
                function_calls.append(function_call)

        if not function_calls:
            self.logger.warning(f"Iter {self.iteration_count}: Model did not call a function.")
            last_content = self.messages[-1].get("content", "")
            pattern = re.compile(r"```(py|python|python3)\s+(.*?)\s*```", re.DOTALL | re.MULTILINE)
            matches = pattern.findall(last_content)

            if matches:
                code = matches[-1][1]
                mock_call = {"name": "submit_program", "arguments": json.dumps({"task_id": task_id, "code": code})}
                function_calls.append(mock_call)
                self.logger.info(f"Iter {self.iteration_count}: Found code. Mocking a 'submit_program' call.")
                self.messages.append({
                    "role": "user",
                    "content": "You did not call any functions, but your reply contained code. We automatically evaluated the last piece of code for you. Please correct or improve your submission based on the following feedback.",
                })
            else:
                self.logger.warning(f"Iter {self.iteration_count}: No function call and no code found in response.")
                self.messages.append({
                    "role": "user",
                    "content": "You did not call any functions and did not provide any code. Your goal is to solve the task. After obtaining the final answer, please use the `submit_program` function to submit it.",
                })

        return function_calls

    def _execute_function_calls(self, function_calls: list[dict]):
        """
        Executes function calls and appends their results to the message history.
        """
        for function_call in function_calls:
            self.logger.info(f"Iter {self.iteration_count}: Executing function call: {repr(function_call)}")

            fn_name: str = function_call.get('name', 'unknown_function')
            fn_res = self.call_tool(function_call)

            if isinstance(fn_res, tuple):
                fn_res = f"({', '.join(map(str, fn_res))})"

            self.messages.append({
                "role": "function",
                "name": fn_name,
                "content": str(fn_res),
            })
            self.logger.info(f"Iter {self.iteration_count}: Got function result: {repr(fn_res)}")

    def _log_run_summary(self, task_id: str):
        """
        Logs a summary of the agent's run for a specific task.
        """
        final_best_solution, final_best_solution_length = solution_manager.get(task_id)

        summary_message = f"""

=========================================================
Agent Run Summary: {self.agent_name} for Task {task_id}
---------------------------------------------------------
Total Iterations: {self.iteration_count} / {self.config.max_iter}
Initial Best Solution Length: {self.initial_best_solution_length if self.initial_best_solution is not None else 'N/A'}
Produced Best Solution Length: {self.produced_best_solution_length if self.produced_best_solution_length < float('inf') else 'N/A'}
Final Best Solution Length:   {final_best_solution_length if final_best_solution is not None else 'N/A'}
=========================================================
"""
        self.logger.info(summary_message)

    def run(self, task_id: str, task: dict[str]):
        """
        The main execution loop for the agent on a given task.
        """
        self.logger.info(f"Start running task {task_id}")
        self.initial_best_solution, self.initial_best_solution_length = solution_manager.get(task_id)

        if self.config.mode == "solve" and self.initial_best_solution is not None:
            self.logger.warning(
                f"Task {task_id} already has a solution with length {self.initial_best_solution_length}. Skipping run based on config."
            )
            return

        user_prompt = self._prepare_initial_prompt(task_id, task)
        self.messages = [
            {
                "role": "system",
                "content": system_prompt_solve if self.config.mode == "solve" else system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ]
        self.save_messages()

        try:
            # Use a while loop with the class member counter
            while self.iteration_count < self.config.max_iter:
                # Increment at the start to make it 1-based for the first iteration
                self.iteration_count += 1

                self.logger.info(f"Starting iteration {self.iteration_count}/{self.config.max_iter}")
                global_logger.info(
                    f"Agent {self.agent_name}: Starting iteration {self.iteration_count}/{self.config.max_iter}")

                # 1. Get completion from the language model
                responses = self.get_completion()
                self.messages.extend(responses)
                self.save_messages()

                # 2. Process response
                function_calls = self._process_model_response(responses, task_id)

                # 3. Execute calls
                if function_calls:
                    self._execute_function_calls(function_calls)

                self.save_messages()

                # 4. check if the task is solved
                if self.config.mode == "solve":
                    if solution_manager.get(task_id)[0] is not None:
                        self.logger.info(f"Task {task_id} solved. Exiting due to 'solve' mode.")
                        break

        except Exception as e:
            self.logger.error(f"An unexpected error occurred during the run for task {task_id}: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.logger.info(f"Finished running task {task_id}.")
            # 4. Log summary
            self._log_run_summary(task_id)


def setup_environment(config: DictConfig):
    """
    Sets up the environment for the run, including resolving configs,
    creating directories, and configuring loggers.
    """
    OmegaConf.resolve(config)
    timestr = time.strftime("%Y%m%d%H%M%S", time.localtime())
    config.agent.save_dir = os.path.join(config.agent.save_dir, timestr)
    os.makedirs(config.agent.save_dir, exist_ok=True)

    # Dump the final resolved config for reproducibility
    with open(os.path.join(config.agent.save_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))

    # Configure global loggers
    global_logger.remove()
    global_logger.add(sys.stderr, level="INFO")
    global_logger.add(os.path.join(config.agent.save_dir, "run.log"), level="DEBUG")
    global_logger.info("Environment setup complete.")


def load_tasks(config: DictConfig) -> List[dict]:
    """
    Loads all task JSON files specified in the configuration.
    
    Returns:
        A list of task dictionaries.
    """
    global_logger.info(f"Loading {config.meta.tot} tasks...")
    tasks = []
    for i in range(1, config.meta.tot + 1):
        task_id = str(i).zfill(3)
        task_file = f'google-code-golf-2025/task{task_id}.json'
        try:
            with open(task_file) as f:
                task = json.load(f)
            tasks.append(task)
        except FileNotFoundError:
            global_logger.error(f"Task file not found: {task_file}. Aborting.")
            sys.exit(1)
    global_logger.info(f"Successfully loaded {len(tasks)} tasks.")
    return tasks


def run_tasks_in_parallel(config: DictConfig, tasks: List[dict]):
    """
    Manages the concurrent execution of agent runs for all tasks.
    """
    # Create a list of all arguments for each run instance.
    # (bon, i) represents the 'bon'-th run on the 'i'-th task.
    run_args = []
    for bon in range(config.meta.bon):
        for i in range(1, config.meta.tot + 1):
            # check if the task has been solved before submitting
            # mainly to avoid the log spam
            if config.meta.skip_existing and solution_manager.get(str(i).zfill(3))[0] is not None:
                continue
            run_args.append((bon, i))

    global_logger.info(f"Total {len(run_args)} tasks to run.")

    # A semaphore to limit the number of concurrently running tasks,
    # preventing resource exhaustion (e.g., too many API calls at once).
    sem = Semaphore(config.meta.max_running_task)

    def agent_run_wrapper(bon: int, i: int):
        """A wrapper function to set up and run a single agent instance."""
        sem.acquire()  # Wait for an available slot
        try:
            task_id = str(i).zfill(3)

            # Deepcopy the config to ensure each agent has its own mutable
            # configuration, preventing race conditions between threads.
            agent_config = deepcopy(config)
            agent_config.agent.name += f"_{task_id}_{bon}"
            agent_config.agent.save_dir = os.path.join(agent_config.agent.save_dir, task_id)

            agent = Agent(agent_config)
            agent.register_function_tool(submit_program, alias="code")

            global_logger.info(f"Agent {agent_config.agent.name} starting its run on task {task_id}.")
            agent.run(task_id, tasks[i - 1])
        finally:
            sem.release()  # Release the slot for another task

    global_logger.info(
        f"Starting parallel execution with max concurrency of {config.meta.max_concurrency} and max running tasks of {config.meta.max_running_task}."
    )
    futures: list[Future[None]] = []
    future_to_name: dict[Future[None], str] = {}
    with ThreadPoolExecutor(max_workers=config.meta.max_concurrency) as executor:
        for bon, i in run_args:
            future = executor.submit(agent_run_wrapper, bon, i)
            task_name = f"task{i}_run{bon}"
            futures.append(future)
            future_to_name[future] = task_name

        # Process futures as they complete to provide real-time feedback
        total_tasks = len(futures)
        for count, future in enumerate(as_completed(futures), 1):
            task_name = future_to_name[future]
            try:
                future.result()  # Raise any exception that occurred in the thread
                global_logger.success(f"({count}/{total_tasks}) Task {task_name} completed successfully.")
            except Exception:
                # The traceback is logged here for central error tracking.
                global_logger.error(
                    f"({count}/{total_tasks}) Task {task_name} failed with an exception:\n{traceback.format_exc()}")
            del future_to_name[future]

    global_logger.info("All tasks have been processed.")


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(config: DictConfig):
    """
    Main entry point for the agent execution script.
    Orchestrates setup, task loading, and parallel execution.
    """
    setup_environment(config)
    tasks = load_tasks(config)
    run_tasks_in_parallel(config, tasks)


if __name__ == "__main__":
    main()
