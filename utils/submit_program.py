import os
import json
import threading
import traceback
import importlib
import tempfile

from loguru import logger
from copy import deepcopy
from functools import lru_cache
from collections import defaultdict

from utils.miscs import dumps_matrix
from utils.best_solution_manager import BestSolutionManager

# Create a single, shared instance of the manager.
# This instance will be used by all threads calling submit_program.
solution_manager = BestSolutionManager(base_path='best_solutions')


@lru_cache(maxsize=256)
def load_task(task_id: str) -> dict:
    with open(f'google-code-golf-2025/task{task_id}.json') as f:
        data = json.load(f)
    return data


def submit_program(task_id: str, code: str) -> tuple[int, str]:
    """Submit your program for the task.
    Inputs:
        task_id: The id of the task.
        code: Your program.
    Returns:
        Your score and the response from the judge.
    """
    # the task id must be three digits between 001 and 400
    if len(task_id) != 3 or not task_id.isdigit() or int(task_id) < 1 or int(task_id) > 400:
        return -1, 'Invalid task id. Please use a three-digit number string between 001 and 400.'
    # save the code to a file
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f'task{task_id}.py')
        os.makedirs(temp_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(code)
    except Exception as e:
        logger.error(f"Failed to save code to file: {e}")
        return -1, f"Failed to save code to file: {e}"
    # with open(f'/tmp/task{task_id}.py', 'w') as f:
    #     f.write(code)
    # load the code from the file
    try:
        # use a separate namespace to execute the code
        code_globals = {}
        exec(code, code_globals)
        program = code_globals.get("p")
        if program is None:
            return -1, "Error: Unable to locate function p() in code."
        if not callable(program):
            return -1, "Error: Function p() in code is not callable."
    except (SyntaxError, IndentationError):
        error_details = traceback.format_exc()
        return -1, f"Syntax errors in your code.\nError message:\n{error_details}"
    except Exception:
        error_details = traceback.format_exc()
        return -1, f"An unknown error occurred while loading your code.\nError message:\n{error_details}"

    # load all the test data
    data = load_task(task_id)
    passed_count, failed_count, except_count = 0, 0, 0
    failed_testcase = {}
    except_testcase = {}
    for testcase in data["train"] + data["test"] + data["arc-gen"]:
        try:
            # only deepcopy the input to avoid side effects
            actual_output = program(deepcopy(testcase['input']))
            if actual_output == testcase['output']:
                passed_count += 1
            else:
                failed_count += 1
                if not failed_testcase:  # first failed testcase
                    failed_testcase = {
                        'input': testcase['input'],
                        'output': testcase['output'],
                        'actual_output': actual_output,
                    }
        except:
            except_count += 1
            if not except_testcase:  # first except testcase
                except_testcase = {
                    'input': testcase['input'],
                    'output': testcase['output'],
                    'error_message': traceback.format_exc(),
                }
        finally:
            if failed_testcase and except_testcase:
                break
    if not failed_testcase and not except_testcase:
        code_length = len(code.encode('utf-8'))
        score = max(1, 2500 - code_length)
        # Use the manager to update the best solution
        updated, prev_best_length = solution_manager.update(task_id, code)
        if prev_best_length == float('inf'):
            prev_best_length_str = "(no previous solution)"
            prev_best_score = 0
        else:
            prev_best_length_str = str(int(prev_best_length))
            prev_best_score = max(1, 2500 - int(prev_best_length))
        if updated:
            message = f"""All testcases passed. New best solution saved!
    Your code length is {code_length} and score is {score}.
    The previous best code length was {prev_best_length_str} (score: {prev_best_score}).
    You can continue to try shortening the code to get an even higher score."""
        else:
            message = f"""All testcases passed. Your code is correct but not shorter than the current best.
    Your code length is {code_length} and score is {score}.
    The current best code length is {prev_best_length_str} (score: {prev_best_score}).
    Try to make your code shorter to beat the best score."""
        return score, message

    message = f"You have {passed_count} passed testcases, {failed_count} failed testcases and {except_count} except testcases."

    if failed_testcase:
        message += f"""\n\nThe first failed testcase is:
Input:
{dumps_matrix(failed_testcase['input'])}

Expected Output:
{dumps_matrix(failed_testcase['output'])}

Your Output:
{dumps_matrix(failed_testcase['actual_output'])}

\nPlease first check if your transformation rule is correct according to the failed testcase and the testcases provided previously. If not, revise your transformation rule accordingly.
\nThen check if your code implements the transformation rule correctly. If not, revise your code accordingly.
\nFinally provide your revised rule and code, and call 'submit_program' function.
"""

    if except_testcase:
        message += f"""\n\nThe first except testcase is:
Input:
{dumps_matrix(except_testcase['input'])}

Expected Output:
{dumps_matrix(except_testcase['output'])}

Error Message:
{except_testcase['error_message']}

\nPlease revise your code according to the error message. Then provide your revised code and call 'submit_program' function.
"""
    return 0.001, message


if __name__ == "__main__":
    programs = [
        "This is errornous.",
        "# no p in program",
        "p = 1# p is not a function",
        "def p(x): return x # not a correct answer",
        r"def p(x): d = len(x); return [[x[i // d][j // d] & x[i % d][j % d] for j in range(d * d)] for i in range(d * d)]",
    ]
    for program in programs:
        logger.info(f"Start to submit program <{program}>")
        score, message = submit_program("001", program)
        logger.info(f"Submit result: {score}, <{message}>")
