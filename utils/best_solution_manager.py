import os
import time
import threading

from loguru import logger
from functools import wraps
from collections import defaultdict


def retry_on_io_error(retries: int = 3, delay: float = 0.1):
    """
    A decorator to retry a function call upon IOError.

    Args:
        retries (int): The maximum number of retries.
        delay (float): The delay in seconds between retries.
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except IOError as e:
                    last_exception = e
                    logger.warning(
                        f"I/O error in '{func.__name__}' on attempt {attempt + 1}/{retries}: {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
            logger.error(f"Function '{func.__name__}' failed after {retries} retries.")
            raise last_exception

        return wrapper

    return decorator


class BestSolutionManager:
    """
    Manages reading and updating best solutions with caching and thread safety.
    This class assumes it is the sole authority for modifying files in its base_path.
    It uses:
    - A lock per task_id to prevent race conditions.
    - An in-memory cache for solution content and length to reduce file I/O.
    - A retry mechanism for file operations.
    """

    def __init__(self, base_path: str = 'best_solutions'):
        self.base_path = base_path
        self.locks = defaultdict(threading.RLock)
        # Cache format: {task_id: (content, length_in_bytes)}
        self.cache = {}
        os.makedirs(self.base_path, exist_ok=True)
        logger.info(f"BestSolutionManager initialized. Storing solutions in '{self.base_path}'.")

    def _get_solution_path(self, task_id: str) -> str:
        return os.path.join(self.base_path, f'task{task_id}.py')

    @retry_on_io_error()
    def _read_file(self, file_path: str) -> tuple[str, int]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        length = len(content.encode('utf-8'))
        return content, length

    @retry_on_io_error()
    def _check_file(self, file_path: str) -> bool:
        return os.path.exists(file_path)

    @retry_on_io_error()
    def _write_file(self, file_path: str, content: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Successfully wrote to file: {file_path}")

    def get(self, task_id: str) -> tuple[str | None, int | float]:
        """
        Gets the best solution's content and length for a task, using a cache.
        """
        # First, check cache without locking for high-performance reads.
        if task_id in self.cache:
            return self.cache[task_id]
        lock = self.locks[task_id]
        with lock:
            # Double-check cache after acquiring lock (another thread might have just populated it).
            if task_id in self.cache:
                return self.cache[task_id]
            # If not in cache, load from disk.
            file_path = self._get_solution_path(task_id)
            if self._check_file(file_path):
                content, length = self._read_file(file_path)
                self.cache[task_id] = (content, length)  # Populate cache
                logger.debug(f"Cache miss for task {task_id}. Loaded solution from disk.")
                return content, length
            else:
                logger.debug(f"No best solution file found for task {task_id}.")
                self.cache[task_id] = (None, float('inf'))  # Cache the non-existence
                return None, float('inf')

    def get_length(self, task_id: str) -> int | float:
        """
        Gets only the length of the best solution, using the cache.
        """
        # This will use the efficient `get` method.
        _, length = self.get(task_id)
        return length

    def update(self, task_id: str, new_code: str) -> tuple[bool, int | float]:
        """
        Attempts to update the best solution if the new code is shorter.
        This operation is atomic and relies on the internal cache as the source of truth.
        """
        new_length = len(new_code.encode('utf-8'))
        lock = self.locks[task_id]
        with lock:
            # The current best length is retrieved from our own state (the cache).
            # The get_length() method ensures the cache is populated if it's the first time.
            current_best_length = self.get_length(task_id)
            if new_length < current_best_length:
                logger.success(
                    f"New best solution found for task {task_id}. "
                    f"New length: {new_length}, Previous length: {current_best_length if current_best_length != float('inf') else 'N/A'}."
                )
                file_path = self._get_solution_path(task_id)
                self._write_file(file_path, new_code)
                # Proactively update the cache with the new best solution.
                # This is more efficient than just invalidating it.
                self.cache[task_id] = (new_code, new_length)
                return True, current_best_length
            else:
                return False, current_best_length
