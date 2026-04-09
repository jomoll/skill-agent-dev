"""
Evaluation function for OS Interaction tasks.

Returns True if the task result indicates a correct solution.
The OS task stores the outcome in result.result["result"] (bool).
"""

from src.typings import TaskOutput


def eval(sample: dict, task_output: TaskOutput) -> bool:
    """
    Args:
        sample: The sample dict from the split file (has "id", "description").
        task_output: The TaskOutput returned by the task worker.

    Returns:
        True if the agent solved the task correctly.
    """
    if task_output is None or task_output.result is None:
        return False
    try:
        return bool(task_output.result.get("result", False))
    except (AttributeError, TypeError):
        return False
