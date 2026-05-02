"""
Evaluation function for ALFWorld (House-Holding) tasks.

Returns True if the agent successfully completed the task (reward=1).
The ALFWorld task stores the outcome in result["result"].
"""

from src.typings import TaskOutput


def eval(sample: dict, task_output: TaskOutput) -> bool:
    """
    Args:
        sample: Split-file dict with keys "id", "description", "type".
        task_output: TaskOutput returned by the task worker.

    Returns:
        True if the agent completed the household task (result==1).
    """
    if task_output is None or task_output.result is None:
        return False
    try:
        return int(task_output.result.get("result", 0)) == 1
    except (AttributeError, TypeError, ValueError):
        return False
