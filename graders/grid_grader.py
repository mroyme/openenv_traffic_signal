"""Grid coordination grader.

Score based on how much the agent reduces mean wait time
compared to the fixed-timing baseline across all 9 intersections.
"""

import numpy as np


def grade(agent_mean_wait: float, baseline_mean_wait: float) -> float:
    """Grade grid coordination performance.

    Args:
        agent_mean_wait: Mean vehicle wait time under agent control.
        baseline_mean_wait: Mean vehicle wait time under fixed-timing baseline.

    Returns:
        Score in [0.0, 1.0].
    """
    if baseline_mean_wait == 0:
        return 0.0
    score = 1.0 - (agent_mean_wait / baseline_mean_wait)
    return round(float(np.clip(score, 0.0, 1.0)), 4)
