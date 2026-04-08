"""Corridor coordination grader.

Score based on how much the agent improves corridor traversal time
(intersections 0->1->2) compared to the fixed-timing baseline.
"""

import numpy as np


def grade(agent_corridor_time: float, baseline_corridor_time: float) -> float:
    """Grade corridor coordination performance.

    Args:
        agent_corridor_time: Mean steps for a vehicle to traverse 0->1->2 under agent control.
        baseline_corridor_time: Mean steps under fixed-timing baseline.

    Returns:
        Score in [0.0, 1.0].
    """
    if baseline_corridor_time == 0:
        return 0.0
    score = 1.0 - (agent_corridor_time / baseline_corridor_time)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


class CorridorGrader:
    """Callable grader class for corridor coordination."""

    def __call__(self, agent_corridor_time: float, baseline_corridor_time: float) -> float:
        return grade(agent_corridor_time, baseline_corridor_time)
