"""Emergency response grader.

Weighted score: 60% emergency vehicle speed, 40% civilian wait time improvement.
"""

import numpy as np


def grade(
    agent_emergency_time: float,
    max_emergency_time: float,
    agent_civilian_wait: float,
    baseline_civilian_wait: float,
) -> float:
    """Grade emergency response performance.

    Args:
        agent_emergency_time: Steps for emergency vehicle to reach destination.
        max_emergency_time: Maximum possible steps (worst case = max_steps).
        agent_civilian_wait: Mean civilian vehicle wait time under agent control.
        baseline_civilian_wait: Mean civilian vehicle wait time under baseline.

    Returns:
        Score in [0.0, 1.0].
    """
    if max_emergency_time == 0:
        emergency_score = 0.0
    else:
        emergency_score = 1.0 - (agent_emergency_time / max_emergency_time)

    if baseline_civilian_wait == 0:
        civilian_score = 0.0
    else:
        civilian_score = 1.0 - (agent_civilian_wait / baseline_civilian_wait)

    score = 0.6 * emergency_score + 0.4 * civilian_score
    return round(float(np.clip(score, 0.0, 1.0)), 4)


class EmergencyGrader:
    """Callable grader class for emergency response."""

    def __call__(
        self,
        agent_emergency_time: float,
        max_emergency_time: float,
        agent_civilian_wait: float,
        baseline_civilian_wait: float,
    ) -> float:
        return grade(agent_emergency_time, max_emergency_time, agent_civilian_wait, baseline_civilian_wait)
