"""
Data models for the Traffic Signal Environment.

The traffic signal environment is a cooperative multi-agent RL environment
for adaptive traffic signal control across a 3x3 grid of intersections.
"""

from typing import Any, Literal

from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    """Action for a single intersection agent.

    Attributes:
        agent_id: Intersection ID (0-8).
        phase_action: Whether to maintain or change the current signal phase.
    """

    agent_id: int
    phase_action: Literal["keep", "switch"]


class TrafficAction(Action):
    """Action for the Traffic Signal environment — one decision per active agent.

    Attributes:
        agent_actions: One keep/switch decision per active intersection agent.
    """

    agent_actions: list[AgentAction]


class IntersectionObservation(BaseModel):
    """Observation for a single intersection agent.

    Attributes:
        agent_id: Intersection ID (0-8).
        queue_lengths: Number of waiting vehicles in N, S, E, W lanes.
        current_phase: Signal phase (0=NS-green, 1=NS-yellow, 2=EW-green, 3=EW-yellow).
        phase_elapsed: Steps spent in the current phase.
        neighbor_queues: Outgoing queue lengths of N/S/E/W neighbors toward this intersection.
        local_wait_time: Mean wait time across all lanes at this intersection.
    """

    agent_id: int
    queue_lengths: list[float]
    current_phase: int
    phase_elapsed: int
    neighbor_queues: list[float]
    local_wait_time: float


class EmergencyVehicleState(BaseModel):
    """State of the emergency vehicle (Task 3 only).

    Attributes:
        position: Current intersection ID (0-8).
        destination: Target intersection ID.
        steps_elapsed: Steps taken since the episode started.
        path_remaining: Ordered list of intersection IDs still to traverse.
    """

    position: int
    destination: int
    steps_elapsed: int
    path_remaining: list[int]


class TrafficObservation(Observation):
    """Observation from the Traffic Signal environment.

    Contains per-agent intersection state and global episode metrics.

    Attributes:
        task_id: Identifier of the current task.
        episode_id: Unique identifier for this episode.
        step: Current step number within the episode.
        agents: Per-agent observations for each active intersection.
        global_wait_time: Mean vehicle wait time across all intersections.
        final_score: Final grader score in [0, 1]; only set at episode end.
        feedback_message: Human-readable status message.
        emergency_vehicle: Emergency vehicle state; present only in emergency_response task.
    """

    task_id: str
    episode_id: str
    step: int
    agents: list[IntersectionObservation]
    global_wait_time: float
    final_score: float | None = None
    feedback_message: str
    emergency_vehicle: EmergencyVehicleState | None = None


class TrafficState(State):
    """Internal state for the Traffic Signal environment.

    Attributes:
        task_id: Identifier of the current task.
        total_wait_time: Accumulated mean wait time over the episode.
        baseline_wait_time: Fixed-timing baseline wait time for scoring.
        is_complete: Whether the episode has ended.
        grid_state: Additional serializable grid metadata.
    """

    task_id: str = ""
    total_wait_time: float = 0.0
    baseline_wait_time: float = 0.0
    is_complete: bool = False
    grid_state: dict[str, Any] = Field(default_factory=dict)
