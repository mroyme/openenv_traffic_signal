"""TrafficEnvironment — OpenEnv-compatible environment for traffic signal control."""

from __future__ import annotations

import logging
import uuid
from collections import deque
from typing import Any, TypedDict

import numpy as np

from openenv.core.env_server.interfaces import Environment

from models import (
    EmergencyVehicleState,
    TrafficAction,
    TrafficObservation,
    TrafficState,
)
from simulator.grid import TrafficGrid, NEIGHBORS
from simulator.vehicle import Vehicle
from graders import corridor_grader, grid_grader, emergency_grader

logger = logging.getLogger(__name__)


class TaskConfig(TypedDict):
    """Configuration for a single task variant.

    Attributes:
        active_intersections: Intersection IDs participating in this task.
        spawn_rate: Mean vehicles spawned per intersection per step (Poisson).
        max_steps: Episode length in steps.
        emergency: Whether an emergency vehicle is present.
    """

    active_intersections: list[int]
    spawn_rate: float
    max_steps: int
    emergency: bool


class TrafficEnvironment(Environment[TrafficAction, TrafficObservation, TrafficState]):
    """Cooperative multi-agent traffic signal control environment.

    A 3x3 grid of intersections, each controlled by an independent agent that
    decides each step whether to keep or switch its current signal phase.
    Agents share a global reward signal to incentivize emergent coordination.

    Attributes:
        SUPPORTS_CONCURRENT_SESSIONS: Enables multiple simultaneous WebSocket
            clients, each receiving their own environment instance when the
            server is running in factory mode.
        TASK_CONFIGS: Maps task IDs to their configuration dicts.

    Example:
        >>> env = TrafficEnvironment()
        >>> obs = env.reset(seed=42, task_id="corridor_coordination")
        >>> print(obs.global_wait_time)
        >>>
        >>> action = TrafficAction(agent_actions=[AgentAction(agent_id=0, phase_action="keep")])
        >>> obs = env.step(action)
        >>> print(obs.step_reward)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    TASK_CONFIGS: dict[str, TaskConfig] = {
        "corridor_coordination": {
            "active_intersections": [0, 1, 2],
            "spawn_rate": 0.6,
            "max_steps": 150,
            "emergency": False,
        },
        "grid_coordination": {
            "active_intersections": list(range(9)),
            "spawn_rate": 0.4,
            "max_steps": 200,
            "emergency": False,
        },
        "emergency_response": {
            "active_intersections": list(range(9)),
            "spawn_rate": 0.4,
            "max_steps": 200,
            "emergency": True,
        },
    }

    def __init__(self):
        """Initialize the traffic signal environment."""
        self._state = TrafficState()
        self._grid: TrafficGrid | None = None
        self._rng: np.random.Generator | None = None
        self._config: TaskConfig = TaskConfig(active_intersections=[], spawn_rate=0.4, max_steps=200, emergency=False)
        self._baseline_wait: float = 0.0
        self._wait_time_history: list[float] = []
        self._cumulative_reward: float = 0.0
        self._prev_wait: float = 0.0

        # Emergency vehicle tracking
        self._emergency_vehicle: Vehicle | None = None
        self._emergency_path: list[int] = []
        self._emergency_steps: int = 0
        self._emergency_arrived: bool = False
        self._emergency_destination: int = -1

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> TrafficObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            episode_id: Optional episode identifier (auto-generated if not provided)
            task_id: Task to run — one of corridor_coordination, grid_coordination, emergency_response

        Returns:
            Initial TrafficObservation with reset grid state
        """
        task_id = task_id or "corridor_coordination"
        if task_id not in self.TASK_CONFIGS:
            raise ValueError(
                f"Unknown task_id: {task_id}. Valid: {list(self.TASK_CONFIGS.keys())}"
            )

        self._config = self.TASK_CONFIGS[task_id]
        seed = (
            seed
            if seed is not None
            else int(np.random.default_rng().integers(0, 2**31))
        )
        self._rng = np.random.default_rng(seed)

        episode_id = episode_id or str(uuid.uuid4())

        active = self._config["active_intersections"]
        self._grid = TrafficGrid(active, seed, self._config["spawn_rate"])
        self._grid.reset()

        # Run baseline
        self._baseline_wait = self._grid.run_baseline(self._config["max_steps"])
        logger.info("Baseline wait time: %s", self._baseline_wait)

        self._wait_time_history = []
        self._cumulative_reward = 0.0
        self._prev_wait = 0.0

        # Emergency vehicle setup for task 3
        self._emergency_vehicle = None
        self._emergency_path = []
        self._emergency_steps = 0
        self._emergency_arrived = False
        self._emergency_destination = -1

        if self._config["emergency"]:
            self._setup_emergency_vehicle(active)

        # Initialize state
        self._state = TrafficState(
            episode_id=episode_id,
            step_count=0,
            task_id=task_id,
            total_wait_time=0.0,
            baseline_wait_time=self._baseline_wait,
            is_complete=False,
        )

        obs = self._build_observation(
            step_reward=0.0,
            feedback_message="Episode started. Control traffic signals to minimize wait times.",
        )
        return obs

    def step(
        self,
        action: TrafficAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> TrafficObservation:
        """
        Execute one step in the environment.

        Args:
            action: TrafficAction with keep/switch decisions for each active agent
            timeout_s: Unused; present for interface compatibility

        Returns:
            TrafficObservation with updated grid state and step reward.
            On the final step, observation.reward contains the graded episode score.
        """
        if self._grid is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._state.is_complete:
            return self._build_observation(
                step_reward=0.0,
                feedback_message="Episode already complete.",
            )

        # Convert agent actions to grid format
        actions: dict[int, str] = {}
        for aa in action.agent_actions:
            actions[aa.agent_id] = aa.phase_action

        prev_wait = self._grid.get_mean_wait_time()

        # Step the grid
        metrics = self._grid.step(actions)

        new_wait = metrics["global_wait_time"]
        self._wait_time_history.append(new_wait)

        # Handle emergency vehicle movement
        emergency_delta = None
        if self._config["emergency"] and not self._emergency_arrived:
            emergency_delta = self._step_emergency_vehicle()

        # Compute reward
        step_reward = self._compute_step_reward(
            prev_wait, new_wait, self._baseline_wait, emergency_delta
        )
        self._cumulative_reward += step_reward

        self._state.step_count += 1
        self._state.total_wait_time = (
            float(np.mean(self._wait_time_history)) if self._wait_time_history else 0.0
        )

        # Check if done
        done = self._state.step_count >= self._config["max_steps"]
        self._state.is_complete = done

        # Build feedback message
        feedback = f"Step {self._state.step_count}: wait={new_wait:.2f}"
        if self._config["emergency"] and not self._emergency_arrived:
            feedback += f", emergency at intersection {self._emergency_path[0] if self._emergency_path else self._emergency_destination}"
        if done:
            feedback += " | Episode complete."

        obs = self._build_observation(
            step_reward=step_reward,
            feedback_message=feedback,
        )

        # Compute final score if done
        if done:
            obs.reward = self._compute_final_score()

        return obs

    @property
    def state(self) -> TrafficState:
        """
        Get the current environment state.

        Returns:
            Current TrafficState with episode_id, step_count, and grid metadata
        """
        return self._state

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    # ---- Private helpers ----

    def _setup_emergency_vehicle(self, active: list[int]) -> None:
        """Spawn an emergency vehicle at a random edge intersection.

        Selects a random origin and destination from edge intersections, computes
        the BFS shortest path, and places the vehicle in the grid.

        Args:
            active: List of active intersection IDs for this task.
        """
        edge_intersections = [i for i in active if len(NEIGHBORS.get(i, {})) < 4]
        if len(edge_intersections) < 2:
            edge_intersections = active

        assert self._rng is not None
        origin = int(self._rng.choice(edge_intersections))
        dest_candidates = [i for i in edge_intersections if i != origin]
        destination = int(self._rng.choice(dest_candidates))

        path = self._bfs_path(origin, destination, active)

        self._emergency_destination = destination
        self._emergency_path = path[1:]  # exclude origin (vehicle starts there)
        self._emergency_steps = 0
        self._emergency_arrived = False

        # Create emergency vehicle and add to grid
        self._emergency_vehicle = Vehicle(
            vehicle_id="emergency_0",
            origin=origin,
            destination=destination,
            current_intersection=origin,
            current_lane="N",  # placeholder
            is_emergency=True,
        )
        logger.info("Emergency vehicle: %d -> %d, path: %s", origin, destination, path)

    def _bfs_path(self, start: int, end: int, active: list[int]) -> list[int]:
        """Find the shortest path between two intersections using BFS.

        Args:
            start: Origin intersection ID.
            end: Destination intersection ID.
            active: Set of traversable intersection IDs.

        Returns:
            Ordered list of intersection IDs from start to end, inclusive.
        """
        active_set = set(active)
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            for direction, neighbor in NEIGHBORS.get(current, {}).items():
                if neighbor not in visited and neighbor in active_set:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # Fallback: direct path (shouldn't happen in connected grid)
        return [start, end]

    def _step_emergency_vehicle(self) -> float:
        """Advance the emergency vehicle one step if the signal allows movement.

        The vehicle moves to the next intersection on its path when the current
        intersection has a green phase in the required direction.

        Returns:
            A reward delta: ``1.0`` on arrival, ``0.5`` on progress, ``-0.1``
            when blocked by a red signal.
        """
        if not self._emergency_path:
            self._emergency_arrived = True
            return 1.0

        self._emergency_steps += 1
        assert self._emergency_vehicle is not None
        assert self._grid is not None
        current = self._emergency_vehicle.current_intersection
        next_intersection = self._emergency_path[0]

        # Determine which direction the emergency vehicle needs to go
        for direction, neighbor in NEIGHBORS.get(current, {}).items():
            if neighbor == next_intersection:
                # Check if this direction has green
                phase = self._grid.phases[current]
                from simulator.grid import GREEN_LANES

                if direction in GREEN_LANES.get(phase, []):
                    # Move emergency vehicle
                    self._emergency_vehicle.current_intersection = next_intersection
                    self._emergency_path = self._emergency_path[1:]
                    if not self._emergency_path:
                        self._emergency_arrived = True
                        return 1.0
                    return 0.5  # Made progress
                break

        return -0.1  # No progress

    def _compute_step_reward(
        self,
        prev_wait: float,
        new_wait: float,
        baseline_wait: float,
        emergency_delta: float | None,
    ) -> float:
        wait_improvement = (prev_wait - new_wait) / (baseline_wait + 1e-6)
        efficiency = max(0.0, 1.0 - (new_wait / (baseline_wait + 1e-6)))
        reward = 0.7 * wait_improvement + 0.3 * efficiency
        if emergency_delta is not None:
            reward = 0.6 * reward + 0.4 * emergency_delta
        return round(float(np.clip(reward, -1.0, 1.0)), 4)

    def _compute_final_score(self) -> float:
        """Compute final score using the appropriate grader."""
        task_id = self._state.task_id

        if task_id == "corridor_coordination":
            agent_wait = (
                float(np.mean(self._wait_time_history))
                if self._wait_time_history
                else 0.0
            )
            return corridor_grader.grade(agent_wait, self._baseline_wait)

        elif task_id == "grid_coordination":
            agent_wait = (
                float(np.mean(self._wait_time_history))
                if self._wait_time_history
                else 0.0
            )
            return grid_grader.grade(agent_wait, self._baseline_wait)

        elif task_id == "emergency_response":
            max_emergency_time = float(self._config["max_steps"])
            emergency_time = (
                float(self._emergency_steps)
                if self._emergency_arrived
                else max_emergency_time
            )
            agent_wait = (
                float(np.mean(self._wait_time_history))
                if self._wait_time_history
                else 0.0
            )
            return emergency_grader.grade(
                emergency_time, max_emergency_time, agent_wait, self._baseline_wait
            )

        return 0.0

    def _build_observation(
        self, step_reward: float, feedback_message: str
    ) -> TrafficObservation:
        """Construct the full TrafficObservation from current environment state.

        Args:
            step_reward: Reward value for the most recent action.
            feedback_message: Human-readable status string for this step.

        Returns:
            TrafficObservation populated with per-agent state and global metrics.
        """
        agents = self._grid.get_observation() if self._grid else []
        global_wait = self._grid.get_mean_wait_time() if self._grid else 0.0

        emergency_state = None
        if self._config.get("emergency") and self._emergency_vehicle is not None:
            emergency_state = EmergencyVehicleState(
                position=self._emergency_vehicle.current_intersection,
                destination=self._emergency_destination,
                steps_elapsed=self._emergency_steps,
                path_remaining=list(self._emergency_path),
            )

        cumulative = round(self._cumulative_reward / max(self._state.step_count, 1), 4)

        return TrafficObservation(
            task_id=self._state.task_id,
            episode_id=self._state.episode_id or "",
            step=self._state.step_count,
            agents=agents,
            global_wait_time=round(global_wait, 4),
            step_reward=round(step_reward, 4),
            cumulative_score=cumulative,
            feedback_message=feedback_message,
            done=self._state.is_complete,
            emergency_vehicle=emergency_state,
        )
