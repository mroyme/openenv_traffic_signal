"""Traffic Signal Environment Client."""

from typing import Any

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import TrafficAction, TrafficObservation, TrafficState


class TrafficSignalEnv(EnvClient[TrafficAction, TrafficObservation, TrafficState]):
    """Client for the Traffic Signal Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency. Each client
    instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> async with TrafficSignalEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(seed=42, task_id="corridor_coordination")
        ...     obs = result.observation
        ...     print(obs.global_wait_time)
        ...
        ...     action = TrafficAction(agent_actions=[AgentAction(agent_id=0, phase_action="keep")])
        ...     result = await env.step(action)
        ...     print(result.observation.step_reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> env = await TrafficSignalEnv.from_docker_image("openenv-traffic-signal:latest")
        >>> try:
        ...     result = await env.reset(seed=42, task_id="grid_coordination")
        ...     result = await env.step(action)
        ... finally:
        ...     await env.close()
    """

    def _step_payload(self, action: TrafficAction) -> dict:
        """Convert a TrafficAction to a JSON payload for the step message.

        Args:
            action: TrafficAction with per-agent phase decisions.

        Returns:
            Dictionary suitable for JSON encoding and transmission to the server.
        """
        return {
            "agent_actions": [
                {"agent_id": a.agent_id, "phase_action": a.phase_action}
                for a in action.agent_actions
            ]
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[TrafficObservation]:
        """Parse a server response into a StepResult.

        Args:
            payload: JSON response data from the server.

        Returns:
            StepResult containing the parsed TrafficObservation, reward, and done flag.
        """
        obs_data = payload.get("observation", {})
        done = payload.get("done", False)
        reward = payload.get("reward")
        # Inject done/reward back since serialize_observation strips them
        obs_data["done"] = done
        if reward is not None:
            obs_data["reward"] = reward
        observation = TrafficObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict[str, Any]) -> TrafficState:
        """Parse a server response into a TrafficState object.

        Args:
            payload: JSON response from the state endpoint.

        Returns:
            TrafficState with episode and grid metadata.
        """
        return TrafficState(**payload)
