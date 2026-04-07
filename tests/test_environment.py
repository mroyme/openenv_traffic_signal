"""Tests for TrafficEnvironment reset/step cycle."""

import pytest
from models import AgentAction, TrafficAction
from server.traffic_signal_environment import TrafficEnvironment


@pytest.fixture
def env():
    return TrafficEnvironment()


@pytest.mark.parametrize(
    "task_id",
    [
        "corridor_coordination",
        "grid_coordination",
        "emergency_response",
    ],
)
def test_reset_returns_valid_observation(env, task_id):
    obs = env.reset(seed=42, task_id=task_id)
    assert obs.step == 0
    assert len(obs.agents) > 0
    assert obs.global_wait_time >= 0.0
    assert not obs.done


@pytest.mark.parametrize(
    "task_id",
    [
        "corridor_coordination",
        "grid_coordination",
        "emergency_response",
    ],
)
def test_five_steps(env, task_id):
    obs = env.reset(seed=42, task_id=task_id)
    for _ in range(5):
        action = TrafficAction(
            agent_actions=[
                AgentAction(agent_id=a.agent_id, phase_action="keep")
                for a in obs.agents
            ]
        )
        obs = env.step(action)
        assert obs.step > 0
        assert obs.global_wait_time >= 0.0


def test_emergency_vehicle_present(env):
    obs = env.reset(seed=42, task_id="emergency_response")
    assert obs.emergency_vehicle is not None
    ev = obs.emergency_vehicle
    assert ev.position is not None
    assert ev.destination is not None
