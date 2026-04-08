"""
FastAPI application for the Traffic Signal Environment.

This module creates an HTTP server that exposes the TrafficEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

import threading
from typing import Literal

from models import AgentAction, TrafficAction, TrafficObservation
from server.traffic_signal_environment import TrafficEnvironment


app = create_app(
    TrafficEnvironment,
    TrafficAction,
    TrafficObservation,
    env_name="openenv-traffic-signal",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# ---------------------------------------------------------------------------
# Extra endpoints required by the OpenEnv hackathon validation pipeline
# ---------------------------------------------------------------------------

_env: TrafficEnvironment | None = None
_env_lock = threading.Lock()


def _get_env() -> TrafficEnvironment:
    """Return the singleton environment, creating one if needed."""
    global _env
    if _env is None:
        _env = TrafficEnvironment()
    return _env


@app.get("/tasks")
async def list_tasks():
    """Return all available tasks with metadata."""
    tasks = []
    for task_id, cfg in TrafficEnvironment.TASK_CONFIGS.items():
        tasks.append(
            {
                "id": task_id,
                "max_steps": cfg["max_steps"],
                "emergency": cfg["emergency"],
                "active_intersections": cfg["active_intersections"],
            }
        )
    return {"tasks": tasks}


@app.get("/grader")
async def get_grader_score():
    """Return the grader score for the current/last episode."""
    with _env_lock:
        env = _get_env()
        if not env._state.is_complete:
            return {"score": None, "message": "Episode not complete yet."}
        score = env._compute_final_score()
        return {
            "task_id": env._state.task_id,
            "score": score,
            "done": True,
        }


@app.post("/baseline")
async def run_baseline():
    """Run a baseline heuristic agent against all tasks and return scores."""
    import os

    from openai import AsyncOpenAI

    from inference import (
        corridor_actions,
        reactive_action,
        get_hold_plan,
        DEFAULT_HOLD_STEPS,
        SUCCESS_SCORE_THRESHOLD,
    )

    api_key = (
        os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key) if api_key else None

    task_ids = list(TrafficEnvironment.TASK_CONFIGS.keys())
    results = []

    for task_id in task_ids:
        with _env_lock:
            env = _get_env()
            obs = env.reset(seed=42, task_id=task_id)

        score = 0.0
        steps = 0
        hold_plan: dict[int, int] = {}
        prev_phase: dict[int, int] = {}
        phase_history: list[str] = []

        while not obs.done:
            agents = obs.agents
            agent_ids = [a.agent_id for a in agents]

            # Detect phase transitions for emergency LLM planning
            needs_plan = [
                a.agent_id
                for a in agents
                if a.current_phase in (0, 2)
                and (
                    a.agent_id not in hold_plan or prev_phase.get(a.agent_id) in (1, 3)
                )
            ]

            if needs_plan and task_id == "emergency_response" and client:
                new_plan = await get_hold_plan(
                    client, obs.model_dump(), task_id, agent_ids, phase_history[-6:]
                )
                for aid in needs_plan:
                    hold_plan[aid] = new_plan.get(aid, DEFAULT_HOLD_STEPS)

            # Decide actions using the same logic as inference.py
            actions_list = []
            if task_id in ("corridor_coordination", "grid_coordination"):
                sync_decisions = corridor_actions(agents)
                for a in agents:
                    pa: Literal["keep", "switch"] = (
                        "switch" if sync_decisions[a.agent_id] == "switch" else "keep"
                    )
                    actions_list.append(
                        AgentAction(agent_id=a.agent_id, phase_action=pa)
                    )
                    prev_phase[a.agent_id] = a.current_phase
            else:
                for a in agents:
                    if task_id == "emergency_response":
                        if a.current_phase in (
                            0,
                            2,
                        ) and a.phase_elapsed >= hold_plan.get(
                            a.agent_id, DEFAULT_HOLD_STEPS
                        ):
                            phase_action: Literal["keep", "switch"] = "switch"
                        else:
                            phase_action = "keep"
                    else:
                        ra = reactive_action(a.model_dump())
                        phase_action = "switch" if ra == "switch" else "keep"
                    actions_list.append(
                        AgentAction(agent_id=a.agent_id, phase_action=phase_action)
                    )
                    prev_phase[a.agent_id] = a.current_phase

            action = TrafficAction(agent_actions=actions_list)

            with _env_lock:
                obs = env.step(action)

            steps += 1

            if needs_plan:
                phase_history.append(
                    f"step={obs.step} global_wait={obs.global_wait_time:.4f}"
                )

            if obs.done and obs.final_score is not None:
                score = obs.final_score

        results.append(
            {
                "task_id": task_id,
                "score": round(score, 4),
                "resolved": score >= SUCCESS_SCORE_THRESHOLD,
                "steps": steps,
            }
        )

    total = sum(r["score"] for r in results)
    resolved = sum(1 for r in results if r["resolved"])
    return {
        "model": model if client else "heuristic",
        "results": results,
        "total_score": round(total, 3),
        "average_score": round(total / len(results), 3) if results else 0.0,
        "resolved": f"{resolved}/{len(results)}",
    }


def main():
    """Start the uvicorn server.

    Entry point for direct execution via ``uv run`` or ``python -m``::

        uv run --project . server
        python -m server.app

    For production with multiple workers use uvicorn directly::

        uvicorn server.app:app --workers 4
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
