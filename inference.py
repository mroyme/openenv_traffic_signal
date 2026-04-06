"""
inference.py -- TrafficSignalEnv Baseline (OpenAI client)

Reads from environment variables:
    API_BASE_URL   -- LLM API endpoint
    MODEL_NAME     -- model identifier
    HF_TOKEN       -- Hugging Face / API key

Stdout format (strictly required):
    [START] {"task_id": "...", "episode": N, "seed": N}
    [STEP]  {"step": N, "actions": [...], "reward": 0.0, "global_wait": 0.0}
    [END]   {"task_id": "...", "episode": N, "score": 0.0}
"""

import os
import sys
import json
import time
import asyncio

from openai import OpenAI
from client import TrafficSignalEnv
from models import TrafficAction, AgentAction

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ["HF_TOKEN"]

EPISODES_PER_TASK = 5
BASE_SEED = 42

TASKS = [
    "corridor_coordination",
    "grid_coordination",
    "emergency_response",
]

ENV_URL = os.environ.get("TRAFFIC_ENV_URL", "http://localhost:8000")

MAX_RUNTIME_SECONDS = 18 * 60  # 18 minutes safety cutoff


def _log(msg: str) -> None:
    """Debug log to stderr only."""
    print(msg, file=sys.stderr, flush=True)


def _build_prompt(obs_data: dict, task_id: str) -> str:
    """Build the LLM prompt from observation data."""
    agents = obs_data.get("agents", [])
    emergency = obs_data.get("emergency_vehicle")

    lines = [
        f"You are controlling traffic signals for task '{task_id}'.",
        f"Current step: {obs_data.get('step', 0)}",
        f"Global wait time: {obs_data.get('global_wait_time', 0):.2f}",
        "",
        "Intersection states:",
    ]

    for a in agents:
        lines.append(
            f"  Agent {a['agent_id']}: queues={a['queue_lengths']}, "
            f"phase={a['current_phase']}, elapsed={a['phase_elapsed']}, "
            f"neighbor_queues={a['neighbor_queues']}, wait={a['local_wait_time']:.2f}"
        )

    if emergency:
        lines.append(
            f"\nEmergency vehicle: pos={emergency['position']}, "
            f"dest={emergency['destination']}, "
            f"path_remaining={emergency['path_remaining']}"
        )

    lines.extend(
        [
            "",
            "For each agent, decide 'keep' (maintain current phase) or 'switch' (change phase).",
            "Rules: switch only works during green phases (0 or 2) after 5+ steps elapsed.",
            "Yellow phases (1, 3) auto-advance and cannot be interrupted.",
            "",
            "Respond with ONLY a JSON array of objects, each with 'agent_id' (int) and 'phase_action' ('keep' or 'switch').",
            'Example: [{"agent_id": 0, "phase_action": "keep"}, {"agent_id": 1, "phase_action": "switch"}]',
        ]
    )

    return "\n".join(lines)


def _parse_llm_response(response_text: str, agent_ids: list) -> list:
    """Parse LLM response into agent actions. Falls back to 'keep' on failure."""
    try:
        # Try to extract JSON array from response
        text = response_text.strip()
        # Find JSON array in response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            actions = json.loads(text[start:end])
            # Validate structure
            result = []
            seen_ids = set()
            for a in actions:
                aid = a.get("agent_id")
                pa = a.get("phase_action", "keep")
                if aid is not None and pa in ("keep", "switch"):
                    result.append({"agent_id": int(aid), "phase_action": pa})
                    seen_ids.add(int(aid))
            # Fill in missing agents with "keep"
            for aid in agent_ids:
                if aid not in seen_ids:
                    result.append({"agent_id": aid, "phase_action": "keep"})
            return result
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        _log(f"Parse error: {e}")

    # Fallback: keep for all agents
    return [{"agent_id": aid, "phase_action": "keep"} for aid in agent_ids]


def _call_llm(client: OpenAI, prompt: str) -> str:
    """Make one LLM call. Returns response text."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        _log(f"LLM call failed: {e}")
        return ""


async def run_inference():
    start_time = time.time()
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_id in TASKS:
        for ep_idx in range(EPISODES_PER_TASK):
            seed = BASE_SEED + ep_idx

            # Check time budget
            elapsed = time.time() - start_time
            if elapsed > MAX_RUNTIME_SECONDS:
                _log(f"Time budget exceeded at {elapsed:.0f}s, truncating")
                print(json.dumps({"task_id": task_id, "episode": ep_idx, "score": 0.0}))
                sys.stdout.flush()
                continue

            print(
                f"[START] {json.dumps({'task_id': task_id, 'episode': ep_idx, 'seed': seed})}"
            )
            sys.stdout.flush()

            env_client = TrafficSignalEnv(base_url=ENV_URL)
            try:
                await env_client.connect()
                result = await env_client.reset(seed=seed, task_id=task_id)
                obs = result.observation

                final_score = 0.0
                step_num = 0

                while not obs.done:
                    # Check time budget
                    if time.time() - start_time > MAX_RUNTIME_SECONDS:
                        _log("Time cutoff during episode")
                        break

                    agent_ids = [a.agent_id for a in obs.agents]
                    obs_dict = obs.model_dump()

                    prompt = _build_prompt(obs_dict, task_id)
                    response_text = _call_llm(llm_client, prompt)
                    actions_list = _parse_llm_response(response_text, agent_ids)

                    action = TrafficAction(
                        agent_actions=[
                            AgentAction(
                                agent_id=a["agent_id"], phase_action=a["phase_action"]
                            )
                            for a in actions_list
                        ]
                    )

                    result = await env_client.step(action)
                    obs = result.observation
                    step_num = obs.step

                    action_summary = [
                        {"agent_id": a["agent_id"], "action": a["phase_action"]}
                        for a in actions_list
                    ]
                    print(
                        f"[STEP]  {json.dumps({'step': step_num, 'actions': action_summary, 'reward': obs.step_reward, 'global_wait': obs.global_wait_time})}"
                    )
                    sys.stdout.flush()

                    if result.reward is not None:
                        final_score = result.reward

                print(
                    f"[END]   {json.dumps({'task_id': task_id, 'episode': ep_idx, 'score': final_score})}"
                )
                sys.stdout.flush()

            except Exception as e:
                _log(f"Episode error: {e}")
                print(
                    f"[END]   {json.dumps({'task_id': task_id, 'episode': ep_idx, 'score': 0.0})}"
                )
                sys.stdout.flush()
            finally:
                try:
                    await env_client.close()
                except Exception:
                    pass


def main():
    asyncio.run(run_inference())


if __name__ == "__main__":
    main()
