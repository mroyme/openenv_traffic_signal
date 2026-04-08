"""
TrafficSignalEnv Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    LOCAL_IMAGE_NAME    The name of the local Docker image if using from_docker_image().
    TRAFFIC_ENV_URL     URL of a running TrafficSignalEnv server (default: http://localhost:8000).
    TRAFFIC_TASK        Task to run: corridor_coordination, grid_coordination, emergency_response.
    TRAFFIC_EPISODE     Episode index (controls random seed, default: 0).

- Defaults for API_BASE_URL and MODEL_NAME reflect a common inference setup:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory.
- Participants must use the OpenAI client for all LLM calls.

STDOUT FORMAT
- The script emits exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards formatted to 2 decimal places; score to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the exception message, or null if none.
    - All fields on a single line with no newlines within a line.
    - score is in [0, 1].

  Example:
    [START] task=corridor_coordination env=traffic_signal model=Qwen2.5-72B-Instruct
    [STEP] step=1 action={0:keep,1:keep,2:keep} reward=0.00 done=false error=null
    [STEP] step=2 action={0:switch,1:keep,2:keep} reward=0.12 done=false error=null
    [END] success=true steps=2 score=0.450 rewards=0.00,0.12
"""

import argparse
import asyncio
import json
import os
import sys
import textwrap
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from openai import AsyncOpenAI

from client import TrafficSignalEnv
from models import AgentAction, TrafficAction

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
ENV_URL = (
    os.getenv("OPENENV_TRAFFIC_SIGNAL_ENV_URL")
    or os.getenv("TRAFFIC_ENV_URL")
    or os.getenv("ENV_URL")
    or "http://localhost:8000"
)
TASK_NAME = (
    os.getenv("OPENENV_TRAFFIC_SIGNAL_TASK")
    or os.getenv("TRAFFIC_TASK")
    or os.getenv("TASK_NAME")
    or ""
)
ALL_TASKS = ["corridor_coordination", "grid_coordination", "emergency_response"]
TASK_ALIASES: dict[str, str] = {
    "easy": "corridor_coordination",
    "medium": "grid_coordination",
    "hard": "emergency_response",
}
TASK_NAME = TASK_ALIASES.get(TASK_NAME, TASK_NAME)
EPISODE = int(
    os.getenv("OPENENV_TRAFFIC_SIGNAL_EPISODE")
    or os.getenv("TRAFFIC_EPISODE")
    or os.getenv("EPISODE")
    or "0"
)
BENCHMARK = (
    os.getenv("OPENENV_TRAFFIC_SIGNAL_BENCHMARK")
    or os.getenv("TRAFFIC_BENCHMARK")
    or os.getenv("BENCHMARK")
    or "traffic_signal"
)
BASE_SEED = 42
MIN_HOLD_STEPS = 10  # minimum steps before any switch is allowed
MAX_HOLD_STEPS = (
    30  # force switch — same as baseline, never hold longer than fixed timing
)
DEFAULT_HOLD_STEPS = 25  # fallback for LLM hold plan
CORRIDOR_MIN_HOLD = 5  # corridor: allow early switch from this point (server min is 5)
CORRIDOR_MAX_HOLD = 30  # corridor: cap at baseline timing
CORRIDOR_RATIO = (
    3.0  # corridor: switch on clear aggregate imbalance (preserves green wave)
)
SUCCESS_SCORE_THRESHOLD = 0.5
TEMPERATURE = 0.2
MAX_TOKENS = 512

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a traffic signal planner for a grid of intersections.

    SIGNAL PHASES:
      0 = NS green  (North↔South traffic flows)
      2 = EW green  (East↔West traffic flows)
      Yellow phases (1, 3) are 2-step automatic transitions — you cannot control them.

    YOUR JOB:
    Agents are entering a new green phase. Decide how many steps each agent should
    hold its current green phase before switching to the other direction.

    DECISION GUIDE:
    - Default: 25 steps.
    - Hold LONGER (up to 45) when:
        * Queue in the current green direction is long.
        * A neighbor upstream is also green — vehicles are en route.
    - Hold SHORTER (down to 10) when:
        * Queue in the current green direction is near zero.
        * Queue in the waiting direction is at least 2x longer.
        * An emergency vehicle needs to pass through in the waiting direction.
    - Each switch costs 2 yellow steps — avoid very short holds (< 15) unless queue
      imbalance is extreme or an emergency vehicle is waiting.
    - For corridor tasks: give ALL agents the SAME hold duration so vehicles can flow
      through multiple intersections without stopping (green wave).
    - For grid tasks: avoid large variation between agents; prefer holds in 20-35 range.

    REASONING: Before answering, briefly think through the queue imbalances and
    neighbor states. Then output ONLY the JSON array — no other text after it.

    OUTPUT FORMAT (nothing else after the array):
    [{"agent_id": 0, "hold_steps": 25}, {"agent_id": 1, "hold_steps": 20}, ...]
    """
).strip()


_output_file: Any | None = None


def _emit(line: str) -> None:
    print(line, flush=True)
    if _output_file is not None:
        _output_file.write(line + "\n")
        _output_file.flush()


def log_start(task: str, env: str, model: str) -> None:
    _emit(f"[START] task={task} env={env} model={model}")


def log_step(
    step: int, action: str, reward: float, done: bool, error: str | None
) -> None:
    error_val = error if error else "null"
    _emit(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}"
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    _emit(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}"
    )


def build_prompt(obs_data: dict, task_id: str, history: list[str] | None = None) -> str:
    agents = obs_data.get("agents", [])
    emergency = obs_data.get("emergency_vehicle")

    lines = [
        f"Task: {task_id}",
        f"Step: {obs_data.get('step', 0)}",
        f"Global wait time: {obs_data.get('global_wait_time', 0):.4f}",
        "",
        "Intersection states (queues: N, S, E, W):",
    ]
    for a in agents:
        q = a["queue_lengths"]  # [N, S, E, W]
        ns = q[0] + q[1] if len(q) >= 2 else 0
        ew = q[2] + q[3] if len(q) >= 4 else 0
        green_dir = "NS" if a["current_phase"] in (0, 1) else "EW"
        green_q = ns if green_dir == "NS" else ew
        wait_q = ew if green_dir == "NS" else ns
        ratio = f"{wait_q / green_q:.1f}x" if green_q > 0.1 else "∞"
        lines.append(
            f"  Agent {a['agent_id']}: queues={a['queue_lengths']} "
            f"[NS={ns:.1f} EW={ew:.1f} | green={green_dir}({green_q:.1f}) wait={ratio}], "
            f"phase={a['current_phase']}, elapsed={a['phase_elapsed']}, "
            f"neighbor_queues={a['neighbor_queues']}, local_wait={a['local_wait_time']:.4f}"
        )

    if emergency:
        lines.append(
            f"\nEmergency vehicle: pos={emergency['position']}, "
            f"dest={emergency['destination']}, "
            f"path_remaining={emergency['path_remaining']}"
        )

    if history:
        lines.append("")
        lines.append("Recent phase history:")
        lines.extend(f"  {h}" for h in history)

    return "\n".join(lines)


def parse_hold_plan(response_text: str, agent_ids: list[int]) -> dict[int, int]:
    """Parse LLM response into hold durations. Falls back to DEFAULT_HOLD_STEPS."""
    try:
        text = response_text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            items = json.loads(text[start:end])
            result = {}
            for item in items:
                aid = item.get("agent_id")
                hold = item.get("hold_steps", DEFAULT_HOLD_STEPS)
                if aid is not None:
                    result[int(aid)] = max(
                        MIN_HOLD_STEPS, min(MAX_HOLD_STEPS, int(hold))
                    )
            return result
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    return {aid: DEFAULT_HOLD_STEPS for aid in agent_ids}


def compute_hold(agent_obs: dict) -> int:
    """Compute hold duration from queue imbalance. Returns steps to hold current green phase."""
    q = agent_obs.get("queue_lengths", [0, 0, 0, 0])
    ns = q[0] + q[1] if len(q) >= 2 else 0
    ew = q[2] + q[3] if len(q) >= 4 else 0
    phase = agent_obs.get("current_phase", 0)
    green_q = ns if phase in (0, 1) else ew
    wait_q = ew if phase in (0, 1) else ns

    # Both directions sparse — no strong signal, use default
    if green_q < 0.5 and wait_q < 1.0:
        return DEFAULT_HOLD_STEPS
    # Current direction empty but waiting direction has traffic — switch soon
    if green_q < 0.5 and wait_q >= 1.0:
        return MIN_HOLD_STEPS
    ratio = wait_q / green_q
    if ratio > 3.0:
        return MIN_HOLD_STEPS  # waiting direction 3x+ longer, switch soon
    if ratio > 1.5:
        return 18  # moderate imbalance, switch a bit early
    if ratio < 0.33:
        return 38  # current direction strongly dominant, hold long
    return DEFAULT_HOLD_STEPS  # balanced, use default


def reactive_action(a: dict) -> str:
    """Queue-adaptive per-step decision: switch when current direction is exhausted."""
    phase = a.get("current_phase", 0)
    elapsed = a.get("phase_elapsed", 0)

    if phase not in (0, 2):
        return "keep"  # yellow — action is ignored anyway
    if elapsed < MIN_HOLD_STEPS:
        return "keep"  # minimum hold not met

    q = a.get("queue_lengths", [0, 0, 0, 0])
    ns = q[0] + q[1] if len(q) >= 2 else 0.0
    ew = q[2] + q[3] if len(q) >= 4 else 0.0
    green_q = ns if phase == 0 else ew
    wait_q = ew if phase == 0 else ns

    # Current direction empty and waiting direction has vehicles — switch now
    if green_q < 0.5 and wait_q > 0.5:
        return "switch"
    # Waiting direction is 2.5x+ longer — switch
    if green_q > 0 and (wait_q / green_q) > 2.5:
        return "switch"
    # Force switch to prevent starvation
    if elapsed >= MAX_HOLD_STEPS:
        return "switch"

    return "keep"


def corridor_actions(agents: list) -> dict[int, str]:
    """Synchronized corridor switch via aggregate queue signal.
    All agents switch together based on total queue balance across the corridor.
    All-or-nothing: switch only when aggregate wait direction is sufficiently dominant.
    """
    agent_dicts = [a.model_dump() for a in agents]
    green_agents = [a for a in agent_dicts if a["current_phase"] in (0, 2)]
    if not green_agents:
        return {a["agent_id"]: "keep" for a in agent_dicts}

    max_elapsed = max(a["phase_elapsed"] for a in green_agents)
    if max_elapsed < CORRIDOR_MIN_HOLD:
        return {a["agent_id"]: "keep" for a in agent_dicts}

    # Aggregate queues across all green-phase agents
    phase = green_agents[0]["current_phase"]
    total_ns = sum(
        (a["queue_lengths"][0] + a["queue_lengths"][1])
        for a in green_agents
        if len(a["queue_lengths"]) >= 2
    )
    total_ew = sum(
        (a["queue_lengths"][2] + a["queue_lengths"][3])
        for a in green_agents
        if len(a["queue_lengths"]) >= 4
    )
    green_q = total_ns if phase in (0, 1) else total_ew
    wait_q = total_ew if phase in (0, 1) else total_ns
    n = len(green_agents)

    should_switch = False
    if green_q < 0.5 * n and wait_q > 0.5 * n:
        should_switch = True
    elif green_q > 0 and (wait_q / green_q) > CORRIDOR_RATIO:
        should_switch = True
    elif max_elapsed >= CORRIDOR_MAX_HOLD:
        should_switch = True

    result = {}
    for a in agent_dicts:
        if should_switch and a["current_phase"] in (0, 2):
            result[a["agent_id"]] = "switch"
        else:
            result[a["agent_id"]] = "keep"
    return result


def action_str(actions: list[dict]) -> str:
    """Compact human-readable action summary."""
    return "{" + ",".join(f"{a['agent_id']}:{a['phase_action']}" for a in actions) + "}"


async def get_hold_plan(
    client: AsyncOpenAI,
    obs_data: dict,
    task_id: str,
    agent_ids: list[int],
    history: list[str] | None = None,
) -> dict[int, int]:
    prompt = build_prompt(obs_data, task_id, history)
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        text = ""
    return parse_hold_plan(text, agent_ids)


async def run_task(
    task_name: str, client: AsyncOpenAI, env: TrafficSignalEnv, seed: int
) -> None:
    """Run a single task episode and emit [START]/[STEP]/[END] to stdout."""
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(seed=seed, task_id=task_name)
        obs = result.observation

        # hold_plan used only for emergency_response task
        hold_plan: dict[int, int] = {}
        prev_phase: dict[int, int] = {}
        phase_history: list[str] = []

        while not obs.done:
            agent_ids = [a.agent_id for a in obs.agents]

            # Detect phase transitions — call LLM to update hold plan (emergency only)
            needs_plan = [
                a.agent_id
                for a in obs.agents
                if a.current_phase in (0, 2)
                and (
                    a.agent_id not in hold_plan or prev_phase.get(a.agent_id) in (1, 3)
                )
            ]

            if needs_plan:
                new_plan = await get_hold_plan(
                    client, obs.model_dump(), task_name, agent_ids, phase_history[-6:]
                )
                if task_name == "emergency_response":
                    for aid in needs_plan:
                        hold_plan[aid] = new_plan.get(aid, DEFAULT_HOLD_STEPS)
                    print(
                        f"[DEBUG] step={obs.step} hold plan agents={needs_plan}: "
                        + ", ".join(f"{aid}→{hold_plan[aid]}" for aid in needs_plan),
                        file=sys.stderr,
                        flush=True,
                    )

            # Decide actions
            actions_list: list[dict] = []
            if task_name in ("corridor_coordination", "grid_coordination"):
                sync_decisions = corridor_actions(obs.agents)
                for a in obs.agents:
                    actions_list.append(
                        {
                            "agent_id": a.agent_id,
                            "phase_action": sync_decisions[a.agent_id],
                        }
                    )
                    prev_phase[a.agent_id] = a.current_phase
            else:
                for a in obs.agents:
                    if task_name == "emergency_response":
                        # Use LLM hold plan
                        if a.current_phase in (
                            0,
                            2,
                        ) and a.phase_elapsed >= hold_plan.get(
                            a.agent_id, DEFAULT_HOLD_STEPS
                        ):
                            phase_action = "switch"
                        else:
                            phase_action = "keep"
                    else:
                        phase_action = reactive_action(a.model_dump())
                    actions_list.append(
                        {"agent_id": a.agent_id, "phase_action": phase_action}
                    )
                    prev_phase[a.agent_id] = a.current_phase

            action = TrafficAction(
                agent_actions=[
                    AgentAction(agent_id=a["agent_id"], phase_action=a["phase_action"])
                    for a in actions_list
                ]
            )

            try:
                result = await env.step(action)
            except Exception as exc:
                log_step(obs.step + 1, action_str(actions_list), 0.0, True, str(exc))
                break

            obs = result.observation
            reward = float(obs.reward or 0.0)
            rewards.append(reward)
            steps_taken = obs.step
            if obs.done and obs.final_score is not None:
                score = obs.final_score

            log_step(obs.step, action_str(actions_list), reward, obs.done, None)

            # Record phase transitions in history for LLM context
            if needs_plan:
                phase_history.append(
                    f"step={obs.step} phase_transition agents={needs_plan} "
                    f"global_wait={obs.global_wait_time:.4f} reward={reward:.4f}"
                )

        score = round(min(max(score, 0.01), 0.99), 2)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main(write: bool = False) -> None:
    global _output_file

    tasks_to_run = [TASK_NAME] if TASK_NAME else ALL_TASKS

    if write:
        os.makedirs("outputs", exist_ok=True)
        label = TASK_NAME or "all"
        output_path = f"outputs/{label}_ep{EPISODE}.txt"
        _output_file = open(output_path, "w")

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    seed = BASE_SEED + EPISODE

    if IMAGE_NAME:
        env = await TrafficSignalEnv.from_docker_image(IMAGE_NAME)
    else:
        env = TrafficSignalEnv(base_url=ENV_URL)

    try:
        await env.connect()
        for task_name in tasks_to_run:
            await run_task(task_name, client, env, seed)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", file=sys.stderr, flush=True)
        if _output_file is not None:
            _output_file.close()
            _output_file = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write output to outputs/<task>_ep<n>.txt in addition to stdout",
    )
    args = parser.parse_args()
    asyncio.run(main(write=args.write))
