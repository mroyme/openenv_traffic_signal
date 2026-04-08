"""Custom Gradio UI for the Traffic Signal Environment.

Provides a task picker, per-agent keep/switch controls, and a visual
grid observation display.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gradio as gr

from server.traffic_signal_environment import TrafficEnvironment

TASK_IDS = list(TrafficEnvironment.TASK_CONFIGS.keys())
PHASE_NAMES = {0: "NS green", 1: "NS yellow", 2: "EW green", 3: "EW yellow"}
PHASE_ICON = {0: "🟢 NS", 1: "🟡 NS", 2: "🟢 EW", 3: "🟡 EW"}
MAX_AGENTS = 9  # largest task uses all 9 intersections

# 3x3 grid layout: row, col for each intersection ID
GRID_POS = {i: (i // 3, i % 3) for i in range(9)}


def _queue_bar(value: float, max_val: float = 8.0) -> str:
    """Render a small horizontal bar for a queue length."""
    filled = min(int(value / max_val * 5), 5) if max_val > 0 else 0
    return "\u2588" * filled + "\u2591" * (5 - filled)


def _fmt_intersection(a: Dict[str, Any], ev_pos: int | None = None) -> str:
    """Format a single intersection as a compact Markdown block."""
    aid = a["agent_id"]
    phase = a["current_phase"]
    elapsed = a["phase_elapsed"]
    q = a["queue_lengths"]  # [N, S, E, W]
    lwait = a["local_wait_time"]

    icon = PHASE_ICON.get(phase, "?")
    ev_marker = " \U0001F691" if ev_pos == aid else ""

    ns_total = q[0] + q[1]
    ew_total = q[2] + q[3]

    lines = [
        f"**[{aid}]** {icon}{ev_marker}",
        f"N {_queue_bar(q[0])} {q[0]:.0f}",
        f"S {_queue_bar(q[1])} {q[1]:.0f}",
        f"E {_queue_bar(q[2])} {q[2]:.0f}",
        f"W {_queue_bar(q[3])} {q[3]:.0f}",
        f"NS={ns_total:.0f} EW={ew_total:.0f} | t={elapsed} | w={lwait:.2f}",
    ]
    return "\n".join(lines)


def _fmt_grid(data: Dict[str, Any]) -> str:
    """Format the full observation with a visual 3x3 grid."""
    obs = data.get("observation", {})
    lines: List[str] = []

    task = obs.get("task_id", "")
    step = obs.get("step", 0)
    gwait = obs.get("global_wait_time", 0.0)

    reward = data.get("reward")
    done = data.get("done")
    final = obs.get("final_score")

    # Header
    reward_str = f" | Reward: **{reward:.4f}**" if reward is not None else ""
    lines.append(
        f"### Step {step} | Task: `{task}` | "
        f"Global wait: **{gwait:.3f}**{reward_str}"
    )

    if done:
        score_str = f" | Score: **{final:.4f}**" if final is not None else ""
        lines.append(f"\n> **Episode complete**{score_str}")

    # Emergency vehicle info
    ev = obs.get("emergency_vehicle")
    ev_pos = ev["position"] if ev else None
    if ev:
        path_str = " -> ".join(str(p) for p in ev["path_remaining"])
        lines.append(
            f"\n\U0001F691 **Emergency**: intersection **{ev['position']}** "
            f"-> dest **{ev['destination']}** | "
            f"elapsed: {ev['steps_elapsed']} steps | "
            f"path: {path_str}"
        )

    # Build agent lookup
    agents = obs.get("agents", [])
    agent_map: Dict[int, Dict[str, Any]] = {a["agent_id"]: a for a in agents}

    # Determine grid dimensions from active agents
    active = set(agent_map.keys())
    if not active:
        return "\n".join(lines)

    # Render grid
    lines.append("\n---\n")

    # Find which rows/cols are active
    rows_used = sorted({GRID_POS[a][0] for a in active})
    cols_used = sorted({GRID_POS[a][1] for a in active})

    for row in rows_used:
        row_blocks: List[str] = []
        for col in cols_used:
            aid = row * 3 + col
            if aid in agent_map:
                row_blocks.append(_fmt_intersection(agent_map[aid], ev_pos))
            else:
                row_blocks.append("*(inactive)*")

        # Render each intersection side by side using a Markdown table
        # Split each block into lines
        split_blocks = [b.split("\n") for b in row_blocks]
        max_lines = max(len(b) for b in split_blocks)
        for b in split_blocks:
            b.extend([""] * (max_lines - len(b)))

        header = "| " + " | ".join(f"Intersection {row * 3 + c}" for c in cols_used) + " |"
        sep = "|" + "|".join("---" for _ in cols_used) + "|"
        lines.append(header)
        lines.append(sep)
        for line_idx in range(max_lines):
            row_line = "| " + " | ".join(
                f"`{split_blocks[ci][line_idx]}`" for ci in range(len(cols_used))
            ) + " |"
            lines.append(row_line)
        lines.append("")

    # Phase legend
    lines.append(
        "*Phase: \U0001F7E2 NS = North-South green, "
        "\U0001F7E2 EW = East-West green, "
        "\U0001F7E1 = yellow transition, "
        "\U0001F691 = emergency vehicle*"
    )

    return "\n".join(lines)


def _fmt_detail_table(data: Dict[str, Any]) -> str:
    """Format a detail table for all agents."""
    obs = data.get("observation", {})
    agents = obs.get("agents", [])
    if not agents:
        return ""

    lines = [
        "| Agent | Phase | Elapsed | N | S | E | W | NS total | EW total | Local wait |",
        "|:-----:|:-----:|:-------:|:-:|:-:|:-:|:-:|:--------:|:--------:|:----------:|",
    ]
    for a in agents:
        q = a["queue_lengths"]
        ns = q[0] + q[1]
        ew = q[2] + q[3]
        phase = PHASE_NAMES.get(a["current_phase"], str(a["current_phase"]))
        lines.append(
            f"| {a['agent_id']} | {phase} | {a['phase_elapsed']} | "
            f"{q[0]:.1f} | {q[1]:.1f} | {q[2]:.1f} | {q[3]:.1f} | "
            f"{ns:.1f} | {ew:.1f} | {a['local_wait_time']:.3f} |"
        )
    return "\n".join(lines)


def build_traffic_ui(
    web_manager: Any,
    action_fields: Any,
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: Optional[str],
) -> gr.Blocks:
    """Build a custom Gradio Blocks UI for TrafficSignalEnv."""

    with gr.Blocks(title=f"OpenEnv: {title}") as demo:
        # -- state -----------------------------------------------------------
        active_ids = gr.State([])  # list[int] of active agent IDs

        # -- layout ----------------------------------------------------------
        gr.Markdown(f"# {title}")

        with gr.Row():
            # Left column: controls
            with gr.Column(scale=1):
                gr.Markdown("## Reset")
                task_dd = gr.Dropdown(
                    choices=TASK_IDS,
                    value=TASK_IDS[0],
                    label="Task",
                )
                seed_num = gr.Number(value=42, label="Seed", precision=0)
                reset_btn = gr.Button("Reset", variant="secondary")

                gr.Markdown("## Agent actions")
                gr.Markdown(
                    "*Set each agent to **keep** or **switch**, then click Step.*"
                )
                agent_dds: List[gr.Dropdown] = []
                for i in range(MAX_AGENTS):
                    dd = gr.Dropdown(
                        choices=["keep", "switch"],
                        value="keep",
                        label=f"Agent {i}",
                        visible=False,
                    )
                    agent_dds.append(dd)

                with gr.Row():
                    all_keep_btn = gr.Button("All keep", size="sm")
                    all_switch_btn = gr.Button("All switch", size="sm")
                step_btn = gr.Button("Step", variant="primary")

            # Right column: observation
            with gr.Column(scale=2):
                obs_md = gr.Markdown("Click **Reset** to start a new episode.")
                with gr.Accordion("Agent details", open=False):
                    detail_md = gr.Markdown("")
                with gr.Accordion("Raw JSON", open=False):
                    raw_json = gr.Code(
                        label="Raw JSON",
                        language="json",
                        interactive=False,
                    )
                status_tb = gr.Textbox(label="Status", interactive=False)

        # -- callbacks -------------------------------------------------------
        async def on_reset(task_id: str, seed: float):
            try:
                data = await web_manager.reset_environment(
                    {"task_id": task_id, "seed": int(seed)}
                )
                obs = data.get("observation", {})
                ids = [a["agent_id"] for a in obs.get("agents", [])]
                updates: list[Any] = [
                    _fmt_grid(data),
                    _fmt_detail_table(data),
                    json.dumps(data, indent=2),
                    f"Reset: {task_id}, seed={int(seed)}",
                    ids,
                ]
                # Show/hide agent dropdowns and reset them to "keep"
                for i in range(MAX_AGENTS):
                    updates.append(
                        gr.update(visible=i < len(ids), value="keep")
                    )
                return updates
            except Exception as e:
                return [
                    "",
                    "",
                    "",
                    f"Error: {e}",
                    [],
                ] + [gr.update() for _ in range(MAX_AGENTS)]

        reset_btn.click(
            fn=on_reset,
            inputs=[task_dd, seed_num],
            outputs=[obs_md, detail_md, raw_json, status_tb, active_ids] + agent_dds,
        )

        async def on_step(ids: list[int], *agent_vals: str):
            if not ids:
                return ("", "", "", "Reset the environment first.")
            agent_actions = []
            for idx, aid in enumerate(ids):
                action = agent_vals[idx] if idx < len(agent_vals) else "keep"
                agent_actions.append(
                    {"agent_id": aid, "phase_action": action}
                )
            try:
                data = await web_manager.step_environment(
                    {"agent_actions": agent_actions}
                )
                return (
                    _fmt_grid(data),
                    _fmt_detail_table(data),
                    json.dumps(data, indent=2),
                    "Step complete.",
                )
            except Exception as e:
                return ("", "", "", f"Error: {e}")

        step_btn.click(
            fn=on_step,
            inputs=[active_ids] + agent_dds,
            outputs=[obs_md, detail_md, raw_json, status_tb],
        )

        def set_all(value: str, ids: list[int]):
            return [gr.update(value=value) for _ in range(MAX_AGENTS)]

        all_keep_btn.click(
            fn=lambda ids: set_all("keep", ids),
            inputs=[active_ids],
            outputs=agent_dds,
        )
        all_switch_btn.click(
            fn=lambda ids: set_all("switch", ids),
            inputs=[active_ids],
            outputs=agent_dds,
        )

    return demo
