"""3x3 traffic grid simulation.

Intersections indexed 0-8:
    0 -- 1 -- 2
    |    |    |
    3 -- 4 -- 5
    |    |    |
    6 -- 7 -- 8
"""

from __future__ import annotations

import copy
import logging

import numpy as np

from models import IntersectionObservation
from simulator.vehicle import Vehicle, VehicleSpawner

logger = logging.getLogger(__name__)

NEIGHBORS: dict[int, dict[str, int]] = {
    0: {"E": 1, "S": 3},
    1: {"W": 0, "E": 2, "S": 4},
    2: {"W": 1, "S": 5},
    3: {"N": 0, "E": 4, "S": 6},
    4: {"N": 1, "W": 3, "E": 5, "S": 7},
    5: {"N": 2, "W": 4, "S": 8},
    6: {"N": 3, "E": 7},
    7: {"N": 4, "W": 6, "E": 8},
    8: {"N": 5, "W": 7},
}

# Which lanes get green in each phase
# Phase 0: NS green, EW red
# Phase 1: NS yellow, EW red
# Phase 2: EW green, NS red
# Phase 3: EW yellow, NS red
GREEN_LANES = {
    0: ["N", "S"],  # NS green
    1: [],  # NS yellow (transition, no movement)
    2: ["E", "W"],  # EW green
    3: [],  # EW yellow (transition, no movement)
}

# Opposite direction mapping: lane a vehicle arrives FROM -> direction it exits TO
# A vehicle in the "N" queue arrived from the north, heading south
# It exits toward the south neighbor
LANE_TO_EXIT_DIR = {
    "N": "S",
    "S": "N",
    "E": "W",
    "W": "E",
}


class TrafficGrid:
    """Manages the 3x3 intersection grid, vehicle queues, and movement.

    Attributes:
        _active: Sorted list of active intersection IDs.
        _seed: Random seed used for vehicle spawning.
        _spawn_rate: Mean vehicles spawned per intersection per step (Poisson).
        _spawner: Vehicle spawner instance; set on reset.
        _queues: Per-intersection, per-lane vehicle queues.
        _phases: Current signal phase per intersection.
        _phase_elapsed: Steps elapsed in the current phase per intersection.
        _step_count: Total steps advanced since the last reset.
        _total_wait_time: Accumulated wait time across all steps.
        _vehicles_cleared: Total vehicles that have exited the grid.
        _cleared_wait_times: Wait time of each vehicle at exit, used for mean calculation.
    """

    def __init__(
        self, active_intersections: list[int], seed: int, spawn_rate: float = 0.4
    ):
        self._active = sorted(active_intersections)
        self._seed = seed
        self._spawn_rate = spawn_rate
        self._spawner: VehicleSpawner | None = None

        # Per-intersection state
        self._queues: dict[int, dict[str, list[Vehicle]]] = {}
        self._phases: dict[int, int] = {}
        self._phase_elapsed: dict[int, int] = {}
        self._step_count = 0

        # Metrics
        self._total_wait_time = 0.0
        self._vehicles_cleared = 0
        self._cleared_wait_times: list[float] = []

    def reset(self) -> None:
        """Clear all queues, reset step counter."""
        self._spawner = VehicleSpawner(self._seed, self._spawn_rate, self._active)
        self._queues = {
            iid: {"N": [], "S": [], "E": [], "W": []} for iid in self._active
        }
        self._phases = {iid: 0 for iid in self._active}
        self._phase_elapsed = {iid: 0 for iid in self._active}
        self._step_count = 0
        self._total_wait_time = 0.0
        self._vehicles_cleared = 0
        self._cleared_wait_times = []

    def step(self, actions: dict[int, str]) -> dict:
        """Apply actions, advance vehicles, return metrics.

        Args:
            actions: Mapping of intersection ID to ``"keep"`` or ``"switch"``.

        Returns:
            A dict with keys:

            - ``global_wait_time``: Mean wait time across all queued vehicles.
            - ``per_intersection_wait``: Per-intersection mean wait times.
            - ``vehicles_cleared``: Vehicles that exited the grid this step.
            - ``total_vehicles_in_grid``: Current number of queued vehicles.
        """
        # 1. Apply phase transitions
        for iid in self._active:
            action = actions.get(iid, "keep")
            self._apply_phase_action(iid, action)

        # 2. Spawn new vehicles
        assert self._spawner is not None
        new_vehicles = self._spawner.spawn(self._step_count)
        for v in new_vehicles:
            if v.current_intersection in self._queues:
                self._queues[v.current_intersection][v.current_lane].append(v)

        # 3. Move vehicles on green lanes
        vehicles_to_move: list[tuple] = []  # (iid, lane, vehicle)
        for iid in self._active:
            phase = self._phases[iid]
            green_lanes = GREEN_LANES[phase]
            for lane in green_lanes:
                queue = self._queues[iid][lane]
                if queue:
                    # One vehicle per step per lane
                    vehicles_to_move.append((iid, lane, queue[0]))

        # Process movements simultaneously
        cleared_this_step = 0
        for iid, lane, vehicle in vehicles_to_move:
            self._queues[iid][lane].remove(vehicle)
            exit_dir = LANE_TO_EXIT_DIR[lane]
            neighbor = NEIGHBORS.get(iid, {}).get(exit_dir)

            if neighbor is not None and neighbor in self._queues:
                # Move to neighbor's incoming queue
                # The vehicle enters from the opposite direction
                entry_lane = _opposite(exit_dir)
                vehicle.current_intersection = neighbor
                vehicle.current_lane = entry_lane
                self._queues[neighbor][entry_lane].append(vehicle)
            else:
                # Vehicle exits the grid
                self._cleared_wait_times.append(vehicle.wait_time)
                self._vehicles_cleared += 1
                cleared_this_step += 1

        # 4. Increment wait time for all vehicles still in queues
        for iid in self._active:
            for lane_vehicles in self._queues[iid].values():
                for v in lane_vehicles:
                    v.wait_time += 1

        self._step_count += 1

        # Compute metrics
        per_intersection_wait = {}
        total_waiting = 0
        total_vehicles = 0
        for iid in self._active:
            iid_wait = 0.0
            iid_count = 0
            for lane_vehicles in self._queues[iid].values():
                for v in lane_vehicles:
                    iid_wait += v.wait_time
                    iid_count += 1
            per_intersection_wait[iid] = iid_wait / max(iid_count, 1)
            total_waiting += iid_wait
            total_vehicles += iid_count

        global_wait_time = total_waiting / max(total_vehicles, 1)

        return {
            "global_wait_time": round(global_wait_time, 4),
            "per_intersection_wait": per_intersection_wait,
            "vehicles_cleared": cleared_this_step,
            "total_vehicles_in_grid": total_vehicles,
        }

    def _apply_phase_action(self, iid: int, action: str) -> bool:
        """Apply a phase action to a single intersection.

        Args:
            iid: Intersection ID.
            action: ``"keep"`` or ``"switch"``.

        Returns:
            True if a phase transition occurred, False otherwise.
        """
        from simulator.traffic import MIN_GREEN_STEPS, YELLOW_DURATION

        phase = self._phases[iid]
        self._phase_elapsed[iid] += 1
        elapsed = self._phase_elapsed[iid]

        if action == "switch":
            # Can only switch during green phases (0 or 2)
            if phase not in (0, 2):
                return False
            if elapsed < MIN_GREEN_STEPS:
                return False
            # Advance to next phase (green -> yellow)
            self._phases[iid] = (phase + 1) % 4
            self._phase_elapsed[iid] = 0
            return True
        else:
            # "keep" — but auto-advance yellow phases after YELLOW_DURATION
            if phase in (1, 3) and elapsed >= YELLOW_DURATION:
                self._phases[iid] = (phase + 1) % 4
                self._phase_elapsed[iid] = 0
            return False

    def get_observation(self) -> list[IntersectionObservation]:
        """Build per-agent observations for all active intersections.

        Returns:
            List of IntersectionObservation, one per active intersection.
        """
        observations = []
        for iid in self._active:
            # Queue lengths for N, S, E, W
            queue_lengths = [
                float(len(self._queues[iid]["N"])),
                float(len(self._queues[iid]["S"])),
                float(len(self._queues[iid]["E"])),
                float(len(self._queues[iid]["W"])),
            ]

            # Neighbor outgoing queues (what's queued toward this intersection)
            neighbor_queues = []
            for direction in ["N", "S", "E", "W"]:
                neighbor_id = NEIGHBORS.get(iid, {}).get(direction)
                if neighbor_id is not None and neighbor_id in self._queues:
                    # The neighbor's queue facing us
                    opposite = _opposite(direction)
                    neighbor_queues.append(
                        float(len(self._queues[neighbor_id][opposite]))
                    )
                else:
                    neighbor_queues.append(0.0)

            # Local mean wait time
            total_wait = 0.0
            count = 0
            for lane_vehicles in self._queues[iid].values():
                for v in lane_vehicles:
                    total_wait += v.wait_time
                    count += 1
            local_wait = round(total_wait / max(count, 1), 4)

            observations.append(
                IntersectionObservation(
                    agent_id=iid,
                    queue_lengths=queue_lengths,
                    current_phase=self._phases[iid],
                    phase_elapsed=self._phase_elapsed[iid],
                    neighbor_queues=neighbor_queues,
                    local_wait_time=local_wait,
                )
            )
        return observations

    def get_mean_wait_time(self) -> float:
        """Get current mean wait time across all vehicles in grid."""
        total_wait = 0.0
        count = 0
        for iid in self._active:
            for lane_vehicles in self._queues[iid].values():
                for v in lane_vehicles:
                    total_wait += v.wait_time
                    count += 1
        return round(total_wait / max(count, 1), 4)

    def get_cleared_mean_wait(self) -> float:
        """Mean wait time of vehicles that have exited the grid."""
        if not self._cleared_wait_times:
            return 0.0
        return round(float(np.mean(self._cleared_wait_times)), 4)

    def get_total_queue_length(self) -> int:
        """Total vehicles currently waiting in all queues."""
        total = 0
        for iid in self._active:
            for lane_vehicles in self._queues[iid].values():
                total += len(lane_vehicles)
        return total

    def run_baseline(self, steps: int) -> float:
        """Run a shadow simulation with a fixed 30-step timer policy.

        Saves and restores all grid state so the main simulation is unaffected.

        Args:
            steps: Number of steps to simulate for the baseline.

        Returns:
            Mean global wait time over all baseline steps.
        """
        # Save current state
        saved_queues = copy.deepcopy(self._queues)
        saved_phases = copy.deepcopy(self._phases)
        saved_elapsed = copy.deepcopy(self._phase_elapsed)
        saved_step = self._step_count
        saved_cleared = self._cleared_wait_times[:]
        saved_vehicles_cleared = self._vehicles_cleared
        saved_total_wait = self._total_wait_time

        # Create a fresh spawner with same seed for reproducibility
        saved_spawner = self._spawner
        self._spawner = VehicleSpawner(self._seed, self._spawn_rate, self._active)

        # Reset
        self._queues = {
            iid: {"N": [], "S": [], "E": [], "W": []} for iid in self._active
        }
        self._phases = {iid: 0 for iid in self._active}
        self._phase_elapsed = {iid: 0 for iid in self._active}
        self._step_count = 0
        self._cleared_wait_times = []
        self._vehicles_cleared = 0

        # Run with fixed timing: switch every 30 steps
        wait_times = []
        for s in range(steps):
            actions = {}
            for iid in self._active:
                phase = self._phases[iid]
                elapsed = self._phase_elapsed[iid]
                if phase in (0, 2) and elapsed >= 30:
                    actions[iid] = "switch"
                else:
                    actions[iid] = "keep"
            metrics = self.step(actions)
            wait_times.append(metrics["global_wait_time"])

        baseline_wait = round(float(np.mean(wait_times)) if wait_times else 0.0, 4)

        # Restore original state
        self._queues = saved_queues
        self._phases = saved_phases
        self._phase_elapsed = saved_elapsed
        self._step_count = saved_step
        self._cleared_wait_times = saved_cleared
        self._vehicles_cleared = saved_vehicles_cleared
        self._total_wait_time = saved_total_wait
        self._spawner = saved_spawner

        return baseline_wait

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def phases(self) -> dict[int, int]:
        return self._phases

    @property
    def queues(self) -> dict[int, dict[str, list[Vehicle]]]:
        return self._queues


def _opposite(direction: str) -> str:
    """Return the opposite compass direction.

    Args:
        direction: One of ``"N"``, ``"S"``, ``"E"``, ``"W"``.

    Returns:
        The direction directly opposite the given one.
    """
    return {"N": "S", "S": "N", "E": "W", "W": "E"}[direction]
