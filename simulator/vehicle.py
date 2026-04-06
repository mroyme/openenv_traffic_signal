from dataclasses import dataclass

import numpy as np


@dataclass
class Vehicle:
    """A single vehicle traversing the traffic grid.

    Attributes:
        vehicle_id: Unique string identifier for this vehicle.
        origin: Intersection ID where the vehicle was spawned.
        destination: Target intersection ID.
        current_intersection: Intersection the vehicle is currently queued at.
        current_lane: Lane direction the vehicle is waiting in (N/S/E/W).
        wait_time: Steps the vehicle has spent waiting in queues.
        is_emergency: Whether this vehicle is an emergency vehicle.
    """

    vehicle_id: str
    origin: int
    destination: int
    current_intersection: int
    current_lane: str
    wait_time: int = 0
    is_emergency: bool = False


class VehicleSpawner:
    """Spawns vehicles deterministically using a seeded RNG.

    Attributes:
        _rng: Seeded numpy random generator.
        _spawn_rate: Mean number of vehicles spawned per intersection per step (Poisson).
        _active: List of active intersection IDs vehicles can spawn at.
        _counter: Running count of spawned vehicles used for unique ID generation.
        _lanes: Available lane directions for vehicle placement.
    """

    def __init__(self, seed: int, spawn_rate: float, active_intersections: list[int]):
        self._rng = np.random.default_rng(seed)
        self._spawn_rate = spawn_rate
        self._active = active_intersections
        self._counter = 0
        self._lanes = ["N", "S", "E", "W"]

    def spawn(self, step: int) -> list[Vehicle]:
        """Return vehicles to inject this step, deterministic given seed.

        Args:
            step: Current simulation step, used for vehicle ID generation.

        Returns:
            List of newly spawned vehicles to add to the grid.
        """
        vehicles: list[Vehicle] = []
        for iid in self._active:
            n = int(self._rng.poisson(self._spawn_rate))
            for _ in range(n):
                lane = self._lanes[int(self._rng.integers(0, 4))]
                # Pick a destination that is different from origin
                dest = iid
                while dest == iid:
                    dest = int(self._rng.choice(self._active))
                    if len(self._active) == 1:
                        # Edge case: only one intersection, vehicle exits immediately
                        dest = -1
                        break
                self._counter += 1
                vehicles.append(
                    Vehicle(
                        vehicle_id=f"v_{step}_{self._counter}",
                        origin=iid,
                        destination=dest,
                        current_intersection=iid,
                        current_lane=lane,
                    )
                )
        return vehicles
