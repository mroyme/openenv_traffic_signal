"""Intersection phase controller.

Phase cycle:
    0 -> NS green, EW red
    1 -> NS yellow, EW red
    2 -> EW green, NS red
    3 -> EW yellow, NS red
"""

PHASES = [
    {"NS": "green", "EW": "red"},  # phase 0
    {"NS": "yellow", "EW": "red"},  # phase 1
    {"NS": "red", "EW": "green"},  # phase 2
    {"NS": "red", "EW": "yellow"},  # phase 3
]

MIN_GREEN_STEPS = 5
YELLOW_DURATION = 1


class IntersectionController:
    """Controls the traffic signal phase for a single intersection.

    Attributes:
        intersection_id: Unique ID of the intersection (0-8).
        phase: Current signal phase (0=NS-green, 1=NS-yellow, 2=EW-green, 3=EW-yellow).
        phase_elapsed: Steps elapsed in the current phase.
    """

    def __init__(self, intersection_id: int):
        self.intersection_id = intersection_id
        self.phase = 0
        self.phase_elapsed = 0

    def reset(self) -> None:
        """Reset phase and elapsed counter to initial state."""
        self.phase = 0
        self.phase_elapsed = 0

    def step(self, action: str) -> dict:
        """Process one time step.

        Args:
            action: ``"keep"`` to maintain the current phase or ``"switch"`` to
                request a phase change. Switch requests are silently ignored if
                the intersection is in a yellow phase or has not met
                ``MIN_GREEN_STEPS``.

        Returns:
            A dict with keys:

            - ``phase``: The phase after this step.
            - ``phase_elapsed``: Steps spent in the current phase after this step.
            - ``phase_switched``: Whether a phase transition occurred.
        """
        phase_switched = False
        self.phase_elapsed += 1

        if action == "switch":
            # Can only switch during green phases (0 or 2)
            if self.phase in (0, 2):
                if self.phase_elapsed >= MIN_GREEN_STEPS:
                    # Transition to yellow
                    self.phase = (self.phase + 1) % 4
                    self.phase_elapsed = 0
                    phase_switched = True
                # else: silently ignored (min green not met)
            # else: silently ignored (in yellow phase)
        else:
            # "keep" — auto-advance yellow after YELLOW_DURATION
            if self.phase in (1, 3) and self.phase_elapsed >= YELLOW_DURATION:
                self.phase = (self.phase + 1) % 4
                self.phase_elapsed = 0

        return {
            "phase": self.phase,
            "phase_elapsed": self.phase_elapsed,
            "phase_switched": phase_switched,
        }

    def get_green_lanes(self) -> list[str]:
        """Return which lane directions currently have green.

        Returns:
            List of direction strings (``"N"``, ``"S"``, ``"E"``, ``"W"``) that
            have a green signal. Empty during yellow phases.
        """
        if self.phase == 0:
            return ["N", "S"]
        elif self.phase == 2:
            return ["E", "W"]
        return []
