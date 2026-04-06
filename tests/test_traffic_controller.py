"""Tests for IntersectionController phase transitions."""

from simulator.traffic import (
    IntersectionController,
    MIN_GREEN_STEPS,
    YELLOW_DURATION,
)


class TestIntersectionController:
    def test_switch_before_min_green_ignored(self):
        ctrl = IntersectionController(0)
        ctrl.reset()
        for _ in range(MIN_GREEN_STEPS - 1):
            result = ctrl.step("switch")
            assert result["phase"] == 0
            assert not result["phase_switched"]

    def test_switch_at_min_green_transitions_to_yellow(self):
        ctrl = IntersectionController(0)
        ctrl.reset()
        for _ in range(MIN_GREEN_STEPS - 1):
            ctrl.step("switch")
        result = ctrl.step("switch")
        assert result["phase"] == 1
        assert result["phase_switched"]

    def test_switch_during_yellow_does_not_switch(self):
        ctrl = IntersectionController(0)
        ctrl.reset()
        for _ in range(MIN_GREEN_STEPS):
            ctrl.step("keep")
        ctrl.step("switch")  # enter yellow
        result = ctrl.step("switch")  # try to switch during yellow
        assert not result["phase_switched"]

    def test_yellow_auto_advances_on_keep(self):
        ctrl = IntersectionController(1)
        ctrl.reset()
        for _ in range(MIN_GREEN_STEPS):
            ctrl.step("keep")
        ctrl.step("switch")  # now at yellow (phase 1)
        assert ctrl.phase == 1
        for _ in range(YELLOW_DURATION):
            result = ctrl.step("keep")
        assert result["phase"] == 2

    def test_get_green_lanes_phase0(self):
        ctrl = IntersectionController(0)
        ctrl.reset()
        assert ctrl.get_green_lanes() == ["N", "S"]

    def test_get_green_lanes_yellow_phase(self):
        ctrl = IntersectionController(0)
        ctrl.reset()
        for _ in range(MIN_GREEN_STEPS):
            ctrl.step("keep")
        ctrl.step("switch")  # go to phase 1 (yellow)
        assert ctrl.get_green_lanes() == []
