"""Tests for grader functions."""

from graders import corridor_grader, emergency_grader, grid_grader


class TestCorridorGrader:
    def test_half_baseline_time(self):
        assert corridor_grader.grade(10.0, 20.0) == 0.5

    def test_zero_time_perfect_score(self):
        assert corridor_grader.grade(0.0, 20.0) == 0.999

    def test_same_as_baseline(self):
        assert corridor_grader.grade(20.0, 20.0) == 0.001

    def test_worse_than_baseline_clipped(self):
        assert corridor_grader.grade(30.0, 20.0) == 0.001

    def test_zero_baseline(self):
        assert corridor_grader.grade(10.0, 0.0) == 0.001

    def test_score_in_valid_range(self):
        assert 0.0 <= corridor_grader.grade(5.0, 15.0) <= 1.0


class TestGridGrader:
    def test_half_baseline_wait(self):
        assert grid_grader.grade(5.0, 10.0) == 0.5

    def test_zero_wait_perfect_score(self):
        assert grid_grader.grade(0.0, 10.0) == 0.999

    def test_same_as_baseline(self):
        assert grid_grader.grade(10.0, 10.0) == 0.001

    def test_worse_than_baseline_clipped(self):
        assert grid_grader.grade(15.0, 10.0) == 0.001

    def test_zero_baseline(self):
        assert grid_grader.grade(5.0, 0.0) == 0.001

    def test_score_in_valid_range(self):
        assert 0.0 <= grid_grader.grade(3.0, 8.0) <= 1.0


class TestEmergencyGrader:
    def test_perfect_emergency_and_civilian(self):
        assert emergency_grader.grade(0.0, 200.0, 0.0, 10.0) == 0.999

    def test_worst_emergency_and_civilian(self):
        assert emergency_grader.grade(200.0, 200.0, 10.0, 10.0) == 0.001

    def test_half_emergency_half_civilian(self):
        assert emergency_grader.grade(100.0, 200.0, 5.0, 10.0) == 0.5

    def test_good_emergency_bad_civilian_in_range(self):
        s = emergency_grader.grade(50.0, 200.0, 12.0, 10.0)
        assert 0.0 <= s <= 1.0

    def test_zero_max_and_baseline(self):
        assert emergency_grader.grade(10.0, 0.0, 5.0, 0.0) == 0.001
