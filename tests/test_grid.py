"""Tests for TrafficGrid simulation."""

from simulator.grid import TrafficGrid


class TestTrafficGrid:
    def test_reset_and_step(self):
        grid = TrafficGrid(active_intersections=list(range(9)), seed=42, spawn_rate=0.4)
        grid.reset()
        for _ in range(10):
            actions = {iid: "keep" for iid in range(9)}
            metrics = grid.step(actions)
            assert "global_wait_time" in metrics
            assert "total_vehicles_in_grid" in metrics
            assert "vehicles_cleared" in metrics
            assert metrics["global_wait_time"] >= 0.0

    def test_observation_structure(self):
        grid = TrafficGrid(active_intersections=list(range(9)), seed=42, spawn_rate=0.4)
        grid.reset()
        grid.step({iid: "keep" for iid in range(9)})
        obs = grid.get_observation()
        assert len(obs) == 9
        for o in obs:
            assert len(o.queue_lengths) == 4
            assert len(o.neighbor_queues) == 4

    def test_baseline_returns_float(self):
        grid = TrafficGrid(active_intersections=list(range(9)), seed=42, spawn_rate=0.4)
        grid.reset()
        baseline = grid.run_baseline(50)
        assert isinstance(baseline, float)
        assert baseline >= 0.0

    def test_corridor_subset(self):
        grid = TrafficGrid(active_intersections=[0, 1, 2], seed=42, spawn_rate=0.4)
        grid.reset()
        metrics = grid.step({0: "keep", 1: "keep", 2: "keep"})
        assert "global_wait_time" in metrics
