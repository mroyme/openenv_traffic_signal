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

from models import TrafficAction, TrafficObservation
from server.traffic_signal_environment import TrafficEnvironment


app = create_app(
    TrafficEnvironment,
    TrafficAction,
    TrafficObservation,
    env_name="openenv-traffic-signal",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Start the uvicorn server.

    Entry point for direct execution via ``uv run`` or ``python -m``::

        uv run --project . server
        python -m server.app --port 8000

    For production with multiple workers use uvicorn directly::

        uvicorn server.app:app --workers 4

    Args:
        host: Network interface to bind to.
        port: Port number to listen on.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
