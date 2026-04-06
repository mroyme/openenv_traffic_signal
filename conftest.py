import sys
from pathlib import Path

# Ensure the repo root is on sys.path so modules are importable
# when running tests without the package being installed.
sys.path.insert(0, str(Path(__file__).parent))
