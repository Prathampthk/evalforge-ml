"""Pytest configuration for tests directory."""

import os
import sys
from pathlib import Path

# Add parent directory to path so tests can import modules
parent_dir = Path(__file__).parent.parent

# TrustedHostMiddleware permits TestClient's host only in test/development mode.
os.environ.setdefault("ENVIRONMENT", "test")
# Unit tests use explicit in-memory telemetry exporters when validating traces.
# Avoid live console/background exporters attached to pytest capture streams.
os.environ.setdefault("TELEMETRY_EXPORTER", "none")
os.environ.setdefault("STRUCTURED_LOGS_ENABLED", "false")
sys.path.insert(0, str(parent_dir))
