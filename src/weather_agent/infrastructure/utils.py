from __future__ import annotations

import os
import sys


def acquire_lock(name: str) -> str:
    pid_dir = os.environ.get("WEATHER_AGENT_PID_DIR", "/tmp")
    pid_file = os.path.join(pid_dir, f"weather-agent-{name}.pid")
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            try:
                old_pid = int(f.read().strip())
            except ValueError:
                old_pid = -1
        if old_pid > 0 and os.path.exists(f"/proc/{old_pid}"):
            print(f"Another {name} instance (PID {old_pid}) is already running.", file=sys.stderr)
            sys.exit(1)
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    return pid_file
