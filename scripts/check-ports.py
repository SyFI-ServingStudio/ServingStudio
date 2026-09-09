"""Check the three service ports without changing running services."""
import os
import socket
from contextlib import ExitStack

try:
    ports = [int(os.environ[key]) for key in ("UI_PORT", "AGENT_PORT", "ANALYZER_PORT")]
    if not all(1024 <= port <= 65535 for port in ports) or len(set(ports)) != 3:
        raise ValueError("Choose VIBESIM_PORT_BASE between 1024 and 65533")
    with ExitStack() as stack:
        for port in ports:
            probe = stack.enter_context(socket.socket())
            # Match server restart behavior while still rejecting active listeners.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("0.0.0.0", port))
except (ValueError, OSError) as error:
    raise SystemExit(f"Port check failed: {error}. Set another VIBESIM_PORT_BASE in .env.")
print("Available UI/Agent/Analyzer ports:", *ports)
