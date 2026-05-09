"""
Global trade lock — ensures only ONE script can trade at a time.

Usage:
    from utils.trade_lock import acquire_trade_lock

    acquire_trade_lock("run_monitor")  # Call at startup, exits if another trader is active.
"""
from __future__ import annotations

import os
import sys
import atexit
import json
from datetime import datetime
from pathlib import Path

LOCK_FILE = Path(__file__).resolve().parent.parent / ".trade.lock"


def acquire_trade_lock(script_name: str) -> None:
    """Acquire the global trade lock. Kills stale holders, exits if contested."""
    my_pid = os.getpid()

    # Check existing lock
    if LOCK_FILE.exists():
        try:
            info = json.loads(LOCK_FILE.read_text())
            old_pid = info.get("pid", 0)
            old_script = info.get("script", "unknown")

            if old_pid and old_pid != my_pid:
                import psutil
                if psutil.pid_exists(old_pid):
                    proc = psutil.Process(old_pid)
                    if proc.is_running() and "python" in proc.name().lower():
                        # Another trader is genuinely running
                        print(f"[TRADE-LOCK] BLOCKED: '{old_script}' (PID {old_pid}) "
                              f"already holds the trade lock.")
                        print(f"[TRADE-LOCK] Kill it first, or delete {LOCK_FILE}")
                        sys.exit(1)

                # PID doesn't exist or isn't Python — stale lock
                print(f"[TRADE-LOCK] Stale lock from '{old_script}' (PID {old_pid}), cleaning up")
        except (json.JSONDecodeError, ValueError, ImportError):
            pass

    # Write our lock
    lock_info = {
        "pid": my_pid,
        "script": script_name,
        "started": datetime.now().isoformat(),
    }
    LOCK_FILE.write_text(json.dumps(lock_info))
    atexit.register(_release_trade_lock, my_pid)
    print(f"[TRADE-LOCK] Acquired by '{script_name}' (PID {my_pid})")


def _release_trade_lock(expected_pid: int) -> None:
    try:
        if LOCK_FILE.exists():
            info = json.loads(LOCK_FILE.read_text())
            if info.get("pid") == expected_pid:
                LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass
