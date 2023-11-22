from __future__ import annotations

import subprocess
import time
from typing import Optional

import gpustat
import loguru

from src.vod_tools import arguantic


class Arguments(arguantic.Arguantic):
    """Arguments for the script. durations are in minutes (`base`=60s)."""

    base: float = 60.0
    freq: float = 1.0
    inactivity: float = 60.0
    cmd: str = "sudo shutdown +5"


def get_user_processes(exclude_users: Optional[list[str]] = None) -> list[dict]:
    """Return the list of user CUDA processes."""
    stats = gpustat.new_query().jsonify()
    processes = [p for gpu in stats["gpus"] for p in gpu["processes"]]
    if exclude_users is not None:
        processes = [p for p in processes if p["username"] not in exclude_users]
    return processes


def run() -> None:  # noqa: D103
    args = Arguments.parse()
    last_active = time.time()
    loguru.logger.info(f"Starting `watch_gpus.py` with args: {args}")
    while True:
        time.sleep(args.base * args.freq)
        user_processes = get_user_processes(exclude_users=["root", "gdm"])
        loguru.logger.info(f"Active processes: {len(user_processes)}")
        if len(user_processes):
            last_active = time.time()

        if time.time() - last_active > args.base * args.inactivity:
            loguru.logger.info("Shutting down...")
            loguru.logger.info(f"Running command: {args.cmd}")
            subprocess.run(args.cmd.split())  # noqa: PLW1510, S603
            break


if __name__ == "__main__":
    run()
