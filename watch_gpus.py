import subprocess
import time
from typing import Optional

import gpustat
import loguru

from raffle_ds_research.tools import arguantic


class Arguments(arguantic.Arguantic):
    freq: float = 60.0
    inactivity_period: float = 3600.0
    shutdown_cmd: str = "shutdown +5"


def get_user_processes(exclude_users: Optional[list[str]] = None) -> list[dict]:
    stats = gpustat.new_query().jsonify()
    processes = [p for gpu in stats["gpus"] for p in gpu["processes"]]
    if exclude_users is not None:
        processes = [p for p in processes if p["username"] not in exclude_users]
    return processes


if __name__ == "__main__":
    args = Arguments.parse()
    last_active = time.time()
    while True:
        time.sleep(args.freq)
        user_processes = get_user_processes(exclude_users=["root", "gdm"])
        loguru.logger.info(f"Active processes: {len(user_processes)}")
        if len(user_processes):
            last_active = time.time()

        if time.time() - last_active > args.inactivity_period:
            loguru.logger.info(f"Shutting down...")
            loguru.logger.info(f"Running command: {args.shutdown_cmd}")
            subprocess.run(args.shutdown_cmd.split())
            break
