import subprocess
from typing import List


def git_check_output(args: List[str]) -> str:
    try:
        output = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        return output.decode("ascii").strip()
    except Exception:
        return "unknown"


def git_revision_hash() -> str:
    return git_check_output(["git", "rev-parse", "HEAD"])


def git_revision_short_hash() -> str:
    return git_check_output(["git", "rev-parse", "--short", "HEAD"])


def git_branch_name() -> str:
    return git_check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
