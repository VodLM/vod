import subprocess
from typing import List


def cmd_check_output(args: List[str]) -> str:
    """Returns the output of a command as a string."""
    try:
        output = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        return output.decode("ascii").strip()
    except Exception:
        return "unknown"


def git_revision_hash() -> str:
    """Return the git revision hash."""
    return cmd_check_output(["git", "rev-parse", "HEAD"])


def git_revision_short_hash() -> str:
    """Return the git revision hash (shortened)."""
    return cmd_check_output(["git", "rev-parse", "--short", "HEAD"])


def git_branch_name() -> str:
    """Return the git branch name."""
    return cmd_check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
