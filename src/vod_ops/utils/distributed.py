import os


def is_gloabl_zero() -> bool:
    """Check if the current process is the global zero."""
    return os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("NODE_RANK", "0") == "0"
