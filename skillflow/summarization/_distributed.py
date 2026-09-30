"""Process ranks used for device placement and generation seeds."""
import os

def rank() -> int:
    return int(os.environ.get("SKILLFLOW_RANK", os.environ.get("RANK", "0")))

def local_rank() -> int:
    return int(os.environ.get("SKILLFLOW_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))
