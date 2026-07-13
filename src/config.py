import os


def get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
LOW_CONFIDENCE_THRESHOLD = get_float_env("LOW_CONFIDENCE_THRESHOLD", 0.6)
CLARIFY_CONFIDENCE_THRESHOLD = get_float_env("CLARIFY_CONFIDENCE_THRESHOLD", 0.72)
