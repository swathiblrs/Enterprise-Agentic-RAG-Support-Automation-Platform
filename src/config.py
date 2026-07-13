import os


def get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
LOW_CONFIDENCE_THRESHOLD = get_float_env("LOW_CONFIDENCE_THRESHOLD", 0.6)
CLARIFY_CONFIDENCE_THRESHOLD = get_float_env("CLARIFY_CONFIDENCE_THRESHOLD", 0.72)
USE_LLM_GENERATION = os.getenv("USE_LLM_GENERATION", "false").lower() == "true"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SUPPORT_API_KEY = os.getenv("SUPPORT_API_KEY", "")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
AUTH_DEMO_ADMIN_USERNAME = os.getenv("AUTH_DEMO_ADMIN_USERNAME", "admin")
AUTH_DEMO_ADMIN_PASSWORD = os.getenv("AUTH_DEMO_ADMIN_PASSWORD", "admin123")
AUTH_DEMO_AGENT_USERNAME = os.getenv("AUTH_DEMO_AGENT_USERNAME", "agent")
AUTH_DEMO_AGENT_PASSWORD = os.getenv("AUTH_DEMO_AGENT_PASSWORD", "agent123")
DEFAULT_DOMAIN = os.getenv("DEFAULT_DOMAIN", "it_support")
