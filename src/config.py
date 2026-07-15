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
PERSISTENCE_BACKEND = os.getenv("PERSISTENCE_BACKEND", "sqlite")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "logs/support_platform.db")
ITSM_INTEGRATION_MODE = os.getenv("ITSM_INTEGRATION_MODE", "mock")
CHAT_INTEGRATION_MODE = os.getenv("CHAT_INTEGRATION_MODE", "mock")
ITSM_WEBHOOK_URL = os.getenv("ITSM_WEBHOOK_URL", "")
CHAT_WEBHOOK_URL = os.getenv("CHAT_WEBHOOK_URL", "")
