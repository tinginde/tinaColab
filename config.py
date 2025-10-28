import os

# Environment variable names
HUGGINGFACE_API_KEY_ENV_VAR = "HUGGINGFACE_API_KEY"

# Base directory of the repository
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for data and vector stores
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "med_vectordata2")
RFP_VECTOR_STORE_DIR = os.path.join(BASE_DIR, "rfp_vectordb")

# Common data files
PATIENT_FILE = os.path.join(BASE_DIR, "patient_c.txt")

# Default model for local LLM examples
DEFAULT_LOCAL_MODEL = "yabi/breeze-7b-instruct-v1_0_q6_k:latest"


def get_env_variable(name: str) -> str:
    """Return the value of an environment variable or raise a helpful error."""

    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Environment variable '{name}' is not set. "
            "Please export it in your shell or add it to your .env file."
        )
    return value


def get_huggingface_api_key() -> str:
    """Return the configured HuggingFace API key."""

    key = os.getenv(HUGGINGFACE_API_KEY_ENV_VAR)
    if key:
        return key

    # Allow advanced users to set the key directly in config.py if desired.
    key = globals().get("HUGGINGFACE_API_KEY")
    if key:
        return key

    raise RuntimeError(
        "HuggingFace API key is not configured. Set the environment variable "
        f"'{HUGGINGFACE_API_KEY_ENV_VAR}' or assign 'HUGGINGFACE_API_KEY' in config.py."
    )
