import os

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
