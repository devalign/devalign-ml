import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API Keys & DB Credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# UMAP Hyperparameters
UMAP_N_NEIGHBORS = int(os.environ.get("UMAP_N_NEIGHBORS", 15))
UMAP_N_COMPONENTS = int(os.environ.get("UMAP_N_COMPONENTS", 15))
UMAP_MIN_DIST = float(os.environ.get("UMAP_MIN_DIST", 0.0))
UMAP_METRIC = os.environ.get("UMAP_METRIC", "cosine")
UMAP_RANDOM_STATE = int(os.environ.get("UMAP_RANDOM_STATE", 42))

# HDBSCAN Hyperparameters
HDBSCAN_MIN_CLUSTER_SIZE = int(os.environ.get("HDBSCAN_MIN_CLUSTER_SIZE", 32))
HDBSCAN_MIN_SAMPLES = int(os.environ.get("HDBSCAN_MIN_SAMPLES", 5))
HDBSCAN_METRIC = os.environ.get("HDBSCAN_METRIC", "euclidean")

# Local Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
