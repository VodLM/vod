"""Index and serve vector bases using faiss."""


from .build import build_faiss_index
from .client import FaissClient, FaissMaster
from .models import FaissInitConfig, SearchFaissQuery
