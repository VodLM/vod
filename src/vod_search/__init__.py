"""Tools for indexing and searching knowledge bases."""

__version__ = "0.1.0"

from vod_types.retrieval import RetrievalBatch, RetrievalData, RetrievalSample

from .base import (
    SearchClient,
    SearchMaster,
)
from .es_search import (
    ElasticsearchClient,
    ElasticSearchMaster,
)
from .factory import (
    build_elasticsearch_index,
    build_faiss_index,
    build_hybrid_search_engine,
    build_search_index,
)
from .faiss_search import (
    FaissClient,
    FaissMaster,
)
from .hybrid_search import (
    HybridSearchClient,
    HyrbidSearchMaster,
)
from .qdrant_search import (
    QdrantSearchClient,
    QdrantSearchMaster,
)
from .sharded_search import (
    ShardedSearchClient,
    ShardedSearchMaster,
    ShardName,
)
