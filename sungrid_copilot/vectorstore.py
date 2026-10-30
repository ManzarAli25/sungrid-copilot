"""Qdrant client and collection setup — Qdrant Cloud if QDRANT_URL is set, else local mode."""

from functools import lru_cache

from qdrant_client import QdrantClient, models

from sungrid_copilot.config import get_settings

# Payload fields we filter on. Indexed so filters stay fast as the collection grows.
INDEXED_FIELDS = ("categories", "doc_id", "chunk_id")


@lru_cache
def get_client() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None, timeout=30)
    return QdrantClient(path=s.qdrant_local_path)


def is_cloud() -> bool:
    return bool(get_settings().qdrant_url)


def ensure_collection(recreate: bool = False) -> None:
    s = get_settings()
    client = get_client()
    name = s.qdrant_collection

    if recreate and client.collection_exists(name):
        client.delete_collection(name)

    if client.collection_exists(name):
        size = client.get_collection(name).config.params.vectors.size
        if size != s.embedding_dim:
            raise RuntimeError(
                f"Collection '{name}' has dim {size} but EMBEDDING_DIM is {s.embedding_dim}. "
                "Re-run ingestion with --recreate."
            )
        return

    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=s.embedding_dim, distance=models.Distance.COSINE),
    )
    if is_cloud():  # local mode ignores payload indexes
        for field_name in INDEXED_FIELDS:
            client.create_payload_index(name, field_name, models.PayloadSchemaType.KEYWORD)
