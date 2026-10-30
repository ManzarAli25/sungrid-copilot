"""Ingestion pipeline: load docs → chunk → tag → embed → upsert into Qdrant."""

import uuid
from dataclasses import asdict

from qdrant_client import models

from sungrid_copilot.config import DOCS_DIR, get_settings, load_category_config
from sungrid_copilot.embeddings import embed_texts
from sungrid_copilot.ingestion.chunker import Chunk, chunk_markdown
from sungrid_copilot.vectorstore import ensure_collection, get_client

# Fixed namespace so the same chunk_id always maps to the same Qdrant point ID.
POINT_NAMESPACE = uuid.UUID("5b0a3a9e-7d55-4b1e-9a63-2f4f3b1c8d10")


def load_chunks() -> list[Chunk]:
    tags = load_category_config()["documents"]
    chunks: list[Chunk] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        doc_id = path.stem
        if doc_id not in tags:
            raise KeyError(f"{doc_id} has no entry in config/categories.yaml")
        markdown = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(doc_id, markdown, tags[doc_id]["categories"]))
    return chunks


def run_ingestion(recreate: bool = False) -> list[Chunk]:
    chunks = load_chunks()
    ensure_collection(recreate=recreate)

    vectors = embed_texts([c.embed_text for c in chunks])
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(POINT_NAMESPACE, c.chunk_id)),
            vector=v,
            payload={**asdict(c), "header": c.header},
        )
        for c, v in zip(chunks, vectors)
    ]

    client = get_client()
    collection = get_settings().qdrant_collection
    client.upsert(collection, points=points, wait=True)

    # Remove points for sections that no longer exist (renamed/deleted headings).
    client.delete(
        collection,
        points_selector=models.Filter(
            must_not=[
                models.FieldCondition(
                    key="chunk_id", match=models.MatchAny(any=[c.chunk_id for c in chunks])
                )
            ]
        ),
        wait=True,
    )
    return chunks
