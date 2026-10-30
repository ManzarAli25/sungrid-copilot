"""Build (or refresh) the vector store from data/docs/.

Usage:
    python -m scripts.ingest            # idempotent upsert
    python -m scripts.ingest --recreate # drop and rebuild the collection
"""

import argparse
from collections import Counter

from sungrid_copilot.config import get_settings
from sungrid_copilot.ingestion.pipeline import run_ingestion
from sungrid_copilot.vectorstore import get_client, is_cloud


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    args = parser.parse_args()

    s = get_settings()
    print(f"Qdrant: {'cloud' if is_cloud() else 'local (' + s.qdrant_local_path + ')'}")
    print(f"Embedding model: {s.embedding_model}")

    chunks = run_ingestion(recreate=args.recreate)

    count = get_client().count(s.qdrant_collection, exact=True).count
    per_category = Counter(cat for c in chunks for cat in c.categories)
    print(f"Ingested {len(chunks)} chunks from {len({c.doc_id for c in chunks})} docs")
    print(f"Points in '{s.qdrant_collection}': {count}")
    for cat, n in sorted(per_category.items()):
        print(f"  {cat:<24}{n:>3} chunks")


if __name__ == "__main__":
    main()
