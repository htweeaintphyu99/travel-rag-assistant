"""
Indexing script for the travel-agent project.

Reads chunks.json (produced by ingest.py), embeds each chunk, and indexes
everything into Elasticsearch with a hybrid mapping that supports both:
  - BM25 keyword search (the "text" field)
  - Dense vector kNN search (the "embedding" field)

This lets you later compare BM25-only vs. vector-only vs. hybrid retrieval
for your evaluation step.

Setup:
    uv add install elasticsearch sentence-transformers tqdm

    Run Elasticsearch locally, e.g. via Docker:
        docker run -p 9200:9200 -e "discovery.type=single-node" \\
            -e "xpack.security.enabled=false" \\
            docker.elastic.co/elasticsearch/elasticsearch:8.15.0

Usage:
    python index.py --chunks data/chunks.json
    python index.py --chunks data/chunks.json --index vietnam-travel --recreate
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DEFAULT_ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")
DEFAULT_INDEX = "travel-chunks"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2's output size


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks:
        raise SystemExit(f"No chunks found in {path} — did ingest.py run successfully?")
    return chunks


def build_index_mapping() -> dict:
    """Mapping supporting both BM25 (text field) and kNN (embedding field)."""
    return {
        "mappings": {
            "properties": {
                "text": {"type": "text"},  # BM25-searchable
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
                "city": {"type": "keyword"},
                "source": {"type": "keyword"},
                "page_title": {"type": "keyword"},
                "section": {"type": "text"},
                "chunk_id": {"type": "keyword"},
            }
        }
    }


def create_index(es: Elasticsearch, index_name: str, recreate: bool) -> None:
    exists = es.indices.exists(index=index_name)
    if exists and recreate:
        print(f"Deleting existing index: {index_name}")
        es.indices.delete(index=index_name)
        exists = False
    if not exists:
        print(f"Creating index: {index_name}")
        es.indices.create(index=index_name, body=build_index_mapping())
    else:
        print(
            f"Index already exists, reusing: {index_name} (use --recreate to rebuild)"
        )


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[list[float]]:
    texts = [
        f"""
        Title: {c.get("page_title", "")}
        Section: {c.get("section", "")}
        Content:
        {c["text"]}
        """
        for c in chunks
    ]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def build_actions(chunks: list[dict], embeddings: list[list[float]], index_name: str):
    for chunk, vector in zip(chunks, embeddings):
        yield {
            "_index": index_name,
            "_id": chunk["id"],
            "_source": {
                "text": chunk["text"],
                "embedding": vector,
                "city": chunk["city"],
                "source": chunk["source"],
                "page_title": chunk["page_title"],
                "section": chunk["section"],
                "chunk_id": chunk["id"],
            },
        }


def run(
    chunks_path: Path,
    es_url: str,
    index_name: str,
    embedding_model: str,
    recreate: bool,
) -> None:
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    print(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    print("Embedding chunks...")
    embeddings = embed_chunks(chunks, model)

    es = Elasticsearch(es_url)
    if not es.ping():
        raise SystemExit(
            f"Could not connect to Elasticsearch at {es_url}. "
            "Is it running? (see the Docker command in this script's docstring)"
        )

    create_index(es, index_name, recreate)

    print("Indexing into Elasticsearch...")
    actions = list(build_actions(chunks, embeddings, index_name))
    success, errors = helpers.bulk(es, actions, stats_only=False, raise_on_error=False)
    print(f"Indexed {success} documents.")
    if errors:
        print(f"{len(errors)} errors occurred, e.g.:")
        for err in errors[:3]:
            print(" ", err)

    es.indices.refresh(index=index_name)
    count = es.count(index=index_name)["count"]
    print(f"Index '{index_name}' now has {count} documents.")


def main():
    parser = argparse.ArgumentParser(
        description="Index chunks.json into Elasticsearch (hybrid: BM25 + kNN)."
    )
    parser.add_argument(
        "--chunks",
        default="data/chunks.json",
        help="Path to chunks.json from ingest.py.",
    )
    parser.add_argument(
        "--index", default=DEFAULT_INDEX, help="Elasticsearch index name."
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="sentence-transformers model name used to embed chunks.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the index if it already exists.",
    )
    args = parser.parse_args()
    run(Path(args.chunks), DEFAULT_ES_URL, args.index, args.embedding_model, args.recreate)


if __name__ == "__main__":
    main()
