from __future__ import annotations

import argparse
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "steam_games_reviews_25.sqlite"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.pkl"

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_REVIEWS_PER_GAME = 6
DEFAULT_MAX_REVIEW_CHARS = 2000
DEFAULT_BATCH_SIZE = 128
SQLITE_IN_CHUNK_SIZE = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for Steam game retrieval.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reviews-per-game", type=int, default=DEFAULT_REVIEWS_PER_GAME)
    parser.add_argument("--max-review-chars", type=int, default=DEFAULT_MAX_REVIEW_CHARS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--create-indexes",
        action="store_true",
        help="Create SQLite helper indexes before loading data (can take time on first run).",
    )
    return parser.parse_args()


def _safe_json_load(payload: str | None, fallback: Any) -> Any:
    if not payload:
        return fallback

    import json

    try:
        return json.loads(payload)
    except Exception:
        return fallback


def _clean_text(value: str | None) -> str:
    text = (value or "").replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())


def _compose_document(metadata: dict[str, Any], review_summary: str) -> str:
    name = metadata.get("name", "")
    genres = ", ".join(metadata.get("genres", []))
    tags = metadata.get("tags", [])
    if isinstance(tags, dict):
        tags = list(tags.keys())
    tags_text = ", ".join(tags[:25])

    description = metadata.get("short_description") or metadata.get("about_the_game") or metadata.get(
        "detailed_description", ""
    )

    parts = [
        f"Title: {name}",
        f"Genres: {genres}",
        f"Tags: {tags_text}",
        f"Description: {_clean_text(description)}",
        f"Recent reviews: {_clean_text(review_summary)}",
    ]
    return "\n".join(parts)


def _ensure_sqlite_indexes(connection: sqlite3.Connection) -> None:
    # These indexes make repeated index builds much faster when filtering by appid/language/time.
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reviews_appid_lang_created
        ON reviews(appid, language, timestamp_created DESC)
        """
    )


def _configure_sqlite_for_read_heavy_workload(connection: sqlite3.Connection) -> None:
    # Read-only tuning. Does not change query results.
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -200000")
    connection.execute("PRAGMA mmap_size = 268435456")


def _chunked(values: list[str], chunk_size: int) -> list[list[str]]:
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def _load_review_summaries(
    connection: sqlite3.Connection,
    app_ids: list[str],
    reviews_per_game: int,
    max_review_chars: int,
) -> dict[str, str]:
    if not app_ids:
        return {}

    per_app: dict[str, list[str]] = {}

    for app_id_chunk in _chunked(app_ids, SQLITE_IN_CHUNK_SIZE):
        placeholders = ",".join("?" for _ in app_id_chunk)
        query = f"""
            SELECT appid, review
            FROM (
                SELECT
                    appid,
                    review,
                    ROW_NUMBER() OVER (
                        PARTITION BY appid
                        ORDER BY timestamp_created DESC
                    ) AS rn
                FROM reviews
                WHERE appid IN ({placeholders})
                  AND language = 'english'
                  AND review IS NOT NULL
                  AND TRIM(review) != ''
            ) ranked
            WHERE rn <= ?
            ORDER BY appid, rn
        """
        params = [*app_id_chunk, reviews_per_game]
        rows = connection.execute(query, params).fetchall()

        for row in rows:
            app_id = str(row["appid"])
            per_app.setdefault(app_id, []).append(_clean_text(row["review"]))

    summary: dict[str, str] = {}
    for app_id, snippets in per_app.items():
        joined = " ".join(snippets)
        summary[app_id] = joined[:max_review_chars]

    return summary


def _load_game_metadata(connection: sqlite3.Connection, limit: int | None) -> list[tuple[str, dict[str, Any]]]:
    query = """
        SELECT
            appid,
            name,
            short_description,
            about_the_game,
            detailed_description,
            release_date,
            price,
            header_image,
            windows,
            mac,
            linux,
            recommendations,
            positive,
            negative,
            peak_ccu,
            genres_json,
            tags_json
        FROM games
        WHERE name IS NOT NULL AND TRIM(name) != ''
        ORDER BY appid
    """

    if limit is not None:
        query = f"{query}\nLIMIT ?"
        rows = connection.execute(query, (limit,)).fetchall()
    else:
        rows = connection.execute(query).fetchall()

    records: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        raw = {
            "name": row["name"] or "Unknown title",
            "short_description": row["short_description"] or "",
            "about_the_game": row["about_the_game"] or "",
            "detailed_description": row["detailed_description"] or "",
            "release_date": row["release_date"],
            "price": row["price"],
            "header_image": row["header_image"],
            "windows": bool(row["windows"]),
            "mac": bool(row["mac"]),
            "linux": bool(row["linux"]),
            "recommendations": row["recommendations"],
            "positive": row["positive"],
            "negative": row["negative"],
            "peak_ccu": row["peak_ccu"],
            "genres": _safe_json_load(row["genres_json"], []),
            "tags": _safe_json_load(row["tags_json"], {}),
        }
        records.append((str(row["appid"]), raw))

    return records


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_index(
    db_path: Path,
    model_name: str,
    reviews_per_game: int,
    max_review_chars: int,
    batch_size: int,
    limit: int | None,
    create_indexes: bool = False,
) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {db_path}")

    start_total = time.perf_counter()
    print("[build_index] Opening SQLite database...", flush=True)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        _configure_sqlite_for_read_heavy_workload(connection)
        if create_indexes:
            print("[build_index] Creating SQLite helper indexes...", flush=True)
            _ensure_sqlite_indexes(connection)

        print("[build_index] Loading game metadata...", flush=True)
        step_start = time.perf_counter()
        games = _load_game_metadata(connection, limit)
        print(f"[build_index] Loaded {len(games)} games in {time.perf_counter() - step_start:.2f}s", flush=True)

        selected_app_ids = [app_id for app_id, _ in games]
        print("[build_index] Loading selected recent reviews...", flush=True)
        step_start = time.perf_counter()
        review_summaries = _load_review_summaries(
            connection,
            selected_app_ids,
            reviews_per_game,
            max_review_chars,
        )
        print(
            f"[build_index] Loaded review summaries for {len(review_summaries)} games in {time.perf_counter() - step_start:.2f}s",
            flush=True,
        )

    metadata: list[dict[str, Any]] = []
    documents: list[str] = []

    for app_id, raw in games:
        review_summary = review_summaries.get(app_id, "")
        metadata.append(
            {
                "app_id": app_id,
                "raw": raw,
                "review_summary": review_summary,
            }
        )
        documents.append(_compose_document(raw, review_summary))

    if not documents:
        raise RuntimeError("No games found in database. Index build aborted.")

    print(
        f"[build_index] Encoding {len(documents)} documents with {model_name} on cpu (batch_size={batch_size})...",
        flush=True,
    )
    step_start = time.perf_counter()
    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True,
    )
    print(f"[build_index] Embedding generation completed in {time.perf_counter() - step_start:.2f}s", flush=True)

    embeddings = embeddings.astype("float32", copy=False)
    embeddings = _l2_normalize(embeddings)

    print("[build_index] Building FAISS index...", flush=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    index_bytes = faiss.serialize_index(index)
    INDEX_PATH.write_bytes(np.asarray(index_bytes, dtype="uint8").tobytes())
    with METADATA_PATH.open("wb") as fh:
        pickle.dump(metadata, fh)

    print(f"Built index for {len(metadata)} games")
    print(f"FAISS index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"[build_index] Total elapsed: {time.perf_counter() - start_total:.2f}s")


def main() -> None:
    args = parse_args()
    build_index(
        db_path=args.db_path,
        model_name=args.model_name,
        reviews_per_game=args.reviews_per_game,
        max_review_chars=args.max_review_chars,
        batch_size=args.batch_size,
        limit=args.limit,
        create_indexes=args.create_indexes,
    )


if __name__ == "__main__":
    main()
