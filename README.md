# Assignment 3: LLM Steam Recommender

This repository contains a Steam game recommender built with a hybrid retrieval + reranking pipeline.
The system returns top game matches and a short explanation based on user intent.

## What Is Implemented

- Semantic retrieval over prebuilt embeddings (FAISS)
- Hybrid candidate scoring (semantic, tags, popularity, sentiment, intent boosts)
- LLM-assisted reranking and concise answer generation
- Flask web app + API endpoint (`POST /api/search`)

## Repository Layout

- `app.py` - Flask entrypoint
- `recommender.py` - retrieval, ranking, and answer generation
- `build_index.py` - offline index/metadata build
- `steam_sqlite.py` - SQLite loading utilities
- `static/` and `templates/` - frontend

## Required Embeddings

Prebuilt embeddings are not included in this repository due to file size limits.

Download `embeddings.zip` from the submission and extract it at the project root.
After extraction, these files must exist:

- `embeddings/faiss.index`
- `embeddings/metadata.pkl`

## Setup

```bash
uv sync
```

## Run

```bash
uv run flask --app app run --debug
```

Open:

`http://127.0.0.1:5000`

## API Example

Request:

```json
{
	"query": "story-driven turn-based RPG"
}
```

Endpoint:

`POST /api/search`
