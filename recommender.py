from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from steam_sqlite import load_games_from_sqlite

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.pkl"

MAX_GAMES = 5000
RETRIEVAL_K = 25
FAISS_CANDIDATE_POOL = 200
MATCH_COUNT = 5
EMBED_MODEL_NAME = os.environ.get("RAGLOOKER_EMBED_MODEL", "all-MiniLM-L6-v2")
OLLAMA_BASE_URL = os.environ.get("RAGLOOKER_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("RAGLOOKER_OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("RAGLOOKER_OLLAMA_TIMEOUT", "30"))
OLLAMA_KEEP_ALIVE = os.environ.get("RAGLOOKER_OLLAMA_KEEP_ALIVE", "15m")
OLLAMA_RERANK_WEIGHT = float(os.environ.get("RAGLOOKER_OLLAMA_RERANK_WEIGHT", "0.75"))
INITIAL_SEMANTIC_WEIGHT = float(os.environ.get("RAGLOOKER_INITIAL_SEMANTIC_WEIGHT", "0.35"))
INITIAL_TAG_WEIGHT = float(os.environ.get("RAGLOOKER_INITIAL_TAG_WEIGHT", "0.30"))
INITIAL_POPULARITY_WEIGHT = float(os.environ.get("RAGLOOKER_INITIAL_POPULARITY_WEIGHT", "0.20"))
INITIAL_REVIEW_VOLUME_WEIGHT = float(os.environ.get("RAGLOOKER_INITIAL_REVIEW_VOLUME_WEIGHT", "0.15"))
TAG_MATCH_BOOST_PER_HIT = float(os.environ.get("RAGLOOKER_TAG_MATCH_BOOST_PER_HIT", "0.07"))
TAG_MATCH_MAX_BOOST = float(os.environ.get("RAGLOOKER_TAG_MATCH_MAX_BOOST", "0.45"))
TAG_EXPANSION_MAX_TAGS = int(os.environ.get("RAGLOOKER_TAG_EXPANSION_MAX_TAGS", "8"))
TAG_PROMPT_ORDER_WEIGHT = float(os.environ.get("RAGLOOKER_TAG_PROMPT_ORDER_WEIGHT", "0.20"))
TAG_PROMPT_ORDER_DECAY = float(os.environ.get("RAGLOOKER_TAG_PROMPT_ORDER_DECAY", "0.10"))
REVERSE_TAG_PENALTY_PER_HIT = float(os.environ.get("RAGLOOKER_REVERSE_TAG_PENALTY_PER_HIT", "0.04"))
REVERSE_TAG_MAX_PENALTY = float(os.environ.get("RAGLOOKER_REVERSE_TAG_MAX_PENALTY", "0.25"))
PERSPECTIVE_MATCH_BOOST = float(os.environ.get("RAGLOOKER_PERSPECTIVE_MATCH_BOOST", "0.12"))
ANCHOR_RECALL_ENABLED = os.environ.get("RAGLOOKER_ANCHOR_RECALL_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
ANCHOR_RECALL_MAX_INJECT = int(os.environ.get("RAGLOOKER_ANCHOR_RECALL_MAX_INJECT", "3"))
ANCHOR_RECALL_RETAIN_QUOTA = int(os.environ.get("RAGLOOKER_ANCHOR_RECALL_RETAIN_QUOTA", "2"))
ANCHOR_RECALL_MIN_TAG_HITS = int(os.environ.get("RAGLOOKER_ANCHOR_RECALL_MIN_TAG_HITS", "1"))
ANCHOR_RECALL_MIN_POPULARITY = float(os.environ.get("RAGLOOKER_ANCHOR_RECALL_MIN_POPULARITY", "0.25"))
ANCHOR_RECALL_SIMILARITY_FLOOR = float(os.environ.get("RAGLOOKER_ANCHOR_RECALL_SIMILARITY_FLOOR", "0.18"))
DYNAMIC_ANCHOR_SUGGEST_ENABLED = os.environ.get("RAGLOOKER_DYNAMIC_ANCHOR_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DYNAMIC_ANCHOR_CANDIDATE_POOL = int(os.environ.get("RAGLOOKER_DYNAMIC_ANCHOR_CANDIDATE_POOL", "60"))
DYNAMIC_ANCHOR_FALLBACK_CANDIDATE_POOL = int(os.environ.get("RAGLOOKER_DYNAMIC_ANCHOR_FALLBACK_POOL", "20"))
DYNAMIC_ANCHOR_MAX_RETURN = int(os.environ.get("RAGLOOKER_DYNAMIC_ANCHOR_MAX_RETURN", "6"))
DYNAMIC_ANCHOR_MIN_CONFIDENCE = float(os.environ.get("RAGLOOKER_DYNAMIC_ANCHOR_MIN_CONFIDENCE", "0.55"))
DYNAMIC_ANCHOR_FALLBACK_MIN_CONFIDENCE = float(os.environ.get("RAGLOOKER_DYNAMIC_ANCHOR_FALLBACK_MIN_CONFIDENCE", "0.40"))

LOCAL_TAG_ALIASES: dict[str, set[str]] = {
    "exploration": {"adventure", "openworld", "sandbox"},
    "explore": {"adventure", "openworld"},
    "cozy": {"casual", "relaxing", "wholesome", "lifesim"},
    "farming": {"farm", "agriculture", "simulation", "lifesim"},
    "farm": {"farming", "simulation", "lifesim"},
    "space": {"sci", "scifi", "adventure", "exploration"},
    "story": {"narrative", "adventure"},
    "builder": {"building", "crafting", "sandbox"},
    "management": {"simulation", "strategy", "tycoon"},
}

LOCAL_REVERSE_TAG_ALIASES: dict[str, set[str]] = {
    "cozy": {"hardcore", "stressful", "intense", "punishing"},
    "simple": {"complex", "deep", "hardcore", "micromanagement"},
    "relaxing": {"stressful", "intense", "competitive"},
    "casual": {"hardcore", "competitive", "difficult"},
    "story": {"grindy", "repetitive"},
    "exploration": {"linear", "corridor"},
}

GENRE_CLUSTER_TERMS: dict[str, set[str]] = {
    "rpg": {"rpg", "jrpg", "arpg", "crpg", "soulslike"},
    "strategy": {"strategy", "tactical", "4x", "towerdefense", "citybuilder"},
    "simulation": {"simulation", "simulator", "lifesim", "management", "tycoon", "farming"},
    "adventure": {"adventure", "exploration", "openworld", "narrative"},
    "action": {"action", "shooter", "hack", "slash", "fighting"},
    "horror": {"horror", "survivalhorror"},
    "puzzle": {"puzzle"},
    "racing": {"racing"},
    "sports": {"sports"},
    "sandbox": {"sandbox", "crafting", "building", "survival"},
}

INTERACTION_MODALITY_TERMS: dict[str, set[str]] = {
    "singleplayer": {"singleplayer", "solo"},
    "coop": {"coop", "cooperative"},
    "multiplayer": {"multiplayer", "online", "mmo"},
    "pvp": {"pvp", "competitive"},
    "turnbased": {"turnbased"},
    "realtime": {"realtime", "action"},
}

PERSPECTIVE_TERMS: dict[str, set[str]] = {
    "firstperson": {"firstperson", "fps"},
    "thirdperson": {"thirdperson", "tps"},
    "topdown": {"topdown", "isometric"},
    "sidescroller": {"sidescroller", "platformer"},
}

SEMANTIC_MATCH_WEIGHT = 0.07
GAMEPLAY_LOOP_WEIGHT = 0.30
MECHANIC_FRICTION_WEIGHT = 0.10
PLAYER_FANTASY_WEIGHT = 0.16
TAG_METADATA_WEIGHT = 0.32
COMMUNITY_SENTIMENT_WEIGHT = 0.02
POPULARITY_SUCCESS_WEIGHT = 0.03


def create_search_engine() -> "GameSearchEngine":
    return GameSearchEngine(DB_PATH)


@dataclass
class GameRecord:
    app_id: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return self.raw.get("name", "Unknown title")

    @property
    def short_description(self) -> str:
        return self.raw.get("short_description", "")

    def to_result(self, score: float) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "name": self.name,
            "score": round(score, 4),
            "short_description": self.short_description,
            "genres": self.raw.get("genres", []),
            "tags": self._normalize_tags(self.raw.get("tags")),
            "price": self.raw.get("price"),
            "release_date": self.raw.get("release_date"),
            "header_image": self.raw.get("header_image"),
            "store_page": f"https://store.steampowered.com/app/{self.app_id}",
            "platforms": {
                "windows": bool(self.raw.get("windows")),
                "mac": bool(self.raw.get("mac")),
                "linux": bool(self.raw.get("linux")),
            },
        }

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if isinstance(tags, dict):
            return list(tags.keys())[:8]
        if isinstance(tags, list):
            return tags[:8]
        return []


@dataclass
class Candidate:
    record: GameRecord
    similarity: float
    popularity: float
    review_volume: float
    tag_match: float
    tag_hit_count: int
    reverse_tag_hit_count: int
    perspective_match: float
    community_sentiment: float
    review_summary: str


@dataclass
class RankedCandidate:
    record: GameRecord
    score: float
    review_summary: str
    reason: str


def _clean_text(value: str | None) -> str:
    text = (value or "").replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())


def _extract_json_block(payload: str) -> str | None:
    payload = payload.strip()
    if payload.startswith("{") and payload.endswith("}"):
        return payload

    match = re.search(r"\{[\s\S]*\}", payload)
    if match:
        return match.group(0)

    return None


def _query_tokens(query: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(token) >= 3}


def _ordered_query_tokens(query: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in re.findall(r"[a-z0-9]+", (query or "").lower()):
        if len(token) < 3 or token in seen:
            continue
        ordered.append(token)
        seen.add(token)
    return ordered


def _normalized_title(value: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _normalized_tag_values(raw: dict[str, Any]) -> set[str]:
    tags = raw.get("tags")
    if isinstance(tags, dict):
        tag_values = list(tags.keys())
    elif isinstance(tags, list):
        tag_values = tags
    else:
        tag_values = []
    return {str(tag).lower() for tag in tag_values if str(tag).strip()}


def _metadata_tokens(raw: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()

    genres = raw.get("genres")
    if isinstance(genres, list):
        for value in genres:
            tokens.update(re.findall(r"[a-z0-9]+", str(value).lower()))

    for value in _normalized_tag_values(raw):
        tokens.update(re.findall(r"[a-z0-9]+", value))

    return {token for token in tokens if len(token) >= 3}


def _cluster_hits(tokens: set[str], mapping: dict[str, set[str]]) -> set[str]:
    hits: set[str] = set()
    for label, terms in mapping.items():
        if tokens & terms:
            hits.add(label)
    return hits


def _query_intent_from_tokens(query_tokens: set[str]) -> tuple[set[str], set[str], set[str]]:
    required_genre_clusters = _cluster_hits(query_tokens, GENRE_CLUSTER_TERMS)
    required_modalities = _cluster_hits(query_tokens, INTERACTION_MODALITY_TERMS)
    preferred_perspectives = _cluster_hits(query_tokens, PERSPECTIVE_TERMS)
    return required_genre_clusters, required_modalities, preferred_perspectives


def _pre_scoring_intent_match(
    raw: dict[str, Any],
    required_genre_clusters: set[str],
    required_modalities: set[str],
    preferred_perspectives: set[str],
) -> float:
    record_tokens = _metadata_tokens(raw)
    record_genre_clusters = _cluster_hits(record_tokens, GENRE_CLUSTER_TERMS)
    record_modalities = _cluster_hits(record_tokens, INTERACTION_MODALITY_TERMS)
    record_perspectives = _cluster_hits(record_tokens, PERSPECTIVE_TERMS)

    perspective_match = 0.0
    if preferred_perspectives and (record_perspectives & preferred_perspectives):
        perspective_match += 1.0
    if required_genre_clusters and (record_genre_clusters & required_genre_clusters):
        perspective_match += 0.5
    if required_modalities and (record_modalities & required_modalities):
        perspective_match += 0.5
    perspective_match = max(0.0, min(perspective_match, 1.0))

    return perspective_match


def _tag_match_score(query_set: set[str], prompt_tokens: list[str], raw: dict[str, Any]) -> float:
    if not query_set:
        return 0.0

    hits = _tag_match_hits(query_set, raw)
    base_score = hits / max(len(query_set), 1)

    if not prompt_tokens:
        return base_score

    normalized_tags = _normalized_tag_values(raw)
    if not normalized_tags:
        return base_score

    decay = max(0.0, min(TAG_PROMPT_ORDER_DECAY, 0.4))
    weights: list[float] = []
    weighted_hits = 0.0
    for idx, token in enumerate(prompt_tokens):
        weight = max(0.5, 1.0 - (decay * idx))
        weights.append(weight)
        if any(token in tag or tag in token for tag in normalized_tags):
            weighted_hits += weight

    order_score = weighted_hits / max(sum(weights), 1e-6)
    order_w = max(0.0, min(TAG_PROMPT_ORDER_WEIGHT, 0.4))
    return ((1.0 - order_w) * base_score) + (order_w * order_score)


def _tag_match_hits(query_set: set[str], raw: dict[str, Any]) -> int:
    if not query_set:
        return 0

    normalized_tags = _normalized_tag_values(raw)
    if not normalized_tags:
        return 0

    hits = 0
    for token in query_set:
        if any(token in tag or tag in token for tag in normalized_tags):
            hits += 1
    return hits


def _to_non_negative_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0.0 else 0.0


def _log_normalize(value: float, cap: float) -> float:
    if value <= 0.0 or cap <= 1.0:
        return 0.0
    return min(np.log1p(value) / np.log1p(cap), 1.0)


def _popularity_prior(raw: dict[str, Any]) -> float:
    positive = _to_non_negative_float(raw.get("positive"))
    negative = _to_non_negative_float(raw.get("negative"))
    recommendations = _to_non_negative_float(raw.get("recommendations"))
    peak_ccu = _to_non_negative_float(raw.get("peak_ccu"))

    review_count = positive + negative
    review_strength = _log_normalize(review_count, 100_000.0)
    recommendation_strength = _log_normalize(recommendations, 100_000.0)
    ccu_strength = _log_normalize(peak_ccu, 250_000.0)

    if review_count >= 20:
        sentiment_strength = max(0.0, min(positive / review_count, 1.0))
    else:
        sentiment_strength = 0.5

    # Popularity is only a prior for tie-breaking, not a primary rank signal.
    prior = (
        0.40 * review_strength
        + 0.30 * recommendation_strength
        + 0.20 * ccu_strength
        + 0.10 * sentiment_strength
    )
    return max(0.0, min(prior, 1.0))


def _community_sentiment_score(raw: dict[str, Any]) -> float:
    positive = _to_non_negative_float(raw.get("positive"))
    negative = _to_non_negative_float(raw.get("negative"))
    total = positive + negative
    if total < 20:
        return 0.5
    return max(0.0, min(positive / total, 1.0))


def _review_volume_score(raw: dict[str, Any]) -> float:
    positive = _to_non_negative_float(raw.get("positive"))
    negative = _to_non_negative_float(raw.get("negative"))
    review_count = positive + negative
    return _log_normalize(review_count, 150_000.0)


def _weighted_rank_score(
    semantic_match: float,
    gameplay_loop: float,
    mechanic_friction: float,
    player_fantasy: float,
    tag_match: float,
    community_sentiment: float,
    popularity: float,
) -> float:
    return (
        (SEMANTIC_MATCH_WEIGHT * semantic_match)
        + (GAMEPLAY_LOOP_WEIGHT * gameplay_loop)
        + (MECHANIC_FRICTION_WEIGHT * mechanic_friction)
        + (PLAYER_FANTASY_WEIGHT * player_fantasy)
        + (TAG_METADATA_WEIGHT * tag_match)
        + (COMMUNITY_SENTIMENT_WEIGHT * community_sentiment)
        + (POPULARITY_SUCCESS_WEIGHT * popularity)
    )


def _blend_rank_signal(
    similarity: float,
    popularity: float,
    review_volume: float,
    tag_match: float,
    tag_hit_count: int,
    reverse_tag_hit_count: int,
    perspective_match: float,
) -> float:
    sem_w = max(0.0, INITIAL_SEMANTIC_WEIGHT)
    tag_w = max(0.0, INITIAL_TAG_WEIGHT)
    pop_w = max(0.0, INITIAL_POPULARITY_WEIGHT)
    rev_w = max(0.0, INITIAL_REVIEW_VOLUME_WEIGHT)
    total = sem_w + tag_w + pop_w + rev_w
    if total <= 0.0:
        return similarity

    sem_w /= total
    tag_w /= total
    pop_w /= total
    rev_w /= total
    base_score = (sem_w * similarity) + (tag_w * tag_match) + (pop_w * popularity) + (rev_w * review_volume)
    per_hit_boost = max(0.0, TAG_MATCH_BOOST_PER_HIT)
    max_boost = max(0.0, TAG_MATCH_MAX_BOOST)
    boost = min(per_hit_boost * max(tag_hit_count, 0), max_boost)
    penalty_per_hit = max(0.0, REVERSE_TAG_PENALTY_PER_HIT)
    max_penalty = max(0.0, REVERSE_TAG_MAX_PENALTY)
    penalty = min(penalty_per_hit * max(reverse_tag_hit_count, 0), max_penalty)
    perspective_boost = max(0.0, min(PERSPECTIVE_MATCH_BOOST, 0.3)) * max(0.0, min(perspective_match, 1.0))
    return max(0.0, min(base_score + boost - penalty + perspective_boost, 1.0))


def _blend_llm_with_base_signal(llm_score: float, base_signal: float) -> float:
    llm_w = max(0.0, min(OLLAMA_RERANK_WEIGHT, 1.0))
    return (llm_w * llm_score) + ((1.0 - llm_w) * base_signal)


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            json_text = _extract_json_block(content)
            if not json_text:
                return None
            return json.loads(json_text)
        except Exception:
            return None

    def chat_text(self, system_prompt: str, user_prompt: str) -> str | None:
        payload = {
            "model": self.model,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "").strip()
            return content or None
        except Exception:
            return None


class GameSearchEngine:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.ollama = OllamaClient(OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS)

        self.records_by_app_id: dict[str, GameRecord] = {}
        self.review_summaries: dict[str, str] = {}
        self.index: faiss.Index | None = None
        self.index_app_ids: list[str] = []
        self.using_cached_index = False
        self._tag_expansion_cache: dict[str, set[str]] = {}
        self._reverse_tag_expansion_cache: dict[str, set[str]] = {}
        self._intent_cache: dict[str, tuple[set[str], set[str], set[str]]] = {}
        self._query_analysis_cache: dict[str, dict[str, Any]] = {}
        self._dynamic_anchor_cache: dict[str, list[dict[str, Any]]] = {}
        self._last_anchor_injected_count = 0
        self._last_anchor_retained_count = 0
        self._last_anchor_injected_app_ids: set[str] = set()
        self._last_dynamic_anchor_suggested = 0
        self._last_dynamic_anchor_validated = 0
        self._last_dynamic_anchor_attempt = "none"

        self._load_cached_index()

        if not self.records_by_app_id:
            self._load_fallback_records()

    def _load_cached_index(self) -> None:
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            return

        try:
            payload = INDEX_PATH.read_bytes()
            vectors = np.frombuffer(payload, dtype="uint8")
            self.index = faiss.deserialize_index(vectors)
            with METADATA_PATH.open("rb") as fh:
                metadata: list[dict[str, Any]] = pickle.load(fh)
        except Exception:
            self.index = None
            return

        for item in metadata:
            app_id = str(item.get("app_id", "")).strip()
            raw = item.get("raw") or {}
            if not app_id or not isinstance(raw, dict):
                continue
            self.records_by_app_id[app_id] = GameRecord(app_id=app_id, raw=raw)
            self.review_summaries[app_id] = _clean_text(item.get("review_summary", ""))
            self.index_app_ids.append(app_id)

        if self.index is not None and self.index.ntotal == len(self.index_app_ids):
            self.using_cached_index = True
        else:
            self.index = None
            self.index_app_ids = []
            self.using_cached_index = False

    def _load_fallback_records(self) -> None:
        for app_id, raw in load_games_from_sqlite(self.db_path, MAX_GAMES):
            self.records_by_app_id[app_id] = GameRecord(app_id=app_id, raw=raw)

    def _analyze_query(self, query: str) -> dict[str, Any]:
        cache_key = (query or "").strip().lower()
        if not cache_key:
            return {
                "expanded_tags": set(),
                "reverse_tags": set(),
                "genre_clusters": set(),
                "interaction_modalities": set(),
                "perspectives": set(),
            }

        cached = self._query_analysis_cache.get(cache_key)
        if cached is not None:
            return {
                "expanded_tags": set(cached.get("expanded_tags", set())),
                "reverse_tags": set(cached.get("reverse_tags", set())),
                "genre_clusters": set(cached.get("genre_clusters", set())),
                "interaction_modalities": set(cached.get("interaction_modalities", set())),
                "perspectives": set(cached.get("perspectives", set())),
            }

        base_tokens = _query_tokens(query)
        expanded = set(base_tokens)
        for token in base_tokens:
            expanded.update(LOCAL_TAG_ALIASES.get(token, set()))

        reverse_tags: set[str] = set()
        for token in base_tokens:
            reverse_tags.update(LOCAL_REVERSE_TAG_ALIASES.get(token, set()))

        local_genres, local_modalities, local_perspectives = _query_intent_from_tokens(expanded)
        genres = set(local_genres)
        modalities = set(local_modalities)
        perspectives = set(local_perspectives)

        system_prompt = (
            "You analyze a game recommendation query and return strict JSON only with this schema: "
            "{\"expanded_tags\": [str], \"reverse_tags\": [str], \"genre_clusters\": [str], "
            "\"interaction_modalities\": [str], \"perspectives\": [str]}. "
            "Use short lowercase tags only. "
            "Allowed genre_clusters: rpg, strategy, simulation, adventure, action, horror, puzzle, racing, sports, sandbox. "
            "Allowed interaction_modalities: singleplayer, coop, multiplayer, pvp, turnbased, realtime. "
            "Allowed perspectives: firstperson, thirdperson, topdown, sidescroller. "
            "Only include strongly implied intent clusters."
        )
        user_prompt = json.dumps(
            {
                "query": query,
                "base_tokens": sorted(base_tokens),
                "positive_tags": sorted(expanded),
                "max_tags": TAG_EXPANSION_MAX_TAGS,
            },
            ensure_ascii=True,
        )

        payload = self.ollama.chat_json(system_prompt, user_prompt)
        if payload and isinstance(payload, dict):
            for value in payload.get("expanded_tags", []):
                for token in re.findall(r"[a-z0-9]+", str(value).lower()):
                    if len(token) >= 3:
                        expanded.add(token)

            for value in payload.get("reverse_tags", []):
                for token in re.findall(r"[a-z0-9]+", str(value).lower()):
                    if len(token) >= 3:
                        reverse_tags.add(token)

            allowed_genres = set(GENRE_CLUSTER_TERMS.keys())
            allowed_modalities = set(INTERACTION_MODALITY_TERMS.keys())
            allowed_perspectives = set(PERSPECTIVE_TERMS.keys())

            for value in payload.get("genre_clusters", []):
                token = str(value).strip().lower()
                if token in allowed_genres:
                    genres.add(token)

            for value in payload.get("interaction_modalities", []):
                token = str(value).strip().lower()
                if token in allowed_modalities:
                    modalities.add(token)

            for value in payload.get("perspectives", []):
                token = str(value).strip().lower()
                if token in allowed_perspectives:
                    perspectives.add(token)

        if len(expanded) > TAG_EXPANSION_MAX_TAGS + len(base_tokens):
            expanded = set(sorted(expanded)[: TAG_EXPANSION_MAX_TAGS + len(base_tokens)])
        if len(reverse_tags) > TAG_EXPANSION_MAX_TAGS:
            reverse_tags = set(sorted(reverse_tags)[:TAG_EXPANSION_MAX_TAGS])

        analysis = {
            "expanded_tags": set(expanded),
            "reverse_tags": set(reverse_tags),
            "genre_clusters": set(genres),
            "interaction_modalities": set(modalities),
            "perspectives": set(perspectives),
        }
        self._query_analysis_cache[cache_key] = {
            "expanded_tags": set(expanded),
            "reverse_tags": set(reverse_tags),
            "genre_clusters": set(genres),
            "interaction_modalities": set(modalities),
            "perspectives": set(perspectives),
        }
        self._tag_expansion_cache[cache_key] = set(expanded)
        self._reverse_tag_expansion_cache[cache_key] = set(reverse_tags)
        self._intent_cache[cache_key] = (set(genres), set(modalities), set(perspectives))
        return analysis

    def _expand_query_tags(self, query: str) -> set[str]:
        analysis = self._analyze_query(query)
        return set(analysis.get("expanded_tags", set()))

    def _expand_reverse_query_tags(self, query: str, positive_tags: set[str]) -> set[str]:
        analysis = self._analyze_query(query)
        return set(analysis.get("reverse_tags", set()))

    def _interpret_query_intent(self, query: str, query_tokens: set[str]) -> tuple[set[str], set[str], set[str]]:
        analysis = self._analyze_query(query)
        return (
            set(analysis.get("genre_clusters", set())),
            set(analysis.get("interaction_modalities", set())),
            set(analysis.get("perspectives", set())),
        )

    def _resolve_anchor_record(self, anchor: dict[str, Any]) -> GameRecord | None:
        app_id = str(anchor.get("app_id", "")).strip()
        if app_id:
            record = self.records_by_app_id.get(app_id)
            if record is not None:
                return record

        aliases = anchor.get("title_aliases")
        if not isinstance(aliases, list):
            return None

        normalized_aliases = [_normalized_title(str(alias)) for alias in aliases if str(alias).strip()]
        if not normalized_aliases:
            return None

        for record in self.records_by_app_id.values():
            normalized_name = _normalized_title(record.name)
            if not normalized_name:
                continue
            if any(alias == normalized_name or alias in normalized_name for alias in normalized_aliases):
                return record

        return None

    def _suggest_dynamic_anchors(
        self,
        query: str,
        query_tokens: set[str],
        required_genre_clusters: set[str],
        required_modalities: set[str],
        preferred_perspectives: set[str],
    ) -> list[dict[str, Any]]:
        self._last_dynamic_anchor_suggested = 0
        self._last_dynamic_anchor_validated = 0
        self._last_dynamic_anchor_attempt = "none"
        if not DYNAMIC_ANCHOR_SUGGEST_ENABLED:
            return []

        cache_key = (query or "").strip().lower()
        if not cache_key:
            return []

        cached = self._dynamic_anchor_cache.get(cache_key)
        if cached is not None:
            self._last_dynamic_anchor_suggested = len(cached)
            self._last_dynamic_anchor_validated = len(cached)
            self._last_dynamic_anchor_attempt = "cache-hit"
            return [dict(item) for item in cached]

        pool_size = max(10, DYNAMIC_ANCHOR_CANDIDATE_POOL)
        max_return = max(1, DYNAMIC_ANCHOR_MAX_RETURN)
        min_confidence = max(0.0, min(DYNAMIC_ANCHOR_MIN_CONFIDENCE, 1.0))

        candidates_for_prompt: list[tuple[float, dict[str, Any]]] = []
        token_count = max(len(query_tokens), 1)
        for record in self.records_by_app_id.values():
            metadata_tokens = _metadata_tokens(record.raw)
            overlap = len(metadata_tokens & query_tokens) / token_count if query_tokens else 0.0
            popularity = _popularity_prior(record.raw)
            score = (0.65 * overlap) + (0.35 * popularity)
            if overlap <= 0.0 and popularity < 0.70:
                continue
            candidates_for_prompt.append(
                (
                    score,
                    {
                        "app_id": record.app_id,
                        "name": record.name,
                        "genres": record.raw.get("genres", [])[:4],
                        "tags": record.to_result(0.0).get("tags", [])[:6],
                        "popularity_prior": round(popularity, 4),
                    },
                )
            )

        if not candidates_for_prompt:
            self._dynamic_anchor_cache[cache_key] = []
            return []

        candidates_for_prompt.sort(key=lambda item: item[0], reverse=True)
        system_prompt = (
            "You select strong anchor games for retrieval recall. "
            "Return strict JSON only with this schema: "
            "{\"anchors\": [{\"app_id\": str, \"title\": str, \"confidence\": float, \"reason\": str}]}. "
            "Only pick from the provided candidate_pool. "
            "Do not invent app_id values. Keep confidence in [0,1]."
        )

        def _run_dynamic_attempt(
            attempt_pool_size: int,
            attempt_min_confidence: float,
            attempt_label: str,
        ) -> tuple[list[dict[str, Any]], int]:
            compact_pool = [row for _, row in candidates_for_prompt[: max(10, attempt_pool_size)]]
            user_prompt = json.dumps(
                {
                    "query": query,
                    "intent": {
                        "genre_clusters": sorted(required_genre_clusters),
                        "interaction_modalities": sorted(required_modalities),
                        "perspectives": sorted(preferred_perspectives),
                    },
                    "query_tokens": sorted(query_tokens),
                    "max_anchors": max_return,
                    "candidate_pool": compact_pool,
                },
                ensure_ascii=True,
            )

            payload = self.ollama.chat_json(system_prompt, user_prompt)
            anchor_rows = payload.get("anchors") if isinstance(payload, dict) else None
            if not isinstance(anchor_rows, list):
                return [], 0

            validated_local: list[dict[str, Any]] = []
            seen_ids_local: set[str] = set()
            conf_threshold = max(0.0, min(attempt_min_confidence, 1.0))
            for row in anchor_rows:
                if not isinstance(row, dict):
                    continue
                app_id = str(row.get("app_id", "")).strip()
                title = str(row.get("title", "")).strip()
                try:
                    confidence = float(row.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                confidence = max(0.0, min(confidence, 1.0))
                if confidence < conf_threshold:
                    continue

                if app_id:
                    if app_id not in self.records_by_app_id or app_id in seen_ids_local:
                        continue
                    validated_local.append({"app_id": app_id})
                    seen_ids_local.add(app_id)
                    if len(validated_local) >= max_return:
                        break
                    continue

                if not title:
                    continue
                resolved = self._resolve_anchor_record({"title_aliases": [title]})
                if resolved is None or resolved.app_id in seen_ids_local:
                    continue
                validated_local.append({"app_id": resolved.app_id, "title_aliases": [title]})
                seen_ids_local.add(resolved.app_id)
                if len(validated_local) >= max_return:
                    break

            if validated_local:
                self._last_dynamic_anchor_attempt = attempt_label

            return validated_local, len(anchor_rows)

        fallback_pool_size = max(10, min(pool_size, DYNAMIC_ANCHOR_FALLBACK_CANDIDATE_POOL))
        fallback_min_confidence = max(0.0, min(DYNAMIC_ANCHOR_FALLBACK_MIN_CONFIDENCE, 1.0))

        attempts = [
            (fallback_pool_size, min_confidence, "primary-smaller-pool"),
            (fallback_pool_size, fallback_min_confidence, "fallback-lower-confidence"),
        ]

        validated: list[dict[str, Any]] = []
        suggested_count = 0
        for attempt_pool_size, attempt_min_confidence, attempt_label in attempts:
            validated, suggested_count = _run_dynamic_attempt(
                attempt_pool_size,
                attempt_min_confidence,
                attempt_label,
            )
            self._last_dynamic_anchor_suggested = suggested_count
            self._last_dynamic_anchor_validated = len(validated)
            if validated:
                break

        self._dynamic_anchor_cache[cache_key] = [dict(item) for item in validated]
        if not validated and self._last_dynamic_anchor_attempt == "none":
            self._last_dynamic_anchor_attempt = "failed-all-attempts"
        return validated

    def _inject_anchor_candidates(
        self,
        query: str,
        candidates: list[Candidate],
        query_tokens: set[str],
        reverse_query_tokens: set[str],
        prompt_tokens: list[str],
        required_genre_clusters: set[str],
        required_modalities: set[str],
        preferred_perspectives: set[str],
    ) -> list[Candidate]:
        self._last_anchor_injected_count = 0
        self._last_anchor_retained_count = 0
        self._last_anchor_injected_app_ids = set()
        self._last_dynamic_anchor_suggested = 0
        self._last_dynamic_anchor_validated = 0
        self._last_dynamic_anchor_attempt = "none"
        if not ANCHOR_RECALL_ENABLED or ANCHOR_RECALL_MAX_INJECT <= 0:
            return candidates

        existing_ids = {item.record.app_id for item in candidates}
        max_inject = max(0, ANCHOR_RECALL_MAX_INJECT)
        min_tag_hits = max(0, ANCHOR_RECALL_MIN_TAG_HITS)
        min_popularity = max(0.0, min(ANCHOR_RECALL_MIN_POPULARITY, 1.0))
        similarity_floor = max(0.0, min(ANCHOR_RECALL_SIMILARITY_FLOOR, 1.0))
        promoted_anchor_ids: set[str] = set()
        anchor_specs = self._suggest_dynamic_anchors(
            query,
            query_tokens,
            required_genre_clusters,
            required_modalities,
            preferred_perspectives,
        )

        for anchor in anchor_specs:
            if self._last_anchor_injected_count >= max_inject:
                break
            if not isinstance(anchor, dict):
                continue

            record = self._resolve_anchor_record(anchor)
            if record is None:
                continue
            if record.app_id in promoted_anchor_ids:
                continue

            tag_hits = _tag_match_hits(query_tokens, record.raw)
            popularity = _popularity_prior(record.raw)
            if tag_hits < min_tag_hits and popularity < min_popularity:
                continue

            # If the game is already in retrieval candidates, still treat it as an injected anchor
            # so retention quota can protect it from cutoff.
            if record.app_id in existing_ids:
                promoted_anchor_ids.add(record.app_id)
                self._last_anchor_injected_count += 1
                self._last_anchor_injected_app_ids.add(record.app_id)
                continue

            perspective_match = _pre_scoring_intent_match(
                record.raw,
                required_genre_clusters,
                required_modalities,
                preferred_perspectives,
            )
            tag_match = _tag_match_score(query_tokens, prompt_tokens, record.raw)
            metadata_overlap = len(_metadata_tokens(record.raw) & query_tokens) / max(len(query_tokens), 1)
            similarity = max(similarity_floor, (0.60 * tag_match) + (0.40 * metadata_overlap))
            similarity = max(0.0, min(similarity, 1.0))

            injected = Candidate(
                record=record,
                similarity=similarity,
                popularity=popularity,
                review_volume=_review_volume_score(record.raw),
                tag_match=tag_match,
                tag_hit_count=tag_hits,
                reverse_tag_hit_count=_tag_match_hits(reverse_query_tokens, record.raw),
                perspective_match=perspective_match,
                community_sentiment=_community_sentiment_score(record.raw),
                review_summary=self.review_summaries.get(record.app_id, ""),
            )
            candidates.append(injected)
            existing_ids.add(record.app_id)
            promoted_anchor_ids.add(record.app_id)
            self._last_anchor_injected_count += 1
            self._last_anchor_injected_app_ids.add(record.app_id)

        return candidates

    def _apply_anchor_retention_quota(self, sorted_candidates: list[Candidate]) -> list[Candidate]:
        self._last_anchor_retained_count = 0
        limit = max(1, RETRIEVAL_K)
        if not self._last_anchor_injected_app_ids:
            return sorted_candidates[:limit]

        anchors = [item for item in sorted_candidates if item.record.app_id in self._last_anchor_injected_app_ids]
        if not anchors:
            return sorted_candidates[:limit]

        non_anchors = [item for item in sorted_candidates if item.record.app_id not in self._last_anchor_injected_app_ids]
        max_inject = max(0, ANCHOR_RECALL_MAX_INJECT)
        keep_anchor_count = min(len(anchors), max_inject, limit)
        kept_anchors = anchors[:keep_anchor_count]
        kept_non_anchors = non_anchors[: max(0, limit - keep_anchor_count)]
        self._last_anchor_retained_count = len(kept_anchors)
        return kept_anchors + kept_non_anchors

    def search(self, query: str) -> dict[str, Any]:
        candidates = self.retrieve_candidates(query)
        ranked_matches = self.rank_candidates(query, candidates)
        results = [match.record.to_result(match.score) for match in ranked_matches[:MATCH_COUNT]]

        if self.using_cached_index:
            retrieval_mode = "semantic-faiss+ollama-rerank"
            note = "Using cached FAISS index and metadata from embeddings/."
        else:
            retrieval_mode = "fallback-keyword"
            note = "Index files missing or invalid. Run build_index.py to enable semantic retrieval."

        return {
            "matches": results,
            "answer": self.generate_answer(query, ranked_matches[:MATCH_COUNT]),
            "meta": {
                "indexed_games": len(self.records_by_app_id),
                "retrieval_mode": retrieval_mode,
                "final_semantic_match_weight": SEMANTIC_MATCH_WEIGHT,
                "final_gameplay_loop_weight": GAMEPLAY_LOOP_WEIGHT,
                "final_mechanic_friction_weight": MECHANIC_FRICTION_WEIGHT,
                "final_player_fantasy_weight": PLAYER_FANTASY_WEIGHT,
                "final_tag_metadata_weight": TAG_METADATA_WEIGHT,
                "final_community_sentiment_weight": COMMUNITY_SENTIMENT_WEIGHT,
                "final_popularity_success_weight": POPULARITY_SUCCESS_WEIGHT,
                "initial_semantic_weight": round(max(0.0, INITIAL_SEMANTIC_WEIGHT), 3),
                "initial_tag_weight": round(max(0.0, INITIAL_TAG_WEIGHT), 3),
                "initial_popularity_weight": round(max(0.0, INITIAL_POPULARITY_WEIGHT), 3),
                "initial_review_volume_weight": round(max(0.0, INITIAL_REVIEW_VOLUME_WEIGHT), 3),
                "anchor_recall_enabled": ANCHOR_RECALL_ENABLED,
                "anchor_recall_max_inject": max(0, ANCHOR_RECALL_MAX_INJECT),
                "anchor_recall_retain_quota": max(0, ANCHOR_RECALL_RETAIN_QUOTA),
                "anchor_recall_min_tag_hits": max(0, ANCHOR_RECALL_MIN_TAG_HITS),
                "anchor_recall_min_popularity": round(max(0.0, min(ANCHOR_RECALL_MIN_POPULARITY, 1.0)), 3),
                "anchor_recall_similarity_floor": round(max(0.0, min(ANCHOR_RECALL_SIMILARITY_FLOOR, 1.0)), 3),
                "anchor_recall_injected": self._last_anchor_injected_count,
                "anchor_recall_retained": self._last_anchor_retained_count,
                "dynamic_anchor_enabled": DYNAMIC_ANCHOR_SUGGEST_ENABLED,
                "dynamic_anchor_candidate_pool": max(10, DYNAMIC_ANCHOR_CANDIDATE_POOL),
                "dynamic_anchor_fallback_pool": max(10, DYNAMIC_ANCHOR_FALLBACK_CANDIDATE_POOL),
                "dynamic_anchor_max_return": max(1, DYNAMIC_ANCHOR_MAX_RETURN),
                "dynamic_anchor_min_confidence": round(max(0.0, min(DYNAMIC_ANCHOR_MIN_CONFIDENCE, 1.0)), 3),
                "dynamic_anchor_fallback_min_confidence": round(
                    max(0.0, min(DYNAMIC_ANCHOR_FALLBACK_MIN_CONFIDENCE, 1.0)),
                    3,
                ),
                "dynamic_anchor_suggested": self._last_dynamic_anchor_suggested,
                "dynamic_anchor_validated": self._last_dynamic_anchor_validated,
                "dynamic_anchor_attempt": self._last_dynamic_anchor_attempt,
                # Legacy aliases retained for backward compatibility.
                "semantic_match_weight": SEMANTIC_MATCH_WEIGHT,
                "gameplay_loop_weight": GAMEPLAY_LOOP_WEIGHT,
                "mechanic_friction_weight": MECHANIC_FRICTION_WEIGHT,
                "player_fantasy_weight": PLAYER_FANTASY_WEIGHT,
                "tag_weight": TAG_METADATA_WEIGHT,
                "community_sentiment_weight": COMMUNITY_SENTIMENT_WEIGHT,
                "popularity_weight": POPULARITY_SUCCESS_WEIGHT,
                "tag_match_boost_per_hit": round(max(0.0, TAG_MATCH_BOOST_PER_HIT), 3),
                "tag_match_max_boost": round(max(0.0, TAG_MATCH_MAX_BOOST), 3),
                "reverse_tag_penalty_per_hit": round(max(0.0, REVERSE_TAG_PENALTY_PER_HIT), 3),
                "reverse_tag_max_penalty": round(max(0.0, REVERSE_TAG_MAX_PENALTY), 3),
                "perspective_match_boost": round(max(0.0, min(PERSPECTIVE_MATCH_BOOST, 0.3)), 3),
                "ollama_rerank_weight": round(max(0.0, min(OLLAMA_RERANK_WEIGHT, 1.0)), 3),
                "note": note,
            },
        }

    def retrieve_candidates(self, query: str) -> list[Candidate]:
        self._last_anchor_injected_count = 0
        self._last_anchor_retained_count = 0
        self._last_anchor_injected_app_ids = set()
        query = (query or "").strip()
        if not query:
            return []
        query_tokens = self._expand_query_tags(query)
        reverse_query_tokens = self._expand_reverse_query_tags(query, query_tokens)
        prompt_tokens = _ordered_query_tokens(query)
        required_genre_clusters, required_modalities, preferred_perspectives = self._interpret_query_intent(
            query,
            query_tokens,
        )

        if self.using_cached_index and self.index is not None:
            query_embedding = self.embed_model.encode(
                [query],
                normalize_embeddings=False,
                convert_to_numpy=True,
            ).astype("float32", copy=False)

            norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            norm[norm == 0.0] = 1.0
            query_embedding = query_embedding / norm

            top_k = min(FAISS_CANDIDATE_POOL, len(self.index_app_ids))
            scores, indices = self.index.search(query_embedding, top_k)

            candidates: list[Candidate] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.index_app_ids):
                    continue
                app_id = self.index_app_ids[int(idx)]
                record = self.records_by_app_id.get(app_id)
                if not record:
                    continue
                perspective_match = _pre_scoring_intent_match(
                    record.raw,
                    required_genre_clusters,
                    required_modalities,
                    preferred_perspectives,
                )
                tag_hits = _tag_match_hits(query_tokens, record.raw)
                reverse_tag_hits = _tag_match_hits(reverse_query_tokens, record.raw)
                candidates.append(
                    Candidate(
                        record=record,
                        similarity=float(score),
                        popularity=_popularity_prior(record.raw),
                        review_volume=_review_volume_score(record.raw),
                        tag_match=_tag_match_score(query_tokens, prompt_tokens, record.raw),
                        tag_hit_count=tag_hits,
                        reverse_tag_hit_count=reverse_tag_hits,
                        perspective_match=perspective_match,
                        community_sentiment=_community_sentiment_score(record.raw),
                        review_summary=self.review_summaries.get(app_id, ""),
                    )
                )
            candidates = self._inject_anchor_candidates(
                query=query,
                candidates=candidates,
                query_tokens=query_tokens,
                reverse_query_tokens=reverse_query_tokens,
                prompt_tokens=prompt_tokens,
                required_genre_clusters=required_genre_clusters,
                required_modalities=required_modalities,
                preferred_perspectives=preferred_perspectives,
            )

            candidates.sort(
                key=lambda item: _blend_rank_signal(
                    item.similarity,
                    item.popularity,
                    item.review_volume,
                    item.tag_match,
                    item.tag_hit_count,
                    item.reverse_tag_hit_count,
                    item.perspective_match,
                ),
                reverse=True,
            )
            return self._apply_anchor_retention_quota(candidates)

        return self._keyword_fallback(query)

    def _keyword_fallback(self, query: str) -> list[Candidate]:
        tokens = self._expand_query_tags(query)
        reverse_tokens = self._expand_reverse_query_tags(query, tokens)
        prompt_tokens = _ordered_query_tokens(query)
        required_genre_clusters, required_modalities, preferred_perspectives = self._interpret_query_intent(
            query,
            tokens,
        )
        scored: list[Candidate] = []

        for record in self.records_by_app_id.values():
            perspective_match = _pre_scoring_intent_match(
                record.raw,
                required_genre_clusters,
                required_modalities,
                preferred_perspectives,
            )
            haystack = " ".join(
                [
                    record.name,
                    record.short_description,
                    ", ".join(record.raw.get("genres", [])),
                    ", ".join(record.to_result(0.0).get("tags", [])),
                ]
            ).lower()
            overlap = sum(1 for token in tokens if token in haystack)
            if overlap <= 0:
                continue
            score = overlap / max(len(tokens), 1)
            tag_hits = _tag_match_hits(tokens, record.raw)
            reverse_tag_hits = _tag_match_hits(reverse_tokens, record.raw)
            scored.append(
                Candidate(
                    record=record,
                    similarity=score,
                    popularity=_popularity_prior(record.raw),
                    review_volume=_review_volume_score(record.raw),
                    tag_match=_tag_match_score(tokens, prompt_tokens, record.raw),
                    tag_hit_count=tag_hits,
                    reverse_tag_hit_count=reverse_tag_hits,
                    perspective_match=perspective_match,
                    community_sentiment=_community_sentiment_score(record.raw),
                    review_summary="",
                )
            )

        scored = self._inject_anchor_candidates(
            query=query,
            candidates=scored,
            query_tokens=tokens,
            reverse_query_tokens=reverse_tokens,
            prompt_tokens=prompt_tokens,
            required_genre_clusters=required_genre_clusters,
            required_modalities=required_modalities,
            preferred_perspectives=preferred_perspectives,
        )

        scored.sort(
            key=lambda item: _blend_rank_signal(
                item.similarity,
                item.popularity,
                item.review_volume,
                item.tag_match,
                item.tag_hit_count,
                item.reverse_tag_hit_count,
                item.perspective_match,
            ),
            reverse=True,
        )
        return self._apply_anchor_retention_quota(scored)

    def rank_candidates(self, query: str, candidates: list[Candidate]) -> list[RankedCandidate]:
        if not candidates:
            return []

        llm_ranked = self._llm_rerank(query, candidates)
        if llm_ranked:
            return llm_ranked

        return [
            RankedCandidate(
                record=item.record,
                score=_weighted_rank_score(
                    semantic_match=item.similarity,
                    gameplay_loop=item.similarity,
                    mechanic_friction=item.similarity,
                    player_fantasy=item.similarity,
                    tag_match=item.tag_match,
                    community_sentiment=item.community_sentiment,
                    popularity=item.popularity,
                ),
                review_summary=item.review_summary,
                reason="High semantic similarity to your request.",
            )
            for item in sorted(
                candidates,
                key=lambda c: _blend_rank_signal(
                    c.similarity,
                    c.popularity,
                    c.review_volume,
                    c.tag_match,
                    c.tag_hit_count,
                    c.reverse_tag_hit_count,
                    c.perspective_match,
                ),
                reverse=True,
            )
        ]

    def _llm_rerank(self, query: str, candidates: list[Candidate]) -> list[RankedCandidate] | None:
        compact_candidates = []
        for item in candidates[:25]:
            compact_candidates.append(
                {
                    "app_id": item.record.app_id,
                    "name": item.record.name,
                    "genres": item.record.raw.get("genres", []),
                    "tags": item.record.to_result(0.0).get("tags", []),
                    "description": item.record.short_description[:280],
                    "review_summary": item.review_summary[:220],
                    "similarity": round(item.similarity, 4),
                    "community_sentiment": round(item.community_sentiment, 4),
                    "popularity_prior": round(item.popularity, 4),
                    "review_volume": round(item.review_volume, 4),
                    "tag_match": round(item.tag_match, 4),
                    "tag_hit_count": item.tag_hit_count,
                    "reverse_tag_hit_count": item.reverse_tag_hit_count,
                    "perspective_match": round(item.perspective_match, 4),
                }
            )

        system_prompt = (
            "You rerank game recommendations. Return strict JSON only with this schema: "
            "{\"ranked\": [{\"app_id\": str, \"gameplay_loop_score\": float, \"mechanic_friction_score\": float, \"player_fantasy_score\": float, \"reason\": str}]}. "
            "Keep reason factual and grounded in provided metadata/reviews. "
            "When judging the subjective dimensions, optimize for intent interpretation, nuance, player expectation alignment, and natural-language understanding. "
            "Also consider the supplied popularity_prior as a supporting signal for broad appeal and proven player reception, without overwhelming query fit. "
            "Use the supplied weighted rubric exactly. "
            "You must only score the three subjective dimensions: gameplay loop similarity, mechanic density or friction fit, and player fantasy fulfillment. "
            "Treat semantic match, tags, community sentiment, and popularity as already-computed inputs supplied in the candidate data. "
            "Use review_summary as supporting evidence, not the sole signal."
        )

        user_prompt = json.dumps(
            {
                "query": query,
                "criteria": [
                    "intent interpretation",
                    "nuance",
                    "player expectation alignment",
                    "natural-language understanding",
                    "popularity as a supporting signal",
                ],
                "candidates": compact_candidates,
                "max_results": MATCH_COUNT,
            },
            ensure_ascii=True,
        )

        payload = self.ollama.chat_json(system_prompt, user_prompt)
        if not payload:
            return None

        ranked_items = payload.get("ranked")
        if not isinstance(ranked_items, list):
            return None

        candidate_map = {item.record.app_id: item for item in candidates}
        reranked: list[RankedCandidate] = []

        for row in ranked_items:
            if not isinstance(row, dict):
                continue
            app_id = str(row.get("app_id", "")).strip()
            if app_id not in candidate_map:
                continue

            source = candidate_map[app_id]
            gameplay_loop_score = row.get("gameplay_loop_score", source.similarity)
            mechanic_friction_score = row.get("mechanic_friction_score", source.similarity)
            player_fantasy_score = row.get("player_fantasy_score", source.similarity)
            try:
                gameplay_loop = float(gameplay_loop_score)
            except (TypeError, ValueError):
                gameplay_loop = source.similarity
            try:
                mechanic_friction = float(mechanic_friction_score)
            except (TypeError, ValueError):
                mechanic_friction = source.similarity
            try:
                player_fantasy = float(player_fantasy_score)
            except (TypeError, ValueError):
                player_fantasy = source.similarity

            # Clamp score to a stable range for frontend display.
            gameplay_loop = max(0.0, min(gameplay_loop, 1.0))
            mechanic_friction = max(0.0, min(mechanic_friction, 1.0))
            player_fantasy = max(0.0, min(player_fantasy, 1.0))
            llm_weighted_score = _weighted_rank_score(
                semantic_match=source.similarity,
                gameplay_loop=gameplay_loop,
                mechanic_friction=mechanic_friction,
                player_fantasy=player_fantasy,
                tag_match=source.tag_match,
                community_sentiment=source.community_sentiment,
                popularity=source.popularity,
            )
            base_signal = _blend_rank_signal(
                source.similarity,
                source.popularity,
                source.review_volume,
                source.tag_match,
                source.tag_hit_count,
                source.reverse_tag_hit_count,
                source.perspective_match,
            )
            score = _blend_llm_with_base_signal(llm_weighted_score, base_signal)
            reason = _clean_text(str(row.get("reason", "")))
            if not reason:
                reason = "Relevant to your requested style and gameplay preferences."

            reranked.append(
                RankedCandidate(
                    record=source.record,
                    score=score,
                    review_summary=source.review_summary,
                    reason=reason,
                )
            )

        if not reranked:
            return None

        # Include unseen candidates to avoid dropping valid retrieval hits.
        seen = {item.record.app_id for item in reranked}
        remaining = [item for item in candidates if item.record.app_id not in seen]
        remaining.sort(
            key=lambda item: _blend_rank_signal(
                item.similarity,
                item.popularity,
                item.review_volume,
                item.tag_match,
                item.tag_hit_count,
                item.reverse_tag_hit_count,
                item.perspective_match,
            ),
            reverse=True,
        )
        for item in remaining:
            reranked.append(
                RankedCandidate(
                    record=item.record,
                    score=_blend_rank_signal(
                        item.similarity,
                        item.popularity,
                        item.review_volume,
                        item.tag_match,
                        item.tag_hit_count,
                        item.reverse_tag_hit_count,
                        item.perspective_match,
                    ),
                    review_summary=item.review_summary,
                    reason="Relevant to your request based on semantic retrieval.",
                )
            )

        return reranked

    def generate_answer(self, query: str, ranked_candidates: list[RankedCandidate]) -> str:
        if not ranked_candidates:
            return "I could not find strong matches for that description. Try adding genres, mood, or gameplay details."

        top = ranked_candidates[:3]
        fallback = self._fallback_answer(query, top)

        context = [
            {
                "name": item.record.name,
                "genres": item.record.raw.get("genres", []),
                "description": item.record.short_description[:220],
                "review_summary": item.review_summary[:180],
                "reason": item.reason,
            }
            for item in top
        ]

        system_prompt = (
            "You write concise recommendation explanations grounded in supplied game metadata and reviews. "
            "Only mention games from top_matches, and mention them in the same order provided. "
            "Do not add or reference any other game titles. "
            "Do not invent mechanics or features not present in context. Keep it under 90 words."
        )
        user_prompt = json.dumps(
            {
                "query": query,
                "top_matches": context,
                "ordered_titles": [item["name"] for item in context],
                "style": "friendly, concise, specific",
            },
            ensure_ascii=True,
        )

        response = self.ollama.chat_text(system_prompt, user_prompt)
        if not response:
            return fallback

        cleaned = _clean_text(response)
        if len(cleaned) < 10:
            return fallback

        return cleaned

    def _fallback_answer(self, query: str, top_matches: list[RankedCandidate]) -> str:
        fragments = []
        for item in top_matches:
            genre_text = ", ".join(item.record.raw.get("genres", [])[:2])
            if genre_text:
                fragments.append(f"{item.record.name} ({genre_text})")
            else:
                fragments.append(item.record.name)

        games_text = ", ".join(fragments)
        return (
            f"For '{query}', the strongest matches are {games_text}. "
            "These picks were selected from semantic similarity, tags, and popularity signals."
        )
