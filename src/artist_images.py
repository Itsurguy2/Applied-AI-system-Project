"""
Artist image resolver for SoundMatch.

Maps each fictional catalog artist to a real artist in the same genre,
then fetches their photo from the Deezer public API (no key required).
Results are cached to data/artist_images.json so the API is only hit once.
"""

import json
import time
from pathlib import Path

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# ── Fictional artist → real artist (same genre / vibe) ───────────────────────
ARTIST_MAP: dict[str, str] = {
    "Neon Echo":      "The Weeknd",
    "LoRoom":         "Joji",
    "Voltline":       "Imagine Dragons",
    "Paper Lanterns": "Nujabes",
    "Max Pulse":      "Calvin Harris",
    "Orbit Bloom":    "Brian Eno",
    "Slow Stereo":    "Norah Jones",
    "Indigo Parade":  "Vampire Weekend",
    "Verse Theory":   "Kendrick Lamar",
    "Clara Voss":     "Yiruma",
    "Shatter Grid":   "Metallica",
    "Sable June":     "SZA",
    "Dusty Holloway": "Chris Stapleton",
    "Coral Drift":    "Bob Marley",
    "Ember & Ash":    "Bon Iver",
    "Zeta Drop":      "Marshmello",
    "Remy Cole":      "Gary Clark Jr.",
    "Starfield Unit": "BTS",
}

_CACHE_PATH = Path(__file__).parent.parent / "data" / "artist_images.json"

# Fallback: neutral dark square so the UI never breaks
FALLBACK_URL = "https://via.placeholder.com/300x300/181818/535353?text=♪"


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── Deezer fetch ──────────────────────────────────────────────────────────────
def _fetch_deezer(real_name: str) -> str:
    """Query Deezer public API for an artist photo URL (picture_xl preferred)."""
    if not _REQUESTS_OK:
        return FALLBACK_URL
    try:
        r = requests.get(
            "https://api.deezer.com/search/artist",
            params={"q": real_name, "limit": 1},
            timeout=6,
        )
        data = r.json().get("data", [])
        if data:
            item = data[0]
            return (
                item.get("picture_xl")
                or item.get("picture_big")
                or item.get("picture_medium")
                or FALLBACK_URL
            )
    except Exception:
        pass
    return FALLBACK_URL


# ── Public API ────────────────────────────────────────────────────────────────
def get_image(fictional_name: str) -> str:
    """Return a real artist photo URL for a fictional SoundMatch artist name."""
    real_name = ARTIST_MAP.get(fictional_name, fictional_name)
    cache = _load_cache()
    if real_name in cache:
        return cache[real_name]
    url = _fetch_deezer(real_name)
    cache[real_name] = url
    _save_cache(cache)
    return url


def preload_all(songs: list) -> dict[str, str]:
    """
    Fetch and cache images for every unique artist in the song list.
    Returns a dict: fictional_artist_name → image_url.
    Safe to call at app startup — skips artists already in cache.
    """
    cache = _load_cache()
    result: dict[str, str] = {}
    changed = False

    for song in songs:
        name = song.get("artist", "")
        if name in result:
            continue
        real = ARTIST_MAP.get(name, name)
        if real not in cache:
            cache[real] = _fetch_deezer(real)
            changed = True
            time.sleep(0.15)   # be polite to Deezer's free tier
        result[name] = cache[real]

    if changed:
        _save_cache(cache)

    return result