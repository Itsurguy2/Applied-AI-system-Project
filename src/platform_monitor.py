"""
SoundMatch Platform Monitor
===========================
Aggregates artist stats across music platforms and scores artists for
potential SoundMatch featuring.

Live integrations (set env vars before launching the app):
  LASTFM_API_KEY   → Last.fm  — listener counts, play counts
  YOUTUBE_API_KEY  → YouTube  — view counts, subscriber counts

Simulated (no public API available):
  SoundCloud  — API closed to new app registrations
  Bandcamp    — no public API
  TikTok      — business-approval API only
  Pitchfork / EARMILK / OnesToWatch — no API; scraping violates ToS
  Spotify     — OAuth required (planned integration)

Scoring formula  (SoundMatch Talent Score 0–100):
  monthly_listeners × 0.35
  total_plays       × 0.25
  total_followers   × 0.25
  engagement_rate   × 0.15   (plays ÷ followers, capped)
  Each term is log-normalised before weighting.
"""

import hashlib
import json
import math
import os
import urllib.parse
import urllib.request

_LASTFM_KEY = os.environ.get("LASTFM_API_KEY", "")
_YT_KEY     = os.environ.get("YOUTUBE_API_KEY", "")

# ── Platform metadata ─────────────────────────────────────────────────────────
PLATFORMS: dict = {
    "youtube":    {"name": "YouTube",    "color": "#ff4444", "live": bool(_YT_KEY),     "icon": "▶"},
    "lastfm":     {"name": "Last.fm",    "color": "#d51007", "live": bool(_LASTFM_KEY), "icon": "♫"},
    "soundcloud": {"name": "SoundCloud", "color": "#ff5500", "live": False,             "icon": "☁"},
    "bandcamp":   {"name": "Bandcamp",   "color": "#1da0c3", "live": False,             "icon": "🏷"},
    "tiktok":     {"name": "TikTok",     "color": "#888",    "live": False,             "icon": "♪"},
    "spotify":    {"name": "Spotify",    "color": "#1db954", "live": False,             "icon": "◎"},
}

# ── Deterministic simulation seed ─────────────────────────────────────────────
def _seed(artist: str, platform: str) -> float:
    """Stable pseudo-random 0–1 float from artist + platform string."""
    h = hashlib.md5(f"{artist}:{platform}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _lerp(lo: float, hi: float, t: float) -> int:
    return int(lo + (hi - lo) * t)


# ── Simulated platform stats ───────────────────────────────────────────────────
def _sim_stats(song: dict) -> dict:
    """
    Generate deterministic simulated stats from a song's audio features.
    Higher energy/danceability → bigger YouTube/TikTok numbers.
    Higher acousticness → stronger Bandcamp/Last.fm underground presence.
    Genre multipliers reflect real-world platform affinities.
    """
    a      = song["artist"]
    energy = song["energy"]
    dance  = song["danceability"]
    acous  = song["acousticness"]
    genre  = song["genre"].lower()

    genre_mult = {
        "pop": 1.8, "k-pop": 2.2, "edm": 2.0, "hip-hop": 1.9,
        "rock": 1.5, "r&b": 1.4, "synthwave": 1.2, "indie pop": 1.1,
        "reggae": 1.0, "country": 1.0, "folk": 0.9, "jazz": 0.8,
        "classical": 0.7, "blues": 0.7, "ambient": 0.75, "lofi": 1.3, "metal": 1.1,
    }.get(genre, 1.0)

    viral = (energy * 0.5 + dance * 0.5) * genre_mult
    indie = (acous  * 0.6 + (1 - energy) * 0.4)

    def s(plat): return _seed(a, plat)

    yt_views  = _lerp(5_000,  50_000_000, s("yt_v") * viral)
    yt_subs   = _lerp(200,    2_000_000,  s("yt_s") * viral * 0.7)
    fm_listen = _lerp(300,    1_500_000,  s("fm_l") * (indie * 0.4 + viral * 0.6))
    fm_plays  = _lerp(2_000,  12_000_000, s("fm_p") * (indie * 0.4 + viral * 0.6))
    sc_follow = _lerp(100,    300_000,    s("sc_f") * (indie * 0.5 + viral * 0.5))
    sc_plays  = _lerp(500,    5_000_000,  s("sc_p") * (indie * 0.5 + viral * 0.5))
    bc_fans   = _lerp(50,     80_000,     s("bc_f") * indie)
    tt_follow = _lerp(200,    8_000_000,  s("tt_f") * viral * (1.5 if dance > 0.75 else 1.0))
    sp_listen = _lerp(500,    3_000_000,  s("sp_l") * viral * 0.8)

    return {
        "youtube_views":       yt_views,
        "youtube_subscribers": yt_subs,
        "lastfm_listeners":    fm_listen,
        "lastfm_plays":        fm_plays,
        "soundcloud_followers":sc_follow,
        "soundcloud_plays":    sc_plays,
        "bandcamp_fans":       bc_fans,
        "tiktok_followers":    tt_follow,
        "spotify_listeners":   sp_listen,
        # Aggregates used for scoring
        "monthly_listeners":   fm_listen + sp_listen,
        "total_plays":         yt_views + fm_plays + sc_plays,
        "total_followers":     yt_subs + sc_follow + tt_follow + bc_fans,
    }


# ── Live: Last.fm ─────────────────────────────────────────────────────────────
def _fetch_lastfm(artist: str) -> dict:
    if not _LASTFM_KEY:
        return {}
    params = urllib.parse.urlencode({
        "method": "artist.getinfo", "artist": artist,
        "api_key": _LASTFM_KEY, "format": "json",
    })
    try:
        with urllib.request.urlopen(
            f"https://ws.audioscrobbler.com/2.0/?{params}", timeout=5
        ) as r:
            data = json.loads(r.read())
        art = data.get("artist", {})
        stats = art.get("stats", {})
        return {
            "lastfm_listeners": int(stats.get("listeners", 0)),
            "lastfm_plays":     int(stats.get("playcount", 0)),
        }
    except Exception:
        return {}


# ── Live: YouTube channel search ──────────────────────────────────────────────
def _fetch_youtube_channel(artist: str) -> dict:
    if not _YT_KEY:
        return {}
    # Search for the artist's channel
    params = urllib.parse.urlencode({
        "part": "snippet", "q": artist, "type": "channel",
        "maxResults": 1, "key": _YT_KEY,
    })
    try:
        with urllib.request.urlopen(
            f"https://www.googleapis.com/youtube/v3/search?{params}", timeout=5
        ) as r:
            data = json.loads(r.read())
        items = data.get("items", [])
        if not items:
            return {}
        channel_id = items[0]["id"]["channelId"]
        # Get channel stats
        params2 = urllib.parse.urlencode({
            "part": "statistics", "id": channel_id, "key": _YT_KEY,
        })
        with urllib.request.urlopen(
            f"https://www.googleapis.com/youtube/v3/channels?{params2}", timeout=5
        ) as r:
            data2 = json.loads(r.read())
        stats = data2["items"][0]["statistics"]
        return {
            "youtube_views":       int(stats.get("viewCount", 0)),
            "youtube_subscribers": int(stats.get("subscriberCount", 0)),
        }
    except Exception:
        return {}


# ── Scoring ────────────────────────────────────────────────────────────────────
def _log_norm(x: float, scale: float) -> float:
    return math.log10(max(x, 1)) / math.log10(scale + 1)


def compute_score(stats: dict) -> float:
    """
    SoundMatch Talent Score (0–100).
    Combines reach (monthly listeners), history (total plays),
    commitment (total followers), and quality (engagement rate).
    All terms are log-normalised to handle the wide dynamic range.
    """
    monthly    = stats.get("monthly_listeners", 0)
    plays      = stats.get("total_plays",       0)
    followers  = max(stats.get("total_followers", 1), 1)
    engagement = min(plays / followers, 200)   # cap at 200× ratio

    raw = (
        _log_norm(monthly,    2_000_000)  * 0.35 +
        _log_norm(plays,     20_000_000)  * 0.25 +
        _log_norm(followers,  5_000_000)  * 0.25 +
        min(engagement / 200, 1.0)        * 0.15
    )
    return round(min(raw * 100, 100), 1)


# ── Public API ─────────────────────────────────────────────────────────────────
def get_artist_stats(songs: list) -> list:
    """
    Build a stats record for every unique artist in the catalog.
    Returns a list of dicts sorted by SoundMatch score (desc).
    """
    # Deduplicate: keep the first song per artist (representative)
    seen: dict = {}
    for song in songs:
        if song["artist"] not in seen:
            seen[song["artist"]] = song

    records = []
    for artist, song in seen.items():
        stats = _sim_stats(song)

        # Overlay live data where available
        live_fm = _fetch_lastfm(artist)
        if live_fm:
            stats.update(live_fm)
            stats["monthly_listeners"] = (
                stats["lastfm_listeners"] + stats.get("spotify_listeners", 0)
            )
            stats["total_plays"] = (
                stats.get("youtube_views", 0) +
                stats["lastfm_plays"] +
                stats.get("soundcloud_plays", 0)
            )

        live_yt = _fetch_youtube_channel(artist)
        if live_yt:
            stats.update(live_yt)
            stats["total_followers"] = (
                stats["youtube_subscribers"] +
                stats.get("soundcloud_followers", 0) +
                stats.get("tiktok_followers", 0) +
                stats.get("bandcamp_fans", 0)
            )
            stats["total_plays"] = (
                stats["youtube_views"] +
                stats.get("lastfm_plays", 0) +
                stats.get("soundcloud_plays", 0)
            )

        stats["score"]    = compute_score(stats)
        stats["artist"]   = artist
        stats["genre"]    = song["genre"]
        stats["mood"]     = song["mood"]
        stats["top_song"] = song["title"]
        stats["live_sources"] = [
            p for p, d in {"lastfm": live_fm, "youtube": live_yt}.items() if d
        ]
        records.append(stats)

    return sorted(records, key=lambda r: r["score"], reverse=True)


def fmt(n: int) -> str:
    """Format a large integer as 1.2K / 3.4M / 7.8B."""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)
