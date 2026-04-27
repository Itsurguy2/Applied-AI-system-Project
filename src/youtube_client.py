"""
SoundMatch YouTube Client

Primary:  YouTube Data API v3 (set YOUTUBE_API_KEY env var)
Fallback: Curated video catalog — safe YouTube links organized by genre and production topic.

Add other platform integrations here as they become available:
  - SoundCloud: awaiting API re-opening (soundcloud.com/developers)
  - TikTok: requires business account approval (developers.tiktok.com)
  - Bandcamp / Pitchfork / EARMILK: no public API — link out only
"""

import json
import os
import urllib.parse
import urllib.request
from functools import lru_cache

_API_KEY  = os.environ.get("YOUTUBE_API_KEY", "")
_SEARCH   = "https://www.googleapis.com/youtube/v3/search"
_VIDEOS   = "https://www.googleapis.com/youtube/v3/videos"


def thumb(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"


def yt_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def has_api_key() -> bool:
    return bool(_API_KEY)


@lru_cache(maxsize=80)
def search_videos(query: str, max_results: int = 6) -> list:
    """Search YouTube via Data API v3. Returns [] if no key or on error."""
    if not _API_KEY:
        return []
    params = urllib.parse.urlencode({
        "part":           "snippet",
        "q":              query,
        "type":           "video",
        "maxResults":     max_results,
        "key":            _API_KEY,
        "safeSearch":     "moderate",
        "videoCategoryId": "10",
    })
    try:
        with urllib.request.urlopen(f"{_SEARCH}?{params}", timeout=6) as r:
            data = json.loads(r.read())
        return [
            {
                "id":      item["id"]["videoId"],
                "title":   item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "thumb":   thumb(item["id"]["videoId"]),
                "url":     yt_url(item["id"]["videoId"]),
            }
            for item in data.get("items", [])
            if item["id"].get("videoId")
        ]
    except Exception:
        return []


# ── Curated fallback data ─────────────────────────────────────────────────────
# Thumbnail URLs are constructed from the video ID — no API key needed.

def _v(vid_id, title, channel):
    return {"id": vid_id, "title": title, "channel": channel,
            "thumb": thumb(vid_id), "url": yt_url(vid_id)}


# Genre → artist/performance videos (shown in Home artist scroll)
GENRE_VIDEOS: dict = {
    "lofi":      [_v("jfKfPfyJRdk", "Lofi Hip Hop Radio 📚 Study Beats", "Lofi Girl"),
                  _v("5qap5aO4i9A", "Lofi Hip Hop Music - Beats to Study", "Lofi Girl")],
    "pop":       [_v("nfWlot6h_JM", "Taylor Swift - Shake It Off", "Taylor Swift"),
                  _v("kTJczUoc26U", "Billie Eilish - bad guy", "Billie Eilish")],
    "rock":      [_v("hTWKbfoikeg", "Nirvana - Smells Like Teen Spirit", "Nirvana"),
                  _v("1w7OgIMMRc4", "Guns N' Roses - Sweet Child O' Mine", "Guns N Roses")],
    "hip-hop":   [_v("tvTRZJ-4EyI", "Kendrick Lamar - HUMBLE.", "Kendrick Lamar"),
                  _v("YVkUvmDQ3HY", "J. Cole - No Role Modelz", "J. Cole")],
    "jazz":      [_v("S6fMJTGBqBo", "Miles Davis - So What (Live)", "Miles Davis"),
                  _v("omSf8xqKBNY", "John Coltrane - A Love Supreme", "John Coltrane")],
    "edm":       [_v("60ItHLz5WEA", "Martin Garrix - Animals (Official)", "Martin Garrix"),
                  _v("IcrbM1l_BoI", "Avicii - Wake Me Up (Official)", "Avicii")],
    "classical": [_v("_CTYymbbEL4", "Beethoven - Moonlight Sonata", "Classical Music"),
                  _v("90WD_ats6eE", "Mozart - Piano Concerto No. 21", "Classical Music")],
    "r&b":       [_v("RB-RcX5DS5A", "SZA - Kill Bill (Official)", "SZA"),
                  _v("CevxZvSJLk8", "H.E.R. - Focus (Live)", "H.E.R.")],
    "indie pop": [_v("u9Dg-g7t2l4", "Clairo - Pretty Girl (Official)", "Clairo"),
                  _v("CHek0-Lu8Og", "Phoebe Bridgers - Savior Complex", "Phoebe Bridgers")],
    "folk":      [_v("8UVNT4wvIGY", "Bon Iver - Skinny Love (Original)", "Bon Iver"),
                  _v("L0MK7qz13bU", "Iron & Wine - Naked As We Came", "Iron and Wine")],
    "metal":     [_v("E0ozmU9cJDg", "Metallica - Master of Puppets", "Metallica"),
                  _v("WM8bTdBs-cw", "System Of A Down - Chop Suey!", "System Of A Down")],
    "ambient":   [_v("hHW1oY26kxQ", "Brian Eno - Music For Airports", "Brian Eno"),
                  _v("UjkpbDnLnsU", "Ambient Music for Focus & Calm", "Ambient Worlds")],
    "synthwave": [_v("MV_3Dpw-BRY", "Kavinsky - Nightcall", "Kavinsky"),
                  _v("1dJMnHiYjGM", "Daft Punk - Get Lucky (Official)", "Daft Punk")],
    "reggae":    [_v("zaGUr64BsId", "Bob Marley - No Woman No Cry", "Bob Marley"),
                  _v("mwRn43cCJ4c", "Toots and the Maytals - Pressure Drop", "Toots")],
    "country":   [_v("avNHDnFzWyI", "Johnny Cash - Ring of Fire", "Johnny Cash"),
                  _v("jPMHDqVoFAc", "Dolly Parton - Jolene (Live)", "Dolly Parton")],
    "blues":     [_v("d8hIL2QDumE", "B.B. King - The Thrill Is Gone", "B.B. King"),
                  _v("R6PeFjbaMzk", "Muddy Waters - Hoochie Coochie Man", "Muddy Waters")],
    "k-pop":     [_v("gdZLi9oWNZg", "BTS - Dynamite (Official)", "HYBE LABELS"),
                  _v("9bZkp7q19f0", "PSY - GANGNAM STYLE (Official)", "officialpsy")],
}

# Indie / underground discovery channels (no genre filter)
DISCOVERY_CHANNELS = [
    {"name": "NPR Tiny Desk",    "url": "https://www.youtube.com/@nprmusic",      "query": "NPR Tiny Desk concert"},
    {"name": "COLORS",           "url": "https://www.youtube.com/@COLORSxSTUDIOS","query": "COLORS music session"},
    {"name": "Sofar Sounds",     "url": "https://www.youtube.com/@sofarsounds",   "query": "Sofar Sounds performance"},
    {"name": "KEXP",             "url": "https://www.youtube.com/@KEXP",          "query": "KEXP live performance"},
    {"name": "OnesToWatch",      "url": "https://www.youtube.com/@OnesToWatch",   "query": "Ones To Watch artist"},
    {"name": "Pitchfork Live",   "url": "https://www.youtube.com/@pitchfork",     "query": "Pitchfork live music"},
]

# Music production learning content
PRODUCTION_LESSONS = [
    {
        "category": "🥁 Beat Making",
        "search_query": "beat making tutorial for beginners 2024",
        "videos": [
            _v("FY3OtFtzGxY", "How To Make Beats For Beginners", "Busy Works Beats"),
            _v("ognIBSIxVXc", "Music Production Complete Guide", "In The Mix"),
            _v("d3x4JQAkAeU", "Beat Making From Scratch", "Produce Like A Pro"),
        ],
    },
    {
        "category": "🎚️ Mixing & Mastering",
        "search_query": "how to mix music tutorial beginners",
        "videos": [
            _v("TEjOdqZFvhY", "How to Mix Music — Beginner's Guide", "In The Mix"),
            _v("GFVhXMEOFKs", "EQ Explained — The Ultimate Guide", "FabFilter"),
            _v("H7dCLnimDOs", "Mastering Your Music at Home", "Produce Like A Pro"),
        ],
    },
    {
        "category": "🎹 Melody & Chords",
        "search_query": "music theory chord progressions producers tutorial",
        "videos": [
            _v("rgaTLrZGlk0", "Music Theory for Producers", "Andrew Huang"),
            _v("JiNKlhspdkg", "How to Write a Melody", "Berklee Online"),
            _v("Qiumzv0HTOY", "Chord Progressions That Work", "Adam Neely"),
        ],
    },
    {
        "category": "🎤 Recording Vocals",
        "search_query": "how to record vocals at home professional sound",
        "videos": [
            _v("V_SpF4QGPCM", "How to Record Vocals at Home", "Produce Like A Pro"),
            _v("6SseK7hUXTU", "Vocal Production Tips", "In The Mix"),
            _v("B9LRLKuhlHE", "Home Studio Recording Setup", "RecordingRevolution"),
        ],
    },
    {
        "category": "☕ Lo-Fi & Indie",
        "search_query": "how to make lofi hip hop music tutorial",
        "videos": [
            _v("oxJ8f6BGLPE", "How to Make Lo-Fi Hip Hop", "Andrew Huang"),
            _v("wStXmSJAloM", "Indie Production Walkthrough", "Produce Like A Pro"),
            _v("KmzHRPyBAmQ", "How to Release Music Independently", "Ari's Take"),
        ],
    },
    {
        "category": "🎧 Independent Artist Tips",
        "search_query": "independent artist music marketing tips 2024",
        "videos": [
            _v("I7HtQgQiRwA", "Music Marketing for Indie Artists", "Damien Keyes"),
            _v("KmzHRPyBAmQ", "Release Music Independently", "Ari's Take"),
            _v("q5G7tKv0Tbc", "Getting Placed on Playlists", "Music Gateway"),
        ],
    },
]


def get_genre_videos(genre: str) -> list:
    """Return curated videos for a genre, falling back to a YouTube search link."""
    return GENRE_VIDEOS.get(genre, GENRE_VIDEOS.get("lofi", []))


def get_lesson_videos(category_index: int) -> list:
    """Return videos for a production lesson category (API or curated fallback)."""
    lesson = PRODUCTION_LESSONS[category_index]
    if _API_KEY:
        live = search_videos(lesson["search_query"])
        return live if live else lesson["videos"]
    return lesson["videos"]
