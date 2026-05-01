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

# Music production learning content — updated 2025
PRODUCTION_LESSONS = [
    {
        "category": "🥁 Beat Making & Drum Programming",
        "search_query": "beat making drum programming tutorial 2025",
        "videos": [
            _v("FY3OtFtzGxY", "How To Make Beats For Beginners (Full Guide)", "Busy Works Beats"),
            _v("ognIBSIxVXc", "Complete Music Production Workflow", "In The Mix"),
            _v("d3x4JQAkAeU", "Beat Making From Scratch — No Experience Needed", "Produce Like A Pro"),
            _v("oxJ8f6BGLPE", "How to Make Lo-Fi Hip Hop Beats", "Andrew Huang"),
        ],
    },
    {
        "category": "🎚️ Mixing & Mastering",
        "search_query": "how to mix and master music at home 2025",
        "videos": [
            _v("TEjOdqZFvhY", "Mixing for Beginners — Full Walkthrough", "In The Mix"),
            _v("GFVhXMEOFKs", "EQ Explained — The Ultimate Producer Guide", "FabFilter"),
            _v("H7dCLnimDOs", "Mastering Your Music at Home (2025)", "Produce Like A Pro"),
            _v("Zv4PIHZsDMg", "Loudness, Compression & Limiting Explained", "In The Mix"),
        ],
    },
    {
        "category": "🎹 Melody, Chords & Music Theory",
        "search_query": "music theory chord progressions melody for producers 2025",
        "videos": [
            _v("rgaTLrZGlk0", "Music Theory for Producers — Essential Concepts", "Andrew Huang"),
            _v("JiNKlhspdkg", "How to Write a Memorable Melody", "Berklee Online"),
            _v("Qiumzv0HTOY", "Chord Progressions Every Producer Should Know", "Adam Neely"),
            _v("b-IOPB6pBpM", "Advanced Harmony for Modern Producers", "12tone"),
        ],
    },
    {
        "category": "🎤 Recording & Vocal Production",
        "search_query": "how to record and produce vocals at home studio 2025",
        "videos": [
            _v("V_SpF4QGPCM", "How to Record Professional Vocals at Home", "Produce Like A Pro"),
            _v("6SseK7hUXTU", "Vocal Processing & Tuning Tips", "In The Mix"),
            _v("B9LRLKuhlHE", "Home Studio Setup on Any Budget", "RecordingRevolution"),
            _v("u7cA0AvZbhc", "Auto-Tune vs Melodyne — When to Use Each", "Produce Like A Pro"),
        ],
    },
    {
        "category": "🔊 Sound Design & Synthesis",
        "search_query": "sound design synthesis tutorial for music producers 2025",
        "videos": [
            _v("sOO1PKPoxyE", "Sound Design From Scratch — Beginner to Pro", "In The Mix"),
            _v("IuV80JfSBKo", "How Synthesizers Work — Full Explanation", "Andrew Huang"),
            _v("Eiw9YnRe64E", "Serum Sound Design Masterclass", "Busy Works Beats"),
            _v("pf1bGhPAcSw", "Designing Bass Sounds for Electronic Music", "Produce Like A Pro"),
        ],
    },
    {
        "category": "🤖 AI Tools in Music Production",
        "search_query": "AI music production tools plugins workflow 2025",
        "videos": [
            _v("6L9R1UYHbPc", "How Producers Are Using AI Right Now", "Andrew Huang"),
            _v("T0_HmWIDe7Q", "iZotope Neutron AI Mixing — Full Walkthrough", "In The Mix"),
            _v("tqByJlK75rM", "AI Mastering vs. Human Mastering — Honest Review", "Produce Like A Pro"),
            _v("KmzHRPyBAmQ", "Protecting Your Music in the Age of AI", "Ari's Take"),
        ],
    },
    {
        "category": "☕ Lo-Fi, Indie & DIY Releases",
        "search_query": "lofi indie music production DIY release tips 2025",
        "videos": [
            _v("wStXmSJAloM", "Indie Production Walkthrough — Full Session", "Produce Like A Pro"),
            _v("5IDifyke-OE", "Making Lo-Fi Beats with Just a Laptop", "Andrew Huang"),
            _v("f4Mc-NYPHaQ", "How to Self-Release Your Music in 2025", "Ari's Take"),
            _v("q5G7tKv0Tbc", "Playlist Pitching Strategy for Independent Artists", "Music Gateway"),
        ],
    },
    {
        "category": "📈 Music Marketing & Growing an Audience",
        "search_query": "music marketing grow audience independent artist 2025",
        "videos": [
            _v("I7HtQgQiRwA", "Music Marketing Blueprint for Indie Artists", "Damien Keyes"),
            _v("v4JMxUbuoQA", "TikTok & Short-Form Video Strategy for Musicians", "Damien Keyes"),
            _v("c8r5C3MXMAE", "Building a Fanbase from Zero in 2025", "Ari's Take"),
            _v("ZdGqHaCJ4MY", "Getting Sync Licensing Deals — Step by Step", "Music Gateway"),
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
