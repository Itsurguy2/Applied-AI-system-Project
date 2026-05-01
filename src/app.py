"""
SoundMatch — AI-powered music discovery app.

Run from project root:
    streamlit run src/app.py
"""

import io
import os
import random
import sys
import wave
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Import recommender from sibling module ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from recommender import MAX_SCORE, SCORE_WEIGHTS, load_songs, recommend_songs, score_song

# ── Import chat agent (RAG + Agentic Workflow) ────────────────────────────────
try:
    from chat_agent import chat_with_history, intent_to_features, rag_retrieve
    _CHAT_OK = True
except ImportError:
    _CHAT_OK = False

# ── Import platform monitor ───────────────────────────────────────────────────
try:
    from platform_monitor import PLATFORMS, fmt, get_artist_stats
    _MON_OK = True
except ImportError:
    _MON_OK = False

# ── Import YouTube client ─────────────────────────────────────────────────────
try:
    from youtube_client import (
        DISCOVERY_CHANNELS, PRODUCTION_LESSONS,
        get_genre_videos, get_lesson_videos, has_api_key, search_videos,
    )
    _YT_OK = True
except ImportError:
    _YT_OK = False

# ── Import artist image resolver ──────────────────────────────────────────────
try:
    from artist_images import ARTIST_MAP, FALLBACK_URL, get_image, preload_all
    _IMG_OK = True
except ImportError:
    ARTIST_MAP = {}
    FALLBACK_URL = ""
    def get_image(name: str) -> str: return ""       # noqa: E704
    def preload_all(songs: list) -> dict: return {}  # noqa: E704
    _IMG_OK = False

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SoundMatch",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
CSV_PATH = _ROOT / "data" / "songs.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
ALPHA = 0.30  # EMA learning rate for taste profile updates


DEFAULT_PROFILE = {
    "genre": "pop",
    "mood": "happy",
    "target_energy": 0.65,
    "target_valence": 0.65,
    "target_acousticness": 0.40,
    "target_danceability": 0.65,
    "target_instrumentalness": 0.30,
    "target_speechiness": 0.05,
    "target_liveness": 0.12,
}

PAGES = ["Home", "Battles", "Discover", "My Taste DNA", "Chat", "Learn", "Monitor"]


# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ── App shell ─────────────────────────────────────────────────────── */
        .stApp {
            background: #0d0d12;
            background-image:
                radial-gradient(ellipse at 15% 0%, #2a0a4a18 0%, transparent 55%),
                radial-gradient(ellipse at 85% 100%, #0a1a3018 0%, transparent 55%);
            color: #ffffff;
        }

        [data-testid="stSidebar"] {
            background: #0d0d18 !important;
            border-right: 1px solid #1a1a28 !important;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
        }
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }
        h1, h2, h3 { color: #ffffff; font-weight: 700; }

        /* ── Section headers ────────────────────────────────────────────────── */
        .section-head {
            font-size: 1.25rem; font-weight: 800; color: #ffffff;
            margin: 32px 0 16px; letter-spacing: -0.02em;
            padding-left: 14px;
            position: relative;
        }
        .section-head::before {
            content: '';
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 3px;
            background: linear-gradient(180deg, #a855f7, #06b6d4);
            border-radius: 2px;
        }

        /* ── Generic card ───────────────────────────────────────────────────── */
        .sm-card { background: #141420; border-radius: 10px; padding: 18px 20px; margin-bottom: 8px; transition: background 0.18s, box-shadow 0.2s; border: 1px solid #1e1e2e; }
        .sm-card:hover { background: #1a1a2a; box-shadow: 0 4px 20px #a855f718; border-color: #2e1e4e; }

        /* ── Song rows ──────────────────────────────────────────────────────── */
        .song-row { display: flex; align-items: center; background: transparent; border-radius: 6px; padding: 8px 16px; margin-bottom: 2px; gap: 12px; transition: background 0.15s, box-shadow 0.15s; position: relative; }
        .song-row:hover { background: #1a1a2a; box-shadow: inset 3px 0 0 #a855f7; }
        .song-rank { font-size: 0.88rem; font-weight: 400; color: #b3b3b3; width: 22px; text-align: center; flex-shrink: 0; }
        .song-emoji-sm { font-size: 1.4rem; flex-shrink: 0; }
        .song-info { flex: 1; min-width: 0; }
        .song-name { font-size: 1.02rem; font-weight: 600; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .song-meta { font-size: 0.82rem; color: #b3b3b3; margin-top: 3px; }
        .song-badge { font-size: 0.58rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; flex-shrink: 0; }
        .score-bar-bg { width: 72px; background: #2a2a3a; border-radius: 2px; height: 4px; flex-shrink: 0; overflow: hidden; }
        .score-bar { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #a855f7, #06b6d4); }
        .score-val { font-size: 0.74rem; color: #b3b3b3; width: 32px; text-align: right; flex-shrink: 0; }

        /* ── Metric cards ───────────────────────────────────────────────────── */
        .metric-card { background: linear-gradient(135deg, #141428, #0f0f1a); border-radius: 10px; padding: 20px 18px; text-align: center; border: 1px solid #2a1a4a; }
        .metric-value { font-size: 1.9rem; font-weight: 900; background: linear-gradient(135deg, #ffffff, #d1a7ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.2; }
        .metric-label { font-size: 0.73rem; color: #888; margin-top: 6px; }

        /* ── Battle cards ───────────────────────────────────────────────────── */
        .battle-card { background: #0f0f1a; border-radius: 16px; padding: 24px 20px 20px; text-align: center; transition: background 0.2s, box-shadow 0.3s; min-height: 480px; border: 1px solid #1e1e2e; }
        .battle-card:hover { background: #141420; box-shadow: 0 12px 40px #a855f725, 0 2px 8px #00000060; }
        .battle-badge { font-size: 0.60rem; font-weight: 700; letter-spacing: 0.12em; padding: 3px 10px; border-radius: 3px; display: inline-block; margin-bottom: 14px; }
        .battle-emoji { font-size: 3rem; margin: 6px 0 10px; }
        .battle-title { font-size: 1.2rem; font-weight: 800; color: #ffffff; margin: 0 0 4px; line-height: 1.25; }
        .battle-artist { font-size: 0.84rem; color: #888; margin-bottom: 12px; }
        .battle-tags { margin-bottom: 12px; }
        .tag { display: inline-block; background: #1e1e2e; color: #888; font-size: 0.68rem; padding: 3px 10px; border-radius: 3px; margin: 2px 3px; border: 1px solid #2a2a3e; }
        .stat-row { display: flex; align-items: center; gap: 8px; margin: 5px 0; font-size: 0.72rem; color: #888; width: 100%; }
        .stat-label { width: 60px; text-align: right; flex-shrink: 0; }
        .stat-bar-bg { flex: 1; background: #1e1e2e; border-radius: 2px; height: 4px; overflow: hidden; }
        .stat-bar { height: 100%; border-radius: 2px; }
        .stat-val { width: 34px; text-align: left; flex-shrink: 0; }
        .match-badge { margin-top: 14px; font-size: 0.80rem; font-weight: 700; background: linear-gradient(90deg, #a855f7, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }

        /* ── VS divider ─────────────────────────────────────────────────────── */
        .vs-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px 0; height: 100%; min-height: 300px; }
        .vs-text { font-size: 2.4rem; font-weight: 900; background: linear-gradient(135deg, #ffffff, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .vs-line { width: 1px; height: 60px; background: linear-gradient(180deg, transparent, #a855f7, transparent); margin: 8px 0; }

        /* ── Battle / result headers ────────────────────────────────────────── */
        .battle-header { background: linear-gradient(135deg, #0f0a1a, #141428); border-radius: 12px; padding: 14px 22px; text-align: center; margin-bottom: 18px; border: 1px solid #2a1a4a; }
        .result-banner { background: linear-gradient(135deg, #0f0a1a, #1a1428); border-radius: 14px; padding: 28px 24px; text-align: center; margin: 12px 0 20px; border: 1px solid #2a1a4a; box-shadow: 0 8px 32px #a855f720; }

        /* ── Hero (hidden — replaced by featured banner) ─────────────────────── */
        .hero { display: none; }

        /* ── CTA card ───────────────────────────────────────────────────────── */
        .cta-card { background: linear-gradient(135deg, #141428, #0f0f1a); border-radius: 12px; padding: 22px 26px; text-align: center; border: 1px solid #2a1a4a; }

        /* ── Sidebar profile ────────────────────────────────────────────────── */
        .sidebar-profile { font-size: 0.76rem; color: #888; padding: 4px; line-height: 1.85; }

        /* ── Featured banner ────────────────────────────────────────────────── */
        .featured-banner { display: flex; gap: 0; border-radius: 16px; overflow: hidden; margin-bottom: 28px; min-height: 270px; background: linear-gradient(135deg, #0f0a1a 0%, #141428 60%, #0a1020 100%); border: 1px solid #2a1a4a; box-shadow: 0 8px 40px #00000070, 0 0 0 1px #a855f710; }
        .featured-content { flex: 1 1 55%; padding: 32px 36px; display: flex; flex-direction: column; justify-content: center; gap: 10px; z-index: 1; }
        .featured-tag { display: inline-block; font-size: 0.60rem; font-weight: 700; letter-spacing: 0.14em; background: linear-gradient(90deg, #a855f7, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; width: fit-content; }
        .featured-news-label { font-size: 0.66rem; font-weight: 600; letter-spacing: 0.08em; color: #666; text-transform: uppercase; }
        .featured-headline { font-size: 1.3rem; font-weight: 700; color: #ffffff; line-height: 1.25; }
        .featured-artist-name { font-size: 2.2rem; font-weight: 900; line-height: 1.1; background: linear-gradient(90deg, #ffffff, #e0c4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .featured-meta { display: flex; flex-wrap: wrap; gap: 8px; }
        .featured-chip { font-size: 0.66rem; font-weight: 500; background: #1e1e2e; color: #888; border-radius: 4px; padding: 4px 10px; border: 1px solid #2a2a3e; }
        .featured-song { font-size: 0.78rem; color: #888; }
        .featured-score-badge { display: inline-block; font-size: 0.66rem; font-weight: 700; background: linear-gradient(90deg, #a855f7, #7c3aed); border-radius: 500px; padding: 5px 16px; color: #ffffff; width: fit-content; box-shadow: 0 2px 12px #a855f750; }
        .featured-image { flex: 0 0 45%; position: relative; overflow: hidden; background: #1a1a2a; }
        .featured-image img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .featured-image-overlay { position: absolute; inset: 0; background: linear-gradient(90deg, #0f0a1a 0%, transparent 40%, transparent 70%, #0f0a1a40 100%); }
        .featured-image-fallback { width: 100%; height: 100%; min-height: 200px; display: flex; align-items: center; justify-content: center; font-size: 5rem; background: #1a1a2a; }

        /* ── Quick-access grid (Spotify-style 2-col) ────────────────────────── */
        .qa-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 32px; }
        .qa-tile { background: #141420; border-radius: 8px; display: flex; align-items: center; gap: 0; cursor: pointer; transition: background 0.15s, box-shadow 0.2s, transform 0.15s; overflow: hidden; height: 64px; border: 1px solid #1e1e2e; }
        .qa-tile:hover { background: #1a1a2a; box-shadow: 0 4px 16px #a855f720; transform: translateY(-1px); border-color: #3a1a5e; }
        .qa-icon { width: 64px; height: 64px; display: flex; align-items: center; justify-content: center; font-size: 1.9rem; flex-shrink: 0; border-radius: 8px 0 0 8px; overflow: hidden; }
        .qa-icon img { width: 64px; height: 64px; object-fit: cover; border-radius: 8px 0 0 8px; display: block; }
        .qa-icon-fb { width: 64px; height: 64px; display: none; align-items: center; justify-content: center; font-size: 1.9rem; background: #1e1e2e; border-radius: 8px 0 0 8px; }
        .qa-text { padding: 0 14px; overflow: hidden; }
        .qa-label { font-size: 0.97rem; font-weight: 700; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .qa-sub { font-size: 0.72rem; color: #888; margin-top: 2px; }

        /* ── Artist profile cards ───────────────────────────────────────────── */
        .artist-profile-grid { display: flex; gap: 14px; overflow-x: auto; padding: 4px 2px 18px; scrollbar-width: thin; scrollbar-color: #3a3a4a transparent; }
        .artist-profile-grid::-webkit-scrollbar { height: 4px; }
        .artist-profile-grid::-webkit-scrollbar-thumb { background: #3a3a4a; border-radius: 2px; }
        .ap-card { flex: 0 0 162px; background: #141420; border-radius: 10px; overflow: hidden; transition: background 0.18s, box-shadow 0.25s, transform 0.2s; cursor: pointer; border: 1px solid #1e1e2e; }
        .ap-card:hover { background: #1a1a2a; box-shadow: 0 8px 28px #a855f730; transform: translateY(-3px); border-color: #3a1a5e; }
        .ap-photo { width: 100%; aspect-ratio: 1/1; overflow: hidden; background: #1e1e2e; position: relative; }
        .ap-photo img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform 0.35s; }
        .ap-card:hover .ap-photo img { transform: scale(1.06); }
        .ap-photo-fallback { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-size: 3.5rem; }
        .ap-overlay { position: absolute; bottom: 0; left: 0; right: 0; height: 50%; background: linear-gradient(transparent, #141420cc); }
        .ap-body { padding: 12px 12px 14px; }
        .ap-name { font-size: 0.84rem; font-weight: 700; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 4px; }
        .ap-badges { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
        .ap-badge { font-size: 0.57rem; font-weight: 600; padding: 2px 6px; border-radius: 3px; background: #1e1e2e; color: #888; border: 1px solid #2a2a3e; }
        .ap-badge.mood { background: #1e1e2e; color: #888; }
        .ap-energy-track { background: #1e1e2e; border-radius: 2px; height: 3px; overflow: hidden; margin-bottom: 6px; }
        .ap-energy-fill { height: 3px; border-radius: 2px; background: linear-gradient(90deg, #a855f7, #06b6d4); }
        .ap-song { font-size: 0.61rem; color: #444; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

        /* ── Mood tiles ─────────────────────────────────────────────────────── */
        .mood-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 4px; }
        .mood-tile { position: relative; overflow: hidden; width: 154px; height: 94px; border-radius: 10px; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; flex-shrink: 0; border: 1px solid #ffffff10; }
        .mood-tile:hover { transform: scale(1.05); box-shadow: 0 10px 30px #00000080; border-color: #ffffff20; }
        .mood-bg-img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0.42; }
        .mood-color-overlay { position: absolute; inset: 0; }
        .mood-tile-body { position: relative; z-index: 1; padding: 12px 14px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; }
        .mood-tile-emoji { font-size: 1.15rem; }
        .mood-tile-name { font-size: 0.82rem; font-weight: 800; color: #ffffff; text-shadow: 0 1px 4px #00000060; }
        .mood-tile-count { font-size: 0.60rem; color: rgba(255,255,255,0.70); }

        /* ── Genre tiles ────────────────────────────────────────────────────── */
        .genre-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; }
        .genre-tile { padding: 14px 18px; border-radius: 10px; cursor: pointer; transition: background 0.15s, box-shadow 0.2s, transform 0.15s; min-width: 110px; text-align: center; background: #141420; border: 1px solid #1e1e2e; }
        .genre-tile:hover { background: #1a1a2a; box-shadow: 0 4px 16px #a855f720; transform: translateY(-2px); border-color: #2a1a4a; }
        .genre-tile-emoji { font-size: 1.5rem; display: block; margin-bottom: 5px; }
        .genre-tile-name { font-size: 0.72rem; font-weight: 700; color: #ffffff; }
        .genre-tile-count { font-size: 0.56rem; color: #888; margin-top: 2px; }

        /* ── Trending song mini-cards ───────────────────────────────────────── */
        .trending-grid { display: flex; gap: 12px; overflow-x: auto; padding: 4px 2px 16px; scrollbar-width: thin; scrollbar-color: #3a3a4a transparent; }
        .trending-grid::-webkit-scrollbar { height: 4px; }
        .trending-grid::-webkit-scrollbar-thumb { background: #3a3a4a; border-radius: 2px; }
        .t-card { flex: 0 0 152px; background: #141420; border-radius: 10px; overflow: hidden; transition: background 0.18s, box-shadow 0.22s, transform 0.2s; cursor: pointer; border: 1px solid #1e1e2e; }
        .t-card:hover { background: #1a1a2a; box-shadow: 0 8px 24px #00000070, 0 0 0 1px #3a1a5e; transform: translateY(-3px); }
        .t-card-img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; background: #1e1e2e; }
        .t-card-fallback { width: 100%; aspect-ratio: 1; display: flex; align-items: center; justify-content: center; font-size: 2.6rem; background: #1e1e2e; }
        .t-card-body { padding: 8px 10px 12px; }
        .t-card-title { font-size: 0.72rem; font-weight: 700; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .t-card-artist { font-size: 0.62rem; color: #888; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .t-card-badge { display: inline-block; margin-top: 5px; font-size: 0.55rem; font-weight: 700; padding: 2px 7px; border-radius: 3px; background: #a855f720; color: #c084fc; border: 1px solid #a855f730; }

        /* ── Legacy artist card ─────────────────────────────────────────────── */
        .artist-card { flex: 0 0 130px; background: #181818; border-radius: 8px; padding: 14px 10px 12px; text-align: center; transition: background 0.15s; cursor: pointer; }
        .artist-card:hover { background: #282828; }
        .artist-avatar { font-size: 2.4rem; margin-bottom: 8px; display: block; }
        .artist-name { font-size: 0.76rem; font-weight: 700; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .artist-genre { font-size: 0.63rem; color: #b3b3b3; margin-top: 3px; }
        .play-pill { display: inline-block; margin-top: 8px; background: #a855f7; border-radius: 500px; padding: 3px 14px; font-size: 0.68rem; color: #fff; font-weight: 700; }

        /* ── Horizontal scroll rows ─────────────────────────────────────────── */
        .scroll-row { display: flex; overflow-x: auto; gap: 12px; padding: 4px 2px 14px; scrollbar-width: thin; scrollbar-color: #535353 transparent; -webkit-overflow-scrolling: touch; }
        .scroll-row::-webkit-scrollbar { height: 4px; }
        .scroll-row::-webkit-scrollbar-thumb { background: #535353; border-radius: 2px; }

        /* ── Video cards ────────────────────────────────────────────────────── */
        .video-card { flex: 0 0 220px; background: #181818; border-radius: 8px; overflow: hidden; transition: background 0.15s; text-decoration: none; }
        .video-card:hover { background: #282828; }
        .video-thumb { width: 100%; height: 124px; object-fit: cover; background: #282828; display: block; }
        .video-thumb-placeholder { width: 100%; height: 124px; background: #282828; display: flex; align-items: center; justify-content: center; font-size: 2rem; }
        .video-info { padding: 10px 12px 14px; }
        .video-title { font-size: 0.76rem; font-weight: 600; color: #ffffff; line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
        .video-channel { font-size: 0.63rem; color: #b3b3b3; margin-top: 5px; }

        /* ── Discovery pills ────────────────────────────────────────────────── */
        .channel-pill { display: inline-flex; align-items: center; gap: 6px; background: #282828; border-radius: 500px; padding: 8px 18px; font-size: 0.76rem; font-weight: 600; color: #ffffff; text-decoration: none; transition: background 0.15s; white-space: nowrap; }
        .channel-pill:hover { background: #3e3e3e; }

        /* ── Learn page ─────────────────────────────────────────────────────── */
        .lesson-card { flex: 0 0 210px; background: #181818; border-radius: 8px; overflow: hidden; transition: background 0.15s; text-decoration: none; display: block; }
        .lesson-card:hover { background: #282828; }
        .lesson-thumb { width: 100%; height: 118px; object-fit: cover; background: #282828; display: block; }
        .lesson-info { padding: 10px 12px 14px; }
        .lesson-title { font-size: 0.74rem; font-weight: 600; color: #ffffff; line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
        .lesson-channel { font-size: 0.61rem; color: #b3b3b3; margin-top: 5px; }

        /* ── Monitor page ───────────────────────────────────────────────────── */
        .platform-status { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
        .plat-pill { display: inline-flex; align-items: center; gap: 6px; padding: 5px 14px; border-radius: 500px; font-size: 0.72rem; font-weight: 600; background: #141420; color: #888; border: 1px solid #1e1e2e; }
        .featured-artist { background: linear-gradient(160deg, #141420, #0f0f1a); border-radius: 12px; padding: 18px 16px; text-align: center; transition: background 0.18s, box-shadow 0.22s, transform 0.2s; border: 1px solid #1e1e2e; }
        .featured-artist:hover { background: linear-gradient(160deg, #1a1a2a, #141428); box-shadow: 0 8px 28px #a855f728; transform: translateY(-2px); border-color: #3a1a5e; }
        .featured-rank { font-size: 0.60rem; font-weight: 800; letter-spacing: 0.14em; background: linear-gradient(90deg, #a855f7, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 8px; }
        .featured-emoji { font-size: 2.2rem; margin-bottom: 8px; display: block; }
        .featured-name { font-size: 0.92rem; font-weight: 700; color: #ffffff; }
        .featured-genre { font-size: 0.68rem; color: #888; margin-top: 2px; }
        .featured-score { font-size: 1.6rem; font-weight: 900; margin: 10px 0 4px; background: linear-gradient(90deg, #a855f7, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .featured-score-label { font-size: 0.58rem; color: #444; letter-spacing: 0.08em; }
        .stat-chip { display: inline-block; background: #1a1a2a; border-radius: 4px; padding: 3px 8px; margin: 2px; font-size: 0.64rem; color: #888; border: 1px solid #2a2a3e; }
        .stat-chip span { color: #ffffff; font-weight: 700; }

        /* ── Sidebar AI assistant ───────────────────────────────────────────── */
        .sbai-header { font-size: 0.66rem; font-weight: 700; color: #555; letter-spacing: 0.10em; margin-bottom: 10px; text-transform: uppercase; }
        .sbai-bubble-user { background: #1e1e2e; border-radius: 6px; padding: 6px 10px; font-size: 0.68rem; color: #ffffff; margin-bottom: 4px; text-align: right; line-height: 1.45; }
        .sbai-bubble-ai { background: #141420; border-radius: 6px; padding: 6px 10px; font-size: 0.68rem; color: #888; margin-bottom: 4px; line-height: 1.45; border-left: 2px solid #a855f740; }
        .sbai-ai-label { font-size: 0.54rem; background: linear-gradient(90deg, #a855f7, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700; margin-bottom: 2px; }
        .sbai-no-key { font-size: 0.65rem; color: #444; line-height: 1.55; padding: 8px 0; }

        /* ── Chat page ──────────────────────────────────────────────────────── */
        .chat-hero { background: linear-gradient(135deg, #0f0a1a 0%, #141428 55%, #0a1020 100%); border-radius: 16px; padding: 30px 36px; margin-bottom: 24px; display: flex; align-items: center; gap: 22px; border: 1px solid #2a1a4a; box-shadow: 0 8px 40px #a855f715; }
        .chat-hero-icon { font-size: 1.6rem; font-weight: 900; background: linear-gradient(135deg, #a855f7, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; padding: 10px 14px; border-radius: 12px; border: 1px solid #a855f730; }
        .chat-hero-title { font-size: 1.5rem; font-weight: 900; background: linear-gradient(90deg, #ffffff, #d1a7ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .chat-hero-sub { font-size: 0.84rem; color: #888; margin-top: 6px; line-height: 1.6; }
        .chat-hero-badge { font-size: 0.58rem; font-weight: 700; letter-spacing: 0.10em; background: linear-gradient(90deg, #a855f720, #06b6d420); color: #a855f7; border: 1px solid #a855f750; border-radius: 4px; padding: 3px 10px; display: inline-block; margin-top: 8px; }
        .chat-info-box { background: linear-gradient(135deg, #141428, #0f0f1a); border-radius: 14px; padding: 22px 26px; margin-bottom: 18px; border: 1px solid #1e1e2e; }
        .chat-example-btn { display: inline-block; background: #1e1e2e; border-radius: 6px; padding: 7px 14px; font-size: 0.80rem; color: #ffffff; margin: 4px; cursor: pointer; border: 1px solid #2a2a3e; }
        .tool-badge { display: inline-block; background: #a855f715; color: #c084fc; font-size: 0.68rem; font-weight: 700; font-family: monospace; padding: 2px 8px; border-radius: 3px; margin-right: 4px; border: 1px solid #a855f730; }
        .rag-badge { display: inline-block; background: #06b6d415; color: #67e8f9; font-size: 0.68rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; border: 1px solid #06b6d430; }

        /* ── Streamlit component overrides ──────────────────────────────────── */
        .stButton > button {
            background: linear-gradient(90deg, #a855f7, #7c3aed) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 500px !important;
            font-weight: 700 !important;
            font-size: 0.86rem !important;
            padding: 10px 24px !important;
            transition: transform 0.1s, opacity 0.15s, box-shadow 0.2s !important;
            letter-spacing: 0.02em !important;
            box-shadow: 0 2px 12px #a855f740 !important;
        }

        .stButton > button:hover {
            opacity: 0.88 !important;
            transform: scale(1.03) !important;
            box-shadow: 0 4px 20px #a855f760 !important;
        }
        .stProgress > div > div { background: linear-gradient(90deg, #a855f7, #06b6d4) !important; }

        /* ── Greeting ───────────────────────────────────────────────────────── */
        .greeting-time { font-size: 0.70rem; font-weight: 700; color: #666; letter-spacing: 0.10em; text-transform: uppercase; margin-bottom: 8px; }
        .greeting-title { font-size: 2.8rem; font-weight: 900; background: linear-gradient(90deg, #ffffff 0%, #e0c4ff 50%, #7dd3fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.08; margin-bottom: 10px; }
        .greeting-accent { color: #a855f7; }
        .greeting-sub { font-size: 0.90rem; color: #888; margin-bottom: 22px; }

        /* ── Made-For-You playlists ─────────────────────────────────────────── */
        .mfy-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 4px; }
        .mfy-tile { background: #141420; border-radius: 10px; overflow: hidden; cursor: pointer; transition: background 0.18s, border-color 0.2s, box-shadow 0.22s, transform 0.2s; border: 1px solid #1e1e2e; }
        .mfy-tile:hover { background: #1a1a2a; border-color: #4a1a7a; box-shadow: 0 6px 24px #a855f725; transform: translateY(-2px); }
        .mfy-cover { width: 100%; aspect-ratio: 1; display: grid; grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(2, 1fr); overflow: hidden; background: #1e1e2e; gap: 1px; }
        .mfy-cover img { width: 100%; height: 100%; object-fit: cover; display: block; filter: brightness(0.90) saturate(1.15); transition: filter 0.25s; }
        .mfy-tile:hover .mfy-cover img { filter: brightness(1.05) saturate(1.25); }
        .mfy-cover-fallback { display: flex; align-items: center; justify-content: center; font-size: 2.5rem; background: #1e1e2e; }
        .mfy-body { padding: 10px 12px 14px; }
        .mfy-title { font-size: 0.88rem; font-weight: 800; color: #ffffff; margin-bottom: 3px; letter-spacing: -0.01em; }
        .mfy-desc { font-size: 0.68rem; color: #888; line-height: 1.4; }
        .mfy-count { font-size: 0.59rem; color: #444; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.04em; }

        /* ── Liked songs shelf ──────────────────────────────────────────────── */
        .liked-shelf { display: flex; gap: 8px; overflow-x: auto; padding: 4px 2px 14px; scrollbar-width: thin; scrollbar-color: #535353 transparent; align-items: center; }
        .liked-shelf::-webkit-scrollbar { height: 4px; }
        .liked-shelf::-webkit-scrollbar-thumb { background: #535353; border-radius: 2px; }
        .liked-item { flex: 0 0 60px; background: #282828; border-radius: 6px; overflow: hidden; cursor: pointer; transition: background 0.15s; }
        .liked-item:hover { background: #3e3e3e; }
        .liked-item-img { width: 60px; height: 60px; object-fit: cover; display: block; }
        .liked-item-fallback { width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; background: #282828; }
        .liked-empty { font-size: 0.80rem; color: #535353; padding: 12px 0; }

        /* ── Battle streak / progress card ──────────────────────────────────── */
        .streak-card { background: linear-gradient(135deg, #0f0a1a, #141428); border-radius: 14px; padding: 20px 24px; display: grid; grid-template-columns: auto 1fr; gap: 20px; align-items: center; border: 1px solid #2a1a4a; box-shadow: 0 4px 20px #a855f715; }
        .streak-big { font-size: 3.4rem; font-weight: 900; background: linear-gradient(135deg, #ffffff, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1; }
        .streak-label { font-size: 0.64rem; font-weight: 600; color: #555; text-transform: uppercase; letter-spacing: 0.10em; margin-top: 2px; }
        .streak-level { font-size: 0.88rem; font-weight: 700; background: linear-gradient(90deg, #a855f7, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 8px; }
        .level-track { background: #1e1e2e; border-radius: 2px; height: 5px; overflow: hidden; }
        .level-fill { height: 5px; border-radius: 2px; background: linear-gradient(90deg, #a855f7, #06b6d4); transition: width 0.5s; }
        .level-meta { display: flex; justify-content: space-between; font-size: 0.60rem; color: #555; margin-top: 4px; }

        /* ── Fresh Finds ────────────────────────────────────────────────────── */
        .fresh-grid { display: flex; gap: 14px; overflow-x: auto; padding: 4px 2px 18px; scrollbar-width: thin; scrollbar-color: #3a3a4a transparent; }
        .fresh-grid::-webkit-scrollbar { height: 4px; }
        .fresh-grid::-webkit-scrollbar-thumb { background: #3a3a4a; border-radius: 2px; }
        .fresh-card { flex: 0 0 165px; background: #141420; border-radius: 10px; overflow: hidden; cursor: pointer; transition: background 0.18s, box-shadow 0.22s, transform 0.2s; border: 1px solid #1e1e2e; }
        .fresh-card:hover { background: #1a1a2a; box-shadow: 0 8px 28px #a855f728; transform: translateY(-3px); border-color: #3a1a5e; }
        .fresh-img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; background: #1e1e2e; }
        .fresh-fallback { width: 100%; aspect-ratio: 1; display: flex; align-items: center; justify-content: center; font-size: 3rem; background: #1e1e2e; }
        .fresh-body { padding: 10px 12px 14px; }
        .fresh-title { font-size: 0.80rem; font-weight: 700; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .fresh-artist { font-size: 0.66rem; color: #888; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .fresh-new-pill { display: inline-block; margin-top: 6px; font-size: 0.55rem; font-weight: 700; padding: 2px 8px; border-radius: 3px; background: linear-gradient(90deg, #a855f7, #7c3aed); color: #ffffff; letter-spacing: 0.06em; box-shadow: 0 1px 6px #a855f750; }

        /* ── Because You Love row ───────────────────────────────────────────── */
        .byl-row { display: flex; gap: 8px; overflow-x: auto; padding: 4px 2px 14px; scrollbar-width: thin; scrollbar-color: #3a3a4a transparent; }
        .byl-row::-webkit-scrollbar { height: 4px; }
        .byl-row::-webkit-scrollbar-thumb { background: #3a3a4a; border-radius: 2px; }
        .byl-card { flex: 0 0 auto; background: #141420; border-radius: 8px; padding: 10px 14px; display: flex; align-items: center; gap: 12px; cursor: pointer; transition: background 0.18s, box-shadow 0.2s; min-width: 220px; border: 1px solid #1e1e2e; }
        .byl-card:hover { background: #1a1a2a; box-shadow: 0 4px 16px #a855f720; border-color: #2e1e4e; }
        .byl-img { width: 48px; height: 48px; object-fit: cover; border-radius: 6px; flex-shrink: 0; background: #1e1e2e; }
        .byl-fallback { width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; border-radius: 6px; background: #1e1e2e; flex-shrink: 0; }
        .byl-info { overflow: hidden; }
        .byl-title { font-size: 0.80rem; font-weight: 600; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .byl-sub { font-size: 0.64rem; color: #888; margin-top: 2px; }

        /* ── Sidebar navigation ─────────────────────────────────────────────── */
        [data-testid="stSidebar"] .stRadio > div { gap: 2px; }
        [data-testid="stSidebar"] .stRadio label {
            font-size: 0.88rem !important; font-weight: 500 !important;
            color: #666 !important; padding: 8px 14px !important;
            border-radius: 6px !important;
            transition: background 0.14s, color 0.14s, box-shadow 0.14s !important;
        }
        [data-testid="stSidebar"] .stRadio label:hover { background: #1a1a28 !important; color: #ffffff !important; }
        [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked + div + div { color: #ffffff !important; font-weight: 700 !important; }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def get_songs() -> list:
    return load_songs(str(CSV_PATH))


@st.cache_data(show_spinner=False)
def _fetch_artist_images(songs_key: tuple) -> dict:
    """Preload real artist photos once and cache for the session."""
    return preload_all(get_songs())


def _aimg(artist: str, size: int = 48, radius: int = 6, css: str = "") -> str:
    """Return an <img> tag for a real artist photo, with emoji initials fallback."""
    imgs = st.session_state.get("artist_images", {})
    url  = imgs.get(artist, FALLBACK_URL)
    init = artist[:2].upper()
    style = f"width:{size}px;height:{size}px;object-fit:cover;border-radius:{radius}px;flex-shrink:0;{css}"
    fb_style = (
        f"width:{size}px;height:{size}px;border-radius:{radius}px;flex-shrink:0;"
        f"background:#282828;display:none;align-items:center;justify-content:center;"
        f"font-size:{max(size//3,10)}px;color:#b3b3b3;font-weight:700;"
    )
    return (
        f'<img src="{url}" alt="{artist}" style="{style}" '
        f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';">'
        f'<div style="{fb_style}">{init}</div>'
    )


# ── Synthetic audio preview ───────────────────────────────────────────────────
@st.cache_data
def make_preview(tempo_bpm: float, energy: float, valence: float) -> bytes:
    """Generate a 5-second synthetic tone whose character reflects the song's audio features."""
    sr = 22050
    dur = 5
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    base_hz = 130.0 + (tempo_bpm - 60) * 1.8
    third = 1.26 if valence > 0.5 else 1.19  # major vs minor third

    sig = (
        energy * 0.45 * np.sin(2 * np.pi * base_hz * t)
        + 0.20 * np.sin(2 * np.pi * base_hz * third * t)
        + 0.12 * np.sin(2 * np.pi * base_hz * 2.0 * t)
        + 0.05 * np.sin(2 * np.pi * base_hz * 3.0 * t)
    )
    bps = tempo_bpm / 60.0
    sig += 0.07 * energy * np.abs(np.sin(np.pi * bps * t)) ** 4

    fade = int(sr * 0.25)
    env = np.ones_like(sig)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    sig *= env

    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.74

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((sig * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


# ── Helpers ───────────────────────────────────────────────────────────────────
def vibe_score(song: dict) -> float:
    return song["energy"] * 0.5 + song["danceability"] * 0.3 + (1 - song["acousticness"]) * 0.2


def classify(song: dict) -> str:
    return "mainstream" if vibe_score(song) > 0.55 else "indie"


def tastemaker_title(n: int) -> str:
    if n < 3:   return "Listener"
    if n < 8:   return "Explorer"
    if n < 15:  return "Tastemaker"
    if n < 25:  return "Trendsetter"
    return "Music Oracle"


def init_state() -> None:
    defaults: dict = {
        "user_profile": dict(DEFAULT_PROFILE),
        "battle_history": [],
        "liked_songs": set(),
        "current_battle": None,
        "battle_result": None,
        "page": "Home",
        "chat_history": [],
        "sidebar_chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def update_profile(chosen: dict) -> None:
    p = st.session_state.user_profile
    a = ALPHA
    for feat, key in [
        ("energy",           "target_energy"),
        ("valence",          "target_valence"),
        ("acousticness",     "target_acousticness"),
        ("danceability",     "target_danceability"),
        ("instrumentalness", "target_instrumentalness"),
        ("speechiness",      "target_speechiness"),
        ("liveness",         "target_liveness"),
    ]:
        p[key] = round(a * chosen[feat] + (1 - a) * p[key], 4)

    recent = st.session_state.battle_history[-5:]
    if recent:
        genres = [b["chosen"]["genre"] for b in recent]
        moods  = [b["chosen"]["mood"]  for b in recent]
        p["genre"] = max(set(genres), key=genres.count)
        p["mood"]  = max(set(moods),  key=moods.count)

    st.session_state.user_profile = p


def fresh_battle(songs: list) -> None:
    mainstream = [s for s in songs if classify(s) == "mainstream"]
    indie      = [s for s in songs if classify(s) == "indie"]
    recent_ids = (
        {b["song1"]["id"] for b in st.session_state.battle_history[-6:]}
        | {b["song2"]["id"] for b in st.session_state.battle_history[-6:]}
    )
    pool_m = [s for s in mainstream if s["id"] not in recent_ids] or mainstream
    pool_i = [s for s in indie      if s["id"] not in recent_ids] or indie
    s1, s2 = random.choice(pool_m), random.choice(pool_i)
    if random.random() < 0.5:
        s1, s2 = s2, s1
    st.session_state.current_battle = {"song1": s1, "song2": s2}
    st.session_state.battle_result = None


# ── HTML builders ─────────────────────────────────────────────────────────────
def song_row_html(rank: int, song: dict, score: float) -> str:
    pct = int((score / MAX_SCORE) * 100)
    is_main = classify(song) == "mainstream"
    badge_col = "#a855f7" if is_main else "#06b6d4"
    badge_txt = "MAINSTREAM" if is_main else "INDIE"
    photo = _aimg(song["artist"], size=40, radius=4)
    return f"""
    <div class="song-row">
        <div class="song-rank">#{rank}</div>
        <div style="display:flex;flex-shrink:0;">{photo}</div>
        <div class="song-info">
            <div class="song-name">{song['title']}</div>
            <div class="song-meta">{song['artist']} &middot; {song['genre']} &middot; {song['mood'].title()}</div>
        </div>
        <span class="song-badge"
              style="color:{badge_col};border:1px solid {badge_col}40;background:{badge_col}12;">
            {badge_txt}
        </span>
        <div class="score-bar-bg">
            <div class="score-bar" style="width:{pct}%;"></div>
        </div>
        <div class="score-val">{pct}%</div>
    </div>"""


def battle_card_html(song: dict, accent: str, label: str) -> str:
    sc, _ = score_song(song, st.session_state.user_profile)
    match_pct = int((sc / MAX_SCORE) * 100)

    def bar(val: float) -> str:
        return (
            f'<div class="stat-bar-bg">'
            f'<div class="stat-bar" style="width:{int(val*100)}%;background:{accent};"></div>'
            f'</div>'
        )

    badge_bg = f"{accent}18"
    photo = _aimg(song["artist"], size=140, radius=10,
                  css=f"border:2px solid {accent}60;margin:10px auto 8px;display:block;")
    return f"""
    <div class="battle-card" style="border-color:{accent}50;">
        <span class="battle-badge"
              style="background:{badge_bg};color:{accent};border:1px solid {accent}40;">
            {label}
        </span>
        <div style="display:flex;justify-content:center;">{photo}</div>
        <div class="battle-title">{song['title']}</div>
        <div class="battle-artist">by {song['artist']}</div>
        <div class="battle-tags">
            <span class="tag">{song['genre']}</span>
            <span class="tag">{song['mood'].title()}</span>
            <span class="tag">{int(song['tempo_bpm'])} BPM</span>
        </div>
        <div style="width:100%;margin:6px 0 0;">
            <div class="stat-row">
                <span class="stat-label">Energy</span>{bar(song['energy'])}<span class="stat-val">{song['energy']:.0%}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Vibe</span>{bar(song['valence'])}<span class="stat-val">{song['valence']:.0%}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Dance</span>{bar(song['danceability'])}<span class="stat-val">{song['danceability']:.0%}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Acoustic</span>{bar(song['acousticness'])}<span class="stat-val">{song['acousticness']:.0%}</span>
            </div>
        </div>
        <div class="match-badge">{match_pct}% match for you</div>
    </div>"""


# ── Radar chart ───────────────────────────────────────────────────────────────
def taste_radar(profile: dict) -> go.Figure:
    labels = ["Energy", "Positivity", "Acoustic", "Danceability",
              "Instrumental", "Speechiness", "Liveness"]
    keys   = ["target_energy", "target_valence", "target_acousticness",
              "target_danceability", "target_instrumentalness",
              "target_speechiness", "target_liveness"]
    vals = [profile[k] for k in keys]
    vals_c  = vals + [vals[0]]
    lbls_c  = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_c, theta=lbls_c,
        fill="toself",
        fillcolor="rgba(168,85,247,0.14)",
        line=dict(color="#a855f7", width=2.5),
        marker=dict(size=6, color="#a855f7"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0e0e1a",
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75],
                ticktext=["25%", "50%", "75%"],
                color="#333", gridcolor="#1e1e2e",
                tickfont=dict(color="#555", size=9),
            ),
            angularaxis=dict(
                color="#888", gridcolor="#1a1a2a",
                tickfont=dict(color="#bbb", size=11),
            ),
        ),
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font=dict(color="#ccc", family="Inter"),
        margin=dict(l=44, r=44, t=30, b=30),
        showlegend=False,
        height=350,
    )
    return fig


# ── Home page helper functions ────────────────────────────────────────────────

_MOOD_COLORS = {
    "happy": "#f59e0b", "chill": "#06b6d4", "intense": "#ef4444",
    "focused": "#a855f7", "relaxed": "#22c55e", "melancholic": "#6366f1",
    "romantic": "#f43f5e", "aggressive": "#dc2626", "euphoric": "#fbbf24",
    "moody": "#7c3aed", "nostalgic": "#d97706", "energetic": "#f97316",
    "dreamy": "#8b5cf6", "sad": "#3b82f6",
}

_GENRE_COLORS = {
    "lofi": "#8b5cf6", "pop": "#ec4899", "rock": "#ef4444", "hip-hop": "#f59e0b",
    "jazz": "#3b82f6", "classical": "#6366f1", "metal": "#dc2626", "edm": "#06b6d4",
    "ambient": "#0ea5e9", "synthwave": "#a855f7", "indie pop": "#f472b6",
    "r&b": "#c084fc", "reggae": "#22c55e", "folk": "#d97706", "country": "#78716c",
    "blues": "#1d4ed8", "k-pop": "#db2777",
}


def _featured_banner_html(songs: list, profile: dict) -> str:
    """Large top banner spotlighting the #1 recommended artist with news-style copy."""
    results = recommend_songs(profile, songs, k=1)
    if not results:
        return ""
    top_song, top_score, _ = results[0]

    genre   = top_song["genre"]
    artist  = top_song["artist"]
    energy  = top_song["energy"]
    energy_label = "High Energy" if energy > 0.7 else ("Mellow" if energy < 0.4 else "Mid-Tempo")

    imgs      = st.session_state.get("artist_images", {})
    thumb_url = imgs.get(artist, FALLBACK_URL)

    headlines = [
        "Your #1 Match Is Here",
        f"Trending in {genre.title()} Right Now",
        "SoundMatch Picked This Just For You",
        "Today's Top Recommended Artist",
    ]
    headline = headlines[abs(hash(artist)) % len(headlines)]

    img_html = (
        f'<img src="{thumb_url}" alt="{artist}" '
        f'style="width:100%;height:100%;object-fit:cover;display:block;" '
        f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';">'
        f'<div class="featured-image-fallback" style="display:none;">{artist[:2].upper()}</div>'
    )

    return f"""
    <div class="featured-banner">
        <div class="featured-content">
            <div class="featured-tag">SOUNDMATCH DAILY</div>
            <div class="featured-news-label">Featured Artist</div>
            <div class="featured-headline">{headline}</div>
            <div class="featured-artist-name">{artist}</div>
            <div class="featured-meta">
                <span class="featured-chip">{genre.title()}</span>
                <span class="featured-chip">{top_song['mood'].title()}</span>
                <span class="featured-chip">{energy_label}</span>
            </div>
            <div class="featured-song">{top_song['title']}</div>
            <div class="featured-score-badge">
                SoundMatch Score &nbsp; {top_score:.1f} / {MAX_SCORE:.1f}
            </div>
        </div>
        <div class="featured-image">
            {img_html}
            <div class="featured-image-overlay"></div>
        </div>
    </div>"""


def _quick_access_html(n_liked: int, n_battles: int, genre: str, mood: str,
                        songs: list) -> str:
    """6-tile quick-jump grid — each tile shows a real artist photo."""
    imgs = st.session_state.get("artist_images", {})

    def _pick(sort_key, reverse=True) -> str:
        """Pick the artist from songs that scores best on sort_key."""
        s = sorted(songs, key=sort_key, reverse=reverse)
        return imgs.get(s[0]["artist"], FALLBACK_URL) if s else FALLBACK_URL

    def _pick_genre(g: str) -> str:
        pool = [s for s in songs if s["genre"] == g]
        if not pool:
            pool = songs
        return imgs.get(pool[0]["artist"], FALLBACK_URL)

    def _pick_mood(m: str) -> str:
        pool = [s for s in songs if s["mood"] == m]
        if not pool:
            pool = songs
        return imgs.get(pool[0]["artist"], FALLBACK_URL)

    # One representative photo per tile
    photos = [
        _pick(lambda s: s["energy"] * 0.4 + s["valence"] * 0.6),   # top picks vibe
        _pick(lambda s: s["valence"] * 0.5 + s["danceability"] * 0.5),  # liked / happy
        _pick(lambda s: s["energy"]),                                 # battle = highest energy
        _pick(lambda s: s["acousticness"]),                           # discover = most organic
        _pick_genre(genre),                                           # user's top genre
        _pick_mood(mood),                                             # user's current mood
    ]

    def _img_html(url: str, fb: str) -> str:
        return (
            f'<img src="{url}" alt="" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';">'
            f'<div class="qa-icon-fb">{fb}</div>'
        )

    items = [
        (_img_html(photos[0], ""),   "Top Picks",           "Your AI matches",            "#a855f7"),
        (_img_html(photos[1], ""),   f"{n_liked} Liked",    "Saved songs",                "#f43f5e"),
        (_img_html(photos[2], ""),   "Battle Arena",        f"{n_battles} battles fought","#f59e0b"),
        (_img_html(photos[3], ""),   "Discover",            "Filter by genre & mood",     "#22c55e"),
        (_img_html(photos[4], ""),   genre.title(),         "Your top genre",             "#06b6d4"),
        (_img_html(photos[5], ""),   mood.title(),          "Your current vibe",          "#8b5cf6"),
    ]
    tiles = ""
    for img_tag, label, sub, color in items:
        tiles += (
            f'<div class="qa-tile" '
            f'style="border-color:{color}30;background:linear-gradient(135deg,{color}15,#13131e);">'
            f'<div class="qa-icon">{img_tag}</div>'
            f'<div class="qa-text">'
            f'<div class="qa-label">{label}</div>'
            f'<div class="qa-sub">{sub}</div>'
            f'</div></div>'
        )
    return f'<div class="qa-grid">{tiles}</div>'


def _mood_tiles_html(songs: list) -> str:
    """Clickable mood exploration tiles with background art."""
    mood_map: dict = {}
    for song in songs:
        mood_map.setdefault(song["mood"], []).append(song)

    tiles = ""
    for mood, mood_songs in sorted(mood_map.items()):
        color  = _MOOD_COLORS.get(mood, "#a855f7")
        count  = len(mood_songs)
        imgs   = st.session_state.get("artist_images", {})
        thumb  = imgs.get(mood_songs[0]["artist"], "")
        bg_img = (
            f'<img class="mood-bg-img" src="{thumb}" alt="" '
            f'onerror="this.style.display=\'none\';">'
        ) if thumb else ""
        tiles += (
            f'<div class="mood-tile">'
            f'{bg_img}'
            f'<div class="mood-color-overlay" style="background:linear-gradient(135deg,{color}55,#0d0d18dd);"></div>'
            f'<div class="mood-tile-body">'
            f'<div><div class="mood-tile-name">{mood.title()}</div>'
            f'<div class="mood-tile-count">{count} song{"s" if count > 1 else ""}</div></div>'
            f'</div></div>'
        )
    return f'<div class="mood-grid">{tiles}</div>'


def _genre_tiles_html(songs: list) -> str:
    """Genre exploration tiles."""
    genre_map: dict = {}
    for song in songs:
        genre_map.setdefault(song["genre"], 0)
        genre_map[song["genre"]] += 1

    tiles = ""
    for genre, count in sorted(genre_map.items()):
        color = _GENRE_COLORS.get(genre, "#a855f7")
        tiles += (
            f'<div class="genre-tile" '
            f'style="border-color:{color}40;background:linear-gradient(135deg,{color}20,#13131e);">'
            f'<div class="genre-tile-name">{genre.title()}</div>'
            f'<div class="genre-tile-count">{count} song{"s" if count > 1 else ""}</div>'
            f'</div>'
        )
    return f'<div class="genre-grid">{tiles}</div>'


def _trending_scroll_html(songs: list) -> str:
    """Horizontal scroll of all catalog songs as mini discovery cards."""
    import random as _rnd
    shuffled = list(songs)
    _rnd.seed(42)
    _rnd.shuffle(shuffled)

    cards = ""
    for song in shuffled:
        img_html = _aimg(song["artist"], size=130, radius=6,
                         css="width:130px;height:130px;object-fit:cover;")

        cards += (
            f'<div class="t-card">'
            f'{img_html}'
            f'<div class="t-card-body">'
            f'<div class="t-card-title">{song["title"]}</div>'
            f'<div class="t-card-artist">{song["artist"]}</div>'
            f'<span class="t-card-badge">{song["mood"].title()}</span>'
            f'</div></div>'
        )
    return f'<div class="trending-grid">{cards}</div>'


# ── Pages ─────────────────────────────────────────────────────────────────────

def _artist_profile_cards_html(songs: list) -> str:
    """Horizontal scroll of artist profile cards with real artist photos."""
    seen: set = set()
    cards = ""
    for song in songs:
        if song["artist"] in seen:
            continue
        seen.add(song["artist"])

        genre  = song["genre"]
        energy = int(song["energy"] * 100)
        photo_inner = _aimg(song["artist"], size=180, radius=0,
                            css="width:100%;height:100%;object-fit:cover;display:block;")

        cards += f"""
        <div class="ap-card">
            <div class="ap-photo">
                {photo_inner}
                <div class="ap-overlay"></div>
            </div>
            <div class="ap-body">
                <div class="ap-name">{song['artist']}</div>
                <div class="ap-badges">
                    <span class="ap-badge">{genre.title()}</span>
                    <span class="ap-badge mood">{song['mood'].title()}</span>
                </div>
                <div class="ap-energy-track">
                    <div class="ap-energy-fill" style="width:{energy}%;"></div>
                </div>
                <div class="ap-song">{song['title']}</div>
            </div>
        </div>"""
    return f'<div class="artist-profile-grid">{cards}</div>'




def _video_scroll_html(videos: list, fallback_label: str = "Video") -> str:
    """Build a horizontal video card scroll from a list of video dicts."""
    cards = ""
    for v in videos:
        thumb_html = (
            f'<img class="video-thumb" src="{v["thumb"]}" alt="{v["title"]}" '
            f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\';">'
            f'<div class="video-thumb-placeholder" style="display:none;">{fallback_label}</div>'
        )
        cards += f"""
        <a class="video-card" href="{v['url']}" target="_blank" rel="noopener">
            {thumb_html}
            <div class="video-info">
                <div class="video-title">{v['title']}</div>
                <div class="video-channel">{v['channel']}</div>
            </div>
        </a>"""
    return f'<div class="scroll-row">{cards}</div>'


# ── Curated playlist definitions ──────────────────────────────────────────────
_CURATED_PLAYLISTS = [
    {
        "name": "Late Night Study",      "emoji": "",
        "desc": "Low energy, instrumental — perfect for deep focus",
        "color": "#6366f1",
        "filter": lambda s: s["energy"] < 0.45 and s["instrumentalness"] > 0.30,
    },
    {
        "name": "Morning Energy",         "emoji": "",
        "desc": "High energy picks to start your day strong",
        "color": "#f59e0b",
        "filter": lambda s: s["energy"] > 0.70 and s["valence"] > 0.55,
    },
    {
        "name": "Dance Floor",            "emoji": "",
        "desc": "Groove-heavy beats made for moving",
        "color": "#ec4899",
        "filter": lambda s: s["danceability"] > 0.72,
    },
    {
        "name": "Acoustic Hour",          "emoji": "",
        "desc": "Warm, organic sounds stripped back to the bone",
        "color": "#d97706",
        "filter": lambda s: s["acousticness"] > 0.60,
    },
    {
        "name": "Indie Underground",      "emoji": "",
        "desc": "Hidden gems far outside the mainstream",
        "color": "#8b5cf6",
        "filter": lambda s: s["genre"] in {
            "indie pop", "folk", "blues", "jazz", "ambient", "synthwave", "classical",
        },
    },
    {
        "name": "Hype Mode",              "emoji": "",
        "desc": "Maximum intensity — no brakes",
        "color": "#ef4444",
        "filter": lambda s: s["energy"] > 0.82,
    },
]

_LEVEL_THRESHOLDS = [
    (0,   "Newcomer",   5),
    (5,   "Listener",  10),
    (10,  "Fan",        20),
    (20,  "Devotee",    35),
    (35,  "Enthusiast", 55),
    (55,  "Connoisseur",80),
    (80,  "Expert",    110),
    (110, "Tastemaker", 150),
    (150, "Legend",    999),
]


def _greeting_html(profile: dict, n_battles: int) -> str:
    from datetime import datetime
    hour = datetime.now().hour
    if hour < 5:
        period = "Late Night"
    elif hour < 12:
        period = "Good Morning"
    elif hour < 17:
        period = "Good Afternoon"
    elif hour < 21:
        period = "Good Evening"
    else:
        period = "Good Night"

    title = tastemaker_title(n_battles)
    return f"""
    <div>
        <div class="greeting-time">{period}</div>
        <div class="greeting-title">
            Welcome back, <span class="greeting-accent">{title}</span>
        </div>
        <div class="greeting-sub">
            Your current vibe is <strong>{profile['genre'].title()}</strong>
            &nbsp;·&nbsp; {n_battles} battles fought
            &nbsp;·&nbsp; taste profile active
        </div>
    </div>"""


def _curated_playlists_html(songs: list) -> str:
    """Build the Made For You 3-column playlist grid."""
    tiles = ""
    for pl in _CURATED_PLAYLISTS:
        matched = [s for s in songs if pl["filter"](s)]
        if not matched:
            matched = songs[:4]
        count = len(matched)

        # Build 2×2 artist-photo mosaic cover
        thumbs = ""
        imgs = st.session_state.get("artist_images", {})
        for s in matched[:4]:
            url = imgs.get(s["artist"], FALLBACK_URL)
            thumbs += (
                f'<img src="{url}" alt="{s["artist"]}" '
                f'onerror="this.style.background=\'{pl["color"]}33\';">'
            )

        tiles += f"""
        <div class="mfy-tile">
            <div class="mfy-cover">
                {thumbs}
            </div>
            <div class="mfy-body">
                <div class="mfy-title">{pl['emoji']} {pl['name']}</div>
                <div class="mfy-desc">{pl['desc']}</div>
                <div class="mfy-count">{count} tracks</div>
            </div>
        </div>"""
    return f'<div class="mfy-grid">{tiles}</div>'


def _liked_shelf_html(songs: list, liked_ids: set) -> str:
    """Horizontal shelf of liked songs (square thumbnails)."""
    if not liked_ids:
        return '<div class="liked-empty">Like songs in Battles to see them here</div>'
    liked_songs = [s for s in songs if s.get("id") in liked_ids]
    items = ""
    for s in liked_songs:
        img = _aimg(s["artist"], size=80, radius=6,
                    css=f'title="{s["title"]} — {s["artist"]}"')
        items += f'<div class="liked-item">{img}</div>'
    return f'<div class="liked-shelf">{items}</div>'


def _streak_card_html(n_battles: int) -> str:
    """Battle progress card with level + progress bar."""
    for lo, name, hi in _LEVEL_THRESHOLDS:
        if n_battles < hi:
            level_name  = name
            level_lo    = lo
            level_hi    = hi
            break
    else:
        level_name, level_lo, level_hi = "Legend", 150, 999

    span    = level_hi - level_lo
    done    = n_battles - level_lo
    pct     = int(min(done / max(span, 1), 1.0) * 100)
    needed  = max(level_hi - n_battles, 0)
    next_lv = _LEVEL_THRESHOLDS[
        min(_LEVEL_THRESHOLDS.index((level_lo, level_name, level_hi)) + 1,
            len(_LEVEL_THRESHOLDS) - 1)
    ][1] if n_battles < 150 else "Legend"

    return f"""
    <div class="streak-card">
        <div>
            <div class="streak-big">{n_battles}</div>
            <div class="streak-label">Battles</div>
        </div>
        <div>
            <div class="streak-level">{level_name}</div>
            <div class="level-track">
                <div class="level-fill" style="width:{pct}%;"></div>
            </div>
            <div class="level-meta">
                <span>{level_lo} battles</span>
                <span>{needed} to {next_lv}</span>
                <span>{level_hi}</span>
            </div>
        </div>
    </div>"""


def _fresh_finds_html(songs: list, profile: dict, battle_history: list) -> str:
    """Songs the user has NEVER encountered in a battle = true discoveries."""
    seen_ids = {
        s["id"]
        for b in battle_history
        for s in (b.get("song1", {}), b.get("song2", {}))
        if isinstance(s, dict) and "id" in s
    }
    fresh = [s for s in songs if s.get("id") not in seen_ids]
    if not fresh:
        fresh = sorted(songs, key=lambda s: score_song(s, profile)[0], reverse=True)[:8]

    fresh_sorted = sorted(fresh, key=lambda s: score_song(s, profile)[0], reverse=True)
    cards = ""
    for s in fresh_sorted[:10]:
        img    = _aimg(s["artist"], size=72, radius=6)
        cards += f"""
        <div class="fresh-card">
            {img}
            <div class="fresh-body">
                <div class="fresh-title">{s['title']}</div>
                <div class="fresh-artist">{s['artist']}</div>
                <span class="fresh-new-pill">NEW</span>
                <span style="font-size:0.60rem;color:#535353;margin-left:4px;">{s['mood'].title()}</span>
            </div>
        </div>"""
    return f'<div class="fresh-grid">{cards}</div>'


def _because_you_love_html(songs: list, profile: dict) -> str:
    """Songs matching the user's top genre, sorted by score."""
    genre = profile.get("genre", "pop")
    matched = [s for s in songs if s["genre"] == genre]
    if not matched:
        matched = songs
    matched_sorted = sorted(matched, key=lambda s: score_song(s, profile)[0], reverse=True)

    cards = ""
    for s in matched_sorted[:8]:
        img   = _aimg(s["artist"], size=48, radius=4)
        sc, _ = score_song(s, profile)
        pct   = int((sc / MAX_SCORE) * 100)
        cards += f"""
        <div class="byl-card">
            {img}
            <div class="byl-info">
                <div class="byl-title">{s['title']}</div>
                <div class="byl-sub">{s['artist']} &nbsp;·&nbsp; {pct}% match</div>
            </div>
        </div>"""
    return f'<div class="byl-row">{cards}</div>'


def page_home(songs: list) -> None:
    profile   = st.session_state.user_profile
    n_battles = len(st.session_state.battle_history)
    n_liked   = len(st.session_state.liked_songs)

    # ════════════════════════════════════════════════════════════════════════════
    # 0 · GREETING — time-aware welcome with taste level badge
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown(_greeting_html(profile, n_battles), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 1 · FEATURED BANNER — top-of-page spotlight with news copy
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown(_featured_banner_html(songs, profile), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 2 · QUICK ACCESS — 6-tile nav grid (like Spotify's top shortcuts)
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-head">Quick Access</div>', unsafe_allow_html=True)
    st.markdown(
        _quick_access_html(n_liked, n_battles, profile["genre"], profile["mood"], songs),
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════════════════
    # 2b · JUMP BACK IN — liked songs shelf
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-head">Jump Back In</div>', unsafe_allow_html=True)
    st.markdown(
        _liked_shelf_html(songs, st.session_state.liked_songs),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 3 · ARTISTS IN CATALOG — photo cards with genre, mood, energy
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-head">Artists in Your Catalog</div>', unsafe_allow_html=True)
    st.markdown(_artist_profile_cards_html(songs), unsafe_allow_html=True)

    # Audio preview selectbox lives just below the artist row
    artist_names = list({s["artist"]: s for s in songs}.keys())
    chosen_artist = st.selectbox(
        "▶  Select an artist to hear a preview",
        artist_names,
        label_visibility="visible",
    )
    artist_song = next(s for s in songs if s["artist"] == chosen_artist)
    st.audio(
        make_preview(artist_song["tempo_bpm"], artist_song["energy"], artist_song["valence"]),
        format="audio/wav",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 3b · MADE FOR YOU — curated playlists built from catalog
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-head">Made For You</div>', unsafe_allow_html=True)
    st.markdown(_curated_playlists_html(songs), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 4 · TOP PICKS FOR YOU — AI-ranked recommendations
    # ════════════════════════════════════════════════════════════════════════════
    results = recommend_songs(profile, songs, k=5)
    st.markdown('<div class="section-head">Top Picks For You</div>', unsafe_allow_html=True)
    rows = "".join(song_row_html(r, s, sc) for r, (s, sc, _) in enumerate(results, 1))
    st.markdown(rows, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 4b · FRESH FINDS — unheard songs matched to taste profile
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-head">Fresh Finds</div>', unsafe_allow_html=True)
    st.markdown(
        _fresh_finds_html(songs, profile, st.session_state.battle_history),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 5 · TRENDING THIS WEEK — shuffled song discovery cards
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-head">Trending This Week</div>', unsafe_allow_html=True)
    st.markdown(_trending_scroll_html(songs), unsafe_allow_html=True)
    st.caption("Browse the full catalog — go to Battles to shape your taste")

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 6 · EXPLORE BY MOOD — colourful mood tiles
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-head">Explore by Mood</div>', unsafe_allow_html=True)
    st.markdown(_mood_tiles_html(songs), unsafe_allow_html=True)
    st.caption("Head to Discover → filter by mood to dive deeper")

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 7 · EXPLORE BY GENRE — genre tile grid
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-head">Explore by Genre</div>', unsafe_allow_html=True)
    st.markdown(_genre_tiles_html(songs), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 7b · BECAUSE YOU LOVE — genre-matched picks row
    # ════════════════════════════════════════════════════════════════════════════
    genre_label = profile["genre"].title()
    st.markdown(
        f'<div class="section-head">Because You Love {genre_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_because_you_love_html(songs, profile), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 8 · ARTIST VIDEOS — genre-matched YouTube scroll
    # ════════════════════════════════════════════════════════════════════════════
    if _YT_OK:
        st.markdown('<div class="section-head">Artist Videos</div>', unsafe_allow_html=True)
        videos = get_genre_videos(profile["genre"]) or get_genre_videos("pop")
        st.markdown(
            _video_scroll_html(videos, profile["genre"].title()),
            unsafe_allow_html=True,
        )
        st.caption(f"Showing {profile['genre']} videos · click any card to watch on YouTube")
        st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 9 · INDIE DISCOVERY CHANNELS
    # ════════════════════════════════════════════════════════════════════════════
    if _YT_OK:
        st.markdown('<div class="section-head">Indie Discovery Channels</div>',
                    unsafe_allow_html=True)
        pills = "".join(
            f'<a class="channel-pill" href="{ch["url"]}" target="_blank" rel="noopener">'
            f'▶ {ch["name"]}</a>'
            for ch in DISCOVERY_CHANNELS
        )
        st.markdown(
            f'<div class="scroll-row" style="flex-wrap:wrap;overflow-x:visible;">{pills}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 9b · BATTLE PROGRESS — level card with XP bar
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-head">Battle Progress</div>', unsafe_allow_html=True)
    st.markdown(_streak_card_html(n_battles), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 10 · YOUR STATS — compact metrics row
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-head">Your Stats</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{n_battles}</div>'
                    f'<div class="metric-label">Battles Fought</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{n_liked}</div>'
                    f'<div class="metric-label">Songs Liked</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{profile["genre"].title()}</div>'
                    f'<div class="metric-label">Top Genre</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{profile["mood"].title()}</div>'
                    f'<div class="metric-label">Current Vibe</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 11 · BATTLE CTA
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="cta-card">
        <div style="font-size:1.45rem;font-weight:900;color:#f0f0ff;margin-bottom:8px;">
            Ready to train your taste?
        </div>
        <div style="color:#667;font-size:0.88rem;line-height:1.55;">
            SoundMatch Battles give your AI 20× cleaner signals than passive listening.
            Pick a side — mainstream vs indie — and watch your recommendations sharpen instantly.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start a Battle Now", use_container_width=True):
        st.session_state.page = "Battles"
        st.rerun()


def page_battles(songs: list) -> None:
    n = len(st.session_state.battle_history)

    st.markdown(f"""
    <div class="battle-header">
        <div style="font-size:1.55rem;font-weight:900;
                    background:linear-gradient(135deg,#a855f7,#06b6d4);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">
            SoundMatch Battles
        </div>
        <div style="color:#556;font-size:0.82rem;margin-top:4px;">
            {tastemaker_title(n)} &nbsp;·&nbsp; {n} battles fought &nbsp;·&nbsp;
            Every pick sharpens your AI
        </div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.current_battle is None:
        fresh_battle(songs)

    result = st.session_state.battle_result

    # ── Result screen ──────────────────────────────────────────────────────────
    if result:
        chosen  = result["chosen"]
        skipped = result["skipped"]
        st.markdown(f"""
        <div class="result-banner">
            <div style="font-size:1.25rem;font-weight:800;color:#f0f0ff;">
                You chose: {chosen['title']}
            </div>
            <div style="color:#778;font-size:0.84rem;margin-top:4px;">
                by {chosen['artist']} &nbsp;·&nbsp; {chosen['genre']} &nbsp;·&nbsp;
                {chosen['mood'].title()}
            </div>
            <div style="margin-top:16px;font-size:0.84rem;color:#a855f7;font-weight:600;">
                Taste profile updated &mdash; energy {chosen['energy']:.0%},
                vibe {chosen['valence']:.0%}, dance {chosen['danceability']:.0%}
            </div>
        </div>""", unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            if st.button("Next Battle", use_container_width=True):
                st.session_state.current_battle = None
                st.session_state.battle_result  = None
                st.rerun()
        with cb:
            if st.button("See My Updated Picks", use_container_width=True):
                st.session_state.current_battle = None
                st.session_state.battle_result  = None
                st.session_state.page = "Home"
                st.rerun()

        if st.session_state.battle_history:
            st.markdown('<div class="section-head">Recent Battles</div>',
                        unsafe_allow_html=True)
            for b in reversed(st.session_state.battle_history[-5:]):
                c = b["chosen"]
                sk = b["skipped"]
                st.markdown(f"""
                <div class="sm-card" style="padding:10px 16px;">
                    <span style="color:#a855f7;font-weight:600;">&#10003; {c['title']}</span>
                    <span style="color:#333;font-size:0.8rem;"> by {c['artist']}</span>
                    <span style="color:#3a3a5a;font-size:0.78rem;margin:0 6px;">vs</span>
                    <span style="color:#444;">{sk['title']}</span>
                </div>""", unsafe_allow_html=True)
        return

    # ── Active battle ──────────────────────────────────────────────────────────
    battle = st.session_state.current_battle
    s1, s2 = battle["song1"], battle["song2"]

    l1 = "MAINSTREAM" if classify(s1) == "mainstream" else "INDIE"
    l2 = "MAINSTREAM" if classify(s2) == "mainstream" else "INDIE"

    col1, col_vs, col2 = st.columns([5, 1, 5])

    with col1:
        st.markdown(battle_card_html(s1, "#a855f7", l1), unsafe_allow_html=True)
        st.audio(make_preview(s1["tempo_bpm"], s1["energy"], s1["valence"]),
                 format="audio/wav")
        chose1 = st.button("Choose This Track  ▶", key="btn1", use_container_width=True)

    with col_vs:
        st.markdown("""
        <div class="vs-wrap">
            <div class="vs-line"></div>
            <div class="vs-text">VS</div>
            <div class="vs-line"></div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(battle_card_html(s2, "#06b6d4", l2), unsafe_allow_html=True)
        st.audio(make_preview(s2["tempo_bpm"], s2["energy"], s2["valence"]),
                 format="audio/wav")
        chose2 = st.button("Choose This Track  ▶", key="btn2", use_container_width=True)

    _, skipcol, _ = st.columns([4, 1, 4])
    with skipcol:
        skip = st.button("Skip ↩", use_container_width=True)

    if chose1 or chose2:
        chosen  = s1 if chose1 else s2
        skipped = s2 if chose1 else s1
        update_profile(chosen)
        st.session_state.battle_history.append(
            {"song1": s1, "song2": s2, "chosen": chosen, "skipped": skipped}
        )
        st.session_state.battle_result = {"chosen": chosen, "skipped": skipped}
        st.session_state.liked_songs.add(chosen["id"])
        st.rerun()

    if skip:
        st.session_state.current_battle = None
        st.rerun()


def page_discover(songs: list) -> None:
    st.markdown("## Discover")

    profile = st.session_state.user_profile

    col_a, col_b = st.columns(2)
    with col_a:
        all_genres = sorted({s["genre"] for s in songs})
        sel_genres = st.multiselect("Filter by genre", all_genres,
                                     placeholder="All genres")
    with col_b:
        all_moods = sorted({s["mood"] for s in songs})
        sel_moods = st.multiselect("Filter by mood", all_moods,
                                    placeholder="All moods")

    e_range = st.slider("Energy range", 0.0, 1.0, (0.0, 1.0), 0.05)

    filtered = [
        s for s in songs
        if (not sel_genres or s["genre"] in sel_genres)
        and (not sel_moods  or s["mood"]  in sel_moods)
        and e_range[0] <= s["energy"] <= e_range[1]
    ]

    scored = sorted(
        [(s, *score_song(s, profile)) for s in filtered],
        key=lambda x: x[1], reverse=True
    )

    st.markdown(f'<div class="section-head">{len(scored)} song{"s" if len(scored) != 1 else ""} found</div>',
                unsafe_allow_html=True)

    if not scored:
        st.info("No songs match these filters — try widening your search.")
        return

    html = "".join(song_row_html(r, s, sc) for r, (s, sc, _) in enumerate(scored, 1))
    st.markdown(html, unsafe_allow_html=True)


def page_profile(songs: list) -> None:
    st.markdown("## My Taste DNA")

    profile    = st.session_state.user_profile
    n_battles  = len(st.session_state.battle_history)

    col_radar, col_stats = st.columns([1.25, 1])

    with col_radar:
        st.markdown('<div class="section-head">Taste Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(taste_radar(profile), use_container_width=True,
                        config={"displayModeBar": False})

    with col_stats:
        st.markdown('<div class="section-head">Profile</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:10px;">
            <div class="metric-value">{tastemaker_title(n_battles)}</div>
            <div class="metric-label">{n_battles} battles fought</div>
        </div>""", unsafe_allow_html=True)

        bars = [
            ("target_energy",      "Energy",        "How intense your music is"),
            ("target_valence",     "Positivity",    "Bright vs dark feel"),
            ("target_acousticness","Acoustic",       "Organic vs electronic"),
            ("target_danceability","Danceability",   "Groove factor"),
        ]
        for key, label, desc in bars:
            v = profile[key]
            st.markdown(f"""
            <div style="margin-bottom:11px;">
                <div style="display:flex;justify-content:space-between;
                            font-size:0.8rem;margin-bottom:4px;">
                    <span style="color:#ccc;font-weight:600;">{label}</span>
                    <span style="color:#a855f7;font-weight:700;">{v:.0%}</span>
                </div>
                <div style="background:#1c1c2e;border-radius:6px;height:7px;overflow:hidden;">
                    <div style="width:{v*100:.0f}%;height:100%;
                                background:linear-gradient(90deg,#a855f7,#06b6d4);
                                border-radius:6px;"></div>
                </div>
                <div style="font-size:0.7rem;color:#444;margin-top:2px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Fine-tune sliders ──────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Fine-Tune Your Profile</div>',
                unsafe_allow_html=True)
    st.caption("Manually adjust what SoundMatch recommends — or just keep battling.")

    all_genres = sorted({s["genre"] for s in songs})
    all_moods  = sorted({s["mood"]  for s in songs})

    c1, c2 = st.columns(2)
    with c1:
        profile["genre"] = st.selectbox(
            "Favorite genre", all_genres,
            index=all_genres.index(profile["genre"]) if profile["genre"] in all_genres else 0,
        )
        profile["target_energy"] = st.slider(
            "Energy", 0.0, 1.0, float(profile["target_energy"]), 0.05)
        profile["target_valence"] = st.slider(
            "Positivity (Valence)", 0.0, 1.0, float(profile["target_valence"]), 0.05)
        profile["target_acousticness"] = st.slider(
            "Acousticness", 0.0, 1.0, float(profile["target_acousticness"]), 0.05)

    with c2:
        profile["mood"] = st.selectbox(
            "Favorite mood", all_moods,
            index=all_moods.index(profile["mood"]) if profile["mood"] in all_moods else 0,
        )
        profile["target_danceability"] = st.slider(
            "Danceability", 0.0, 1.0, float(profile["target_danceability"]), 0.05)
        profile["target_instrumentalness"] = st.slider(
            "Instrumentalness", 0.0, 1.0, float(profile["target_instrumentalness"]), 0.05)
        profile["target_speechiness"] = st.slider(
            "Speechiness", 0.0, 1.0, float(profile["target_speechiness"]), 0.05)

    st.session_state.user_profile = profile

    cr1, cr2 = st.columns(2)
    with cr1:
        if st.button("Reset Profile to Defaults", use_container_width=True):
            st.session_state.user_profile = dict(DEFAULT_PROFILE)
            st.rerun()
    with cr2:
        if st.button("Clear Battle History", use_container_width=True):
            st.session_state.battle_history = []
            st.session_state.liked_songs    = set()
            st.rerun()

    # ── Battle history ─────────────────────────────────────────────────────────
    if st.session_state.battle_history:
        st.markdown('<div class="section-head">Battle History</div>',
                    unsafe_allow_html=True)
        for i, b in enumerate(reversed(st.session_state.battle_history), 1):
            c  = b["chosen"]
            sk = b["skipped"]
            st.markdown(f"""
            <div class="sm-card" style="padding:9px 15px;display:flex;
                        align-items:center;gap:9px;">
                <span style="color:#333;font-size:0.72rem;min-width:22px;">#{i}</span>
                <span style="color:#a855f7;font-weight:600;">&#10003; {c['title']}</span>
                <span style="color:#2a2a44;font-size:0.78rem;">by {c['artist']}</span>
                <span style="color:#2a2a44;font-size:0.75rem;margin:0 4px;">vs</span>
                <span style="color:#383858;">{sk['title']}</span>
                <span style="margin-left:auto;font-size:0.7rem;color:#445;">
                    {c['mood'].title()}
                </span>
            </div>""", unsafe_allow_html=True)


# ── Monitor page ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_stats(_songs_key: tuple) -> list:
    """Cache artist stats for 5 minutes so charts don't re-compute on every interaction."""
    return get_artist_stats(get_songs())


def page_monitor(songs: list) -> None:
    st.markdown("## Artist Intelligence Monitor")
    st.caption(
        "SoundMatch scans artist stats across platforms and scores each artist "
        "on reach, plays, followers, and engagement. Top scorers get featured."
    )

    if not _MON_OK:
        st.error("platform_monitor import failed — check src/platform_monitor.py.")
        return

    # ── Platform status row ───────────────────────────────────────────────────
    st.markdown('<div class="platform-status">', unsafe_allow_html=True)
    pills = ""
    for _, meta in PLATFORMS.items():
        color  = meta["color"]
        status = "LIVE" if meta["live"] else "SIMULATED"
        dot_style = "color:#22c55e;" if meta["live"] else "color:#ef4444;"
        pills += (
            f'<span class="plat-pill" '
            f'style="color:{color};border-color:{color}40;background:{color}0d;">'
            f'<span style="{dot_style}font-size:0.55rem;">&#9679;</span> '
            f'{meta["name"]} <em style="font-weight:400;'
            f'font-size:0.64rem;color:#666;">{status}</em></span>'
        )
    st.markdown(pills + "</div>", unsafe_allow_html=True)

    if not any(m["live"] for m in PLATFORMS.values()):
        st.info(
            "All data is **simulated** from catalog audio features. "
            "Set `LASTFM_API_KEY` or `YOUTUBE_API_KEY` to pull live numbers.",
            icon=None,
        )

    # ── Load stats ────────────────────────────────────────────────────────────
    with st.spinner("Scanning platforms…"):
        records = _load_stats(tuple(frozenset(s.items()) for s in songs))
        if not records:
            records = get_artist_stats(songs)

    if st.button("Refresh Data"):
        _load_stats.clear()
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Featured artists (top 5) ──────────────────────────────────────────────
    st.markdown('<div class="section-head">Featured Artists — Top SoundMatch Scores</div>',
                unsafe_allow_html=True)

    top5 = records[:5]
    cols = st.columns(5)
    ranks = ["1st", "2nd", "3rd", "4th", "5th"]
    for col, rec, rank in zip(cols, top5, ranks):
        with col:
            photo = _aimg(rec["artist"], size=80, radius=8,
                          css="margin:8px auto 6px;display:block;")
            live_badge = (
                ' <span style="font-size:0.6rem;color:#22c55e;">&#9679; LIVE</span>'
                if rec["live_sources"] else ""
            )
            yt  = fmt(rec.get("youtube_subscribers", 0))
            sc  = fmt(rec.get("soundcloud_followers", 0))
            tt  = fmt(rec.get("tiktok_followers", 0))
            plays = fmt(rec.get("total_plays", 0))
            st.markdown(f"""
            <div class="featured-artist">
                <div class="featured-rank">{rank} &mdash; RANK #{records.index(rec)+1}{live_badge}</div>
                <div style="display:flex;justify-content:center;">{photo}</div>
                <div class="featured-name">{rec['artist']}</div>
                <div class="featured-genre">{rec['genre']} · {rec['mood']}</div>
                <div class="featured-score">{rec['score']}</div>
                <div class="featured-score-label">SOUNDMATCH SCORE</div>
                <div style="margin-top:10px;">
                    <span class="stat-chip">▶ <span>{plays}</span> plays</span>
                    <span class="stat-chip">YT <span>{yt}</span></span>
                    <span class="stat-chip">SC <span>{sc}</span></span>
                    <span class="stat-chip">TT <span>{tt}</span></span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 1: SoundMatch Score leaderboard ─────────────────────────────────
    st.markdown('<div class="section-head">SoundMatch Score Leaderboard</div>',
                unsafe_allow_html=True)

    artists_sorted = [r["artist"] for r in records]
    scores         = [r["score"]  for r in records]
    genres         = [r["genre"]  for r in records]

    genre_colors = {
        "pop":"#a855f7","k-pop":"#ec4899","edm":"#06b6d4","hip-hop":"#f59e0b",
        "rock":"#ef4444","r&b":"#8b5cf6","lofi":"#6366f1","jazz":"#10b981",
        "classical":"#f0abfc","folk":"#84cc16","metal":"#dc2626","ambient":"#67e8f9",
        "synthwave":"#c084fc","reggae":"#4ade80","country":"#fbbf24","blues":"#60a5fa",
        "indie pop":"#f9a8d4",
    }
    bar_colors = [genre_colors.get(g, "#888") for g in genres]

    fig_bar = go.Figure(go.Bar(
        x=artists_sorted, y=scores,
        marker=dict(color=bar_colors, line=dict(color="#0a0a0f", width=1)),
        text=[f"{s}" for s in scores],
        textposition="outside",
        textfont=dict(color="#aaa", size=10),
        hovertemplate="<b>%{x}</b><br>Score: %{y}<extra></extra>",
    ))
    fig_bar.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(color="#aaa", family="Inter"),
        xaxis=dict(tickangle=-40, color="#666", gridcolor="#1a1a2e", tickfont=dict(size=10)),
        yaxis=dict(range=[0, 105], color="#666", gridcolor="#1a1a2e", title="Score (0–100)"),
        margin=dict(l=40, r=20, t=20, b=80),
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Chart 2: Followers vs Plays scatter ───────────────────────────────────
    st.markdown('<div class="section-head">Total Followers vs Total Plays</div>',
                unsafe_allow_html=True)
    st.caption("Bubble size = SoundMatch score. Colour = genre. Each dot is one catalog artist.")

    fig_scatter = go.Figure(go.Scatter(
        x=[r["total_followers"] for r in records],
        y=[r["total_plays"]     for r in records],
        mode="markers+text",
        marker=dict(
            size=[max(r["score"] * 0.6, 8) for r in records],
            color=[genre_colors.get(r["genre"], "#888") for r in records],
            line=dict(color="#0a0a0f", width=1),
            opacity=0.85,
        ),
        text=[r["artist"] for r in records],
        textposition="top center",
        textfont=dict(size=9, color="#888"),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Followers: %{x:,.0f}<br>"
            "Plays: %{y:,.0f}<extra></extra>"
        ),
    ))
    fig_scatter.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(color="#aaa", family="Inter"),
        xaxis=dict(title="Total Followers (all platforms)", color="#666",
                   gridcolor="#1a1a2e", tickformat=".2s"),
        yaxis=dict(title="Total Plays (all platforms)", color="#666",
                   gridcolor="#1a1a2e", tickformat=".2s"),
        margin=dict(l=60, r=20, t=20, b=60),
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

    # ── Chart 3: Platform breakdown stacked bar ───────────────────────────────
    st.markdown('<div class="section-head">Follower Breakdown by Platform</div>',
                unsafe_allow_html=True)
    st.caption("How each artist's audience is split across platforms.")

    top10 = records[:10]
    names  = [r["artist"] for r in top10]

    fig_stack = go.Figure()
    plat_fields = [
        ("youtube_subscribers", "YouTube",    "#ff4444"),
        ("soundcloud_followers","SoundCloud", "#ff5500"),
        ("tiktok_followers",    "TikTok",     "#aaaaaa"),
        ("bandcamp_fans",       "Bandcamp",   "#1da0c3"),
    ]
    for field, label, color in plat_fields:
        fig_stack.add_trace(go.Bar(
            name=label,
            x=names,
            y=[r.get(field, 0) for r in top10],
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:,.0f}}<extra></extra>",
        ))
    fig_stack.update_layout(
        barmode="stack",
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(color="#aaa", family="Inter"),
        xaxis=dict(tickangle=-35, color="#666", gridcolor="#1a1a2e", tickfont=dict(size=10)),
        yaxis=dict(color="#666", gridcolor="#1a1a2e", title="Followers", tickformat=".2s"),
        legend=dict(bgcolor="#0a0a0f", bordercolor="#1a1a2e", font=dict(color="#aaa")),
        margin=dict(l=60, r=20, t=20, b=80),
        height=360,
    )
    st.plotly_chart(fig_stack, use_container_width=True, config={"displayModeBar": False})

    # ── Full stats table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Full Artist Stats Table</div>',
                unsafe_allow_html=True)

    import pandas as pd
    rows = []
    for r in records:
        rows.append({
            "Rank":       records.index(r) + 1,
            "Artist":     r["artist"],
            "Genre":      r["genre"],
            "Score":      r["score"],
            "YT Views":   fmt(r.get("youtube_views", 0)),
            "YT Subs":    fmt(r.get("youtube_subscribers", 0)),
            "Last.fm":    fmt(r.get("lastfm_listeners", 0)),
            "SoundCloud": fmt(r.get("soundcloud_followers", 0)),
            "TikTok":     fmt(r.get("tiktok_followers", 0)),
            "Bandcamp":   fmt(r.get("bandcamp_fans", 0)),
            "Total Plays":fmt(r.get("total_plays", 0)),
            "Live Data":  ", ".join(r["live_sources"]) or "—",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Learn page ────────────────────────────────────────────────────────────────
def _lesson_scroll_html(videos: list) -> str:
    cards = ""
    for v in videos:
        thumb_html = (
            f'<img class="lesson-thumb" src="{v["thumb"]}" alt="{v["title"]}" '
            f'onerror="this.style.display=\'none\';">'
        )
        cards += f"""
        <a class="lesson-card" href="{v['url']}" target="_blank" rel="noopener">
            {thumb_html}
            <div class="lesson-info">
                <div class="lesson-title">{v['title']}</div>
                <div class="lesson-channel">{v['channel']}</div>
            </div>
        </a>"""
    return f'<div class="scroll-row">{cards}</div>'


def page_learn() -> None:
    st.markdown("## 🎓 Music Production — Learn")
    st.caption(
        "Eight curated tracks covering everything from beat making to AI tools — "
        "all from working producers and educators. Click any video to watch on YouTube."
    )

    if not _YT_OK:
        st.info("YouTube client unavailable — check that youtube_client.py is in src/.")
        return

    # ── Platform links row ────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Discover Independent Artists On</div>',
                unsafe_allow_html=True)
    platform_links = [
        ("SoundCloud",    "https://soundcloud.com/discover"),
        ("Bandcamp",      "https://bandcamp.com"),
        ("Pitchfork",     "https://pitchfork.com"),
        ("OnesToWatch",   "https://www.ones-to-watch.com"),
        ("EARMILK",       "https://www.earmilk.com"),
        ("Indie Shuffle", "https://www.indieshuffle.com"),
        ("Musikerpool",   "https://www.musikerpool.com"),
        ("TikTok Music",  "https://www.tiktok.com/music"),
        ("YouTube Music", "https://music.youtube.com"),
    ]
    pills = "".join(
        f'<a class="channel-pill" href="{url}" target="_blank" rel="noopener">{name}</a>'
        for name, url in platform_links
    )
    st.markdown(
        f'<div class="scroll-row" style="flex-wrap:wrap;overflow-x:visible;gap:10px;">{pills}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Production lesson categories ──────────────────────────────────────────
    api_note = "YouTube API key not set — showing curated videos"
    if has_api_key():
        api_note = "YouTube API active — showing live search results"
    st.caption(api_note)

    for i, lesson in enumerate(PRODUCTION_LESSONS):
        st.markdown(f'<div class="section-head">{lesson["category"]}</div>',
                    unsafe_allow_html=True)
        videos = get_lesson_videos(i)
        st.markdown(_lesson_scroll_html(videos), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── YouTube API key setup ─────────────────────────────────────────────────
    if not has_api_key():
        with st.expander("Enable live YouTube search (optional)"):
            st.markdown(
                "Get a free YouTube Data API v3 key at "
                "[console.cloud.google.com](https://console.cloud.google.com) "
                "→ APIs & Services → YouTube Data API v3 → Create credentials.\n\n"
                "Then launch the app with:"
            )
            st.code(
                "$env:YOUTUBE_API_KEY = 'AIza...'\n"
                "C:\\Users\\jesse\\miniconda3\\envs\\ml\\Scripts\\streamlit.exe run src\\app.py",
                language="powershell",
            )


# ── Chat page ─────────────────────────────────────────────────────────────────
_EXAMPLE_QUERIES = [
    "Find me something to focus while coding",
    "I need high energy workout music",
    "Something romantic for a dinner playlist",
    "Recommend songs based on my taste profile",
    "What's a good chill song I might have missed?",
]


def page_chat(songs: list) -> None:
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY")) and _CHAT_OK

    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="chat-hero">
        <div class="chat-hero-icon">AI</div>
        <div>
            <div class="chat-hero-title">SoundMatch AI</div>
            <div class="chat-hero-sub">
                Ask anything — music recommendations, song details, app help,
                or just talk music. I remember everything you say in this conversation.
            </div>
            <span class="chat-hero-badge">RAG · Agentic · Multi-turn</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── AI not available ──────────────────────────────────────────────────────
    if not has_key:
        st.markdown("""
        <div class="chat-info-box" style="text-align:center;padding:56px 32px;">
            <div style="font-size:1.5rem;font-weight:900;margin-bottom:16px;color:#a855f7;">SoundMatch AI</div>
            <div style="font-size:1.3rem;font-weight:900;color:#f0f0ff;margin-bottom:10px;">
                AI Chat Coming Soon
            </div>
            <div style="color:#556;font-size:0.88rem;line-height:1.7;max-width:400px;margin:0 auto;">
                SoundMatch AI is being fine-tuned to give you the most personalised
                music conversations possible. Check back soon.
            </div>
            <div style="margin-top:24px;color:#445;font-size:0.78rem;">
                In the meantime, sharpen your taste with
                <strong style="color:#a855f7;">Battles</strong> —
                every pick makes your recommendations smarter.
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Start a Battle", use_container_width=False):
            st.session_state.page = "Battles"
            st.rerun()
        return

    # ── Example chips (empty history only) ───────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown('<div class="section-head">Try asking…</div>',
                    unsafe_allow_html=True)
        cols = st.columns(len(_EXAMPLE_QUERIES))
        for i, (col, q) in enumerate(zip(cols, _EXAMPLE_QUERIES)):
            with col:
                if st.button(q, key=f"ex_{i}", use_container_width=True):
                    st.session_state._prefill = q
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Render full conversation history ──────────────────────────────────────
    for msg in st.session_state.chat_history:
        avatar = "A" if msg["role"] == "assistant" else "U"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Chat input ─────────────────────────────────────────────────────────────
    prefill = st.session_state.pop("_prefill", "")
    prompt  = st.chat_input("Ask about music, artists, moods…", key="chat_input") or prefill

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="U"):
            st.markdown(prompt)

        # Build prior history for multi-turn context (role + content only)
        prior = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history[:-1]
        ]

        with st.chat_message("assistant", avatar="A"):
            with st.spinner("Thinking…"):
                try:
                    reply = chat_with_history(
                        prompt, prior, songs,
                        st.session_state.user_profile,
                    )
                    # Also run RAG for the expandable context panel
                    rag_songs = rag_retrieve(
                        intent_to_features(prompt), songs, k=4
                    )
                except Exception as exc:
                    reply     = f"Something went wrong: {exc}"
                    rag_songs = []

            st.markdown(reply)

            if rag_songs:
                with st.expander("RAG context — songs retrieved by audio similarity"):
                    for s in rag_songs:
                        st.markdown(
                            f"- **{s['title']}** by {s['artist']} "
                            f"— {s['genre']}, {s['mood']}, energy:{s['energy']:.0%}"
                        )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply}
        )

    # ── Clear button ──────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear Conversation", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
def render_topnav() -> None:
    _PAGES = [
        ("Home",         "🏠 Home"),
        ("Battles",      "⚔️ Battles"),
        ("Discover",     "🔍 Discover"),
        ("My Taste DNA", "🧬 Taste DNA"),
        ("Chat",         "💬 Chat"),
        ("Learn",        "🎓 Learn"),
        ("Monitor",      "📊 Monitor"),
    ]
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>"
        "<span style='font-size:1.3rem;font-weight:900;color:#ffffff;"
        "letter-spacing:-0.02em;margin-right:8px;'>🎵 SoundMatch</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(_PAGES))
    for col, (key, label) in zip(cols, _PAGES):
        with col:
            if st.button(label, key=f"topnav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()
    st.divider()


def main() -> None:
    inject_css()
    init_state()
    songs = get_songs()

    # Preload real artist photos once per session (cached to disk after first run)
    if not st.session_state.get("artist_images"):
        st.session_state.artist_images = _fetch_artist_images(
            tuple(s["id"] for s in songs)
        )

    render_topnav()
    page = st.session_state.page

    if   page == "Home":          page_home(songs)
    elif page == "Battles":       page_battles(songs)
    elif page == "Discover":      page_discover(songs)
    elif page == "My Taste DNA":  page_profile(songs)
    elif page == "Chat":          page_chat(songs)
    elif page == "Learn":         page_learn()
    elif page == "Monitor":       page_monitor(songs)


main()