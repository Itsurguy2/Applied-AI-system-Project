"""
SoundMatch Chat Agent — RAG + Agentic Workflow

RAG step:    Cosine similarity on audio features pre-retrieves context songs
             before the LLM sees the query.
Agentic step: Claude uses tool-calling to search the catalog, get ranked
             recommendations, or look up specific songs — deciding on its own
             which tools to call and in what order.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from recommender import MAX_SCORE, load_songs, recommend_songs, score_song

try:
    import anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"

_FEATURE_KEYS = [
    "energy", "valence", "acousticness", "danceability",
    "instrumentalness", "speechiness", "liveness",
]

SYSTEM_PROMPT = """You are SoundMatch AI — an expert music recommendation assistant.
You have access to a curated 20-song catalog spanning 17 genres and 14 moods.

Your job is to help users discover music using your tools. Rules:
- Always call at least one tool before answering so your response is grounded in real data.
- Explain WHY each song fits (energy, mood, vibe, genre) — not just the name.
- Mention the artist and one standout audio characteristic per song.
- Be conversational and enthusiastic — you love music.
- Be fair to both mainstream and indie artists: surface hidden gems when relevant.
- If a request is vague, make a reasonable interpretation and state it briefly."""

TOOLS = [
    {
        "name": "search_songs",
        "description": (
            "Search the music catalog by mood, genre, and/or energy range. "
            "Returns matching songs ranked by how well they fit the user's taste profile."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mood":       {"type": "string",  "description": "Mood label, e.g. 'focused', 'happy', 'chill', 'intense'"},
                "genre":      {"type": "string",  "description": "Genre label, e.g. 'lofi', 'jazz', 'rock', 'edm'"},
                "energy_min": {"type": "number",  "description": "Minimum energy level (0.0–1.0)"},
                "energy_max": {"type": "number",  "description": "Maximum energy level (0.0–1.0)"},
                "limit":      {"type": "integer", "description": "Max songs to return (default 3)"},
            },
        },
    },
    {
        "name": "get_top_recommendations",
        "description": (
            "Return the top-k songs personalized to the user's current taste profile. "
            "Use when the user wants general picks or says 'what should I listen to'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "description": "Number of recommendations (default 3, max 5)"},
            },
        },
    },
    {
        "name": "get_song_details",
        "description": "Get full audio feature details for a specific song by title.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Exact or approximate song title"},
            },
            "required": ["title"],
        },
    },
]


# ── RAG: cosine similarity retrieval ─────────────────────────────────────────
def _vec(song: dict) -> np.ndarray:
    return np.array([song[k] for k in _FEATURE_KEYS], dtype=float)


def rag_retrieve(query_features: dict, songs: list, k: int = 4) -> list:
    """
    Retrieve the k songs whose feature vectors are most similar to query_features
    using cosine similarity. This is the RAG retrieval step.
    """
    q = np.array([query_features.get(key, 0.5) for key in _FEATURE_KEYS], dtype=float)
    q_norm = np.linalg.norm(q) + 1e-9
    scored = []
    for song in songs:
        s = _vec(song)
        sim = float(np.dot(q, s) / (q_norm * (np.linalg.norm(s) + 1e-9)))
        scored.append((song, sim))
    return [s for s, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]


def intent_to_features(message: str) -> dict:
    """Map natural language cues to approximate feature targets for RAG retrieval."""
    msg = message.lower()
    p = {
        "energy": 0.60, "valence": 0.60, "acousticness": 0.40,
        "danceability": 0.60, "instrumentalness": 0.35,
        "speechiness": 0.05, "liveness": 0.12,
    }
    if any(w in msg for w in ["chill", "relax", "study", "focus", "calm", "quiet", "sleep", "ambient", "lofi"]):
        p["energy"] = 0.33
        p["acousticness"] = 0.78
        p["instrumentalness"] = 0.72
    if any(w in msg for w in ["intense", "workout", "gym", "hype", "pump", "loud", "aggressive", "metal"]):
        p["energy"] = 0.92
        p["danceability"] = 0.82
    if any(w in msg for w in ["dance", "party", "club", "groove", "upbeat", "edm"]):
        p["energy"] = 0.82
        p["danceability"] = 0.90
        p["valence"] = 0.85
    if any(w in msg for w in ["sad", "melancholic", "heartbreak", "dark", "moody", "blue"]):
        p["valence"] = 0.25
        p["energy"] = min(p["energy"], 0.50)
    if any(w in msg for w in ["happy", "joyful", "euphoric", "positive", "bright", "cheerful"]):
        p["valence"] = 0.88
    if any(w in msg for w in ["acoustic", "unplugged", "folk", "organic", "live"]):
        p["acousticness"] = 0.87
    if any(w in msg for w in ["electronic", "synth", "beats", "bass"]):
        p["acousticness"] = 0.05
        p["energy"] = max(p["energy"], 0.75)
    if any(w in msg for w in ["jazz", "coffee", "smooth", "mellow"]):
        p["energy"] = 0.38
        p["acousticness"] = 0.80
        p["valence"] = 0.68
    if any(w in msg for w in ["romantic", "love", "date", "r&b"]):
        p["valence"] = 0.78
        p["energy"] = 0.48
        p["danceability"] = 0.72
    return p


# ── Tool execution ─────────────────────────────────────────────────────────────
def _format_songs(songs: list, profile: dict) -> str:
    if not songs:
        return "No songs found matching those criteria."
    lines = []
    for s in songs:
        sc, _ = score_song(s, profile)
        pct = int((sc / MAX_SCORE) * 100)
        lines.append(
            f"• \"{s['title']}\" by {s['artist']} | {s['genre']} | mood:{s['mood']} | "
            f"energy:{s['energy']:.0%} valence:{s['valence']:.0%} "
            f"dance:{s['danceability']:.0%} acoustic:{s['acousticness']:.0%} | "
            f"profile match:{pct}%"
        )
    return "\n".join(lines)


def _execute_tool(name: str, inputs: dict, songs: list, profile: dict) -> str:
    if name == "search_songs":
        filtered = list(songs)
        if "mood" in inputs:
            filtered = [s for s in filtered if s["mood"] == inputs["mood"]]
        if "genre" in inputs:
            filtered = [s for s in filtered if s["genre"] == inputs["genre"]]
        if "energy_min" in inputs:
            filtered = [s for s in filtered if s["energy"] >= float(inputs["energy_min"])]
        if "energy_max" in inputs:
            filtered = [s for s in filtered if s["energy"] <= float(inputs["energy_max"])]
        limit = int(inputs.get("limit", 3))
        filtered.sort(key=lambda s: score_song(s, profile)[0], reverse=True)
        return _format_songs(filtered[:limit], profile)

    if name == "get_top_recommendations":
        k = min(int(inputs.get("k", 3)), 5)
        results = recommend_songs(profile, songs, k=k)
        return _format_songs([s for s, _, _ in results], profile)

    if name == "get_song_details":
        title = inputs.get("title", "").lower().strip()
        match = next(
            (s for s in songs if title in s["title"].lower() or s["title"].lower() in title),
            None,
        )
        if not match:
            return f"No song matching '{inputs.get('title')}' found in the catalog."
        sc, expl = score_song(match, profile)
        return (
            f"\"{match['title']}\" by {match['artist']}\n"
            f"Genre: {match['genre']} | Mood: {match['mood']} | "
            f"Tempo: {int(match['tempo_bpm'])} BPM\n"
            f"Energy:{match['energy']:.0%}  Valence:{match['valence']:.0%}  "
            f"Danceability:{match['danceability']:.0%}  Acousticness:{match['acousticness']:.0%}\n"
            f"Instrumentalness:{match['instrumentalness']:.0%}  "
            f"Speechiness:{match['speechiness']:.0%}  Liveness:{match['liveness']:.0%}\n"
            f"Profile match: {int((sc / MAX_SCORE) * 100)}%\n"
            f"Why: {expl}"
        )

    return f"Unknown tool: {name}"


# ── Agent entry point ─────────────────────────────────────────────────────────
def run_agent(
    user_message: str,
    songs: list,
    user_profile: dict,
    *,
    max_iterations: int = 5,
) -> dict:
    """
    Run the SoundMatch agentic workflow.

    1. RAG step:    cosine similarity retrieves context songs from the catalog.
    2. Agentic step: Claude decides which tools to call (search, recommend, detail).
    3. Return the final text reply, tool call log, and RAG context.

    Returns:
        {
            "reply":        str,    # final text for display
            "tools_called": list,   # [(tool_name, inputs_dict, result_str), ...]
            "rag_context":  list,   # songs retrieved by cosine similarity
        }
    """
    if not _ANTHROPIC_OK:
        return {
            "reply": "anthropic package not installed. Run: pip install anthropic",
            "tools_called": [],
            "rag_context": [],
        }

    # ── RAG retrieval ─────────────────────────────────────────────────────────
    query_features = intent_to_features(user_message)
    rag_songs = rag_retrieve(query_features, songs, k=4)
    rag_context_str = "\n".join(
        f"  [{i+1}] \"{s['title']}\" — {s['genre']}, {s['mood']}, "
        f"energy:{s['energy']:.0%}, acoustic:{s['acousticness']:.0%}"
        for i, s in enumerate(rag_songs)
    )

    # Inject RAG context into the user message as grounding
    augmented = (
        f"{user_message}\n\n"
        f"[System — RAG pre-retrieval via cosine similarity on audio features:]\n"
        f"{rag_context_str}\n"
        f"(Use your tools to verify or expand on these — they are starting context only.)"
    )

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": augmented}]
    tools_called: list = []

    for _ in range(max_iterations):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            reply = " ".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return {"reply": reply, "tools_called": tools_called, "rag_context": rag_songs}

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _execute_tool(block.name, block.input, songs, user_profile)
                    tools_called.append((block.name, dict(block.input), result))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            continue

        break

    return {
        "reply": "I couldn't complete that request — please try again.",
        "tools_called": tools_called,
        "rag_context": rag_songs,
    }


# ── Conversational multi-turn assistant ───────────────────────────────────────

CONVO_SYSTEM_PROMPT = """You are SoundMatch AI — a warm, knowledgeable music assistant built into the SoundMatch app.

You help users in three ways:
1. Music discovery — recommend songs, explain audio characteristics, surface hidden gems
2. Music conversations — talk about genres, artists, production, vibes, history
3. App guidance — explain Battles, Taste DNA, Discover, Monitor, and the Learn page

About the SoundMatch app:
- Battles: users pick mainstream vs indie tracks; each choice updates their Taste Profile via EMA (α=0.30) — 20× stronger signal than passive listening
- AI Recommendations: a weighted scorer ranks all catalog songs against the evolving taste profile
- Taste DNA: live radar chart showing how preferences shift over time
- Monitor: cross-platform artist stats — YouTube, Last.fm, SoundCloud, TikTok, Spotify
- Catalog: 20 songs across 17 genres (lofi, pop, rock, hip-hop, jazz, edm, classical, r&b, indie pop, folk, metal, ambient, synthwave, reggae, country, blues, k-pop) and 14 moods

Personality: warm, enthusiastic, genuinely passionate about music — like a knowledgeable friend who works at a record store. You remember what was said earlier in this conversation. When you need catalog data, use your tools. For general music chat, talk naturally."""


def chat_with_history(
    message: str,
    history: list,
    songs: list,
    profile: dict,
) -> str:
    """
    Multi-turn conversation — history is a list of {"role": str, "content": str}.
    Returns the assistant reply as a plain string.
    """
    if not _ANTHROPIC_OK:
        return "The anthropic package isn't installed. Run: pip install anthropic"

    # Build profile context string (injected on the first user turn)
    profile_ctx = (
        f"[User taste profile — genre:{profile.get('genre','pop')}, "
        f"mood:{profile.get('mood','happy')}, "
        f"energy:{profile.get('target_energy',0.65):.0%}, "
        f"acousticness:{profile.get('target_acousticness',0.4):.0%}, "
        f"danceability:{profile.get('target_danceability',0.65):.0%}]"
    )

    # Rebuild message list from history
    api_messages: list = []
    for i, turn in enumerate(history):
        content = turn["content"]
        # Prefix profile context on the very first user message only
        if i == 0 and turn["role"] == "user":
            content = f"{profile_ctx}\n\n{content}"
        api_messages.append({"role": turn["role"], "content": content})

    # RAG-augment the new message
    query_features = intent_to_features(message)
    rag_songs      = rag_retrieve(query_features, songs, k=3)
    rag_ctx        = "\n".join(
        f'  "{s["title"]}" — {s["genre"]}, {s["mood"]}, energy:{s["energy"]:.0%}'
        for s in rag_songs
    )

    # First turn ever → prepend profile context here too
    prefix = f"{profile_ctx}\n\n" if not api_messages else ""
    augmented = f"{prefix}{message}\n\n[Catalog context — top audio similarity matches:]\n{rag_ctx}"
    api_messages.append({"role": "user", "content": augmented})

    client = anthropic.Anthropic()

    for _ in range(5):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=CONVO_SYSTEM_PROMPT,
            tools=TOOLS,
            messages=api_messages,
        )

        if response.stop_reason == "end_turn":
            return " ".join(
                block.text for block in response.content if hasattr(block, "text")
            )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _execute_tool(block.name, block.input, songs, profile)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            api_messages.append({"role": "assistant", "content": response.content})
            api_messages.append({"role": "user",      "content": tool_results})
            continue

        break

    return "I couldn't complete that — please try again."
