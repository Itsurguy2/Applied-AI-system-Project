"""
SoundMatch Chat Agent — RAG + Agentic Workflow

RAG step:    Cosine similarity on audio features pre-retrieves context songs
             before the LLM sees the query.
Agentic step: Claude uses tool-calling to search the catalog, get ranked
             recommendations, or look up specific songs — deciding on its own
             which tools to call and in what order.
"""

import json
import sys
import time
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

# ── Genre Knowledge RAG (second data source) ──────────────────────────────────
_GENRE_KNOWLEDGE_PATH = Path(__file__).parent.parent / "data" / "genre_knowledge.json"

# Maps genre keys to query keywords used for document retrieval
_GENRE_KEYWORDS: dict = {
    "lofi":      ["lofi", "lo-fi", "study", "coding", "focus", "rain", "café", "cafe"],
    "jazz":      ["jazz", "coffee", "improvisation", "bebop", "swing", "smooth", "bossa"],
    "classical": ["classical", "orchestral", "piano", "nocturne", "symphony", "baroque", "opus"],
    "metal":     ["metal", "heavy", "headbang", "shred", "aggressive", "brutal", "distortion"],
    "edm":       ["edm", "electronic", "rave", "festival", "drop", "banger", "techno", "house"],
    "pop":       ["pop", "mainstream", "catchy", "radio", "chart", "hook"],
    "hip-hop":   ["hip-hop", "hip hop", "rap", "trap", "verse", "bars", "beat"],
    "r&b":       ["r&b", "rnb", "soul", "romance", "groove", "rhythm and blues"],
    "ambient":   ["ambient", "meditation", "space", "atmospheric", "drone", "soundscape"],
    "synthwave": ["synthwave", "retro", "80s", "neon", "synth", "retrowave", "vaporwave"],
    "rock":      ["rock", "guitar", "riff", "band", "electric", "distorted"],
    "folk":      ["folk", "acoustic", "singer-songwriter", "fingerpicking", "storytelling"],
    "indie pop": ["indie", "alternative", "jangly", "diy", "bedroom pop"],
    "reggae":    ["reggae", "caribbean", "island", "jamaica", "dub", "skank"],
    "country":   ["country", "western", "porch", "fiddle", "americana", "twang"],
    "blues":     ["blues", "blue note", "delta", "pentatonic", "soulful", "slide guitar"],
    "k-pop":     ["k-pop", "kpop", "korean", "k pop", "idol", "bts", "blackpink"],
}


def load_genre_knowledge() -> dict:
    """Load genre knowledge documents from JSON. Returns empty dict if file missing."""
    if not _GENRE_KNOWLEDGE_PATH.exists():
        return {}
    with open(_GENRE_KNOWLEDGE_PATH, encoding="utf-8") as f:
        return json.load(f)


def rag_retrieve_docs(query_text: str, knowledge: dict, top_n: int = 2) -> list:
    """
    RAG Enhancement: retrieve genre knowledge documents relevant to query_text.
    Uses keyword matching against _GENRE_KEYWORDS to score each genre doc.
    Returns up to top_n docs sorted by keyword hit count.
    """
    msg = query_text.lower()
    matches: list = []
    for genre, keywords in _GENRE_KEYWORDS.items():
        if genre not in knowledge:
            continue
        hits = sum(1 for kw in keywords if kw in msg)
        if hits > 0:
            matches.append((hits, knowledge[genre]))
    matches.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in matches[:top_n]]

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

# ── Agentic Workflow: plan-first tool ─────────────────────────────────────────
# Adding this to TOOLS_WITH_PLAN forces Claude to emit a search plan before
# calling any data tools, creating three observable reasoning steps:
#   Step 1 — Plan:   Claude states how it interprets the query and what it will do
#   Step 2 — Search: Claude executes the actual tool calls
#   Step 3 — Synth:  Claude synthesizes data into a grounded response
_PLAN_TOOL = {
    "name": "plan_reasoning",
    "description": (
        "CALL THIS FIRST before any other tool. "
        "State your search plan: how you interpret the request, which tools you will use, "
        "and what audio feature profile you expect the ideal result to have. "
        "This creates a transparent reasoning trace."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query_interpretation": {
                "type": "string",
                "description": "How you interpret the user's request — mood, context, use case",
            },
            "search_strategy": {
                "type": "string",
                "description": "Which tool(s) you plan to call and the parameters you will use",
            },
            "expected_audio_profile": {
                "type": "string",
                "description": "Target audio features: approximate energy, acousticness, tempo, mood label",
            },
        },
        "required": ["query_interpretation", "search_strategy", "expected_audio_profile"],
    },
}

TOOLS_WITH_PLAN = [_PLAN_TOOL] + TOOLS

# ── Few-Shot Specialization ───────────────────────────────────────────────────
# These three examples train Claude to use SoundMatch's voice:
#   • Reference exact audio feature percentages
#   • Use music production vocabulary (sidechain, vinyl warmth, blast-beat, etc.)
#   • Compare songs within the catalog when multiple options exist
#   • Ground every pick in at least one specific numeric feature
FEW_SHOT_EXAMPLES = """
## SoundMatch Response Style — Examples

User: "I want something chill to work to"
You: For focused work, Library Rain by Paper Lanterns is the call — lofi at 35% energy, acousticness 86%, instrumentalness 92%. That vinyl warmth keeps your brain steady without triggering alert mode. 72 BPM slots into a flow state naturally. If you need a touch more forward motion, Focus Flow by LoRoom sits at 80 BPM and 40% energy with 91% instrumental — same focused zone, slightly more pulse.

User: "give me something to get hyped for the gym"
You: Pulse Horizon by Zeta Drop — EDM at 95% energy, 128 BPM, danceability at 94%. The four-on-the-floor kick with sidechain compression is engineered for exactly this. If you want something more aggressive and less dance-floor, Iron Collapse by Shatter Grid goes to 97% energy at 168 BPM — blast-beat drums, wall-of-sound distortion, metal. Different texture, same intensity.

User: "something late night, cinematic and moody"
You: Night Drive Loop by Neon Echo — synthwave at 75% energy, 89% instrumental, 110 BPM. Analog synth pads and gated reverb drums create that neon-highway atmosphere. Valence sits at 49% — emotionally ambiguous, not sad or happy, just atmospheric. If you want deeper immersion, Spacewalk Thoughts by Orbit Bloom goes full ambient: 28% energy, 92% acousticness, nearly completely instrumental.
"""

SPECIALIZED_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\n\n## Response Style Guide\n"
    + FEW_SHOT_EXAMPLES
    + "\nAlways cite at least two specific audio feature percentages per song you recommend."
)


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

    if name == "plan_reasoning":
        # Confirms the plan and returns it so Claude can see it was accepted
        return (
            f"[PLAN ACCEPTED] "
            f"Interpretation: {inputs.get('query_interpretation', '')} | "
            f"Strategy: {inputs.get('search_strategy', '')} | "
            f"Expecting: {inputs.get('expected_audio_profile', '')}"
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


# ── Agentic Workflow: plan-first agent ────────────────────────────────────────

_PLAN_FIRST_SYSTEM = (
    SYSTEM_PROMPT
    + """

REASONING PROTOCOL — follow this order strictly:
1. Call plan_reasoning FIRST. State your interpretation of the request, which tools
   you will call, and what audio features you expect in the ideal result.
2. After the plan is confirmed, call the appropriate search/recommendation tools.
3. Synthesize the tool results into your final answer.

This three-step process (Plan → Search → Synthesize) makes your reasoning transparent."""
)


def run_agent_with_plan(
    user_message: str,
    songs: list,
    user_profile: dict,
    *,
    max_iterations: int = 6,
) -> dict:
    """
    Agentic Workflow Enhancement — plan-first multi-step reasoning.

    Observable intermediate steps:
      Step 1  plan        Claude calls plan_reasoning to declare its strategy
      Step 2  tool_call   Claude calls search/recommendation tools
      Step 3  synthesis   Claude generates the final grounded response

    Returns:
        {
            "reply":           str,   # final text
            "reasoning_trace": list,  # [{"step": int, "type": str, "name": str,
                                      #   "content": str, "elapsed_ms": int}, ...]
            "tools_called":    list,  # same format as run_agent()
            "rag_context":     list,  # cosine-retrieved songs
        }
    """
    if not _ANTHROPIC_OK:
        return {
            "reply": "anthropic package not installed.",
            "reasoning_trace": [],
            "tools_called": [],
            "rag_context": [],
        }

    t0 = time.time()

    # RAG step — same as run_agent()
    query_features = intent_to_features(user_message)
    rag_songs = rag_retrieve(query_features, songs, k=4)
    rag_context_str = "\n".join(
        f"  [{i+1}] \"{s['title']}\" — {s['genre']}, {s['mood']}, "
        f"energy:{s['energy']:.0%}, acoustic:{s['acousticness']:.0%}"
        for i, s in enumerate(rag_songs)
    )

    # Genre doc RAG — second data source
    knowledge = load_genre_knowledge()
    genre_docs = rag_retrieve_docs(user_message, knowledge, top_n=2)
    doc_context_str = ""
    if genre_docs:
        doc_context_str = "\n[Genre Knowledge Documents:]\n" + "\n".join(
            f"  {doc['name']}: {doc['description']} "
            f"(typical energy {doc['typical_energy']:.0%}, "
            f"acousticness {doc['typical_acousticness']:.0%})"
            for doc in genre_docs
        )

    augmented = (
        f"{user_message}\n\n"
        f"[RAG — cosine similarity on audio features:]\n{rag_context_str}"
        f"{doc_context_str}\n"
        f"(Verify or expand on these with your tools.)"
    )

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": augmented}]
    tools_called: list = []
    reasoning_trace: list = []
    step = 0

    for _ in range(max_iterations):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=_PLAN_FIRST_SYSTEM,
            tools=TOOLS_WITH_PLAN,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            reply = " ".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            step += 1
            reasoning_trace.append({
                "step": step,
                "type": "synthesis",
                "name": "final_response",
                "content": reply[:200] + ("…" if len(reply) > 200 else ""),
                "elapsed_ms": int((time.time() - t0) * 1000),
            })
            return {
                "reply": reply,
                "reasoning_trace": reasoning_trace,
                "tools_called": tools_called,
                "rag_context": rag_songs,
            }

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _execute_tool(block.name, block.input, songs, user_profile)
                    tools_called.append((block.name, dict(block.input), result))
                    step += 1
                    trace_type = "plan" if block.name == "plan_reasoning" else "tool_call"
                    reasoning_trace.append({
                        "step": step,
                        "type": trace_type,
                        "name": block.name,
                        "content": result[:300] + ("…" if len(result) > 300 else ""),
                        "elapsed_ms": int((time.time() - t0) * 1000),
                    })
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
        "reasoning_trace": reasoning_trace,
        "tools_called": tools_called,
        "rag_context": rag_songs,
    }


# ── Few-Shot Specialization: specialized agent ────────────────────────────────

def run_agent_specialized(
    user_message: str,
    songs: list,
    user_profile: dict,
    *,
    max_iterations: int = 5,
) -> dict:
    """
    Few-Shot Specialization — uses SPECIALIZED_SYSTEM_PROMPT with 3 example
    Q&A pairs that demonstrate the SoundMatch voice:
      • Cites specific audio feature percentages (energy 35%, acousticness 86%)
      • Uses production vocabulary (sidechain, vinyl warmth, blast-beat)
      • Compares catalog songs when multiple options exist

    Compare output quality against run_agent() using measure_response_quality().
    """
    if not _ANTHROPIC_OK:
        return {"reply": "anthropic package not installed.", "tools_called": [], "rag_context": []}

    # Same RAG preamble as run_agent()
    query_features = intent_to_features(user_message)
    rag_songs = rag_retrieve(query_features, songs, k=4)
    rag_context_str = "\n".join(
        f"  [{i+1}] \"{s['title']}\" — {s['genre']}, {s['mood']}, "
        f"energy:{s['energy']:.0%}, acoustic:{s['acousticness']:.0%}"
        for i, s in enumerate(rag_songs)
    )

    # Hybrid RAG: add genre docs
    knowledge = load_genre_knowledge()
    genre_docs = rag_retrieve_docs(user_message, knowledge, top_n=2)
    doc_context_str = ""
    if genre_docs:
        doc_context_str = "\n[Genre Knowledge:]\n" + "\n".join(
            f"  {doc['name']}: {doc['description'][:120]}…"
            for doc in genre_docs
        )

    augmented = (
        f"{user_message}\n\n"
        f"[RAG pre-retrieval:]\n{rag_context_str}{doc_context_str}\n"
        f"(Use tools to verify.)"
    )

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": augmented}]
    tools_called: list = []

    for _ in range(max_iterations):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SPECIALIZED_SYSTEM_PROMPT,   # <-- few-shot prompt
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

    return {"reply": "I couldn't complete that request.", "tools_called": tools_called, "rag_context": rag_songs}


# ── Response Quality Measurement ──────────────────────────────────────────────

_AUDIO_FEATURE_VOCAB = {
    "energy", "valence", "acousticness", "danceability",
    "instrumentalness", "speechiness", "liveness", "bpm", "tempo",
}
_PRODUCTION_VOCAB = {
    "vinyl", "reverb", "sidechain", "compression", "distortion",
    "analog", "four-on-the-floor", "808", "sample", "drum", "synth",
    "pad", "bass", "blast-beat", "gated", "acoustic", "arpegg",
}


def measure_response_quality(text: str) -> dict:
    """
    Measure how grounded and vocabulary-rich a chat response is.
    Used to compare baseline run_agent() vs specialized run_agent_specialized().

    Returns:
        feature_mentions   — count of audio feature terms (energy, valence, …)
        production_vocab   — count of production/instrument terms
        song_references    — number of quoted song/artist names
        numeric_values     — count of percentage or decimal figures cited
        quality_score      — weighted composite (higher = richer, more grounded)
    """
    lower = text.lower()
    feature_mentions = sum(1 for w in _AUDIO_FEATURE_VOCAB if w in lower)
    production_vocab = sum(1 for w in _PRODUCTION_VOCAB if w in lower)
    # Count pairs of double-quotes as song/artist name references
    song_references = text.count('"') // 2
    # Count numeric percentage-style references ("35%", "0.86", "128 bpm")
    import re
    numeric_values = len(re.findall(r'\d+%|\d\.\d+|\d+ bpm', lower))
    quality_score = (
        feature_mentions * 3
        + production_vocab * 2
        + song_references * 4
        + min(numeric_values, 10) * 2
    )
    return {
        "feature_mentions": feature_mentions,
        "production_vocab": production_vocab,
        "song_references": song_references,
        "numeric_values": numeric_values,
        "quality_score": quality_score,
    }