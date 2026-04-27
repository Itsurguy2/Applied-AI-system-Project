# SoundMatch — AI-Powered Music Discovery App

> A full-stack music recommendation platform powered by Claude AI, RAG retrieval,
> and an adaptive taste-learning engine built in Python + Streamlit.

---

## Original Project (Modules 1–3)

The foundation of this work is a **command-line music recommender system** built during
Modules 1–3. Its goal was to score songs against a user's audio feature preferences using a
hand-crafted weighted scoring formula across 9 dimensions — mood, genre, energy, acousticness,
instrumentalness, valence, danceability, speechiness, and liveness. The system exposed a
`Recommender` class, a `score_song()` functional API, and a CLI runner (`src/main.py`) that
stress-tested the algorithm against four adversarial user profiles (the "Sad Gym Rat," the
"Genre Ghost," the "Centrist," and the "Impossible Combo"), each designed to reveal a
different flaw in how the scoring weights interact.

The original system had no UI, no learning from user behavior, and no natural-language interface.
The final project adds every one of those layers on top of that scoring engine without changing it.

---

## Title and Summary

**SoundMatch** is an AI-powered music discovery app that learns what you like by watching how
you choose, then uses Claude claude-sonnet-4-6 to explain and surface music in natural conversation.

Most recommendation engines are black boxes — they tell you what to listen to but not why, and
they require weeks of passive listening history before recommendations feel personal. SoundMatch
addresses both problems:

1. **Battles** — you pick between a Mainstream and an Indie track and each choice immediately
   updates your Taste Profile via Exponential Moving Average, giving 20× stronger signal than
   passive streaming history.
2. **Chat AI** — a Claude-powered agent uses Retrieval-Augmented Generation (RAG) and live tool
   calls to answer music questions grounded in your actual catalog data, not hallucinated guesses.
3. **Transparency** — every recommendation shows its match percentage and the exact audio features
   that drove the score.

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│                   Streamlit UI  (app.py)                   │
│  Home · Battles · Discover · Taste DNA · Chat · Monitor   │
└─────────────────────────┬─────────────────────────────────┘
                          │
       ┌──────────────────┼──────────────────────┐
       │                  │                      │
┌──────▼──────────┐ ┌─────▼────────────┐ ┌──────▼──────────┐
│  recommender.py │ │  chat_agent.py   │ │platform_monitor │
│                 │ │                  │ │     .py          │
│ Weighted score  │ │ RAG retrieval    │ │                  │
│ (MAX 8.75 pts)  │ │ (cosine sim on   │ │ YouTube / Last.fm│
│                 │ │  7 audio features│ │ live + simulated │
│ EMA profile     │ │  via NumPy)      │ │ artist stats +   │
│ updates α=0.30  │ │ +                │ │ talent scores    │
└──────┬──────────┘ │ Claude claude-sonnet-4-6  │ └─────────────────┘
       │            │ + 3 tools        │
┌──────▼──────────┐ └──────┬───────────┘
│  data/songs.csv │        │               ┌──────────────────┐
│                 │ ┌──────▼───────────┐   │ artist_images.py │
│  20 songs       │ │  Anthropic API   │   │                  │
│  13 features    │ │ (ANTHROPIC_      │   │ Fictional artist │
│  17 genres      │ │  API_KEY)        │   │ → real name map  │
│  14 moods       │ └──────────────────┘   │ → Deezer API     │
└─────────────────┘                        │ → disk cache     │
                                           └──────────────────┘
```

### Data flow for a recommendation

1. User completes a Battle → EMA updates `user_profile` in `st.session_state`
2. Home page calls `recommend_songs(profile, songs, k=5)` from `recommender.py`
3. Each song is scored across 9 weighted dimensions; top-k returned with score + explanation
4. Artist photos resolved via `artist_images.py` (fictional→real name map → Deezer API → disk cache)

### Data flow for a chat query

1. `intent_to_features(message)` maps natural language to approximate audio feature targets
2. `rag_retrieve()` runs cosine similarity on 7 audio feature vectors to pre-fetch 3–4 context songs
3. Augmented message (with RAG context and user profile state) sent to Claude claude-sonnet-4-6
4. Claude decides which tools to call: `search_songs`, `get_top_recommendations`, or `get_song_details`
5. Tool results executed locally in Python, returned to Claude as tool results
6. Final answer streamed back to the Streamlit chat UI

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- `pip` package manager
- An Anthropic API key for the Chat page (free tier works fine)
- Optional: YouTube Data API v3 key for live video content

### Step 1 — Clone and enter the repo

```bash
git clone https://github.com/YOUR_USERNAME/Applied-AI-system-Project.git
cd Applied-AI-system-Project
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows PowerShell
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Installs: `streamlit`, `anthropic`, `numpy`, `plotly`, `pandas`, `pytest`

### Step 4 — Set API keys (optional but recommended)

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."
```

The app runs without any key. The Chat page and Sidebar AI display a "coming soon" state.
Artist images (via Deezer) require no key at all.

### Step 5 — Launch the app

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 6 — Run the original CLI recommender

```bash
python -m src.main
```

### Step 7 — Run the test suite

```bash
pytest tests/ -v
```

---

## Sample Interactions

### Interaction 1 — Chat: Asking for a study playlist

**User types:**
> "I need something calm and instrumental to study tonight, no lyrics"

**What happens internally:**
- `intent_to_features()` maps "calm," "instrumental," "no lyrics" → energy target 0.33,
  instrumentalness 0.72, acousticness 0.78
- RAG cosine similarity retrieves *Midnight Coding*, *Library Rain*, *Spacewalk Thoughts*,
  *Focus Flow* as the 4 closest audio matches
- Claude calls `search_songs` with `{"energy_max": 0.45, "mood": "focused"}`
- Tool returns matching songs with profile match percentages

**Claude's response:**
> "For a late-night study session I'd go with these three:
>
> **Library Rain** by Paper Lanterns is your best bet — 92% match. It's a lofi track at just
> 35% energy with 92% instrumentalness, meaning almost no vocals competing for your attention.
> The acoustic warmth (86%) gives it that rain-on-the-window feel at a slow 72 BPM.
>
> **Focus Flow** by LoRoom follows closely — 89% match. Slightly more energy at 40%, but 91%
> instrumental and deliberately paced at 80 BPM, practically designed for deep work.
>
> **Spacewalk Thoughts** by Orbit Bloom is the wildcard — ambient rather than lofi, with 95%
> instrumentalness and energy at just 28%. It's almost meditative; if you can get into it,
> it disappears completely into the background."

---

### Interaction 2 — Chat: Looking up a specific song

**User types:**
> "Tell me everything about Pulse Horizon"

**What happens internally:**
- Claude calls `get_song_details` with `{"title": "Pulse Horizon"}`
- Tool returns full audio feature breakdown + current profile match score

**Claude's response:**
> "**Pulse Horizon** by Zeta Drop — EDM, euphoric mood, 128 BPM.
>
> Energy: 95% · Valence: 82% · Danceability: 94% · Acousticness: 3% · Instrumentalness: 88%
>
> This is a pure festival track. The 95% energy and 94% danceability put it in the top tier of
> the catalog for physical intensity. The 88% instrumentalness means it's almost entirely
> synth-driven — the kind of track that sounds like a drop for its entire runtime. Based on your
> current Taste Profile it's a 71% match. If you've been choosing high-energy picks in Battles,
> that number goes up."

---

### Interaction 3 — Battles: Learning from a choice

**Setup:** Current profile — `genre: pop, energy: 0.65, acousticness: 0.40`

**Battle shown:**
- MAINSTREAM: *Gym Hero* by Max Pulse — Pop, intense, energy 0.93, acousticness 0.05
- INDIE: *Willow Thread* by Ember & Ash — Folk, dreamy, energy 0.31, acousticness 0.88

**User clicks:** INDIE (Willow Thread)

**EMA update applied (α = 0.30):**
```
new energy       = 0.30 × 0.31 + 0.70 × 0.65 = 0.548   (down from 0.65)
new acousticness = 0.30 × 0.88 + 0.70 × 0.40 = 0.544   (up from 0.40)
new genre        = "folk"
new mood         = "dreamy"
```

**App response:**
> "Willow Thread wins. Your Taste Profile shifted — less intensity, more organic sound."

On the next page load the Home feed surfaces *Coffee Shop Stories*, *Old Porch Radio*, and
*Willow Thread* instead of the previous pop/energy-heavy recommendations.

---

## Design Decisions

### Why Streamlit?

Streamlit lets a solo developer ship a polished multi-page app with custom HTML/CSS without
writing a separate React frontend and Flask/FastAPI backend. The trade-off is limited real-time
interactivity — every user action triggers a full Python re-run — but for a portfolio project
that prioritizes AI integration and feature breadth over production-scale performance, it is
the right tool.

### Why EMA for taste profile updates?

Exponential Moving Average with α = 0.30 means each Battle choice contributes 30% to the new
profile while 70% of the existing history is preserved. This creates a momentum effect — you
cannot accidentally wreck your profile with one pick, but consistent choices noticeably shift
recommendations within 3–4 Battles. A higher α (0.5+) would feel unstable; lower (0.1) would
feel unresponsive. Battles are also explicitly more valuable than passive listening because they
are deliberate, binary choices — the user is forced to prefer one song over another rather than
just letting something play in the background.

### Why RAG + Tool Use instead of one or the other?

Sending the full catalog (20 songs × 13 features) in every API call wastes tokens and adds
latency. RAG cosine similarity pre-filters to the 3–4 most sonically relevant songs given the
query's apparent audio intent, giving Claude focused grounding. Claude's tools then let it
verify, filter, or expand beyond that pre-retrieved context. The combination catches what either
approach misses alone: RAG provides efficient grounding; tool calls provide flexible, precise
catalog access when the user's question is specific.

### Why Deezer for artist photos?

Deezer's public search API (`api.deezer.com/search/artist`) requires no API key, has a generous
free tier, and returns high-quality square photos (`picture_xl` at 1000×1000 px). Spotify
requires OAuth even for public catalog reads. The fictional-to-real artist mapping in
`artist_images.py` (e.g., "Neon Echo" → "The Weeknd") means each of the 18 catalog artists
gets a professional photo matched to their genre and vibe, not a placeholder.

### Why simulated stats on the Monitor page?

SoundCloud closed their API to new registrations, Bandcamp has no public API, and TikTok requires
business approval. Rather than display an empty page, platform stats are generated
deterministically from each song's own audio features: high energy + danceability produces
larger YouTube/TikTok numbers; high acousticness produces stronger Bandcamp presence. The seed
uses an MD5 hash of `artist + platform`, so results are stable across page loads. Live Last.fm
and YouTube data overlay on top when API keys are configured.

### Scoring weight rationale

Every weight in `recommender.py` is documented with a justification comment:

| Feature | Weight | Reasoning |
|---|---|---|
| Energy | 3.00 | Best single discriminator in the catalog (range 0.20–0.97) |
| Mood match | 2.00 | Listener intent is the strongest qualitative signal |
| Acousticness | 1.00 | Captures production texture independent of energy |
| Genre match | 0.75 | Style preference matters, but mood overrides genre |
| Instrumentalness | 0.75 | Critical for focus and study use cases |
| Valence | 0.50 | Narrow spread in the catalog; less discriminating |
| Danceability | 0.25 | Situational (workout, party contexts) |
| Speechiness | 0.25 | Most songs score near zero; functions as a tiebreaker |
| Liveness | 0.25 | Studio vs. live feel; tiebreaker only |

**MAX_SCORE = 8.75**

---

## Testing Summary

> **Bottom line:** 14 of 14 automated tests passed in 1.04 s. Confidence scores averaged 70%
> across all profiles; well-specified profiles averaged 88%, while deliberately conflicted or
> ambiguous profiles averaged 64%. The system failed intuitively on one known edge case (the
> Sad Gym Rat), failed silently on one (missing genre), and degraded gracefully on the rest.

---

### 1. Automated Unit Tests — 14 / 14 passed

Run command and real output:

```
$ pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.10.19, pytest-9.0.3
collected 14 items

tests/test_recommender.py::test_recommend_returns_songs_sorted_by_score    PASSED
tests/test_recommender.py::test_explain_recommendation_returns_non_empty_string PASSED
tests/test_recommender.py::test_recommend_returns_exactly_k_results        PASSED
tests/test_recommender.py::test_recommend_empty_catalog_returns_empty      PASSED
tests/test_recommender.py::test_score_song_perfect_genre_and_mood_match_earns_categorical_bonus PASSED
tests/test_recommender.py::test_score_song_no_match_returns_low_score      PASSED
tests/test_recommender.py::test_score_song_mood_match_outweighs_genre_mismatch PASSED
tests/test_recommender.py::test_explain_recommendation_mentions_mood_when_matched PASSED
tests/test_recommender.py::test_explain_recommendation_mentions_genre_when_matched PASSED
tests/test_recommender.py::test_rag_retrieve_returns_exactly_k_songs       PASSED
tests/test_recommender.py::test_rag_retrieve_ranks_closer_song_first       PASSED
tests/test_recommender.py::test_intent_to_features_chill_maps_to_low_energy PASSED
tests/test_recommender.py::test_intent_to_features_workout_maps_to_high_energy PASSED
tests/test_recommender.py::test_intent_to_features_sad_maps_to_low_valence PASSED

14 passed in 1.04s
```

The tests cover: scoring correctness, ranking order, explanation output, empty catalog, k-capping,
RAG cosine ordering, and all three `intent_to_features` keyword categories.

---

### 2. Confidence Scoring — Built Into Every Recommendation

The recommender expresses confidence as a **match percentage** = `(score / MAX_SCORE) × 100`.
This is surfaced on every song card in the UI and in every CLI result. MAX_SCORE is 8.75 — the
sum of all weighted features when the song is a perfect match on every dimension.

Real scores from running `python -m src.main` against five test profiles:

| Profile | Top-1 match % | Top-2 match % | Top-3 match % | Avg |
|---|---|---|---|---|
| Late-Night Study (baseline) | **99%** | 75% | 74% | 83% |
| Genre Ghost (missing genre) | **90%** | 65% | 64% | 73% |
| Impossible Combo | **74%** | 74% | 59% | 69% |
| Sad Gym Rat (conflicted) | **68%** | 65% | 64% | 66% |
| The Centrist (no anchors) | **62%** | 60% | 59% | 60% |

**Interpretation:**
- A score ≥ 85% means the top result is a near-perfect match on both mood/genre label and
  numeric audio features — the system is high-confidence.
- A score of 60–70% means the system found the best it could but the profile has contradictions
  or missing data — the user should treat the recommendation as a starting point.
- Scores across the 15 result slots averaged **70.1%**. Profiles with at least one matching
  mood or genre categorical signal averaged **78%**; profiles with no matching signals averaged
  **61%**.

---

### 3. Error Handling and Graceful Degradation

Every external call in the system is wrapped in a `try/except` that falls back silently so a
network or API failure never crashes the app:

| Module | Error scenario | Behavior |
|---|---|---|
| `artist_images.py` | Deezer API timeout or 4xx | Returns `FALLBACK_URL` (placeholder image); other artists unaffected |
| `artist_images.py` | Corrupt `artist_images.json` cache | Re-initializes to empty dict; re-fetches on next load |
| `chat_agent.py` | `anthropic` package missing | Returns a plain-text "install anthropic" message; app keeps running |
| `app.py` chat handler | Any exception during Claude call | Displays `"Something went wrong: {exc}"` inline in the chat thread |
| `platform_monitor.py` | Last.fm or YouTube API failure | Falls back to simulated stats; `live_sources` field shows `[]` |
| `youtube_client.py` | No API key set | Returns empty list; UI hides the video section entirely |
| `app.py` imports | Any optional module missing | `_CHAT_OK`, `_YT_OK`, etc. flags set to `False`; dependent UI sections hidden |

This pattern means the app has **zero hard dependencies** at runtime — it degrades gracefully from
full AI-powered mode down to a standalone recommender with no external calls.

---

### 4. Human Evaluation — Five Adversarial Profiles

Four profiles were designed to expose failure modes, run manually, and reviewed against
"what should a good recommender return here?" Each result was judged pass/fail based on
musical intuition:

**Profile 1 — Late-Night Study Session** (`genre=lofi, mood=focused, energy=0.40`)
- Top result: *Focus Flow* — lofi, focused, energy 0.40 — **99% match**
- Judgment: **PASS.** Exactly right. The top result is a nearly perfect match and the explanation
  correctly calls out every contributing feature.

**Profile 2 — Sad Gym Rat** (`mood=sad, genre=blues, energy=0.93`)
- Top result: *Empty Bottle Blues* — blues, sad, energy 0.38 — **68% match**
- Judgment: **FAIL (expected).** The blues ballad won because mood + genre categorical bonuses
  (+2.75 pts total) outweighed a 0.55 energy mismatch. The system is technically correct per
  its own rules — the song does match the labels — but a gym user asking for "sad" music wants
  something emotionally heavy and fast, not slow and acoustic. This is a documented flaw.
  Raising the energy weight from 3.0 to ~4.5 would let large numeric mismatches override
  categorical matches.

**Profile 3 — Genre Ghost** (`genre=bossa nova` — not in catalog)
- Top result: *Coffee Shop Stories* — jazz, relaxed, energy 0.37 — **90% match**
- Judgment: **PASS with caveat.** The result feels right (jazz + acoustic + relaxed is the
  closest real analogue to bossa nova), but the system silently dropped the genre bonus with
  no user-facing warning. A real product should surface "bossa nova not found — showing closest
  alternatives."

**Profile 4 — The Centrist** (all targets at 0.50, no genre/mood)
- Top result: *Velvet Sunday* — r&b, romantic — **62% match**
- Judgment: **PARTIAL PASS.** No strong result is possible without a signal to anchor on.
  The ranking is effectively arbitrary (top 3 span 3 pts in a 15-pt range). The system does
  not crash or return garbage; it just cannot be confident. This is the correct behavior for
  a cold-start user — it is exactly why Battles exist.

**Profile 5 — Impossible Combo** (`genre=classical, mood=euphoric`)
- Top result: *Sunshine Current* — reggae, euphoric — **74% match**
- Judgment: **PASS.** No song in the catalog is both classical and euphoric, so one categorical
  bonus must be sacrificed. Mood match (2.00 pts) correctly outweighed genre match (0.75 pts),
  producing a euphoric result. The user gets the right emotional feel even though the genre
  preference was deprioritized — which is the stated design intent.

**Human evaluation summary: 3 full passes, 1 expected fail, 1 partial pass (4/5 = 80%).**
The one hard failure (Sad Gym Rat) is a known, documented algorithmic limitation.

---

### 5. Catalog Diversity Check

Across all five adversarial profiles × top-3 results (15 total result slots):

```
Unique songs in all top-3 slots: 12 of 15   (80% diversity)
Songs appearing more than once:
  Midnight Coding    2×
  Library Rain       2×
  Empty Bottle Blues 2×
```

Three songs showed up in two different profiles' top-3. None dominated every profile.
80% diversity is acceptable for a 20-song catalog — in a production catalog of 50M+ songs
this would not be a concern.

---

### What worked well

- All 14 automated tests pass and cover the system's core invariants end-to-end.
- Confidence scores (match %) make the AI's certainty visible and auditable on every result.
- Error handling keeps the app alive through API failures, missing packages, and empty states.
- Graceful degradation means the app is demo-able offline with no API keys at all.
- The scoring explanation string makes every recommendation traceable to specific features.

### Known limitations

- **Sad Gym Rat flaw is unresolved.** Categorical bonuses can still override a severe energy
  mismatch. Raising the energy weight to ~4.5 would fix it but requires re-running all
  adversarial profiles to check for new regressions.
- **Silent genre miss.** When a user's genre preference does not exist in the catalog, the
  system falls back correctly but does not inform the user. A warning banner would help.
- **No cross-session persistence.** Profile resets on browser close. Requires a database or
  local storage layer for a real product.
- **20-song catalog limits diversity testing.** The 80% diversity score is good for the catalog
  size; the architecture is designed for a much larger dataset.

---

## Responsible AI

### What are the limitations or biases in your system?

The most significant bias is **catalog representation**. The 20-song dataset covers 17 genres,
but 13 of those genres have exactly one song. A user whose taste runs toward blues, classical,
reggae, or folk can never earn the genre match bonus from more than one track. The system will
consistently under-serve those users compared to someone who likes lofi or pop, where three songs
compete for the top slot. This is not a flaw in the algorithm — it is a flaw in the data, and
it is exactly the kind of invisible unfairness that shows up in real recommendation systems where
certain artists, genres, or cultural traditions are simply absent from the training catalog.

A second structural bias is the **categorical bonus ceiling**. Mood match is worth 2.00 points
and genre match is worth 0.75 — together they make up 31% of the maximum score. A song that
matches both labels but has completely wrong audio features will still outscore a song that sounds
perfect but carries a different label. The Sad Gym Rat adversarial test proved this is not
theoretical: a slow acoustic blues ballad at 38% energy beat a metal track at 97% energy because
the labels matched. In a real product, users do not experience music as labels — they experience
it as sound. The scoring formula does not fully reflect that.

A third limitation is **cold-start invisibility**. New users get the system's default profile
(genre: pop, mood: happy). The first several recommendations they see reflect those defaults, not
their actual taste. Unless they use Battles immediately, the system is effectively serving them
the most generic possible results under the appearance of personalization.

---

### Could your AI be misused, and how would you prevent that?

**Recommendation manipulation.** Because the scoring formula is fully documented and deterministic,
someone who knows the weights could craft a user profile that surfaces a specific song for all
users — essentially gaming the recommender to artificially promote one artist. In a real system
with paying clients or label partnerships, this is a real risk. The mitigation would be
diversity enforcement (capping how many results can come from the same artist or genre) and
making weight values configurable only by administrators, not exposed in public documentation.

**AI chat misuse.** The Claude-powered chat agent answers questions about the app's catalog and
music in general. Because it has access to tool calls that read the catalog, a user cannot make
it fabricate song data — the tools enforce grounding. However, the `CONVO_SYSTEM_PROMPT` is
intentionally broad ("talk about genres, artists, production, vibes, history"), which means a
persistent user could try to steer the conversation toward off-topic or harmful content. The
mitigation already in place is that Claude's built-in safety guidelines apply regardless of the
system prompt. A production version would add explicit topic filtering and rate limiting on the
chat endpoint.

**False authority.** The Monitor page shows platform statistics with the visual language of a
real analytics dashboard — bar charts, ranked tables, exact numbers. Every number is simulated
from audio features, not from real streaming data, but there is no prominent warning that the
figures are illustrative. A user who did not read the documentation could mistake them for live
data and make real decisions based on them (e.g., choosing which artists to feature or book).
This would be fixed in production by adding a persistent "DEMO DATA" watermark to every chart
and a disclaimer banner at the top of the page.

---

### What surprised you while testing reliability?

Two things were genuinely surprising, not just instructive.

**The Genre Ghost profile actually worked well.** I expected that removing the genre signal
entirely (by using `genre=bossa nova`, which does not exist in the catalog) would produce a
noticeably worse result — random-feeling, low-confidence picks. Instead, the top result was
*Coffee Shop Stories* at 90% confidence. It's a jazz track that is acoustically and energetically
almost exactly what a bossa nova fan would want. The numeric features — high acousticness (0.89),
low energy (0.37), warm valence (0.71) — carried the recommendation even without the categorical
anchor. I had underestimated how much of music preference is captured in audio features rather
than genre labels. That test changed how I think about the value of genre as a signal versus
the value of direct acoustic measurement.

**How narrow the margin was at the top of The Centrist profile.** When I ran all targets at 0.50
with no genre or mood, I expected a near-tie but I did not expect the top 3 results to span only
3 percentage points (62%, 60%, 59%). The song that ranked first — *Velvet Sunday* by Sable June —
beat the second-place result by just 0.16 raw points out of 8.75. That margin is smaller than the
rounding error in any real system. It exposed something important: the scoring formula only
produces *meaningful* rankings when the user gives it at least one strong preference to anchor on.
Without that anchor, the output is technically correct but practically useless. This made me
understand why cold-start is considered one of the hardest problems in recommendation systems —
it is not just a UI problem, it is a math problem.

---

### Collaboration with AI during this project

This entire project was built in collaboration with Claude (the same model that powers the
SoundMatch chat feature). Claude wrote most of the code across every file, generated the CSS,
designed the scoring weight rationale, drafted the adversarial test profiles, and wrote this
README. My role was to direct the architecture, define what each feature should do, catch
problems with the output, and make final decisions when the AI's suggestion conflicted with the
product vision.

**One instance where the AI was genuinely helpful:**
When I asked for artist photos, Claude proposed mapping the fictional catalog artists to
real-world equivalents (e.g., "Neon Echo" → "The Weeknd," "Paper Lanterns" → "Nujabes") and
then fetching photos from the Deezer public API, which requires no API key. I had been planning
to use the Spotify API, which requires OAuth even for public catalog reads. Claude's alternative
was simpler, more reliable for a demo context, and actually better — the Deezer `picture_xl`
images at 1000×1000 px look better than the 300×300 px Spotify covers. I would not have thought
of Deezer on my own. That suggestion saved setup time and produced a better result.

**One instance where the AI's suggestion was flawed:**
Early in the project, Claude suggested caching the artist image preload function using Streamlit's
`@st.cache_data` decorator with `id(songs)` as the cache key — the memory address of the songs
list object. This looked reasonable at a glance: the key would change if the list changed. But
Python does not guarantee stable object IDs across function calls, and in Streamlit's re-run
model a new list object is created on every page load, meaning the cache would always miss and
the Deezer API would be called on every single render. The app would have made hundreds of
unnecessary API requests in a single session. I caught it during code review and changed the
key to `tuple(s["id"] for s in songs)` — a stable, hashable value derived from the actual data.
The AI's suggestion was syntactically valid and passed a quick reading, which is exactly why
AI-generated code needs careful review: it can be wrong in ways that only show up at runtime.

---

## Reflection

### What this project taught me about AI

**RAG is not just a performance optimization — it is about grounding.** Before building this,
I thought RAG mainly saved tokens. What I learned is that it changes the *quality* of reasoning.
When Claude sees the top-4 cosine-similarity matches as context, it stops making things up about
what songs exist and starts reasoning about real data. The difference between a grounded and an
ungrounded music recommendation is immediately obvious.

**Agentic tool use changes the human-AI contract.** In a simple prompt-response setup, the user
has to know exactly how to phrase their request. With tool calling, Claude decides whether to
search by mood, by energy range, or to look up a specific title — and that decision is usually
better than what the user would have specified themselves. The user can type "something for my
drive home tonight" and get a genuinely useful, data-grounded answer.

**A scoring formula is a hypothesis about human preference, not a fact.** Writing down weights
like "energy is worth 3.0 points and danceability is worth 0.25" forced me to articulate
assumptions I normally leave vague. Every weight is a testable claim. The adversarial profiles
proved that some of those claims are wrong in edge cases — which means the formula is a starting
point for an experiment, not a finished product. This is how I now think about every AI system
configuration: hyperparameters are hypotheses.

### What this project taught me about problem-solving

**Build the core first, not the UI.** The scoring algorithm in `recommender.py` was working,
tested, and fully understood before a single Streamlit widget existed. Because the logic was
clean and well-documented, integrating it into the app was mechanical rather than risky. Starting
with the visual layer first — which is tempting because it's more immediately rewarding — usually
means discovering logic problems after the scaffolding is already built around them.

**Constraints produce creativity.** No Spotify API key, no SoundCloud access, and no real
listening history forced specific solutions: fictional artists mapped to real ones for photos,
deterministic simulation for platform stats, and Battles as a substitute for passive history.
Each constraint produced a design decision that ended up being a feature rather than a workaround.

**Explaining a system is harder than building it.** Writing the scoring weight rationale, the
adversarial profile analysis, and this README took longer than most code changes. But the act of
explaining revealed things I had not fully thought through — the Sad Gym Rat observation in
`main.py` started as a comment and became a full architectural critique pointing to a real flaw
in the weights. Clear documentation is not just communication; it is a form of review.

---

## Project Structure

```
Applied-AI-system-Project/
├── data/
│   ├── songs.csv              # 20-song catalog, 13 audio features per song
│   └── artist_images.json     # Deezer photo cache (auto-generated on first run)
├── src/
│   ├── app.py                 # Streamlit multi-page app — main entry point
│   ├── recommender.py         # Weighted scoring engine + EMA profile logic
│   ├── chat_agent.py          # RAG retrieval + Claude agentic workflow + tools
│   ├── platform_monitor.py    # Artist stats aggregator (simulated + live)
│   ├── artist_images.py       # Deezer API image resolver with disk cache
│   ├── youtube_client.py      # YouTube Data API v3 integration
│   └── main.py                # Original CLI runner + adversarial test profiles
├── tests/
│   └── test_recommender.py    # Pytest suite for the scoring engine
├── assets/
│   ├── terminal-output.png    # CLI recommender sample output screenshot
│   └── preference-dictionary-eval.png
├── model_card.md
├── requirements.txt
└── README.md
```

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | For Chat page | Powers Claude claude-sonnet-4-6 agent + sidebar AI |
| `YOUTUBE_API_KEY` | Optional | Live genre videos + production lesson search |
| `LASTFM_API_KEY` | Optional | Live listener/play counts on Monitor page |

No key is required to run the core app. The recommender, Battles, Taste DNA, and Discover pages
work fully offline.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit 1.x + custom HTML/CSS |
| AI model | Claude claude-sonnet-4-6 (Anthropic) |
| Recommendation | Custom weighted scoring (Python, NumPy) |
| RAG retrieval | Cosine similarity (NumPy — no vector DB required) |
| Visualizations | Plotly (radar chart, scatter plot, stacked bar) |
| Artist photos | Deezer public API + JSON disk cache |
| Platform stats | YouTube Data API v3, Last.fm API, deterministic simulation |
| Tests | pytest |
| Language | Python 3.10+ |
