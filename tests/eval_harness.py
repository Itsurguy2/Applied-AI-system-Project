"""
SoundMatch Evaluation Harness
==============================
Runs 15 predefined scenarios through the recommender engine and RAG utilities.
No API key required — tests pure algorithmic logic only.

Usage:
  python tests/eval_harness.py

Exit code 0  → all scenarios pass
Exit code 1  → one or more scenarios fail
"""

import io
import sys
import time
from pathlib import Path

# Force UTF-8 stdout so ANSI colour codes and any Unicode print cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from recommender import MAX_SCORE, load_songs, recommend_songs, score_song
from chat_agent import intent_to_features, rag_retrieve, load_genre_knowledge, rag_retrieve_docs

SONGS_CSV = ROOT / "data" / "songs.csv"

# ── Colour helpers (graceful fallback on terminals without ANSI) ──────────────
def _green(s):  return f"\033[32m{s}\033[0m"
def _red(s):    return f"\033[31m{s}\033[0m"
def _bold(s):   return f"\033[1m{s}\033[0m"
def _dim(s):    return f"\033[2m{s}\033[0m"

PASS_LABEL = _green("PASS")
FAIL_LABEL = _red("FAIL")

# ── Scenario runner ───────────────────────────────────────────────────────────

class Result:
    def __init__(self, name, passed, confidence=None, note=""):
        self.name       = name
        self.passed     = passed
        self.confidence = confidence   # float 0-100, or None
        self.note       = note


def run_scenario(idx, name, fn) -> Result:
    """Execute one scenario function and return a Result."""
    try:
        return fn()
    except Exception as exc:
        return Result(name, False, note=f"EXCEPTION: {exc}")


# ── Individual scenarios ──────────────────────────────────────────────────────

def songs():
    return load_songs(str(SONGS_CSV))


def s01_pop_happy_baseline(catalog):
    profile = {
        "genre": "pop", "mood": "happy",
        "target_energy": 0.80, "target_valence": 0.80,
        "target_acousticness": 0.20, "target_danceability": 0.80,
        "target_instrumentalness": 0.05, "target_speechiness": 0.05,
        "target_liveness": 0.10,
    }
    results = recommend_songs(profile, catalog, k=5)
    top = results[0][0]
    sc, _ = score_song(top, profile)
    confidence = round(sc / MAX_SCORE * 100, 1)
    passed = top["genre"] == "pop" or top["mood"] == "happy"
    return Result(
        "Pop/Happy Baseline", passed, confidence,
        f'"{top["title"]}" ({top["genre"]}/{top["mood"]})',
    )


def s02_lofi_focused(catalog):
    profile = {
        "genre": "lofi", "mood": "focused",
        "target_energy": 0.40, "target_acousticness": 0.78,
        "target_instrumentalness": 0.88, "target_valence": 0.58,
        "target_danceability": 0.60, "target_speechiness": 0.03,
        "target_liveness": 0.10,
    }
    results = recommend_songs(profile, catalog, k=5)
    top = results[0][0]
    sc, _ = score_song(top, profile)
    confidence = round(sc / MAX_SCORE * 100, 1)
    passed = top["acousticness"] > 0.65 and top["energy"] < 0.55
    return Result(
        "Lofi/Focused Specialist", passed, confidence,
        f'acousticness={top["acousticness"]:.2f} > 0.65  |  energy={top["energy"]:.2f} < 0.55',
    )


def s03_high_energy(catalog):
    profile = {
        "genre": "edm", "mood": "euphoric",
        "target_energy": 0.95, "target_valence": 0.85,
        "target_danceability": 0.92, "target_acousticness": 0.04,
        "target_instrumentalness": 0.80, "target_speechiness": 0.04,
        "target_liveness": 0.08,
    }
    results = recommend_songs(profile, catalog, k=5)
    top = results[0][0]
    sc, _ = score_song(top, profile)
    confidence = round(sc / MAX_SCORE * 100, 1)
    passed = top["energy"] > 0.85
    return Result(
        "High-Energy Seeker", passed, confidence,
        f'"{top["title"]}" energy={top["energy"]:.2f} > 0.85',
    )


def s04_classical_melancholic(catalog):
    profile = {
        "genre": "classical", "mood": "melancholic",
        "target_energy": 0.20, "target_acousticness": 0.95,
        "target_instrumentalness": 0.95, "target_valence": 0.35,
        "target_danceability": 0.25, "target_speechiness": 0.02,
        "target_liveness": 0.13,
    }
    results = recommend_songs(profile, catalog, k=5)
    top = results[0][0]
    sc, _ = score_song(top, profile)
    confidence = round(sc / MAX_SCORE * 100, 1)
    passed = top["energy"] < 0.30 or top["acousticness"] > 0.85
    return Result(
        "Classical/Melancholic", passed, confidence,
        f'"{top["title"]}" energy={top["energy"]:.2f}  acousticness={top["acousticness"]:.2f}',
    )


def s05_genre_ghost(catalog):
    """Genre not in catalog — numeric matching should still surface relevant songs."""
    profile = {
        "genre": "bossa nova", "mood": "relaxed",
        "target_energy": 0.38, "target_acousticness": 0.85,
        "target_instrumentalness": 0.70, "target_valence": 0.68,
        "target_danceability": 0.55, "target_speechiness": 0.04,
        "target_liveness": 0.18,
    }
    results = recommend_songs(profile, catalog, k=5)
    top = results[0][0]
    sc, _ = score_song(top, profile)
    confidence = round(sc / MAX_SCORE * 100, 1)
    # No genre match possible, but acoustic/low-energy song should still surface
    passed = len(results) == 5 and top["energy"] < 0.55
    return Result(
        "Genre Ghost (bossa nova)", passed, confidence,
        f'{len(results)} results via numeric matching  |  top energy={top["energy"]:.2f}',
    )


def s06_centrist(catalog):
    """All features at midpoint — tests cold-start behavior (narrow score band expected)."""
    profile = {
        "genre": "ambient", "mood": "mellow",
        "target_energy": 0.50, "target_acousticness": 0.50,
        "target_instrumentalness": 0.50, "target_valence": 0.50,
        "target_danceability": 0.50, "target_speechiness": 0.05,
        "target_liveness": 0.12,
    }
    results = recommend_songs(profile, catalog, k=5)
    top_score = results[0][1]
    bottom_score = results[-1][1]
    band = round((top_score - bottom_score) / MAX_SCORE * 100, 1)
    confidence = round(top_score / MAX_SCORE * 100, 1)
    # Should return results without crashing; narrow band confirms cold-start problem
    passed = len(results) == 5 and band < 20.0
    return Result(
        "Centrist (all 0.50, cold-start)", passed, confidence,
        f"score band={band}% across top 5  (< 20% expected for no-anchor query)",
    )


def s07_impossible_combo(catalog):
    """Contradictory features — no crash, graceful degradation."""
    profile = {
        "genre": "metal", "mood": "melancholic",
        "target_energy": 0.97, "target_acousticness": 0.97,
        "target_instrumentalness": 0.97, "target_valence": 0.10,
        "target_danceability": 0.10, "target_speechiness": 0.01,
        "target_liveness": 0.30,
    }
    results = recommend_songs(profile, catalog, k=5)
    passed = len(results) == 5
    return Result(
        "Impossible Combo (contradictory features)", passed, None,
        f"no crash, {len(results)} results returned",
    )


def s08_k_limit_one(catalog):
    profile = {"genre": "pop", "mood": "happy", "target_energy": 0.80,
               "target_valence": 0.80, "target_acousticness": 0.20,
               "target_danceability": 0.80, "target_instrumentalness": 0.05,
               "target_speechiness": 0.05, "target_liveness": 0.10}
    results = recommend_songs(profile, catalog, k=1)
    passed = len(results) == 1
    return Result("K-Limiting: k=1", passed, None, f"returned {len(results)} result")


def s09_k_limit_three(catalog):
    profile = {"genre": "pop", "mood": "happy", "target_energy": 0.80,
               "target_valence": 0.80, "target_acousticness": 0.20,
               "target_danceability": 0.80, "target_instrumentalness": 0.05,
               "target_speechiness": 0.05, "target_liveness": 0.10}
    results = recommend_songs(profile, catalog, k=3)
    passed = len(results) == 3
    return Result("K-Limiting: k=3", passed, None, f"returned {len(results)} results")


def s10_score_separation(catalog):
    """Top result should meaningfully outscore 5th result — proves ranking is not random."""
    profile = {
        "genre": "lofi", "mood": "chill",
        "target_energy": 0.40, "target_acousticness": 0.75,
        "target_instrumentalness": 0.85, "target_valence": 0.60,
        "target_danceability": 0.58, "target_speechiness": 0.03,
        "target_liveness": 0.09,
    }
    results = recommend_songs(profile, catalog, k=5)
    gap = results[0][1] - results[-1][1]
    gap_pct = round(gap / MAX_SCORE * 100, 1)
    passed = gap_pct >= 5.0
    return Result(
        "Score Separation (rank quality)", passed, None,
        f"gap={gap:.3f} pts ({gap_pct}% of MAX_SCORE)  threshold=5%",
    )


def s11_empty_catalog():
    profile = {"genre": "pop", "mood": "happy", "target_energy": 0.80,
               "target_valence": 0.80, "target_acousticness": 0.20,
               "target_danceability": 0.80, "target_instrumentalness": 0.05,
               "target_speechiness": 0.05, "target_liveness": 0.10}
    results = recommend_songs(profile, [], k=5)
    passed = results == []
    return Result("Empty Catalog", passed, None, "returned empty list as expected")


def s12_rag_high_energy(catalog):
    query = {"energy": 0.95, "valence": 0.82, "acousticness": 0.04,
             "danceability": 0.92, "instrumentalness": 0.80,
             "speechiness": 0.04, "liveness": 0.08}
    retrieved = rag_retrieve(query, catalog, k=3)
    top = retrieved[0]
    passed = top["energy"] > 0.80
    return Result(
        "RAG Ordering: high-energy query", passed, None,
        f'cosine sim -> "{top["title"]}" energy={top["energy"]:.2f} (> 0.80)',
    )


def s13_rag_low_energy(catalog):
    query = {"energy": 0.20, "valence": 0.38, "acousticness": 0.97,
             "danceability": 0.22, "instrumentalness": 0.97,
             "speechiness": 0.02, "liveness": 0.13}
    retrieved = rag_retrieve(query, catalog, k=3)
    top = retrieved[0]
    passed = top["energy"] < 0.40
    return Result(
        "RAG Ordering: low-energy/acoustic query", passed, None,
        f'cosine sim -> "{top["title"]}" energy={top["energy"]:.2f} (< 0.40)',
    )


def s14_intent_chill_study():
    feats = intent_to_features("I want something chill to study to")
    passed = feats["energy"] < 0.50 and feats["acousticness"] > 0.50
    return Result(
        "Intent Mapping: chill study", passed, None,
        f'energy={feats["energy"]:.2f} < 0.50  |  acousticness={feats["acousticness"]:.2f} > 0.50',
    )


def s15_intent_gym():
    feats = intent_to_features("hard gym workout playlist, intense")
    passed = feats["energy"] > 0.80 and feats["danceability"] > 0.70
    return Result(
        "Intent Mapping: gym workout", passed, None,
        f'energy={feats["energy"]:.2f} > 0.80  |  danceability={feats["danceability"]:.2f} > 0.70',
    )


def s_bonus_rag_docs():
    """RAG Enhancement: genre knowledge docs loaded and retrieved correctly."""
    knowledge = load_genre_knowledge()
    if not knowledge:
        return Result("RAG Docs: genre knowledge loaded", False, None, "genre_knowledge.json not found")
    # A jazz query should retrieve the jazz doc
    docs = rag_retrieve_docs("something jazzy for a coffee shop", knowledge, top_n=2)
    retrieved_genres = {d["name"] for d in docs}
    passed = len(docs) > 0 and any("jazz" in g.lower() or "Jazz" in g for g in retrieved_genres)
    return Result(
        "RAG Enhancement: genre doc retrieval", passed, None,
        f"{len(knowledge)} genres loaded  |  retrieved: {[d['name'] for d in docs]}",
    )


# ── Main runner ───────────────────────────────────────────────────────────────

SECTION_1 = "Recommender Quality"
SECTION_2 = "RAG & Intent Mapping"
SECTION_3 = "RAG Enhancement (Genre Docs)"

SCENARIOS = [
    # Section 1
    (SECTION_1, "Pop/Happy Baseline",             s01_pop_happy_baseline),
    (SECTION_1, "Lofi/Focused Specialist",         s02_lofi_focused),
    (SECTION_1, "High-Energy Seeker",              s03_high_energy),
    (SECTION_1, "Classical/Melancholic",           s04_classical_melancholic),
    (SECTION_1, "Genre Ghost (bossa nova)",        s05_genre_ghost),
    (SECTION_1, "Centrist (all 0.50, cold-start)", s06_centrist),
    (SECTION_1, "Impossible Combo",                s07_impossible_combo),
    (SECTION_1, "K-Limiting: k=1",                s08_k_limit_one),
    (SECTION_1, "K-Limiting: k=3",                s09_k_limit_three),
    (SECTION_1, "Score Separation (rank quality)", s10_score_separation),
    (SECTION_1, "Empty Catalog",                   s11_empty_catalog),
    # Section 2
    (SECTION_2, "RAG Ordering: high-energy",       s12_rag_high_energy),
    (SECTION_2, "RAG Ordering: low-energy",        s13_rag_low_energy),
    (SECTION_2, "Intent Mapping: chill study",     s14_intent_chill_study),
    (SECTION_2, "Intent Mapping: gym workout",     s15_intent_gym),
    # Section 3
    (SECTION_3, "RAG Enhancement: genre docs",     s_bonus_rag_docs),
]

COL_W = 42   # scenario name column width


def _row(idx, result):
    label = PASS_LABEL if result.passed else FAIL_LABEL
    conf  = f"{result.confidence:5.1f}%" if result.confidence is not None else "    —  "
    note  = _dim(result.note) if result.note else ""
    return f"  #{idx:02d}  {result.name:<{COL_W}}  {label}  {conf}   {note}"


def main():
    t_start = time.time()
    catalog = load_songs(str(SONGS_CSV))

    results: list[tuple[str, Result]] = []
    current_section = None

    print()
    print(_bold("=" * 72))
    print(_bold("  SOUNDMATCH EVALUATION HARNESS"))
    print(f"  {len(SCENARIOS)} scenarios  |  catalog: {len(catalog)} songs  |  MAX_SCORE={MAX_SCORE}")
    print(_bold("=" * 72))

    idx = 0
    for section, name, fn in SCENARIOS:
        if section != current_section:
            current_section = section
            print()
            print(f"  {_bold(section)}")
            print("  " + "-" * 68)

        idx += 1
        # Inject catalog where needed
        if fn.__code__.co_argcount > 0:
            result = fn(catalog)
        else:
            result = fn()

        results.append((section, result))
        print(_row(idx, result))

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    passed_all = [r for _, r in results if r.passed]
    failed_all = [r for _, r in results if not r.passed]
    conf_vals  = [r.confidence for _, r in results if r.confidence is not None]
    avg_conf   = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0

    print()
    print(_bold("=" * 72))
    passed_str = _green(f"{len(passed_all)}/{len(results)} passed ({len(passed_all)/len(results)*100:.0f}%)")
    print(f"  RESULTS: {passed_str}   Avg confidence: {avg_conf:.1f}%   Runtime: {elapsed:.3f} s")
    if failed_all:
        print()
        print(_red("  FAILURES:"))
        for r in failed_all:
            print(f"    • {r.name} — {r.note}")
    print(_bold("=" * 72))
    print()

    sys.exit(0 if not failed_all else 1)


if __name__ == "__main__":
    main()
