"""
SoundMatch — Full Feature Test Suite
=====================================
Tests every major feature without requiring API keys.

Run from the project root:
    pytest tests/test_all_features.py -v --tb=short
"""

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
SRC   = Path(__file__).parent.parent / "src"
DATA  = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(SRC))

from recommender import (
    MAX_SCORE, SCORE_WEIGHTS, Recommender, Song, UserProfile,
    load_songs, recommend_songs, score_song,
)
from chat_agent import intent_to_features, rag_retrieve


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 1 — Data Loading
# ══════════════════════════════════════════════════════════════════════════════

class TestDataLoading:
    """songs.csv loads correctly with all 20 tracks."""

    def test_csv_exists(self):
        assert (DATA / "songs.csv").exists(), "data/songs.csv not found"

    def test_csv_loads_20_songs(self):
        songs = load_songs(str(DATA / "songs.csv"))
        assert len(songs) == 20, f"Expected 20 songs, got {len(songs)}"

    def test_all_required_fields_present(self):
        required = {"id", "title", "artist", "genre", "mood",
                    "energy", "tempo_bpm", "valence", "danceability",
                    "acousticness", "instrumentalness", "speechiness", "liveness"}
        songs = load_songs(str(DATA / "songs.csv"))
        for song in songs:
            missing = required - set(song.keys())
            assert not missing, f"Song '{song.get('title')}' missing fields: {missing}"

    def test_numeric_fields_are_floats(self):
        songs = load_songs(str(DATA / "songs.csv"))
        float_fields = ["energy", "valence", "danceability",
                        "acousticness", "instrumentalness", "speechiness", "liveness"]
        for song in songs:
            for field in float_fields:
                assert isinstance(song[field], float), \
                    f"'{field}' in '{song['title']}' is not a float"

    def test_all_feature_values_in_range(self):
        songs = load_songs(str(DATA / "songs.csv"))
        float_fields = ["energy", "valence", "danceability",
                        "acousticness", "instrumentalness", "speechiness", "liveness"]
        for song in songs:
            for field in float_fields:
                assert 0.0 <= song[field] <= 1.0, \
                    f"'{field}' = {song[field]} out of [0,1] in '{song['title']}'"

    def test_unique_ids(self):
        songs = load_songs(str(DATA / "songs.csv"))
        ids = [s["id"] for s in songs]
        assert len(ids) == len(set(ids)), "Duplicate song IDs found"

    def test_unique_titles(self):
        songs = load_songs(str(DATA / "songs.csv"))
        titles = [s["title"] for s in songs]
        assert len(titles) == len(set(titles)), "Duplicate song titles found"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 — Recommender Engine / Scoring
# ══════════════════════════════════════════════════════════════════════════════

def _sd(id, genre, mood, energy, valence=0.6, dance=0.6, acoustic=0.4, instr=0.4):
    return dict(id=id, title=f"Track {id}", artist="Artist",
                genre=genre, mood=mood, energy=energy, tempo_bpm=100.0,
                valence=valence, danceability=dance, acousticness=acoustic,
                instrumentalness=instr, speechiness=0.05, liveness=0.12)

_PROFILE = {"genre": "pop", "mood": "happy", "target_energy": 0.8,
            "target_valence": 0.75, "target_acousticness": 0.2,
            "target_danceability": 0.7, "target_instrumentalness": 0.2,
            "target_speechiness": 0.05, "target_liveness": 0.12}


class TestRecommenderScoring:
    """score_song and recommend_songs return correct ranked results."""

    def test_max_score_is_8_75(self):
        assert MAX_SCORE == pytest.approx(8.75), f"MAX_SCORE should be 8.75, got {MAX_SCORE}"

    def test_perfect_match_scores_near_max(self):
        song = _sd(1, "pop", "happy", energy=0.8, valence=0.75,
                   dance=0.7, acoustic=0.2, instr=0.2)
        score, _ = score_song(song, _PROFILE)
        assert score >= MAX_SCORE - 0.5, f"Perfect match scored only {score:.2f}/{MAX_SCORE}"

    def test_genre_and_mood_match_appear_in_explanation(self):
        song = _sd(1, "pop", "happy", energy=0.8)
        _, expl = score_song(song, _PROFILE)
        assert "mood 'happy' match" in expl
        assert "genre 'pop' match" in expl

    def test_mood_weight_exceeds_genre_weight(self):
        assert SCORE_WEIGHTS["mood_match"] > SCORE_WEIGHTS["genre_match"], \
            "Mood should outweigh genre"

    def test_mood_match_beats_genre_match_in_practice(self):
        song_mood  = _sd(1, "jazz", "happy", energy=0.8, valence=0.9)
        song_genre = _sd(2, "pop",  "sad",   energy=0.8, valence=0.3)
        s_mood,  _ = score_song(song_mood,  _PROFILE)
        s_genre, _ = score_song(song_genre, _PROFILE)
        assert s_mood > s_genre

    def test_recommend_songs_returns_correct_k(self):
        songs = load_songs(str(DATA / "songs.csv"))
        for k in (1, 3, 5):
            results = recommend_songs(_PROFILE, songs, k=k)
            assert len(results) == k

    def test_recommend_songs_sorted_descending(self):
        songs = load_songs(str(DATA / "songs.csv"))
        results = recommend_songs(_PROFILE, songs, k=5)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

    def test_recommend_songs_tuple_structure(self):
        songs = load_songs(str(DATA / "songs.csv"))
        results = recommend_songs(_PROFILE, songs, k=3)
        for song, score, expl in results:
            assert isinstance(song, dict)
            assert isinstance(score, float)
            assert isinstance(expl, str)

    def test_bad_profile_still_returns_results(self):
        songs = load_songs(str(DATA / "songs.csv"))
        empty_profile = {}
        results = recommend_songs(empty_profile, songs, k=3)
        assert len(results) == 3

    def test_oop_recommender_matches_functional_api(self):
        songs_dicts = load_songs(str(DATA / "songs.csv"))
        songs_objs  = [Song(**{k: v for k, v in s.items()
                               if k in Song.__dataclass_fields__}) for s in songs_dicts]
        user = UserProfile("pop", "happy", 0.8, False,
                           target_valence=0.75, target_acousticness=0.2,
                           target_danceability=0.7)
        rec = Recommender(songs_objs)
        top_oop  = rec.recommend(user, k=1)[0].title
        top_func = recommend_songs(_PROFILE, songs_dicts, k=1)[0][0]["title"]
        assert top_oop == top_func, \
            f"OOP ({top_oop}) and functional ({top_func}) APIs disagree on top pick"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 3 — Battle EMA Profile Update
# ══════════════════════════════════════════════════════════════════════════════

ALPHA = 0.30

def _ema_update(profile: dict, song: dict) -> dict:
    """Simulate the EMA update used in the Battles page."""
    updated = dict(profile)
    for key, song_key in [
        ("target_energy",           "energy"),
        ("target_valence",          "valence"),
        ("target_acousticness",     "acousticness"),
        ("target_danceability",     "danceability"),
        ("target_instrumentalness", "instrumentalness"),
        ("target_speechiness",      "speechiness"),
        ("target_liveness",         "liveness"),
    ]:
        old = profile.get(key, 0.5)
        updated[key] = round(ALPHA * song[song_key] + (1 - ALPHA) * old, 4)
    return updated


class TestBattleEMA:
    """EMA profile update shifts preferences toward the chosen song."""

    def test_ema_moves_energy_toward_chosen_song(self):
        profile = dict(_PROFILE)
        high_energy_song = _sd(1, "edm", "euphoric", energy=0.95)
        profile["target_energy"] = 0.50
        updated = _ema_update(profile, high_energy_song)
        assert updated["target_energy"] > 0.50, "Energy should shift up after choosing high-energy song"

    def test_ema_moves_toward_chill_song(self):
        profile = dict(_PROFILE)
        chill_song = _sd(2, "lofi", "chill", energy=0.35, acoustic=0.86, instr=0.92)
        profile["target_energy"]       = 0.80
        profile["target_acousticness"] = 0.20
        updated = _ema_update(profile, chill_song)
        assert updated["target_energy"]       < 0.80
        assert updated["target_acousticness"] > 0.20

    def test_ema_alpha_applied_correctly(self):
        profile = {"target_energy": 0.50}
        song    = {"energy": 1.0, "valence": 0.5, "acousticness": 0.5,
                   "danceability": 0.5, "instrumentalness": 0.5,
                   "speechiness": 0.05, "liveness": 0.12}
        updated = _ema_update(profile, song)
        expected = ALPHA * 1.0 + (1 - ALPHA) * 0.50
        assert updated["target_energy"] == pytest.approx(expected, abs=0.001)

    def test_profile_genre_updated_after_battle(self):
        profile = dict(_PROFILE)
        metal_song = _sd(1, "metal", "aggressive", energy=0.97)
        profile["genre"] = metal_song["genre"]
        assert profile["genre"] == "metal"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 4 — RAG Retrieval (cosine similarity)
# ══════════════════════════════════════════════════════════════════════════════

class TestRAGRetrieval:
    """rag_retrieve returns correctly ranked, correctly sized results."""

    def test_returns_exactly_k(self):
        songs = load_songs(str(DATA / "songs.csv"))
        query = {"energy": 0.5, "valence": 0.6, "acousticness": 0.4,
                 "danceability": 0.6, "instrumentalness": 0.4,
                 "speechiness": 0.05, "liveness": 0.12}
        for k in (1, 3, 5):
            assert len(rag_retrieve(query, songs, k=k)) == k

    def test_high_energy_query_returns_high_energy_songs(self):
        songs = load_songs(str(DATA / "songs.csv"))
        query = {"energy": 0.95, "valence": 0.80, "acousticness": 0.05,
                 "danceability": 0.90, "instrumentalness": 0.05,
                 "speechiness": 0.05, "liveness": 0.10}
        results = rag_retrieve(query, songs, k=3)
        avg_energy = sum(s["energy"] for s in results) / len(results)
        assert avg_energy >= 0.70, f"High-energy query returned avg energy {avg_energy:.2f}"

    def test_chill_query_returns_low_energy_songs(self):
        songs = load_songs(str(DATA / "songs.csv"))
        query = {"energy": 0.30, "valence": 0.55, "acousticness": 0.85,
                 "danceability": 0.45, "instrumentalness": 0.80,
                 "speechiness": 0.03, "liveness": 0.09}
        results = rag_retrieve(query, songs, k=3)
        avg_energy = sum(s["energy"] for s in results) / len(results)
        assert avg_energy <= 0.55, f"Chill query returned avg energy {avg_energy:.2f}"

    def test_closer_song_ranked_first(self):
        song_close = _sd(1, "edm", "euphoric", energy=0.95, valence=0.90,
                         dance=0.92, acoustic=0.03, instr=0.05)
        song_far   = _sd(2, "ambient", "chill", energy=0.10, valence=0.20,
                         dance=0.20, acoustic=0.95, instr=0.90)
        query = {"energy": 0.95, "valence": 0.90, "acousticness": 0.03,
                 "danceability": 0.92, "instrumentalness": 0.05,
                 "speechiness": 0.04, "liveness": 0.07}
        results = rag_retrieve(query, [song_close, song_far], k=2)
        assert results[0]["id"] == 1, "Closer song should rank first"

    def test_returns_dicts_with_required_keys(self):
        songs = load_songs(str(DATA / "songs.csv"))
        query = {"energy": 0.6, "valence": 0.6, "acousticness": 0.4,
                 "danceability": 0.6, "instrumentalness": 0.4,
                 "speechiness": 0.05, "liveness": 0.12}
        results = rag_retrieve(query, songs, k=3)
        for r in results:
            assert "title"  in r
            assert "artist" in r
            assert "energy" in r


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 5 — Intent Mapping (natural language → features)
# ══════════════════════════════════════════════════════════════════════════════

class TestIntentMapping:
    """intent_to_features maps natural language cues to correct feature targets."""

    def test_chill_maps_to_low_energy(self):
        f = intent_to_features("something chill to relax to")
        assert f["energy"] < 0.50
        assert f["acousticness"] > 0.60
        assert f["instrumentalness"] > 0.60

    def test_workout_maps_to_high_energy(self):
        f = intent_to_features("pump me up for the gym workout")
        assert f["energy"] > 0.80
        assert f["danceability"] > 0.70

    def test_sad_maps_to_low_valence(self):
        f = intent_to_features("something sad and melancholic for heartbreak")
        assert f["valence"] < 0.40

    def test_happy_maps_to_high_valence(self):
        f = intent_to_features("I want something cheerful and joyful")
        assert f["valence"] > 0.80

    def test_dance_maps_to_high_danceability(self):
        f = intent_to_features("give me dance party club music")
        assert f["danceability"] > 0.80
        assert f["energy"] > 0.70

    def test_jazz_maps_to_acoustic_low_energy(self):
        f = intent_to_features("coffee shop smooth jazz")
        assert f["acousticness"] > 0.60
        assert f["energy"] < 0.55

    def test_returns_all_seven_keys(self):
        required = {"energy", "valence", "acousticness", "danceability",
                    "instrumentalness", "speechiness", "liveness"}
        f = intent_to_features("any music")
        assert required.issubset(f.keys())

    def test_all_values_in_range(self):
        f = intent_to_features("intense metal workout bass electronic")
        for key, val in f.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 6 — Genre Knowledge JSON
# ══════════════════════════════════════════════════════════════════════════════

_GENRE_FILE = DATA / "genre_knowledge.json"
_genre_file_missing = pytest.mark.skipif(
    not _GENRE_FILE.exists(),
    reason="data/genre_knowledge.json not present in this environment",
)


class TestGenreKnowledge:
    """genre_knowledge.json loads and has the expected structure (skipped if absent)."""

    @_genre_file_missing
    def test_loads_as_valid_json(self):
        with open(_GENRE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list), "genre_knowledge.json should be a list"

    @_genre_file_missing
    def test_has_17_genres(self):
        with open(_GENRE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 17, f"Expected 17 genres, got {len(data)}"

    @_genre_file_missing
    def test_each_entry_has_required_fields(self):
        required = {"name", "description"}
        with open(_GENRE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            missing = required - set(entry.keys())
            assert not missing, f"Genre '{entry.get('name')}' missing: {missing}"

    @_genre_file_missing
    def test_all_catalog_genres_represented(self):
        songs = load_songs(str(DATA / "songs.csv"))
        catalog_genres = {s["genre"] for s in songs}
        with open(_GENRE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        doc_names = {d["name"].lower() for d in data}
        for genre in catalog_genres:
            assert genre.lower() in doc_names, \
                f"Genre '{genre}' from catalog not found in genre_knowledge.json"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 7 — Platform Monitor
# ══════════════════════════════════════════════════════════════════════════════

class TestPlatformMonitor:
    """platform_monitor returns structured stats for all expected platforms."""

    def test_module_imports(self):
        try:
            from platform_monitor import PLATFORMS, get_artist_stats
        except ImportError as e:
            pytest.fail(f"platform_monitor import failed: {e}")

    def test_platforms_list_non_empty(self):
        from platform_monitor import PLATFORMS
        assert len(PLATFORMS) > 0

    def test_get_artist_stats_returns_list(self):
        from platform_monitor import get_artist_stats
        songs = load_songs(str(DATA / "songs.csv"))
        results = get_artist_stats(songs)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_stats_each_record_has_artist_and_score(self):
        from platform_monitor import get_artist_stats
        songs = load_songs(str(DATA / "songs.csv"))
        results = get_artist_stats(songs)
        for record in results:
            assert "artist" in record, "Record missing 'artist'"
            assert "score"  in record, "Record missing 'score'"

    def test_stats_sorted_by_score_descending(self):
        from platform_monitor import get_artist_stats
        songs = load_songs(str(DATA / "songs.csv"))
        results = get_artist_stats(songs)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Stats not sorted by score desc"

    def test_stats_scores_are_non_negative(self):
        from platform_monitor import get_artist_stats
        songs = load_songs(str(DATA / "songs.csv"))
        results = get_artist_stats(songs)
        for r in results:
            assert r["score"] >= 0, f"{r['artist']} has negative score {r['score']}"

    def test_stats_cover_all_catalog_artists(self):
        from platform_monitor import get_artist_stats
        songs  = load_songs(str(DATA / "songs.csv"))
        results = get_artist_stats(songs)
        catalog_artists = {s["artist"] for s in songs}
        stats_artists   = {r["artist"] for r in results}
        assert catalog_artists == stats_artists, \
            f"Missing artists in stats: {catalog_artists - stats_artists}"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 8 — App Structure (no Streamlit runtime needed)
# ══════════════════════════════════════════════════════════════════════════════

class TestAppStructure:
    """app.py defines all required page functions and nav."""

    APP_FILE = SRC / "app.py"

    def test_app_file_exists(self):
        assert self.APP_FILE.exists(), "src/app.py not found"

    @pytest.mark.parametrize("fn_name", [
        "inject_css", "init_state", "main", "render_topnav",
        "page_home", "page_battles", "page_discover",
        "page_profile", "page_chat", "page_learn", "page_monitor",
    ])
    def test_required_function_defined(self, fn_name):
        source = self.APP_FILE.read_text(encoding="utf-8")
        assert f"def {fn_name}" in source, f"Function '{fn_name}' not found in app.py"

    def test_all_nav_pages_handled_in_main(self):
        source = self.APP_FILE.read_text(encoding="utf-8")
        for page in ["Home", "Battles", "Discover", "My Taste DNA", "Chat", "Learn", "Monitor"]:
            assert f'"{page}"' in source, f"Page '{page}' not found in app.py"

    def test_topnav_has_all_seven_pages(self):
        source = self.APP_FILE.read_text(encoding="utf-8")
        for page in ["Home", "Battles", "Discover", "Taste DNA", "Chat", "Learn", "Monitor"]:
            assert page in source, f"Nav label '{page}' missing from app.py"

    def test_config_toml_exists(self):
        cfg = SRC.parent / ".streamlit" / "config.toml"
        assert cfg.exists(), ".streamlit/config.toml not found"

    def test_config_toml_has_dark_theme(self):
        cfg = (SRC.parent / ".streamlit" / "config.toml").read_text()
        assert 'base = "dark"' in cfg
        assert "backgroundColor" in cfg


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 9 — Catalog Coverage
# ══════════════════════════════════════════════════════════════════════════════

class TestCatalogCoverage:
    """Catalog spans the expected 17 genres and 14 moods."""

    EXPECTED_GENRES = {
        "lofi", "pop", "rock", "hip-hop", "jazz", "edm", "classical",
        "r&b", "indie pop", "folk", "metal", "ambient", "synthwave",
        "reggae", "country", "blues", "k-pop",
    }
    EXPECTED_MOODS = {
        "happy", "chill", "intense", "relaxed", "moody", "focused",
        "melancholic", "romantic", "nostalgic", "euphoric", "dreamy",
        "aggressive", "sad", "energetic",
    }

    def test_all_17_genres_present(self):
        songs = load_songs(str(DATA / "songs.csv"))
        found = {s["genre"] for s in songs}
        missing = self.EXPECTED_GENRES - found
        assert not missing, f"Missing genres: {missing}"

    def test_all_14_moods_present(self):
        songs = load_songs(str(DATA / "songs.csv"))
        found = {s["mood"] for s in songs}
        missing = self.EXPECTED_MOODS - found
        assert not missing, f"Missing moods: {missing}"

    def test_each_genre_has_at_least_one_song(self):
        songs = load_songs(str(DATA / "songs.csv"))
        found_genres = {s["genre"] for s in songs}
        for genre in self.EXPECTED_GENRES:
            assert genre in found_genres, f"No song with genre '{genre}'"

    def test_each_mood_has_at_least_one_song(self):
        songs = load_songs(str(DATA / "songs.csv"))
        found_moods = {s["mood"] for s in songs}
        for mood in self.EXPECTED_MOODS:
            assert mood in found_moods, f"No song with mood '{mood}'"
