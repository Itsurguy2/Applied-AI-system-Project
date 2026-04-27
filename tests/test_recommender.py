"""
Test suite for the SoundMatch recommender engine and chat agent utilities.

Covers:
  - Core recommender: scoring, ranking, explanation
  - Edge cases: empty catalog, k-limiting, zero-score scenarios
  - RAG utilities: cosine retrieval ordering, intent mapping
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recommender import MAX_SCORE, Recommender, Song, UserProfile, score_song
from chat_agent import intent_to_features, rag_retrieve


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _song(id, genre, mood, energy, valence=0.6, danceability=0.6,
          acousticness=0.4, instrumentalness=0.4):
    return Song(
        id=id, title=f"Track {id}", artist="Test Artist",
        genre=genre, mood=mood,
        energy=energy, tempo_bpm=100,
        valence=valence, danceability=danceability,
        acousticness=acousticness, instrumentalness=instrumentalness,
    )


def _song_dict(id, genre, mood, energy, valence=0.6, danceability=0.6,
               acousticness=0.4, instrumentalness=0.4):
    return dict(
        id=id, title=f"Track {id}", artist="Test Artist",
        genre=genre, mood=mood,
        energy=energy, tempo_bpm=100.0,
        valence=valence, danceability=danceability,
        acousticness=acousticness, instrumentalness=instrumentalness,
        speechiness=0.05, liveness=0.12,
    )


def _pop_happy_profile():
    return UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )


# ── Original tests (preserved) ────────────────────────────────────────────────

def test_recommend_returns_songs_sorted_by_score():
    songs = [
        _song(1, "pop", "happy", energy=0.8, valence=0.9),
        _song(2, "lofi", "chill", energy=0.4, acousticness=0.9),
    ]
    rec = Recommender(songs)
    results = rec.recommend(_pop_happy_profile(), k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    songs = [_song(1, "pop", "happy", energy=0.8)]
    rec = Recommender(songs)
    expl = rec.explain_recommendation(_pop_happy_profile(), rec.songs[0])
    assert isinstance(expl, str)
    assert expl.strip() != ""


# ── New tests ─────────────────────────────────────────────────────────────────

def test_recommend_returns_exactly_k_results():
    songs = [_song(i, "pop", "happy", energy=0.7) for i in range(10)]
    rec = Recommender(songs)
    for k in (1, 3, 5, 10):
        assert len(rec.recommend(_pop_happy_profile(), k=k)) == k


def test_recommend_empty_catalog_returns_empty():
    rec = Recommender([])
    assert rec.recommend(_pop_happy_profile(), k=5) == []


def test_score_song_perfect_genre_and_mood_match_earns_categorical_bonus():
    song = _song_dict(1, "pop", "happy", energy=0.8)
    profile = {"genre": "pop", "mood": "happy", "target_energy": 0.8,
               "target_valence": 0.6, "target_acousticness": 0.4,
               "target_danceability": 0.6, "target_instrumentalness": 0.4,
               "target_speechiness": 0.05, "target_liveness": 0.12}
    score, expl = score_song(song, profile)
    # Must include both categorical bonuses
    assert "mood 'happy' match" in expl
    assert "genre 'pop' match" in expl
    # Score should be near MAX (within 1 point for slightly-off numeric features)
    assert score >= MAX_SCORE - 1.5


def test_score_song_no_match_returns_low_score():
    song = _song_dict(1, "metal", "aggressive", energy=0.95,
                      valence=0.2, danceability=0.5, acousticness=0.05)
    profile = {"genre": "lofi", "mood": "focused", "target_energy": 0.35,
               "target_valence": 0.6, "target_acousticness": 0.8,
               "target_danceability": 0.55, "target_instrumentalness": 0.85,
               "target_speechiness": 0.03, "target_liveness": 0.09}
    score, _ = score_song(song, profile)
    # No categorical matches, large numeric gap → low score
    assert score < MAX_SCORE * 0.45


def test_score_song_mood_match_outweighs_genre_mismatch():
    # Song matches mood but not genre
    song_mood = _song_dict(1, "jazz", "happy", energy=0.8, valence=0.9)
    # Song matches genre but not mood
    song_genre = _song_dict(2, "pop", "sad", energy=0.8, valence=0.3)
    profile = {"genre": "pop", "mood": "happy", "target_energy": 0.8,
               "target_valence": 0.8, "target_acousticness": 0.3,
               "target_danceability": 0.7, "target_instrumentalness": 0.3,
               "target_speechiness": 0.05, "target_liveness": 0.12}
    score_mood, _  = score_song(song_mood, profile)
    score_genre, _ = score_song(song_genre, profile)
    # Mood weight (2.0) > genre weight (0.75), so mood match should win
    assert score_mood > score_genre


def test_explain_recommendation_mentions_mood_when_matched():
    songs = [_song(1, "pop", "happy", energy=0.8)]
    rec = Recommender(songs)
    expl = rec.explain_recommendation(_pop_happy_profile(), rec.songs[0])
    assert "mood" in expl.lower()
    assert "happy" in expl


def test_explain_recommendation_mentions_genre_when_matched():
    songs = [_song(1, "pop", "happy", energy=0.8)]
    rec = Recommender(songs)
    expl = rec.explain_recommendation(_pop_happy_profile(), rec.songs[0])
    assert "genre" in expl.lower()
    assert "pop" in expl


# ── RAG utility tests ─────────────────────────────────────────────────────────

def test_rag_retrieve_returns_exactly_k_songs():
    songs = [_song_dict(i, "pop", "happy", energy=i * 0.05) for i in range(10)]
    result = rag_retrieve({"energy": 0.5, "valence": 0.6, "acousticness": 0.4,
                            "danceability": 0.6, "instrumentalness": 0.4,
                            "speechiness": 0.05, "liveness": 0.12},
                           songs, k=3)
    assert len(result) == 3


def test_rag_retrieve_ranks_closer_song_first():
    # Song A: very high energy/valence → close to query
    song_a = _song_dict(1, "edm", "euphoric", energy=0.95, valence=0.90,
                        danceability=0.90, acousticness=0.05)
    # Song B: very low energy/valence → far from query
    song_b = _song_dict(2, "ambient", "chill", energy=0.10, valence=0.20,
                        danceability=0.20, acousticness=0.95)

    high_energy_query = {"energy": 0.95, "valence": 0.90, "acousticness": 0.05,
                         "danceability": 0.90, "instrumentalness": 0.10,
                         "speechiness": 0.04, "liveness": 0.08}
    result = rag_retrieve(high_energy_query, [song_a, song_b], k=2)
    assert result[0]["id"] == song_a["id"]


def test_intent_to_features_chill_maps_to_low_energy():
    features = intent_to_features("I want something chill to study to")
    assert features["energy"] < 0.5
    assert features["acousticness"] > 0.5
    assert features["instrumentalness"] > 0.5


def test_intent_to_features_workout_maps_to_high_energy():
    features = intent_to_features("give me a hype gym workout playlist")
    assert features["energy"] > 0.8
    assert features["danceability"] > 0.7


def test_intent_to_features_sad_maps_to_low_valence():
    features = intent_to_features("something sad and melancholic")
    assert features["valence"] < 0.4
