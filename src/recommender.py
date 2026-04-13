import csv
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py

    The three new fields (instrumentalness, speechiness, liveness) have
    defaults so existing test constructors that pass only the original
    10 arguments keep working unchanged.
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Extended features — optional, catalog-midpoint defaults
    instrumentalness: float = 0.50
    speechiness: float = 0.05
    liveness: float = 0.12


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py

    Core fields (positional, no defaults) must stay first so existing
    test constructors — UserProfile("pop","happy",0.8,False) — keep working.

    Extended numeric targets use defaults so they are optional.
    Each target_* field is proximity-scored: closer = better.
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    # Extended targets — all optional, safe defaults are catalog midpoints
    target_valence: float = 0.65
    target_acousticness: float = 0.50
    target_danceability: float = 0.65
    target_instrumentalness: float = 0.50
    target_speechiness: float = 0.05
    target_liveness: float = 0.12
    target_tempo_bpm: float = 100.0


# ── Scoring recipe ────────────────────────────────────────────────────────────
#
#  CATEGORICAL (binary match — full points or zero):
#    Mood match   +2.0   Listener intent is the strongest signal.  A user who
#                        wants "focused" music cares more about *how it feels*
#                        than which genre it belongs to.
#    Genre match  +1.5   Style preference matters, but two songs of the same
#                        genre with opposite moods are worse than two songs of
#                        different genres with the same mood.
#
#  NUMERIC (proximity — full points for exact match, scales down linearly):
#    Energy       +1.5   Best single numeric discriminator in the catalog.
#                        Separates "intense rock" (0.91) from "chill lofi" (0.35)
#                        more reliably than any other single feature.
#    Acousticness +1.0   Second-strongest discriminator; captures production
#                        texture independently of energy level.
#    Instrumental +0.75  Critical for focus/study use-cases; penalises vocal
#                        tracks that compete for the listener's attention.
#    Valence      +0.50  Emotional positivity axis. Narrow spread in this
#                        catalog, so weighted lower than the above three.
#    Danceability +0.25  Situational — useful for workout/party personas.
#    Speechiness  +0.25  Identifies rap/narration tracks. Low weight because
#                        most songs score near zero anyway.
#    Liveness     +0.25  Studio vs live feel. Tiebreaker-level weight.
#
#  MAX POSSIBLE SCORE = 2.0 + 1.5 + 1.5 + 1.0 + 0.75 + 0.50 + 0.25 + 0.25 + 0.25
#                     = 8.0
#
# ─────────────────────────────────────────────────────────────────────────────
SCORE_WEIGHTS: Dict[str, float] = {
    "mood_match":       2.00,
    "genre_match":      1.50,
    "energy":           1.50,
    "acousticness":     1.00,
    "instrumentalness": 0.75,
    "valence":          0.50,
    "danceability":     0.25,
    "speechiness":      0.25,
    "liveness":         0.25,
}
MAX_SCORE: float = sum(SCORE_WEIGHTS.values())  # 8.0

# Tempo normalization range (broadened past catalog extremes for robustness)
_BPM_MIN: float = 50.0
_BPM_MAX: float = 200.0


def _proximity(song_val: float, target_val: float) -> float:
    """
    Linear proximity on a [0, 1]-scaled feature.
    Returns 1.0 for a perfect match, 0.0 for maximum possible distance.
    Clamped to [0, 1] to guard against out-of-range values.
    """
    return max(0.0, 1.0 - abs(song_val - target_val))


def _normalize_bpm(bpm: float) -> float:
    return (bpm - _BPM_MIN) / (_BPM_MAX - _BPM_MIN)


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """

    def __init__(self, songs: List[Song]):
        self.songs = songs

    # ── internal scoring ──────────────────────────────────────────────────────

    def _score(self, song: Song, user: UserProfile) -> float:
        """Return the raw weighted score for one (song, user) pair."""
        s = 0.0

        # Categorical — binary match
        if song.mood == user.favorite_mood:
            s += SCORE_WEIGHTS["mood_match"]
        if song.genre == user.favorite_genre:
            s += SCORE_WEIGHTS["genre_match"]

        # Numeric — proximity
        s += SCORE_WEIGHTS["energy"]           * _proximity(song.energy,           user.target_energy)
        s += SCORE_WEIGHTS["acousticness"]     * _proximity(song.acousticness,     user.target_acousticness)
        s += SCORE_WEIGHTS["instrumentalness"] * _proximity(song.instrumentalness, user.target_instrumentalness)
        s += SCORE_WEIGHTS["valence"]          * _proximity(song.valence,          user.target_valence)
        s += SCORE_WEIGHTS["danceability"]     * _proximity(song.danceability,     user.target_danceability)
        s += SCORE_WEIGHTS["speechiness"]      * _proximity(song.speechiness,      user.target_speechiness)
        s += SCORE_WEIGHTS["liveness"]         * _proximity(song.liveness,         user.target_liveness)

        return s

    # ── public API ────────────────────────────────────────────────────────────

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the k highest-scoring songs for this user, best first."""
        return sorted(self.songs, key=lambda s: self._score(s, user), reverse=True)[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Return a human-readable string explaining why this song was recommended.
        Every entry shows the points actually earned so the score is fully traceable.
        Only contributions >= 0.10 pts are listed to avoid noise from near-zero terms.
        """
        reasons = []

        # Categorical — binary, so earned == max or 0
        if song.mood == user.favorite_mood:
            reasons.append(
                f"mood '{song.mood}' match (+{SCORE_WEIGHTS['mood_match']:.2f})"
            )
        if song.genre == user.favorite_genre:
            reasons.append(
                f"genre '{song.genre}' match (+{SCORE_WEIGHTS['genre_match']:.2f})"
            )

        # Numeric — show earned points for every feature above the noise floor
        numeric_pairs = [
            ("energy",           song.energy,           user.target_energy,           "energy"),
            ("acousticness",     song.acousticness,     user.target_acousticness,     "acousticness"),
            ("instrumentalness", song.instrumentalness, user.target_instrumentalness, "instrumentalness"),
            ("valence",          song.valence,          user.target_valence,          "valence"),
            ("danceability",     song.danceability,     user.target_danceability,     "danceability"),
            ("speechiness",      song.speechiness,      user.target_speechiness,      "speechiness"),
            ("liveness",         song.liveness,         user.target_liveness,         "liveness"),
        ]
        for label, song_val, target_val, weight_key in numeric_pairs:
            earned = SCORE_WEIGHTS[weight_key] * _proximity(song_val, target_val)
            if earned >= 0.10:
                reasons.append(
                    f"{label} {song_val:.2f} ~= {target_val:.2f} (+{earned:.2f})"
                )

        if not reasons:
            reasons.append("closest overall match across all features")

        return " | ".join(reasons)


# ── Functional API (used by src/main.py) ─────────────────────────────────────

def load_songs(csv_path: str) -> List[Dict]:
    """
    Load songs from a CSV file into a list of dicts.
    Required by src/main.py
    """
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            songs.append({
                "id":               int(row["id"]),
                "title":            row["title"],
                "artist":           row["artist"],
                "genre":            row["genre"],
                "mood":             row["mood"],
                "energy":           float(row["energy"]),
                "tempo_bpm":        float(row["tempo_bpm"]),
                "valence":          float(row["valence"]),
                "danceability":     float(row["danceability"]),
                "acousticness":     float(row["acousticness"]),
                # .get() with fallback so older CSVs without these columns still load
                "instrumentalness": float(row.get("instrumentalness", 0.50)),
                "speechiness":      float(row.get("speechiness",      0.05)),
                "liveness":         float(row.get("liveness",         0.12)),
            })
    return songs


# Feature mapping used by score_song: (song_key, user_prefs_key, weight_key)
_NUMERIC_FEATURES: List[Tuple[str, str, str]] = [
    ("energy",           "target_energy",           "energy"),
    ("acousticness",     "target_acousticness",     "acousticness"),
    ("instrumentalness", "target_instrumentalness", "instrumentalness"),
    ("valence",          "target_valence",          "valence"),
    ("danceability",     "target_danceability",     "danceability"),
    ("speechiness",      "target_speechiness",      "speechiness"),
    ("liveness",         "target_liveness",         "liveness"),
]


def score_song(song: Dict, user_prefs: Dict) -> Tuple[float, str]:
    """
    Judge a single song against the user's taste profile.

    Scoring rules
    -------------
    Categorical (binary — full points or zero):
        mood  match  +2.00
        genre match  +1.50

    Numeric proximity (scales linearly from 0 to max weight):
        earned = weight * (1.0 - |song_value - user_target|)

        energy           max +1.50
        acousticness     max +1.00
        instrumentalness max +0.75
        valence          max +0.50
        danceability     max +0.25
        speechiness      max +0.25
        liveness         max +0.25

    Max possible total: 8.00

    Returns
    -------
    (score, explanation)
        score       float  — total weighted points, rounded to 3 decimal places
        explanation str    — pipe-separated reasons, each showing points earned
    """
    score = 0.0
    reasons: List[str] = []

    # ── Categorical scoring ───────────────────────────────────────────────────
    if song["mood"] == user_prefs.get("mood"):
        pts = SCORE_WEIGHTS["mood_match"]
        score += pts
        reasons.append(f"mood '{song['mood']}' match (+{pts:.2f})")

    if song["genre"] == user_prefs.get("genre"):
        pts = SCORE_WEIGHTS["genre_match"]
        score += pts
        reasons.append(f"genre '{song['genre']}' match (+{pts:.2f})")

    # ── Numeric proximity scoring ─────────────────────────────────────────────
    for song_key, pref_key, weight_key in _NUMERIC_FEATURES:
        if pref_key in user_prefs:
            earned = SCORE_WEIGHTS[weight_key] * _proximity(song[song_key], user_prefs[pref_key])
            score += earned
            # Include every feature that earned >= 0.10 pts so the score is
            # fully traceable — low contributors are in the number but not the text
            if earned >= 0.10:
                reasons.append(
                    f"{song_key} {song[song_key]:.2f} ~= {user_prefs[pref_key]:.2f} (+{earned:.2f})"
                )

    explanation = " | ".join(reasons) if reasons else "closest overall match"
    return round(score, 3), explanation


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """
    Rank every song in the catalog and return the top k.

    This function has exactly three responsibilities:
      1. Call score_song on every song  (judge the whole catalog)
      2. Sort the scored results descending  (ranking rule)
      3. Return the top k  (output slice)

    The scoring logic lives entirely in score_song — this function
    never touches weights or feature math directly.
    """
    # Step 1 — score every song in the catalog
    # score_song returns (score, explanation); the * unpacks that pair directly
    # into the tuple so each element is (song, score, explanation).
    scored: List[Tuple[Dict, float, str]] = [
        (song, *score_song(song, user_prefs))
        for song in songs
    ]

    # Step 2 — rank by score, highest first
    # sorted() returns a NEW list; the original `scored` is left untouched.
    # key=lambda item: item[1]  →  sort by the score (index 1 of each tuple)
    # reverse=True              →  descending (highest score first)
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)

    # Step 3 — return the top k results
    return ranked[:k]
