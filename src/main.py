"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from .recommender import load_songs, recommend_songs

# Terminal display width — all borders and padding fit within this
WIDTH = 62


def _score_bar(score: float, max_score: float = 8.0, width: int = 20) -> str:
    """Return an ASCII progress bar, e.g. [################....] 95%"""
    filled = round((score / max_score) * width)
    bar = "#" * filled + "." * (width - filled)
    pct = round((score / max_score) * 100)
    return f"[{bar}] {pct}%"


def _divider(char: str = "-") -> str:
    return char * WIDTH


def _print_header(profile_name: str, catalog_size: int, k: int) -> None:
    print(_divider("="))
    print(f"  MUSIC RECOMMENDER  |  Profile: {profile_name}")
    print(f"  Catalog: {catalog_size} songs{f'  |  Showing top {k}':>{WIDTH - 12 - len(str(catalog_size))}}")
    print(_divider("="))


def _print_result(rank: int, song: dict, score: float, explanation: str) -> None:
    # ── Title row: rank + title (left) + score (right-aligned) ───────────────
    score_str = f"{score:.2f} / 8.00"
    rank_title = f"  #{rank}  {song['title']}"
    padding = WIDTH - len(rank_title) - len(score_str)
    print(rank_title + " " * max(padding, 1) + score_str)

    # ── Subtitle: artist / genre / mood ──────────────────────────────────────
    print(f"      by {song['artist']}  |  {song['genre']}  |  {song['mood']}")

    # ── Score bar ─────────────────────────────────────────────────────────────
    print(f"      {_score_bar(score)}")

    # ── Reasons: label (left) + points earned (right-aligned) ────────────────
    print("      Why:")
    for reason in explanation.split(" | "):
        # Split on the last '(' to isolate the points token e.g. "(+1.50)"
        if "(+" in reason:
            label, pts = reason.rsplit("(+", 1)
            pts_str = f"+{pts.rstrip(')')}"
            pad = WIDTH - 8 - len(label.rstrip()) - len(pts_str)
            print(f"        {label.rstrip()}{' ' * max(pad, 1)}{pts_str}")
        else:
            print(f"        {reason}")

    print(_divider())


def main() -> None:
    songs = load_songs("data/songs.csv")

    # ── Taste Profile: "Late-Night Study Session" ─────────────────────────────
    # Persona: someone winding down after class, wants low-distraction background
    # music that keeps them alert without pulling focus away from the work.
    #
    # Categorical anchors
    #   genre -> lofi    (familiar, consistent texture)
    #   mood  -> focused (intent-driven, not just vibe-driven)
    #
    # Numeric targets (all on 0-1 scale except tempo_bpm)
    #   target_energy           0.40  low-stimulation; avoids hype or aggression
    #   target_valence          0.58  mildly positive but not euphoric
    #   target_acousticness     0.80  warm, organic sound over heavy production
    #   target_danceability     0.58  some groove to stay awake, not a dance track
    #   target_instrumentalness 0.88  vocals compete for attention; avoid them
    #   target_speechiness      0.03  no rap or narration in the background
    #   target_liveness         0.09  clean studio recording, no crowd noise
    #   target_tempo_bpm        79    slow-to-mid cadence; matches reading pace
    # ─────────────────────────────────────────────────────────────────────────
    profile_name = "Late-Night Study Session"
    user_prefs = {
        "genre":                   "lofi",
        "mood":                    "focused",
        "target_energy":           0.40,
        "target_valence":          0.58,
        "target_acousticness":     0.80,
        "target_danceability":     0.58,
        "target_instrumentalness": 0.88,
        "target_speechiness":      0.03,
        "target_liveness":         0.09,
        "target_tempo_bpm":        79,
    }

    k = 5
    recommendations = recommend_songs(user_prefs, songs, k=k)

    _print_header(profile_name, len(songs), k)
    print()
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        _print_result(rank, song, score, explanation)
    print()


if __name__ == "__main__":
    main()
