"""
Command line runner for the Music Recommender Simulation.

Run with:  py -m src.main
"""

from .recommender import load_songs, recommend_songs

WIDTH = 62


def _score_bar(score: float, max_score: float = 8.0, width: int = 20) -> str:
    """Return an ASCII progress bar, e.g. [################....] 95%"""
    filled = round((score / max_score) * width)
    bar = "#" * filled + "." * (width - filled)
    pct = round((score / max_score) * 100)
    return f"[{bar}] {pct}%"


def _divider(char: str = "-") -> str:
    return char * WIDTH


def _print_result(rank: int, song: dict, score: float, explanation: str) -> None:
    """Print one ranked result card with score bar and per-feature breakdown."""
    score_str = f"{score:.2f} / 8.00"
    rank_title = f"  #{rank}  {song['title']}"
    padding = WIDTH - len(rank_title) - len(score_str)
    print(rank_title + " " * max(padding, 1) + score_str)
    print(f"      by {song['artist']}  |  {song['genre']}  |  {song['mood']}")
    print(f"      {_score_bar(score)}")
    print("      Why:")
    for reason in explanation.split(" | "):
        if "(+" in reason:
            label, pts = reason.rsplit("(+", 1)
            pts_str = f"+{pts.rstrip(')')}"
            pad = WIDTH - 8 - len(label.rstrip()) - len(pts_str)
            print(f"        {label.rstrip()}{' ' * max(pad, 1)}{pts_str}")
        else:
            print(f"        {reason}")
    print(_divider())


def run_profile(
    name: str,
    prefs: dict,
    songs: list,
    *,
    k: int = 3,
    observe: str = "",
) -> None:
    """Print a labelled recommendation run for one user profile."""
    print(_divider("="))
    print(f"  PROFILE: {name}")
    print(f"  genre={prefs.get('genre')}  mood={prefs.get('mood')}"
          f"  energy={prefs.get('target_energy')}")
    print(_divider("="))
    if observe:
        print(f"  OBSERVE: {observe}")
        print(_divider("."))
    print()
    for rank, (song, score, explanation) in enumerate(
        recommend_songs(prefs, songs, k=k), start=1
    ):
        _print_result(rank, song, score, explanation)
    print()


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs\n")

    # Bind the loaded catalog so every call below omits the `songs` argument
    def show(name, prefs, *, k=3, observe=""):
        run_profile(name, prefs, songs, k=k, observe=observe)

    # ── Baseline profile ──────────────────────────────────────────────────────
    show(
        name="Late-Night Study Session  (baseline)",
        prefs={
            "genre":                   "lofi",
            "mood":                    "focused",
            "target_energy":           0.40,
            "target_valence":          0.58,
            "target_acousticness":     0.80,
            "target_danceability":     0.58,
            "target_instrumentalness": 0.88,
            "target_speechiness":      0.03,
            "target_liveness":         0.09,
        },
        k=3,
        observe="Expected: lofi/focused songs dominate. Score near 8.0 for best match.",
    )

    # ── Adversarial Profile 1: The Sad Gym Rat ────────────────────────────────
    # Conflict: mood:sad aligns with low energy, but target_energy:0.93 demands
    # the opposite. The only "sad" song (Empty Bottle Blues) has energy 0.38 —
    # it earns +2.00 mood match but loses ~0.83 pts on energy proximity alone.
    show(
        name="Sad Gym Rat  (conflicting mood vs energy)",
        prefs={
            "genre":                   "blues",
            "mood":                    "sad",
            "target_energy":           0.93,
            "target_valence":          0.22,
            "target_acousticness":     0.08,
            "target_danceability":     0.90,
            "target_instrumentalness": 0.10,
            "target_speechiness":      0.05,
            "target_liveness":         0.15,
        },
        k=3,
        observe="Trap: does the sad mood match (+2.00) beat high-energy songs "
                "that ignore mood entirely? Watch whether blues/sad or metal/aggressive wins.",
    )

    # ── Adversarial Profile 2: The Genre Ghost ────────────────────────────────
    # 'bossa nova' does not exist in the catalog — no song will ever earn the
    # +1.50 genre bonus. The system silently falls back to numeric features only.
    show(
        name="Genre Ghost  (genre not in catalog)",
        prefs={
            "genre":                   "bossa nova",
            "mood":                    "relaxed",
            "target_energy":           0.38,
            "target_valence":          0.70,
            "target_acousticness":     0.85,
            "target_danceability":     0.55,
            "target_instrumentalness": 0.70,
            "target_speechiness":      0.04,
            "target_liveness":         0.20,
        },
        k=3,
        observe="Trap: genre 'bossa nova' is never matched, so +1.50 is permanently "
                "off the table. The max achievable score drops to 6.50. Does the "
                "output still feel reasonable despite the missing genre signal?",
    )

    # ── Adversarial Profile 3: The Centrist ───────────────────────────────────
    # All numeric targets at 0.5 and no genre/mood anchor that exists in the
    # catalog. Every song receives the same numeric contribution per feature.
    # The ranking becomes determined almost entirely by which song is closest
    # to the midpoint on the features with the highest weights.
    show(
        name="The Centrist  (all targets at midpoint, no anchors)",
        prefs={
            "genre":                   "none",
            "mood":                    "none",
            "target_energy":           0.50,
            "target_valence":          0.50,
            "target_acousticness":     0.50,
            "target_danceability":     0.50,
            "target_instrumentalness": 0.50,
            "target_speechiness":      0.05,
            "target_liveness":         0.12,
        },
        k=3,
        observe="Trap: no categorical bonuses ever fire. All scores cluster near "
                "4.0 pts. The top result wins by tiny margins on energy and "
                "acousticness proximity — the ranking is nearly arbitrary.",
    )

    # ── Adversarial Profile 4: The Impossible Combo ───────────────────────────
    # No song in the 20-song catalog is both classical AND euphoric.
    # Nocturne in Blue (classical) is melancholic; Sunshine Current and Pulse
    # Horizon are euphoric but not classical. The user can never get both
    # categorical bonuses — maximum achievable score is capped at 6.50.
    show(
        name="Impossible Combo  (classical + euphoric — no such song)",
        prefs={
            "genre":                   "classical",
            "mood":                    "euphoric",
            "target_energy":           0.75,
            "target_valence":          0.90,
            "target_acousticness":     0.85,
            "target_danceability":     0.80,
            "target_instrumentalness": 0.90,
            "target_speechiness":      0.03,
            "target_liveness":         0.12,
        },
        k=3,
        observe="Trap: no song earns both +2.00 mood and +1.50 genre. The winner "
                "gets one or the other, not both. Does genre match or mood match "
                "produce a better-feeling top result?",
    )


if __name__ == "__main__":
    main()
