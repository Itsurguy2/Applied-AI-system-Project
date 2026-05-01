"""
Command line runner for the Music Recommender Simulation.

Run with:  py -m src.main
"""

from collections import Counter
from .recommender import load_songs, recommend_songs, SCORE_WEIGHTS, MAX_SCORE

WIDTH = 62


def _score_bar(score: float, max_score: float = MAX_SCORE, width: int = 20) -> str:
    """Return an ASCII progress bar, e.g. [################....] 95%"""
    filled = round((score / max_score) * width)
    bar = "#" * filled + "." * (width - filled)
    pct = round((score / max_score) * 100)
    return f"[{bar}] {pct}%"


def _divider(char: str = "-") -> str:
    return char * WIDTH


def _print_result(rank: int, song: dict, score: float, explanation: str) -> None:
    """Print one ranked result card with score bar and per-feature breakdown."""
    score_str = f"{score:.2f} / {MAX_SCORE:.2f}"
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
) -> list:
    """Print a labelled recommendation run and return the scored results list."""
    print(_divider("="))
    print(f"  PROFILE: {name}")
    print(f"  genre={prefs.get('genre')}  mood={prefs.get('mood')}"
          f"  energy={prefs.get('target_energy')}")
    print(_divider("="))
    if observe:
        print(f"  OBSERVE: {observe}")
        print(_divider("."))
    print()
    results = recommend_songs(prefs, songs, k=k)
    for rank, (song, score, explanation) in enumerate(results, start=1):
        _print_result(rank, song, score, explanation)
    print()
    return results


def print_diversity_report(all_runs: list) -> None:
    """Print cross-profile #1 table, repeat-song flags, and intuition check."""
    print(_divider("="))
    print("  CROSS-PROFILE DIVERSITY REPORT")
    print(_divider("="))
    print()

    # ── Table: who ranked #1 in each profile ─────────────────────────────────
    print("  #1 result per profile:")
    print(_divider("."))
    all_titles = []
    for profile_name, results in all_runs:
        top_song, top_score, _ = results[0]
        label = f"  {profile_name[:36]}"
        value = f"{top_song['title']} ({top_score:.2f})"
        pad = WIDTH - len(label) - len(value)
        print(label + " " * max(pad, 1) + value)
        for song, _, _ in results:
            all_titles.append(song["title"])
    print()

    # ── Repeat-song frequency across all top-k slots ─────────────────────────
    counts = Counter(all_titles)
    repeats = {t: c for t, c in counts.items() if c > 1}
    unique = len(counts)
    total  = len(all_titles)
    print(f"  Unique songs across all top-3 slots: {unique} of {total}")
    if repeats:
        print("  Songs appearing more than once (check for dominance):")
        for title, freq in sorted(repeats.items(), key=lambda x: -x[1]):
            bar = "#" * freq + "." * (5 - freq)
            print(f"    [{bar}] {title}  ({freq}x)")
    else:
        print("  No song repeated -- catalog variety looks healthy.")
    print()

    # ── Weight transparency: what the categorical bonuses are worth ───────────
    mood_w  = SCORE_WEIGHTS["mood_match"]
    genre_w = SCORE_WEIGHTS["genre_match"]
    max_cat = mood_w + genre_w
    max_num = MAX_SCORE - max_cat
    print(_divider("."))
    print("  WEIGHT BREAKDOWN (explains why categorical beats numeric):")
    print(_divider("."))
    print(f"  mood match             {mood_w:.2f} pts  ({mood_w/MAX_SCORE:.0%} of max)")
    print(f"  genre match            {genre_w:.2f} pts  ({genre_w/MAX_SCORE:.0%} of max)")
    print(f"  categorical subtotal   {max_cat:.2f} pts  ({max_cat/MAX_SCORE:.0%} of max)")
    print(f"  all numeric features   {max_num:.2f} pts  ({max_num/MAX_SCORE:.0%} of max)")
    print()

    # ── Musical intuition check: Sad Gym Rat ─────────────────────────────────
    # Why did Empty Bottle Blues (energy 0.38) beat Iron Collapse (energy 0.97)
    # for a user who set target_energy: 0.93?
    #
    # Empty Bottle Blues earned:
    #   mood 'sad'   match  +2.00   <- profile mood matched exactly
    #   genre 'blues' match +1.50   <- profile genre matched exactly
    #   energy penalty      -0.83   <- 1.50 * (1 - |0.38 - 0.93|) = 0.67 earned
    #   categorical total   +3.50   <- larger than the max energy contribution
    #
    # Iron Collapse earned:
    #   mood mismatch        0.00
    #   genre mismatch       0.00
    #   energy near-match   +1.44   <- 1.50 * (1 - |0.97 - 0.93|) = 1.44 earned
    #   categorical total   +0.00
    #
    # Verdict: the 3.50 categorical head-start is unbeatable even when every
    # numeric feature goes against the matched song.  A user who says "sad blues"
    # gets a slow acoustic track regardless of their energy target -- which FEELS
    # wrong.  The fix would be raising the energy weight or soft-capping how much
    # categorical bonuses can override a large numeric mismatch.
    print(_divider("."))
    print("  INTUITION CHECK: Sad Gym Rat -- why did Empty Bottle Blues win?")
    print(_divider("."))
    print("  The user wanted:  mood=sad  genre=blues  target_energy=0.93")
    print("  Empty Bottle Blues has energy 0.38 -- a 0.55 mismatch.")
    print()
    print("  Its categorical bonus (+3.50) vs. its energy penalty (-0.83):")
    print(f"    mood match   +{mood_w:.2f}")
    print(f"    genre match  +{genre_w:.2f}")
    print(f"    energy loss  -{(1.50 - 1.50 * (1 - abs(0.38 - 0.93))):.2f}"
          f"  (earned 0.67 of max 1.50)")
    print(f"    net gain from categorical alone: "
          f"+{max_cat - (1.50 - 1.50*(1-abs(0.38-0.93))):.2f}")
    print()
    print("  Iron Collapse (energy 0.97) earned 0.00 on mood+genre")
    print("  and only +1.44 on energy -- never enough to close the gap.")
    print()
    print("  Intuition verdict: WRONG FEEL. A gym user asking for 'sad'")
    print("  music probably means emotionally heavy, not acoustically slow.")
    print("  The system delivers the genre/mood label correctly but ignores")
    print("  the lived experience that high energy + sad = post-hardcore,")
    print("  not acoustic blues.  Raising the energy weight from 1.50 to")
    print("  2.00+ would let numeric features override a bad categorical fit.")
    print(_divider("="))
    print()


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs\n")

    # Bind the catalog; collect (name, results) so the diversity report can read them
    all_runs = []
    def show(name, prefs, *, k=3, observe=""):
        results = run_profile(name, prefs, songs, k=k, observe=observe)
        all_runs.append((name, results))

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


    print_diversity_report(all_runs)


if __name__ == "__main__":
    main()