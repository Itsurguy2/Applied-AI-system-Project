# 🎧 Model Card: Music Recommender Simulation

---

## 1. Model Name

**ScoreShuffle 1.0**

---

## 2. Goal / Task

The goal is to suggest songs a user will probably enjoy.

It does this by comparing each song in a fixed catalog to a user's stated preferences. The user describes what kind of music they want — genre, mood, energy level, and a few other audio qualities. The system scores every song and returns the top matches.

It does not learn from listening history. It does not adapt over time. It just scores and ranks.

---

## 3. Data Used

The catalog has 20 songs in `data/songs.csv`.

Each song has 13 fields: ID, title, artist, genre, mood, and 8 audio features. The audio features are energy, tempo, valence, danceability, acousticness, instrumentalness, speechiness, and liveness. All features are on a 0–1 scale except tempo, which is in beats per minute.

The catalog covers 15 genres and 14 moods. But the coverage is uneven — lofi and pop each have 3 songs, while 9 other genres have only 1 song each. Most moods also appear only once.

The songs were hand-written, not pulled from real streaming data. The numeric values are rough estimates, not measured audio analysis. The catalog skews toward Western popular music. Genres like bossa nova, folk, or world music are either missing or underrepresented.

---

## 4. Algorithm Summary

The system checks two things for each song.

**First: categorical matches.** Does the song's mood match the user's requested mood? If yes, the song gets +2.00 points. Does the genre match? If yes, +0.75 points. These are all-or-nothing — no partial credit.

**Second: numeric closeness.** For each audio feature (energy, acousticness, instrumentalness, valence, danceability, speechiness, liveness), the system measures how close the song's value is to what the user asked for. A perfect match earns the full weight for that feature. A big gap earns fewer points. The farther away, the less you earn.

Energy has the highest weight (3.00 pts). Liveness has the lowest (0.25 pts). Mood match is worth more than genre match because how a song feels matters more than its style label.

Every song gets a total score. The catalog is sorted by score, and the top results are returned with a point-by-point explanation.

The maximum possible score is **8.75**.

---

## 5. Observed Behavior / Biases

**Energy dominates the ranking.**

Energy is worth 3.00 out of 8.75 points — about 34% of the total. This means the system splits users into two groups early: low-energy listeners always see acoustic or lofi songs, and high-energy listeners always see loud or electronic tracks. Other preferences don't move the needle much.

**Conflicting preferences cause bad results.**

The "Sad Gym Rat" profile asked for mood: sad, genre: blues, and energy: 0.93. "Sad" music tends to be slow and quiet, but the energy target demanded something intense. The system returned a slow blues ballad (energy 0.38) as the top result because the mood and genre bonus (+3.50 combined) was too big to overcome. The result had the right labels but felt completely wrong.

**Genre and mood matching is binary.**

A song labeled "chill" gets zero points toward a "focused" preference, even though those moods describe similar listening situations. There is no partial credit for close matches.

**The catalog has gaps.**

Nine of fifteen genres appear only once. A user who prefers classical, reggae, or folk can earn the genre bonus from at most one song in the entire catalog. The ranking for those users is mostly determined by numeric features, not genre preference.

---

## 6. Evaluation Process

Five user profiles were tested. Each was designed to expose a different edge case.

**Profile 1 — Late-Night Study Session (baseline).** Genre: lofi, mood: focused, low energy. Every preference pointed the same direction. The top result scored 8.70 / 8.75. The system worked well because nothing contradicted anything else.

**Profile 2 — Sad Gym Rat (conflicting preferences).** Genre: blues, mood: sad, energy: 0.93. The mood said "slow and quiet" but the energy target said "loud and intense." The system picked a slow blues song because the mood + genre bonus outweighed the energy penalty. The result was technically correct but experientially wrong.

**Profile 3 — Genre Ghost (genre not in catalog).** Genre: "bossa nova" — a genre with no songs in the catalog. The genre bonus never fired. The system still returned a reasonable result (a jazz song) because the mood matched and the numeric features were similar. But the system gave no warning that the genre preference was being ignored.

**Profile 4 — The Centrist (no real signal).** All numeric targets at 0.50, no valid genre or mood. All top-3 results scored within 0.24 points of each other. The ranking was almost random. This showed that the system needs at least one strong preference to produce meaningful results.

**Profile 5 — Impossible Combo (classical + euphoric).** No song in the catalog is both classical and euphoric. Mood match (+2.00) beat genre match (+0.75), so an EDM song ranked above the classical song. This showed that mood always wins over genre when only one can match.

**One sensitivity experiment was also run.** The energy weight was doubled (1.50 → 3.00) and the genre weight was halved (1.50 → 0.75). For the Sad Gym Rat profile, the winning margin for the blues song shrank from 1.77 to 0.22 points — almost flipping the result. A small change in one number nearly changed who won.

---

## 7. Intended Use and Non-Intended Use

**Intended use:**

This system is for classroom learning only. It demonstrates how a content-based recommendation algorithm works and where it breaks down. It is meant to be inspected, modified, and experimented with.

**Not intended for:**

- Real music applications or real users
- Making decisions that affect anyone's actual experience
- Any situation where accuracy, fairness, or diversity in results actually matters
- Catalogs larger than a few hundred songs — the system is not optimized for scale

---

## 8. Ideas for Improvement

**1. Add more songs in underrepresented genres and moods.**
Nine genres have only one song. Genre-based matching is nearly meaningless for most users. A minimum of 3–5 songs per genre would make the genre bonus actually useful.

**2. Add partial credit for similar moods.**
Right now "focused" and "chill" are treated as completely different. A table that gives 0.5–0.8 credit for moods that feel similar would reduce the penalty for songs that are close but not an exact label match.

**3. Enforce variety in the top results.**
The system can return 5 songs from the same genre if that genre scores consistently high. A simple rule — like penalizing a second song from the same genre — would force more variety without changing the underlying scores.

---

## 9. Personal Reflection

**Biggest learning moment**

The biggest thing I learned is that every weight in the scoring dictionary is a value judgment, not a neutral fact. When I set energy to 3.00 points, I was deciding — on behalf of every user — that energy matters more than genre, mood, acousticness, and everything else combined. That felt like a small technical choice when I was writing it. But when I ran the adversarial profiles, I saw the real consequences. A user asking for "sad gym music" got a quiet blues ballad because the mood label matched and the energy weight wasn't high enough to override it. The math was correct. The result was wrong. That gap between "correct by the rules" and "actually what the person wanted" is the most important thing I took away from this project.

**How AI tools helped — and when I had to check them**

AI tools helped me move fast on things I would have gotten stuck on. They generated the initial scoring logic, the formatted terminal output, and the adversarial profiles faster than I could have designed them from scratch. That gave me more time to actually think about *why* the system behaved the way it did.

But I had to double-check the output constantly. A few times the AI edited the wrong parameter name and introduced a silent bug. Once it suggested a weight change that looked reasonable in isolation but broke the score balance I had deliberately set up. The AI doesn't know what I was trying to test — it only knows the current state of the code. I learned to treat AI suggestions like a first draft: useful starting point, needs review before trusting it.

**What surprised me about simple algorithms "feeling" like recommendations**

I expected a weighted sum to feel mechanical. What surprised me was how much it *didn't* feel that way when the profile was well-matched. The baseline Late-Night Study Session profile returned exactly the songs I would have picked by hand. It felt like the system understood what I wanted.

But then the Sad Gym Rat profile broke that illusion completely. The algorithm didn't fail — it did exactly what the math said to do. The problem was that the math didn't match what "sad gym music" actually means to a real person. That contrast — a system that feels smart when inputs are clean and feels completely broken when inputs are ambiguous — is exactly what makes real recommendation systems hard. The algorithm isn't intelligent. It's just fast at applying rules I wrote.

**What I'd try next**

If I kept working on this, I'd want to try two things. First, I'd replace the fixed weights with learned weights — run a set of user ratings on recommendations, then adjust the weights until the scores better match what people actually preferred. Right now the weights are my best guess. They should be the output of a process, not the input.

Second, I'd add a feedback loop inside the session. After seeing the top results, the user could say "more like this one" or "less of that." The system would shift the target values in real time without the user having to re-enter their whole profile. That would make it feel much more like an actual recommendation tool and much less like a static scoring spreadsheet.
