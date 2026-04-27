# SoundMatch — Project Reflection

---

## What Are the Limitations or Biases in Your System?

The most significant bias is **catalog representation**. The 20-song dataset covers 17 genres,
but 13 of those genres have exactly one song. A user whose taste runs toward blues, classical,
reggae, or folk can never earn the genre match bonus from more than one track. The system will
consistently under-serve those users compared to someone who likes lofi or pop, where three songs
compete for the top slot. This is not a flaw in the algorithm — it is a flaw in the data, and it
is exactly the kind of invisible unfairness that shows up in real recommendation systems when
certain artists, genres, or cultural traditions are simply absent from the training catalog.

A second structural bias is the **categorical bonus ceiling**. Mood match is worth 2.00 points
and genre match is worth 0.75 — together 31% of the maximum score. A song that matches both
labels but has completely wrong audio features will still outscore a song that sounds perfect but
carries a different label. The Sad Gym Rat adversarial test proved this is not theoretical: a
slow acoustic blues ballad at 38% energy beat a metal track at 97% energy because the labels
matched. In a real product, users do not experience music as labels — they experience it as sound.
The scoring formula does not fully reflect that.

A third limitation is **cold-start invisibility**. New users get the default profile
(genre: pop, mood: happy). The first several recommendations they see reflect those defaults, not
their actual taste. Unless they use Battles immediately, the system is effectively serving generic
results under the appearance of personalization.

A fourth limitation is the **binary mood and genre matching**. A song labeled "chill" earns zero
points toward a "focused" preference, even though those two moods describe very similar listening
situations. There is no partial credit for adjacent or related labels. This means the system
treats small semantic differences in labeling as if they were large musical differences.

---

## Could Your AI Be Misused, and How Would You Prevent That?

**Recommendation manipulation.** Because the scoring formula is fully documented and
deterministic, someone who knows the weights could craft a user profile that surfaces a specific
song for all users — essentially gaming the recommender to artificially promote one artist. In a
real system with paying clients or label partnerships, this is a genuine business risk. The
mitigation would be diversity enforcement (capping how many results can come from the same artist
or genre) and making the weight values configurable only by administrators, not exposed in public
documentation.

**AI chat misuse.** The Claude-powered chat agent answers questions about the catalog and music
in general. Because it uses tool calls that read real catalog data, a user cannot make it
fabricate song entries — the tools enforce grounding. However, the conversational system prompt
is intentionally broad ("talk about genres, artists, production, vibes, history"), which means a
persistent user could try to steer the conversation off-topic. The mitigation already in place is
that Claude's built-in safety guidelines apply regardless of the system prompt. A production
version would add explicit topic filtering and rate limiting on the chat endpoint.

**False authority from the Monitor page.** The Monitor page shows platform statistics with the
visual language of a real analytics dashboard — bar charts, ranked tables, exact numbers like
"1.4M listeners." Every number is simulated from audio features, not from real streaming data,
but there is no prominent disclaimer. A user who did not read the documentation could mistake the
figures for live data and make real decisions (e.g., choosing which artists to book or feature)
based on them. The fix is a persistent "DEMO DATA — illustrative only" watermark on every chart
and a banner at the top of the Monitor page.

---

## What Surprised You While Testing Your AI's Reliability?

**The Genre Ghost profile worked better than expected.** I designed that adversarial test assuming
that removing the genre signal entirely (by using `genre=bossa nova`, which does not exist in the
catalog) would produce noticeably weak results — low-confidence, random-feeling picks. Instead,
the top result was *Coffee Shop Stories* at 90% match confidence. It is a jazz track that is
acoustically and energetically almost exactly what a bossa nova fan would want: high acousticness
(0.89), low energy (0.37), warm valence (0.71). The numeric features carried the recommendation
even without the categorical anchor. That test changed how I think about the value of genre
labels versus direct acoustic measurement as a preference signal. The math found the right answer
through a different route than I expected.

**How dangerously narrow the margin was in The Centrist profile.** When I ran all targets at 0.50
with no genre or mood anchor, I expected a near-tie — but I did not expect the top 3 results to
span only 3 percentage points (62%, 60%, 59%). The first-place song beat second place by just
0.16 raw points out of 8.75. That gap is smaller than any reasonable rounding tolerance. It
exposed something important: the scoring formula only produces *meaningful* rankings when the
user gives it at least one strong anchor. Without that anchor, the output is technically correct
but practically useless — the equivalent of a recommendation engine that says "I genuinely have
no idea." This made me understand why cold-start is considered one of the hardest problems in
recommendation systems: it is not just a UX challenge, it is a mathematics problem.

**The Sad Gym Rat failure was the right kind of failure.** I expected this test to fail — that
was the point of designing it. What I did not expect was how clean the failure mode was. The
system did not crash, return an error, or produce a nonsense result. It returned a completely
valid blues song that matched every label perfectly. The failure was invisible from the
algorithm's perspective; only a human listener would recognize that a quiet acoustic ballad is
the wrong answer for a gym session. That distinction — a system that fails correctly by its own
rules but wrongly by human experience — is the clearest illustration I have seen of why AI
systems need human review, not just automated testing.

---

## Describing My Collaboration with AI During This Project

This entire project was built in active collaboration with Claude (the same model that powers the
SoundMatch chat feature). Claude wrote the majority of the code across every source file,
generated the CSS, designed the scoring weight rationale, drafted the adversarial test profiles,
and wrote most of the README. My role was to direct the architecture, define what each feature
should do, catch problems with the AI's output, and make final decisions when its suggestions
conflicted with the product vision. The division was roughly: Claude wrote the first draft of
almost everything; I defined, reviewed, redirected, and decided.

### One instance where the AI gave a genuinely helpful suggestion

When I asked for real artist photos in the UI, Claude proposed mapping the fictional catalog
artists to real-world equivalents — for example, "Neon Echo" → "The Weeknd," "Paper Lanterns" →
"Nujabes," "Sable June" → "SZA" — and then fetching their photos from the **Deezer public API**,
which requires no API key and no OAuth setup. I had been planning to use the Spotify API, which
requires a developer account and OAuth even for basic public catalog reads.

Claude's alternative was simpler, more reliable for a demo context, and actually produced better
images — Deezer's `picture_xl` format is 1000×1000 px, whereas Spotify's catalog images are
typically 300×300 px. The fictional-to-real mapping also solved a problem I had not fully thought
through: the fictional artists have no real photos anywhere, but their real-world genre
equivalents do. That suggestion — unprompted, practical, and better than my original plan — saved
setup time and produced a cleaner result.

### One instance where the AI's suggestion was flawed

Early in the project, Claude suggested caching the artist image preload function using
Streamlit's `@st.cache_data` decorator with `id(songs)` as the cache key — the memory address
of the songs list object. The reasoning looked sound at first glance: if the list changed, its
memory address would change, so the cache would correctly invalidate.

The problem is that Python does not guarantee stable object IDs across function calls, and in
Streamlit's re-run model a **new list object is created on every page interaction**. This means
the memory address changes on every render, the cache never hits, and the Deezer API would be
called every single time any user did anything in the app. A session with 20 page interactions
would make 20 × 18 = 360 unnecessary API calls.

The bug was syntactically valid, passed a quick code reading, and would only reveal itself at
runtime under realistic usage conditions. I caught it during review and changed the key to
`tuple(s["id"] for s in songs)` — a stable, hashable value derived from the actual song data,
not from the object's location in memory.

This was the most important technical lesson of the collaboration: AI-generated code can be
wrong in ways that are invisible to static analysis and only surface under the specific runtime
conditions of the framework you are using. It is not enough to ask "does this code look right?"
You have to ask "does this code behave correctly given how this specific system re-runs?"

---

## What This Project Taught Me About AI

**RAG is not just a performance optimization — it is about grounding.** Before building this, I
thought RAG mainly saved tokens. What I learned is that it changes the *quality* of reasoning.
When Claude sees the top-4 cosine-similarity matches as context before generating a response, it
stops inventing facts about what songs exist and starts reasoning about real data. The difference
between a grounded and an ungrounded music recommendation is immediately obvious in the output.

**Agentic tool use changes the human-AI contract.** In a simple prompt-response setup, the user
has to know exactly how to phrase their request. With tool calling, Claude decides whether to
search by mood, by energy range, or to look up a specific title — and that decision is usually
better than what the user would have specified. The user can type "something for my drive home
tonight" and get a genuinely useful, data-grounded answer without knowing what search parameters
to provide.

**A scoring formula is a hypothesis about human preference, not a fact.** Writing down weights
like "energy is worth 3.0 points and danceability is worth 0.25" forced me to articulate
assumptions I normally leave vague. Every weight is a testable claim. The adversarial profiles
proved that some of those claims are wrong in edge cases — which means the formula is a starting
point for an experiment, not a finished product. This is how I now think about every AI system
configuration: hyperparameters are hypotheses, not settings.

---

## What This Project Taught Me About Problem-Solving

**Build the core first, not the UI.** The scoring algorithm in `recommender.py` was working,
tested, and fully understood before a single Streamlit widget existed. Because the logic was
clean and well-documented, integrating it into the app was mechanical rather than risky.

**Constraints produce creativity.** No Spotify API key, no SoundCloud access, no real listening
history — each constraint forced a specific solution that ended up being a feature rather than a
workaround: fictional-to-real artist mapping for photos, deterministic simulation for platform
stats, Battles as a substitute for passive streaming history.

**Explaining a system is harder than building it.** Writing the scoring weight rationale, the
adversarial profile analysis, and this document took longer than most code changes. But the act
of explaining revealed things I had not fully thought through. The Sad Gym Rat analysis started
as a one-line comment and became a full architectural critique pointing to a real flaw in the
weights. Clear documentation is not just communication — it is a form of review that surfaces
problems the code itself cannot show you.
