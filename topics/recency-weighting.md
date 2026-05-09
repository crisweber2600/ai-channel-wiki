# Recency-weighting

## What this page is for

This wiki is built from a rolling month of source material. To avoid treating every item as equally important, the ingestion pipeline uses a *recency-weighted memory pass* before writing the topic pages.

The idea is similar to a "dreaming" or consolidation cycle:

- keep the freshest items near the top of the active memory
- compress older material into durable themes
- preserve repeated ideas across multiple sources
- let recent evidence tilt the synthesis without erasing the long tail

## Weighting model

Each item gets a score from several signals:

- **Recency** — newer items get more weight
- **Source authority** — some channels are treated as higher-signal for certain topics
- **Cross-source repetition** — ideas repeated across multiple sources get boosted
- **Topical centrality** — items that connect to multiple sections matter more
- **Semantic matching** — the ranking pass uses an expanded topic lexicon plus local embedding-style similarity to catch paraphrases
- **Novelty** — brand-new claims are highlighted, but only if they are grounded

A simple version looks like this:

```text
weight = source_weight × topicality × repetition_boost × recency_decay × novelty_factor
```

Where `recency_decay` can be modeled with exponential decay over a 30-day window.

## Practical memory tiers

- **Hot memory** — last 7 days; drives the main synthesis
- **Warm memory** — days 8–21; supports continuity and trend detection
- **Cold memory** — days 22–30; retained as context, but less influential

## How the wiki uses it

- The monthly graph emphasizes recent clusters while still linking older anchor events.
- Topic pages should surface the newest representative items first.
- Repeated claims only get promoted if they recur across sources or survive multiple passes.
- New daily digests should update the active window rather than rewrite the entire history.

## Human-readable rule of thumb

If two items say similar things, the newer one should usually win *unless* the older one has stronger cross-source confirmation or better evidence.

## Linked pages

- [Graph map](map.md)
- [Algorithmic topic re-ranking](auto-ranked.md)
- [Wes Roth](../channels/wes-roth.md)
- [AI.Dot Engineer](../channels/ai-dot-engineer.md)
- [The AI Daily Brief](../podcasts/the-ai-daily-brief.md)
