# AI Channel Wiki

A wiki-style knowledge base built from a one-month window of:

- **Wes Roth** — frontier-model news, risk, and market commentary
- **AI.Dot Engineer** — agent engineering, observability, context, multimodal systems, and shipping AI
- **The AI Daily Brief** — podcast transcript and episode digestion via `https://pod.link/1680633614`

This repo is meant to feel like a real wiki instead of a flat summary. The content is split into pages with deep internal links, topic pages, source pages, a graph map, and a recency-weighted synthesis layer.

## How to navigate

- Start with the [graph map](topics/map.md)
- Then jump into [channels](channels/wes-roth.md) or [topics](topics/interpretability.md)
- Use the sidebar for fast topic switching

## Core idea

The month’s story is a shift from:

- black-box capability → observable behavior
- single-agent demos → agent workflows
- chat products → embedded systems
- compute spend → compute as strategic capital

The wiki also uses a recency-weighted consolidation pass so the newest evidence has more influence than older material, without throwing away durable themes. The ranking helper now blends an expanded topic lexicon with local embedding-style semantic matching so paraphrases and partial matches still land on the right page. It can also discover new topic slugs from source pages when they recur often enough.

## Source pages

- [Wes Roth](channels/wes-roth.md)
- [AI.Dot Engineer](channels/ai-dot-engineer.md)
- [The AI Daily Brief](podcasts/the-ai-daily-brief.md)

## Topic graph

- [Interpretability](topics/interpretability.md)
- [Security & systemic risk](topics/security.md)
- [Agentic systems](topics/agents.md)
- [Multimodal, voice, vision & edge](topics/multimodal.md)
- [Distribution & product surfaces](topics/distribution.md)
- [Compute economics](topics/compute.md)
- [Recency-weighting](topics/recency-weighting.md)
- [Algorithmic topic re-ranking](topics/auto-ranked.md)

## Live processing

A daily digest job is already configured to ingest the channels above and the podcast feed.
The job also regenerates [Algorithmic topic re-ranking](topics/auto-ranked.md), which is the machine-scored ordering used to re-rank topics from transcript evidence.

For the older compressed month summary, see the [source gist](https://gist.github.com/crisweber2600/231df3009d553b1fce61781db9e7d206).
