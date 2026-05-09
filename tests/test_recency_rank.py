from datetime import date
from pathlib import Path
import sys
import types

from recency_rank import (
    SourceItem,
    TOPIC_LEXICON,
    parse_source_markdown,
    rank_topics,
    score_topic_item,
    render_markdown_report,
    topic_embedding_signal,
    discover_topics_from_items,
)


def make_item(*, topic, days_ago, source="Wes Roth", text="", title="Sample", weight=1.0):
    published = date(2026, 5, 9) - __import__("datetime").timedelta(days=days_ago)
    return SourceItem(
        source=source,
        title=title,
        text=text or topic,
        published=published,
        topic_hint=topic,
        source_weight=weight,
    )


def test_score_prefers_newer_items_when_everything_else_matches():
    older = make_item(topic="interpretability", days_ago=30)
    newer = make_item(topic="interpretability", days_ago=1)

    older_score = score_topic_item(older, "interpretability", reference_date=date(2026, 5, 9))
    newer_score = score_topic_item(newer, "interpretability", reference_date=date(2026, 5, 9))

    assert newer_score > older_score


def test_repeated_topic_mentions_outscore_single_mentions():
    items = [
        make_item(topic="agents", days_ago=2, source="AI.Dot Engineer", text="agents orchestration context observability"),
        make_item(topic="agents", days_ago=5, source="AI.Dot Engineer", text="agents orchestration workflow"),
        make_item(topic="compute", days_ago=1, source="Wes Roth", text="compute economics and market structure"),
    ]

    ranking = rank_topics(items, reference_date=date(2026, 5, 9))
    assert ranking[0].topic == "agents"
    assert ranking[0].score > ranking[1].score


def test_rendered_report_shows_scores_and_order():
    items = [
        make_item(topic="security", days_ago=1, source="Wes Roth", text="cyber systemic risk banks"),
        make_item(topic="security", days_ago=3, source="The AI Daily Brief", text="cyber attacks and financial stability"),
        make_item(topic="distribution", days_ago=10, source="AI.Dot Engineer", text="product surfaces browsers slack github"),
    ]

    ranking = rank_topics(items, reference_date=date(2026, 5, 9))
    report = render_markdown_report(ranking, reference_date=date(2026, 5, 9))

    assert report.startswith("# Algorithmic topic re-ranking")
    assert "## 1. security" in report.lower()
    assert "distribution" in report.lower()
    assert "score" in report.lower()


def test_expanded_lexicon_scores_topic_synonyms():
    assert TOPIC_LEXICON["agents"]
    assert "tool use" in TOPIC_LEXICON["agents"]
    assert "feature attribution" in TOPIC_LEXICON["interpretability"]
    assert score_topic_item(
        SourceItem(source="Demo", title="Tool use and task routing", text="tool use and task routing", topic_hint=None),
        "agents",
        reference_date=date(2026, 5, 9),
    ) > 0
    assert score_topic_item(
        SourceItem(source="Demo", title="Feature attribution and probes", text="feature attribution and probes", topic_hint=None),
        "interpretability",
        reference_date=date(2026, 5, 9),
    ) > 0


def test_semantic_assignment_works_without_exact_keywords():
    item = SourceItem(
        source="Demo",
        title="Coordinating worker loops with tools and memory",
        text="A planner coordinates worker loops, tool use, and memory across steps.",
        published=date(2026, 5, 8),
        topic_hint=None,
    )

    score = score_topic_item(item, "agents", reference_date=date(2026, 5, 9))
    assert score > 0


def test_embedding_signal_handles_semantic_paraphrases():
    text = "A system routes tasks between helpers, keeps state, and coordinates step-by-step work."
    assert topic_embedding_signal(text, "agents") > topic_embedding_signal(text, "compute")


def test_generic_markdown_parser_extracts_frontmatter_and_bullets(tmp_path: Path):
    path = tmp_path / "sample.md"
    path.write_text(
        """---
source: Demo Podcast
published: 2026-05-08
topic: agents
---

# Episode notes

## Highlights

- [Coordinating worker loops with tools](https://example.com/1)
- [Vision planning on device](https://example.com/2)
""",
        encoding="utf-8",
    )

    items = parse_source_markdown(path, source_name="Demo Podcast")

    assert len(items) == 2
    assert items[0].source == "Demo Podcast"
    assert items[0].topic_hint == "agents"
    assert items[0].published == date(2026, 5, 8)
    assert items[0].url == "https://example.com/1"
    assert items[0].title == "Coordinating worker loops with tools"


def test_sentence_transformer_backend_is_used_when_available(monkeypatch):
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            calls["model_name"] = model_name

        def encode(self, texts, normalize_embeddings=False):
            calls["texts"] = list(texts)
            calls["normalize_embeddings"] = normalize_embeddings
            if len(texts) == 1:
                return [[1.0, 0.0, 0.0]]
            return [[1.0, 0.0, 0.0] for _ in texts]

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    import recency_rank as rr

    rr._embedding_backend.cache_clear()
    rr._topic_embedding_profile.cache_clear()
    score = rr.topic_embedding_signal("tool use and task routing", "agents")

    assert score > 0
    assert calls["model_name"] == "all-MiniLM-L6-v2"
    assert calls["texts"]
    assert calls["normalize_embeddings"] is True


def test_discovered_topics_extend_rank_order():
    items = [
        SourceItem(source="Demo", title="New topic: model distillation at the edge", text="model distillation at the edge", published=date(2026, 5, 8), topic_hint="deployment", confidence=1.0),
        SourceItem(source="Demo", title="Another note on deployment and distillation", text="deployment distillation", published=date(2026, 5, 7), topic_hint="deployment", confidence=1.0),
    ]

    topics = discover_topics_from_items(items)
    assert "deployment" in topics
    assert any(topic == "deployment" for topic in topics)
