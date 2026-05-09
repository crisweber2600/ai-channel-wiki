from datetime import date

from recency_rank import SourceItem, rank_topics, score_topic_item, render_markdown_report


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
