from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

TOPIC_ORDER = [
    "interpretability",
    "security",
    "agents",
    "multimodal",
    "distribution",
    "compute",
]

TOPIC_LABELS = {
    "interpretability": "Interpretability & hidden behavior",
    "security": "Security & systemic risk",
    "agents": "Agentic systems & orchestration",
    "multimodal": "Multimodal, voice, vision & edge",
    "distribution": "Distribution & product surfaces",
    "compute": "Compute economics & market structure",
}

TOPIC_KEYWORDS = {
    "interpretability": [
        "interpretability",
        "activations",
        "hidden behavior",
        "hidden motives",
        "thoughts",
        "black box",
        "alignment",
        "auditor",
        "evaluation",
        "deception",
        "nla",
        "autoencoder",
    ],
    "security": [
        "security",
        "systemic risk",
        "cyber",
        "attack",
        "banks",
        "financial",
        "liquidity",
        "risk",
        "threat",
        "blackmail",
        "defense",
    ],
    "agents": [
        "agents",
        "agentic",
        "orchestration",
        "workflow",
        "skills",
        "context",
        "observability",
        "eval",
        "mcp",
        "plan and review",
        "production",
    ],
    "multimodal": [
        "multimodal",
        "voice",
        "vision",
        "image",
        "video",
        "audio",
        "edge",
        "camera",
        "local inference",
    ],
    "distribution": [
        "distribution",
        "product",
        "surface",
        "browser",
        "chrome",
        "slack",
        "github",
        "embedded",
        "workflow",
        "interface",
    ],
    "compute": [
        "compute",
        "latency",
        "pricing",
        "economics",
        "market",
        "capital",
        "infra",
        "spend",
        "tokens",
        "asset",
    ],
}

SOURCE_WEIGHTS = {
    "Wes Roth": 1.12,
    "AI.Dot Engineer": 1.0,
    "The AI Daily Brief": 0.94,
}

SOURCE_NAME_OVERRIDES = {
    "wes-roth": "Wes Roth",
    "ai-dot-engineer": "AI.Dot Engineer",
    "the-ai-daily-brief": "The AI Daily Brief",
}

SECTION_TO_TOPIC = {
    "interpretability and hidden behavior": "interpretability",
    "security and systemic risk": "security",
    "agentic systems and orchestration": "agents",
    "multimodal voice vision and edge": "multimodal",
    "multimodal, voice, vision, and edge": "multimodal",
    "distribution and product surfaces": "distribution",
    "distribution platforms and product surfaces": "distribution",
    "compute economics and market structure": "compute",
}

MONTH_WINDOW_DAYS = 30
RECENCY_HALF_LIFE_DAYS = 10.0

TOPIC_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:\.\./)?topics/([a-z0-9\-]+)\.md\)")
CHANNEL_BULLET_RE = re.compile(r"^-\s+\[(.+?)\]\((https?://[^)]+)\)")
MONTH_BULLET_RE = re.compile(r"^-\s+(\d{4}-\d{2}-\d{2})\s+—\s+\[(.+?)\]\((https?://[^)]+)\)")


@dataclass(frozen=True)
class SourceItem:
    source: str
    title: str
    text: str
    published: date | None = None
    topic_hint: str | None = None
    url: str | None = None
    source_weight: float = 1.0
    confidence: float = 1.0


@dataclass
class TopicRanking:
    topic: str
    score: float
    item_count: int
    sources: list[str] = field(default_factory=list)
    representative_items: list[SourceItem] = field(default_factory=list)
    evidence_score: float = 0.0


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def topic_from_heading(heading: str) -> str | None:
    return SECTION_TO_TOPIC.get(slugify(heading))


def _count_phrase(text: str, phrase: str) -> int:
    phrase = phrase.strip().lower()
    if not phrase:
        return 0
    if " " in phrase:
        return text.count(phrase)
    return len(re.findall(rf"\b{re.escape(phrase)}\b", text))


def topic_signal(text: str, topic: str) -> float:
    normalized = text.lower()
    score = 0.0
    for phrase in TOPIC_KEYWORDS.get(topic, []):
        hits = _count_phrase(normalized, phrase)
        if hits:
            score += hits * (1.8 if " " in phrase else 1.0)
    if topic in normalized:
        score += 1.0
    return score


def recency_decay(published: date | None, reference_date: date) -> float:
    if published is None:
        return 0.62
    age_days = max((reference_date - published).days, 0)
    age_days = min(age_days, MONTH_WINDOW_DAYS)
    return math.exp(-age_days / RECENCY_HALF_LIFE_DAYS)


def score_topic_item(item: SourceItem, topic: str, reference_date: date | None = None) -> float:
    if reference_date is None:
        reference_date = date.today()

    signal = topic_signal(f"{item.title}\n{item.text}", topic)
    if signal <= 0 and item.topic_hint == topic:
        signal = 1.2
    if signal <= 0:
        return 0.0

    source_weight = item.source_weight if item.source_weight != 1.0 else SOURCE_WEIGHTS.get(item.source, 1.0)
    recency = recency_decay(item.published, reference_date)
    hint_boost = 1.18 if item.topic_hint == topic else 1.0
    return signal * source_weight * recency * hint_boost * item.confidence


def rank_topics(items: Sequence[SourceItem], reference_date: date | None = None) -> list[TopicRanking]:
    if reference_date is None:
        reference_date = date.today()

    base_scores: dict[str, float] = defaultdict(float)
    representative: dict[str, list[tuple[float, SourceItem]]] = defaultdict(list)
    sources_by_topic: dict[str, set[str]] = defaultdict(set)
    days_by_topic: dict[str, set[date]] = defaultdict(set)

    for item in items:
        for topic in TOPIC_ORDER:
            score = score_topic_item(item, topic, reference_date=reference_date)
            if score <= 0:
                continue
            base_scores[topic] += score
            representative[topic].append((score, item))
            sources_by_topic[topic].add(item.source)
            if item.published is not None:
                days_by_topic[topic].add(item.published)

    results: list[TopicRanking] = []
    for topic in TOPIC_ORDER:
        base = base_scores.get(topic, 0.0)
        if base <= 0:
            continue

        distinct_sources = len(sources_by_topic[topic])
        distinct_days = len(days_by_topic[topic])
        repetition_boost = 1.0 + min(0.18 * max(distinct_sources - 1, 0) + 0.08 * max(distinct_days - 1, 0), 0.7)
        centrality_boost = 1.0 + {
            "interpretability": 0.08,
            "security": 0.1,
            "agents": 0.12,
            "multimodal": 0.1,
            "distribution": 0.1,
            "compute": 0.1,
        }.get(topic, 0.0)
        score = base * repetition_boost * centrality_boost
        ranked_items = [item for _, item in sorted(representative[topic], key=lambda pair: pair[0], reverse=True)[:5]]
        results.append(
            TopicRanking(
                topic=topic,
                score=score,
                item_count=len(representative[topic]),
                sources=sorted(sources_by_topic[topic]),
                representative_items=ranked_items,
                evidence_score=base,
            )
        )

    results.sort(key=lambda ranking: (-ranking.score, TOPIC_ORDER.index(ranking.topic)))
    return results


def render_markdown_report(rankings: Sequence[TopicRanking], reference_date: date | None = None) -> str:
    if reference_date is None:
        reference_date = date.today()

    lines = [
        "# Algorithmic topic re-ranking",
        "",
        f"Generated from a recency-weighted pass on {reference_date.isoformat()}.",
        "",
        "## How it works",
        "",
        "- newer items get higher decay-adjusted weight",
        "- repeated items across sources get boosted",
        "- topic hints are reinforced by keyword evidence from transcripts and notes",
        "",
        "## Ranked topics",
        "",
    ]

    for index, ranking in enumerate(rankings, start=1):
        label = TOPIC_LABELS.get(ranking.topic, ranking.topic)
        lines.append(f"## {index}. {ranking.topic}")
        lines.append("")
        lines.append(f"- **Label:** {label}")
        lines.append(f"- **Score:** {ranking.score:.3f}")
        lines.append(f"- **Evidence score:** {ranking.evidence_score:.3f}")
        lines.append(f"- **Items:** {ranking.item_count}")
        lines.append(f"- **Sources:** {', '.join(ranking.sources) if ranking.sources else 'n/a'}")
        if ranking.representative_items:
            lines.append("- **Representative items:**")
            for item in ranking.representative_items[:3]:
                when = item.published.isoformat() if item.published else "undated"
                suffix = f" — {item.url}" if item.url else ""
                lines.append(f"  - {when} · {item.source} · {item.title}{suffix}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_month_graph_markdown(path: str | Path, source_name: str) -> list[SourceItem]:
    path = Path(path)
    items: list[SourceItem] = []
    current_topic = None
    in_representative = False

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            current_topic = topic_from_heading(line[3:])
            in_representative = False
            continue
        if line.startswith("### "):
            in_representative = slugify(line[4:]) == "representative videos"
            continue
        if not in_representative or not line.startswith("-"):
            continue
        match = MONTH_BULLET_RE.match(line)
        if not match or current_topic is None:
            continue
        published = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        title = match.group(2)
        url = match.group(3)
        items.append(
            SourceItem(
                source=source_name,
                title=title,
                text=title,
                published=published,
                topic_hint=current_topic,
                url=url,
            )
        )
    return items


def parse_channel_page(path: str | Path, source_name: str) -> list[SourceItem]:
    path = Path(path)
    items: list[SourceItem] = []
    current_topic = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("### "):
            current_topic = topic_from_heading(line[4:])
            continue
        if current_topic is None or not line.startswith("-"):
            continue
        match = CHANNEL_BULLET_RE.match(line)
        if not match:
            continue
        items.append(
            SourceItem(
                source=source_name,
                title=match.group(1),
                text=match.group(1),
                published=None,
                topic_hint=current_topic,
                url=match.group(2),
                confidence=0.82,
            )
        )
    return items


def parse_link_surface_bullets(path: str | Path, source_name: str) -> list[SourceItem]:
    path = Path(path)
    items: list[SourceItem] = []
    in_link_section = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            in_link_section = slugify(line[3:]) in {
                "links into the graph",
                "how it links outward",
                "links into graph",
            }
            continue
        if not in_link_section or not line.startswith("-"):
            continue
        match = TOPIC_LINK_RE.search(line)
        if not match:
            continue
        topic = match.group(2)
        if topic not in TOPIC_ORDER:
            continue
        items.append(
            SourceItem(
                source=source_name,
                title=f"graph link: {match.group(1)}",
                text=match.group(1),
                published=None,
                topic_hint=topic,
                confidence=0.38,
            )
        )
    return items


def _dedupe_items(items: Iterable[SourceItem]) -> list[SourceItem]:
    seen: set[tuple[str, str]] = set()
    deduped: list[SourceItem] = []
    for item in items:
        key = (item.url or "", item.title.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_ranking_from_paths(month_graph: str | Path | None = None, channel_pages: Iterable[str | Path] = ()) -> list[TopicRanking]:
    items: list[SourceItem] = []
    if month_graph:
        items.extend(parse_month_graph_markdown(month_graph, source_name="month graph"))
    for path in channel_pages:
        path = Path(path)
        source_name = SOURCE_NAME_OVERRIDES.get(path.stem, path.stem.replace("-", " ").title())
        items.extend(parse_channel_page(path, source_name=source_name))
        items.extend(parse_link_surface_bullets(path, source_name=source_name))
    return rank_topics(_dedupe_items(items))


def write_outputs(rankings: Sequence[TopicRanking], markdown_path: str | Path | None = None, json_path: str | Path | None = None, reference_date: date | None = None) -> None:
    if reference_date is None:
        reference_date = date.today()
    if markdown_path:
        markdown_file = Path(markdown_path)
        markdown_file.parent.mkdir(parents=True, exist_ok=True)
        markdown_file.write_text(render_markdown_report(rankings, reference_date=reference_date), encoding="utf-8")
    if json_path:
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": reference_date.isoformat(),
            "topics": [
                {
                    "topic": ranking.topic,
                    "label": TOPIC_LABELS.get(ranking.topic, ranking.topic),
                    "score": round(ranking.score, 6),
                    "evidence_score": round(ranking.evidence_score, 6),
                    "item_count": ranking.item_count,
                    "sources": ranking.sources,
                    "representative_items": [
                        {
                            "source": item.source,
                            "title": item.title,
                            "published": item.published.isoformat() if item.published else None,
                            "url": item.url,
                        }
                        for item in ranking.representative_items
                    ],
                }
                for ranking in rankings
            ],
        }
        json_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recency-weighted topic ranking helper")
    parser.add_argument("--month-graph", help="Path to the month graph markdown file")
    parser.add_argument(
        "--channel-page",
        action="append",
        default=[],
        help="Additional channel or podcast markdown page to include (repeatable)",
    )
    parser.add_argument("--output-md", help="Write a markdown ranking report to this path")
    parser.add_argument("--output-json", help="Write machine-readable ranking output to this path")
    parser.add_argument("--reference-date", help="Override reference date as YYYY-MM-DD")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    reference_date = date.fromisoformat(args.reference_date) if args.reference_date else date.today()
    rankings = build_ranking_from_paths(args.month_graph, args.channel_page)
    if args.output_md or args.output_json:
        write_outputs(rankings, markdown_path=args.output_md, json_path=args.output_json, reference_date=reference_date)
    else:
        print(render_markdown_report(rankings, reference_date=reference_date))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
