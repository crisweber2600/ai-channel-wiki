from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
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

TOPIC_LEXICON = {
    "interpretability": [
        "interpretability",
        "mechanistic interpretability",
        "feature attribution",
        "feature attribution and probes",
        "probes",
        "circuits",
        "activation patterns",
        "activations",
        "neurons",
        "neuron",
        "hidden behavior",
        "hidden motives",
        "model internals",
        "black box",
        "alignment",
        "evaluation",
        "deception",
        "steering",
        "autoencoder",
        "sparse autoencoder",
        "latent representation",
        "circulation of thoughts",
    ],
    "security": [
        "security",
        "systemic risk",
        "cyber",
        "cybersecurity",
        "attack",
        "attacks",
        "prompt injection",
        "exfiltration",
        "jailbreak",
        "jailbreaking",
        "fraud",
        "blackmail",
        "defense",
        "threat",
        "misuse",
        "abuse",
        "banks",
        "financial",
        "liquidity",
        "risk",
        "model abuse",
    ],
    "agents": [
        "agents",
        "agentic",
        "orchestration",
        "orchestrating",
        "workflow",
        "workflows",
        "tool use",
        "tool-use",
        "tool calling",
        "task routing",
        "planner",
        "planning",
        "task loops",
        "worker loops",
        "memory",
        "state",
        "context",
        "observability",
        "eval",
        "evaluations",
        "mcp",
        "runtime",
        "worker",
        "workers",
        "multi-step",
        "autonomy",
        "delegation",
        "helper",
        "helpers",
        "coordination",
    ],
    "multimodal": [
        "multimodal",
        "voice",
        "vision",
        "image",
        "images",
        "video",
        "audio",
        "speech",
        "edge",
        "on-device",
        "local inference",
        "camera",
        "realtime",
        "vision-language",
        "vision language",
        "image generation",
        "video generation",
    ],
    "distribution": [
        "distribution",
        "product",
        "products",
        "surface",
        "surfaces",
        "browser",
        "chrome",
        "slack",
        "github",
        "embedded",
        "embedding",
        "interface",
        "workflow",
        "ux",
        "platform",
        "adoption",
        "shipping",
        "surface area",
    ],
    "compute": [
        "compute",
        "latency",
        "pricing",
        "economics",
        "market",
        "capital",
        "infra",
        "infrastructure",
        "spend",
        "tokens",
        "gpu",
        "capacity",
        "data center",
        "datacenter",
        "utilization",
        "inference cost",
        "cost",
        "scale",
        "scaling",
    ],
}

# Backward-compatible alias used by older tests and docs.
TOPIC_KEYWORDS = TOPIC_LEXICON

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
    "memory management": "agents",
}

MONTH_WINDOW_DAYS = 30
RECENCY_HALF_LIFE_DAYS = 10.0

TOPIC_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:\.\./)?topics/([a-z0-9\-]+)\.md\)")
CHANNEL_BULLET_RE = re.compile(r"^-\s+\[(.+?)\]\((https?://[^)]+)\)")
MONTH_BULLET_RE = re.compile(r"^-\s+(\d{4}-\d{2}-\d{2})\s+—\s+\[(.+?)\]\((https?://[^)]+)\)")
FRONTMATTER_RE = re.compile(r"^---\s*$")
BULLET_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
TOKEN_RE = re.compile(r"[a-z0-9]+")


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


def normalize_topic_name(value: str | None) -> str | None:
    if value is None:
        return None
    slug = slugify(str(value))
    if slug in TOPIC_ORDER:
        return slug
    if slug in SECTION_TO_TOPIC:
        return SECTION_TO_TOPIC[slug]
    for topic, label in TOPIC_LABELS.items():
        if slug == slugify(label):
            return topic
    return None


def topic_from_heading(heading: str) -> str | None:
    return normalize_topic_name(heading)


def topic_from_link_target(target: str) -> str | None:
    target = target.strip()
    match = TOPIC_LINK_RE.search(f"[x]({target})")
    if match:
        return match.group(2)
    return None


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
    for phrase in TOPIC_LEXICON.get(topic, []):
        hits = _count_phrase(normalized, phrase)
        if hits:
            score += hits * (1.9 if " " in phrase else 1.0)
    if topic in normalized:
        score += 1.0
    return score


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=None)
def _topic_embedding_profile(topic: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    descriptor_tokens = tokenize(TOPIC_LABELS.get(topic, topic))
    for token in descriptor_tokens:
        counts[token] += 2
    for phrase in TOPIC_LEXICON.get(topic, []):
        tokens = tokenize(phrase)
        if not tokens:
            continue
        weight = 3 if len(tokens) > 1 else 2
        for token in tokens:
            counts[token] += weight
    return counts


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = 0.0
    for token, value in left.items():
        if token in right:
            dot += value * right[token]
    if dot <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def topic_embedding_signal(text: str, topic: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    text_vec = Counter(tokens)
    return _cosine_similarity(text_vec, _topic_embedding_profile(topic))


def recency_decay(published: date | None, reference_date: date) -> float:
    if published is None:
        return 0.62
    age_days = max((reference_date - published).days, 0)
    age_days = min(age_days, MONTH_WINDOW_DAYS)
    return math.exp(-age_days / RECENCY_HALF_LIFE_DAYS)


def _parse_date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    lines = text.splitlines()
    if not lines or not FRONTMATTER_RE.match(lines[0]):
        return {}, text

    meta: dict[str, object] = {}
    body_start = 0
    i = 1
    while i < len(lines):
        if FRONTMATTER_RE.match(lines[i]):
            body_start = i + 1
            break
        raw = lines[i].rstrip()
        i += 1
        if not raw or raw.lstrip().startswith("#"):
            continue
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if value in {"", "|", ">"}:
            collected: list[str] = []
            while i < len(lines):
                next_line = lines[i]
                if not next_line.strip():
                    break
                if next_line.startswith(" ") or next_line.startswith("\t"):
                    collected.append(next_line.strip())
                    i += 1
                    continue
                break
            value = "\n".join(collected)
        elif value.startswith("[") and value.endswith("]"):
            value = [part.strip().strip('"\'') for part in value[1:-1].split(",") if part.strip()]
        meta[key] = value
    body = "\n".join(lines[body_start:])
    return meta, body


def _extract_topic_hint_from_line(line: str) -> str | None:
    match = TOPIC_LINK_RE.search(line)
    if match:
        return match.group(2)
    return None


def _clean_markdown_text(text: str) -> str:
    text = BULLET_LINK_RE.sub(lambda match: match.group(1), text)
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_bullet_line(line: str) -> tuple[str, str | None] | None:
    stripped = line.strip()
    if not (stripped.startswith("- ") or stripped.startswith("* ")):
        return None
    content = stripped[2:].strip()
    if not content:
        return None
    match = BULLET_LINK_RE.search(content)
    url = None
    if match:
        title = match.group(1).strip()
        url = match.group(2).strip()
    else:
        title = content
    return _clean_markdown_text(title), url


def _frontmatter_default_source(meta: dict[str, object], fallback: str) -> str:
    source = meta.get("source")
    if source:
        return str(source)
    return fallback


def _frontmatter_default_topic(meta: dict[str, object]) -> str | None:
    for key in ("topic", "primary_topic", "subject"):
        if key in meta:
            topic = normalize_topic_name(meta[key])
            if topic:
                return topic
    topics = meta.get("topics")
    if isinstance(topics, list):
        for value in topics:
            topic = normalize_topic_name(value)
            if topic:
                return topic
    elif topics is not None:
        topic = normalize_topic_name(topics)
        if topic:
            return topic
    return None


def _frontmatter_default_published(meta: dict[str, object]) -> date | None:
    for key in ("published", "date", "updated"):
        if key in meta:
            parsed = _parse_date_value(meta[key])
            if parsed is not None:
                return parsed
    return None


def parse_source_markdown(path: str | Path, source_name: str) -> list[SourceItem]:
    path = Path(path)
    meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    source = _frontmatter_default_source(meta, source_name)
    default_topic = _frontmatter_default_topic(meta)
    default_published = _frontmatter_default_published(meta)

    items: list[SourceItem] = []
    current_topic = default_topic
    current_section = None
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_section = slugify(line.lstrip("# "))
            mapped = topic_from_heading(line.lstrip("# "))
            if mapped:
                current_topic = mapped
            continue
        if line.startswith(("-", "*")):
            parsed = _parse_bullet_line(line)
            if not parsed:
                continue
            title, url = parsed
            link_topic = _extract_topic_hint_from_line(line)
            topic_hint = link_topic or current_topic or default_topic
            confidence = 0.84
            if link_topic:
                confidence = 0.38
            if current_section in {"how it links outward", "links into the graph", "connective logic"} and link_topic:
                confidence = 0.35
            items.append(
                SourceItem(
                    source=source,
                    title=title,
                    text=title,
                    published=default_published,
                    topic_hint=topic_hint,
                    url=url,
                    confidence=confidence,
                )
            )
    return items


def score_topic_item(item: SourceItem, topic: str, reference_date: date | None = None) -> float:
    if reference_date is None:
        reference_date = date.today()

    text = f"{item.title}\n{item.text}"
    signal = topic_signal(text, topic)
    semantic = topic_embedding_signal(text, topic)
    combined = signal + (semantic * 4.0)
    if combined <= 0 and item.topic_hint == topic:
        combined = 1.2
    elif item.topic_hint == topic:
        combined *= 1.18

    if combined <= 0:
        return 0.0

    source_weight = item.source_weight if item.source_weight != 1.0 else SOURCE_WEIGHTS.get(item.source, 1.0)
    recency = recency_decay(item.published, reference_date)
    return combined * source_weight * recency * item.confidence


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
        "- semantic similarity from local embeddings helps catch paraphrases and partial matches",
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
    return parse_source_markdown(path, source_name=source_name)


def parse_link_surface_bullets(path: str | Path, source_name: str) -> list[SourceItem]:
    return parse_source_markdown(path, source_name=source_name)


def build_ranking_from_paths(month_graph: str | Path | None = None, channel_pages: Iterable[str | Path] = ()) -> list[TopicRanking]:
    items: list[SourceItem] = []
    if month_graph:
        items.extend(parse_month_graph_markdown(month_graph, source_name="month graph"))
    for path in channel_pages:
        path = Path(path)
        source_name = SOURCE_NAME_OVERRIDES.get(path.stem, path.stem.replace("-", " ").title())
        items.extend(parse_source_markdown(path, source_name=source_name))
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
