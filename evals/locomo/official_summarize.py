"""CLI and helpers for summarizing public LoCoMo evaluation results."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from export import load_json_or_jsonl_records

PRIMARY_METRIC_PREFERENCE = ("llm_score", "f1_score", "bleu_score")


def summarize_public_scores(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    normalized_records = [item for item in records if isinstance(item, dict)]
    summary = summarize_metrics(normalized_records)
    primary_metric = _select_primary_metric(summary.get("metric_fields") or [])
    text_only_records = [item for item in normalized_records if not bool(item.get("is_multi_modality"))]
    multimodal_records = [item for item in normalized_records if bool(item.get("is_multi_modality"))]

    summary["primary_metric"] = primary_metric
    summary["record_count"] = len(normalized_records)
    summary["by_is_multi_modality"] = {
        "text_only": summarize_metrics(text_only_records).get("overall", {}),
        "multimodal_available": summarize_metrics(multimodal_records).get("overall", {}),
    }
    summary["metadata"] = {
        "llm_judge": _detect_judge(normalized_records),
        "text_only_count": len(text_only_records),
        "multimodal_available_count": len(multimodal_records),
    }
    return summary


def load_and_summarize_public_scores(input_path: str | Path) -> dict[str, Any]:
    return summarize_public_scores(load_json_or_jsonl_records(input_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize public LoCoMo evaluation results")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--markdown-output-path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = args.output_path or _default_output_path(args.input_path)
    markdown_output_path = args.markdown_output_path or _default_markdown_output_path(output_path)
    summary = load_and_summarize_public_scores(args.input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_output = Path(markdown_output_path)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_public_score_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_path": str(output),
                "markdown_output_path": str(markdown_output),
                "primary_metric": summary.get("primary_metric", ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _default_output_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    return path.with_name(f"{path.stem}_summary.json")


def _default_markdown_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(".md")


def render_public_score_markdown(summary: dict[str, Any]) -> str:
    sections: list[str] = ["# Score Summary", ""]
    metric_fields = [str(item) for item in summary.get("metric_fields") or []]

    overall = summary.get("overall")
    if isinstance(overall, dict):
        sections.append("## Overall")
        sections.append("")
        sections.extend(_render_metric_table(overall, metric_fields))
        sections.append("")

    by_category = summary.get("by_category")
    if isinstance(by_category, dict) and by_category:
        sections.append("## By Category")
        sections.append("")
        sections.extend(_render_group_table(by_category, metric_fields, group_name="Category"))
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def _select_primary_metric(metric_fields: Sequence[str]) -> str:
    normalized = [str(field).strip() for field in metric_fields if str(field).strip()]
    for field in PRIMARY_METRIC_PREFERENCE:
        if field in normalized:
            return field
    return normalized[0] if normalized else ""


def _detect_judge(records: Sequence[dict[str, Any]]) -> str:
    for item in records:
        value = str(item.get("llm_judge") or "").strip()
        if value:
            return value
    return ""


def _render_metric_table(group_summary: dict[str, Any], metric_fields: list[str]) -> list[str]:
    lines = [
        f"- Count: {int(group_summary.get('count', 0) or 0)}",
        "",
        "| Metric | Mean | Count | Missing | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    metrics = group_summary.get("metrics") or {}
    for field in metric_fields:
        metric_summary = metrics.get(field) if isinstance(metrics, dict) else None
        if not isinstance(metric_summary, dict):
            continue
        lines.append(
            "| {metric} | {mean} | {count} | {missing} | {minv} | {maxv} |".format(
                metric=field,
                mean=_format_float(metric_summary.get("mean")),
                count=int(metric_summary.get("count", 0) or 0),
                missing=int(metric_summary.get("missing_count", 0) or 0),
                minv=_format_float(metric_summary.get("min")),
                maxv=_format_float(metric_summary.get("max")),
            )
        )
    return lines


def _render_group_table(
    groups: dict[str, Any],
    metric_fields: list[str],
    *,
    group_name: str,
) -> list[str]:
    headers = [group_name, "Count", *metric_fields]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---:", *(["---:"] * len(metric_fields))]) + " |",
    ]
    for group_key, payload in sorted(groups.items(), key=lambda item: item[0]):
        if not isinstance(payload, dict):
            continue
        row = [
            str(group_key),
            str(int(payload.get("count", 0) or 0)),
            *[_format_float(payload.get(field)) for field in metric_fields],
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-category and overall metric summaries."""
    metric_fields = _discover_metric_fields(records)
    return {
        "metric_fields": metric_fields,
        "overall": _summarize_group(records, metric_fields),
        "by_category": _summarize_groups(_group_records(records, "category"), metric_fields),
        "by_conversations": _summarize_conversations(records, metric_fields),
    }


def _summarize_group(records: list[dict[str, Any]], metric_fields: list[str]) -> dict[str, Any]:
    count = len(records)
    summary: dict[str, Any] = {"count": count, "metrics": {}}

    for field in metric_fields:
        values = [float(record[field]) for record in records if isinstance(record.get(field), (int, float))]
        metric_summary = {
            "count": len(values),
            "missing_count": count - len(values),
            "mean": sum(values) / len(values) if values else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
        summary[field] = metric_summary["mean"]
        summary["metrics"][field] = metric_summary

    return summary


def _summarize_conversations(
    records: list[dict[str, Any]],
    metric_fields: list[str],
) -> dict[str, dict[str, Any]]:
    summarized: dict[str, dict[str, Any]] = {}
    for conversation_idx, items in sorted(_group_records(records, "conversation_idx", skip_missing=True).items()):
        conversation_summary = _summarize_group(items, metric_fields)
        conversation_summary["by_category"] = _summarize_groups(_group_records(items, "category"), metric_fields)
        summarized[conversation_idx] = conversation_summary
    return summarized


def _summarize_groups(
    grouped_records: dict[str, list[dict[str, Any]]],
    metric_fields: list[str],
) -> dict[str, dict[str, Any]]:
    return {
        group_key: _summarize_group(items, metric_fields)
        for group_key, items in sorted(grouped_records.items(), key=lambda item: item[0])
    }


def _group_records(
    records: list[dict[str, Any]],
    field: str,
    *,
    skip_missing: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        raw_value = record.get(field)
        if raw_value is None and skip_missing:
            continue
        group_key = "" if raw_value is None else str(raw_value)
        grouped.setdefault(group_key, []).append(record)
    return grouped


def _discover_metric_fields(records: list[dict[str, Any]]) -> list[str]:
    fields: set[str] = set()
    for record in records:
        for key, value in record.items():
            if not _is_metric_field(key, value):
                continue
            fields.add(key)
    preferred_order = [
        "llm_score",
        "f1_score",
        "bleu_score",
    ]
    ordered = [field for field in preferred_order if field in fields]
    ordered.extend(sorted(field for field in fields if field not in ordered))
    return ordered


def _is_metric_field(field: str, value: Any) -> bool:
    return field.endswith("_score") and isinstance(value, (int, float))


if __name__ == "__main__":
    main()
