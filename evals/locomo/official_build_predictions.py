#!/usr/bin/env python3
"""Minimal reference script for building predictions.jsonl from questions.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a LoCoMo predictions.jsonl file from questions.jsonl")
    parser.add_argument("--questions-path", required=True, help="Path to questions.jsonl")
    parser.add_argument("--output-path", required=True, help="Path to write predictions.jsonl")
    parser.add_argument(
        "--default-answer",
        default="",
        help="Fallback answer used by this reference script. Replace run_target_system() with your own system call.",
    )
    return parser


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def run_target_system(question_record: dict[str, Any], default_answer: str) -> str:
    """Replace this stub with a call into your own system."""
    _ = question_record
    return default_answer


def write_predictions(
    question_records: list[dict[str, Any]],
    output_path: str | Path,
    default_answer: str,
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as handle:
        for item in question_records:
            qa_id = str(item["qa_id"])
            predicted_answer = run_target_system(item, default_answer)
            handle.write(
                json.dumps(
                    {
                        "qa_id": qa_id,
                        "predicted_answer": predicted_answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    args = build_parser().parse_args()
    question_records = load_jsonl(args.questions_path)
    write_predictions(
        question_records=question_records,
        output_path=args.output_path,
        default_answer=args.default_answer,
    )


if __name__ == "__main__":
    main()
