#!/usr/bin/env python3
"""Convert RULER's PaulGrahamEssays.json into VeOmni plaintext JSONL.

The In-Place-TTT README recommends:
  - data.data_type=plaintext
  - data.datasets_type=iterable
  - data.text_keys=content_split

This script reads the single-text JSON blob shipped by RULER and emits a
newline-delimited JSON file whose records contain a `content_split` field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = (
        repo_root.parent / "RULER" / "scripts" / "data" / "synthetic" / "json" / "PaulGrahamEssays.json"
    )
    default_output = repo_root / "data" / "paul_graham_essays.jsonl"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to RULER's PaulGrahamEssays.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Destination JSONL path. Each line will contain `content_split`.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help=(
            "Optional maximum character count per output record. "
            "Use 0 to emit a single record."
        ),
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=1000,
        help="Minimum chunk size before flushing when --max-chars is enabled.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def split_paragraphs(text: str) -> Iterable[str]:
    return (paragraph.strip() for paragraph in text.split("\n\n"))


def chunk_text(text: str, max_chars: int, min_chars: int) -> Iterator[str]:
    if max_chars <= 0:
        yield text
        return

    parts = [part for part in split_paragraphs(text) if part]
    if not parts:
        return

    buffer: list[str] = []
    buffer_len = 0

    for part in parts:
        separator_len = 2 if buffer else 0
        projected_len = buffer_len + separator_len + len(part)
        if buffer and projected_len > max_chars and buffer_len >= min_chars:
            yield "\n\n".join(buffer)
            buffer = [part]
            buffer_len = len(part)
            continue

        buffer.append(part)
        buffer_len = projected_len

    if buffer:
        yield "\n\n".join(buffer)


def load_ruler_text(path: Path) -> str:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "text" not in payload or not isinstance(payload["text"], str):
        raise ValueError(f"Expected a JSON object with a string `text` field in {path}")
    return normalize_text(payload["text"])


def write_jsonl(records: Iterable[str], output_path: Path, input_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for index, record in enumerate(records):
            line = {
                "content_split": record,
                "source": "RULER/PaulGrahamEssays",
                "record_index": index,
                "source_file": str(input_path),
            }
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    text = load_ruler_text(args.input)
    chunks = list(chunk_text(text, max_chars=args.max_chars, min_chars=args.min_chars))
    if not chunks:
        raise ValueError(f"No text records were produced from {args.input}")

    count = write_jsonl(chunks, args.output, args.input)
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
