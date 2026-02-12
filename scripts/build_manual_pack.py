#!/usr/bin/env python3
import argparse
import csv
import io
import json
from pathlib import Path


def read_feedback_errors(path: Path):
    text = path.read_text(encoding="utf-8")
    if text.count("\n") <= 1 and "\\n" in text:
        text = text.replace("\\n", "\n")
    reader = csv.DictReader(io.StringIO(text))
    errors = set()
    for row in reader:
        fb = (row.get("feedback") or "").strip().lower()
        if fb != "error":
            continue
        try:
            errors.add(int(row.get("page", "").strip()))
        except Exception:
            continue
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    errors = read_feedback_errors(Path(args.feedback))
    if not errors:
        raise SystemExit("No error pages found in feedback.")

    final_data = json.loads(Path(args.final).read_text(encoding="utf-8"))
    summary = {p["page"]: p for p in json.loads(Path(args.summary).read_text(encoding="utf-8"))}

    pages_out = []
    for p in final_data.get("puzzles", []):
        page = int(p["id"].split("p")[-1])
        if page not in errors:
            continue
        board_path = summary.get(page, {}).get("board_path", "")
        givens = p.get("givens", "")
        pages_out.append({
            "page": page,
            "id": p.get("id"),
            "givens": givens,
            "board_path": board_path
        })

    pages_out.sort(key=lambda x: x["page"])
    Path(args.out).write_text(json.dumps({"pages": pages_out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} with {len(pages_out)} pages")


if __name__ == "__main__":
    main()
