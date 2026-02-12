#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    final_data = json.loads(Path(args.final).read_text())
    summary = {p["page"]: p for p in json.loads(Path(args.summary).read_text())}

    review_rows = []
    with open(args.review, newline="", encoding="utf-8") as f:
        review_rows = list(csv.DictReader(f))

    by_page = {}
    for row in review_rows:
        try:
            page = int(row["page"])
        except Exception:
            continue
        by_page.setdefault(page, []).append(row)

    pages_out = []
    for p in final_data.get("puzzles", []):
        page = int(p["id"].split("p")[-1])
        rows = by_page.get(page, [])
        pending = [r for r in rows if not r.get("correct")]
        board_path = summary.get(page, {}).get("board_path", "")
        pages_out.append({
            "page": page,
            "id": p["id"],
            "status": p.get("status", "pending"),
            "givens": p.get("givens", ""),
            "givens_count": sum(1 for ch in p.get("givens", "") if ch in "123456789"),
            "review_total": len(rows),
            "review_pending": len(pending),
            "board_path": board_path
        })

    pages_out.sort(key=lambda x: x["page"])
    Path(args.out).write_text(json.dumps({"pages": pages_out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} with {len(pages_out)} pages")


if __name__ == "__main__":
    main()
