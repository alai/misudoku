#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import date
from pathlib import Path


def grid_to_string(grid):
    return "".join(grid[r][c] for r in range(9) for c in range(9))


def is_valid_grid(grid):
    # check rows, cols, boxes for conflicts
    for r in range(9):
        seen = set()
        for c in range(9):
            v = grid[r][c]
            if v == "0":
                continue
            if v in seen:
                return False
            seen.add(v)
    for c in range(9):
        seen = set()
        for r in range(9):
            v = grid[r][c]
            if v == "0":
                continue
            if v in seen:
                return False
            seen.add(v)
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            seen = set()
            for r in range(br, br + 3):
                for c in range(bc, bc + 3):
                    v = grid[r][c]
                    if v == "0":
                        continue
                    if v in seen:
                        return False
                    seen.add(v)
    return True


def solve_grid(grid):
    def find_empty():
        for r in range(9):
            for c in range(9):
                if grid[r][c] == "0":
                    return r, c
        return None

    def is_valid(r, c, val):
        for cc in range(9):
            if grid[r][cc] == val:
                return False
        for rr in range(9):
            if grid[rr][c] == val:
                return False
        br = (r // 3) * 3
        bc = (c // 3) * 3
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                if grid[rr][cc] == val:
                    return False
        return True

    empty = find_empty()
    if not empty:
        return True, grid
    r, c = empty
    for val in "123456789":
        if is_valid(r, c, val):
            grid[r][c] = val
            ok, solved = solve_grid(grid)
            if ok:
                return True, solved
            grid[r][c] = "0"
    return False, grid


def load_discarded(path: Path):
    if not path or not path.exists():
        return set()
    discarded = set()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        status = (row.get("status") or "").strip().lower()
        if status != "discarded":
            continue
        try:
            discarded.add(int(row.get("page", "").strip()))
        except Exception:
            continue
    return discarded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", required=True)
    parser.add_argument("--corrections", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--solve", action="store_true", help="Solve puzzles to fill solution (slow).")
    parser.add_argument("--status", help="Optional manual-page-status.csv to exclude discarded pages.")
    args = parser.parse_args()

    final_data = json.loads(Path(args.final).read_text(encoding="utf-8"))
    corrections = {}

    with open(args.corrections, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        try:
            page = int(row.get("page", "").strip())
            r = int(row.get("row", "").strip()) - 1
            c = int(row.get("col", "").strip()) - 1
        except Exception:
            continue
        val = (row.get("value") or "0").strip()
        if val == "":
            val = "0"
        if val not in list("0123456789"):
            continue
        corrections.setdefault(page, []).append((r, c, val))

    discarded = load_discarded(Path(args.status)) if args.status else set()

    applied = 0
    puzzles = []
    for p in final_data.get("puzzles", []):
        page = int(p["id"].split("p")[-1])
        if page in discarded:
            continue
        givens = p.get("givens", "0" * 81)
        grid = [["0"] * 9 for _ in range(9)]
        for i, ch in enumerate(givens):
            grid[i // 9][i % 9] = ch
        for r, c, val in corrections.get(page, []):
            if 0 <= r < 9 and 0 <= c < 9:
                grid[r][c] = val
                applied += 1

        if not is_valid_grid(grid):
            status = "invalid"
            solution = ""
        elif args.solve:
            grid_copy = [row[:] for row in grid]
            ok, solved = solve_grid(grid_copy)
            status = "ok" if ok else "unsolved"
            solution = grid_to_string(solved) if ok else ""
        else:
            status = "pending"
            solution = ""

        puzzles.append({
            "id": p.get("id"),
            "givens": grid_to_string(grid),
            "solution": solution,
            "source": p.get("source", "book"),
            "difficulty": p.get("difficulty", 0),
            "createdAt": p.get("createdAt", date.today().isoformat()),
            "status": status
        })

    out = {
        "version": 1,
        "exportedAt": date.today().isoformat(),
        "puzzles": puzzles
    }

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Applied {applied} cell updates")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
