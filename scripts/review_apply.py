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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--solve", action="store_true", help="Solve puzzles to fill solution (slow).")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    pages = {p["page"]: p for p in summary if p.get("status") == "ok"}

    # build grid map from givens
    grids = {}
    for page, info in pages.items():
        givens = info.get("givens", "0" * 81)
        grid = [["0"] * 9 for _ in range(9)]
        for i, ch in enumerate(givens):
            grid[i // 9][i % 9] = ch
        grids[page] = grid

    applied = 0
    skipped = 0
    with open(args.review, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if not row.get("correct"):
            skipped += 1
            continue
        try:
            page = int(row["page"])
            r = int(row["row"]) - 1
            c = int(row["col"]) - 1
        except Exception:
            skipped += 1
            continue
        val = row["correct"].strip()
        if val not in list("123456789"):
            skipped += 1
            continue
        if page not in grids:
            skipped += 1
            continue
        grids[page][r][c] = val
        applied += 1

    puzzles = []
    for page, grid in grids.items():
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
            "id": f"xsudoku-p{page}",
            "givens": grid_to_string(grid),
            "solution": solution,
            "source": "book",
            "difficulty": 0,
            "createdAt": date.today().isoformat(),
            "status": status
        })

    out = {
        "version": 1,
        "exportedAt": date.today().isoformat(),
        "puzzles": puzzles
    }

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Applied {applied} corrections, skipped {skipped} rows")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
