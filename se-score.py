#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

DIGITS = "123456789"
DIGIT_SET = set(DIGITS)

Idx = int
Pos = Tuple[int, int]


class ContradictionError(Exception):
    pass


def rc_to_idx(r: int, c: int) -> Idx:
    return r * 9 + c


def idx_to_rc(i: Idx) -> Pos:
    return divmod(i, 9)


ROW_UNITS: List[List[Idx]] = [[rc_to_idx(r, c) for c in range(9)] for r in range(9)]
COL_UNITS: List[List[Idx]] = [[rc_to_idx(r, c) for r in range(9)] for c in range(9)]
BOX_UNITS: List[List[Idx]] = []
for br in range(0, 9, 3):
    for bc in range(0, 9, 3):
        BOX_UNITS.append([rc_to_idx(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)])
UNITS: List[List[Idx]] = ROW_UNITS + COL_UNITS + BOX_UNITS

PEERS: List[Set[Idx]] = [set() for _ in range(81)]
for i in range(81):
    r, c = idx_to_rc(i)
    b = (r // 3) * 3 + (c // 3)
    for j in ROW_UNITS[r] + COL_UNITS[c] + BOX_UNITS[b]:
        if j != i:
            PEERS[i].add(j)


@dataclass
class Step:
    technique: str
    rating: float
    fills: List[Tuple[Idx, str]] = field(default_factory=list)
    eliminations: List[Tuple[Idx, str]] = field(default_factory=list)
    note: str = ""


@dataclass
class Technique:
    name: str
    rating: float
    finder: Callable[[List[str], List[Set[str]]], Optional[Step]]


@dataclass
class SolveResult:
    solved: bool
    stuck: bool
    contradiction: bool
    final_grid: str
    steps: List[Step]
    technique_counts: Dict[str, int]
    hardest_technique: str
    score: float
    difficulty: int
    error: str = ""


def normalize_givens(givens: str) -> str:
    s = givens.strip()
    if len(s) != 81:
        raise ValueError("givens must be exactly 81 chars")
    out = []
    for ch in s:
        if ch in DIGIT_SET:
            out.append(ch)
        elif ch in ("0", "."):
            out.append(".")
        else:
            raise ValueError(f"invalid char in givens: {ch}")
    return "".join(out)


def grid_to_string(values: List[str]) -> str:
    return "".join(values)


def validate_values(values: List[str]) -> None:
    for unit in UNITS:
        seen: Set[str] = set()
        for idx in unit:
            v = values[idx]
            if v == ".":
                continue
            if v in seen:
                r, c = idx_to_rc(idx)
                raise ContradictionError(f"duplicate solved digit {v} at r{r+1}c{c+1}")
            seen.add(v)


def init_state(givens: str) -> Tuple[List[str], List[Set[str]]]:
    values = list(normalize_givens(givens))
    validate_values(values)

    cands: List[Set[str]] = []
    for idx in range(81):
        if values[idx] != ".":
            cands.append({values[idx]})
            continue
        used = {values[p] for p in PEERS[idx] if values[p] in DIGIT_SET}
        cand = DIGIT_SET - used
        if not cand:
            r, c = idx_to_rc(idx)
            raise ContradictionError(f"no candidate at r{r+1}c{c+1}")
        cands.append(set(cand))
    return values, cands


def validate_candidates(values: List[str], cands: List[Set[str]]) -> None:
    for idx in range(81):
        if values[idx] == ".":
            if not cands[idx]:
                r, c = idx_to_rc(idx)
                raise ContradictionError(f"empty candidate at r{r+1}c{c+1}")
        else:
            if cands[idx] != {values[idx]}:
                cands[idx] = {values[idx]}


def apply_step(step: Step, values: List[str], cands: List[Set[str]]) -> bool:
    changed = False

    for idx, d in sorted(set(step.eliminations), key=lambda x: (x[0], x[1])):
        if values[idx] != ".":
            continue
        if d in cands[idx]:
            cands[idx].discard(d)
            changed = True
            if not cands[idx]:
                r, c = idx_to_rc(idx)
                raise ContradictionError(f"candidate wipeout at r{r+1}c{c+1}")

    newly_filled: List[Tuple[Idx, str]] = []
    for idx, v in sorted(set(step.fills), key=lambda x: (x[0], x[1])):
        if values[idx] == ".":
            if v not in cands[idx]:
                r, c = idx_to_rc(idx)
                raise ContradictionError(f"invalid fill {v} at r{r+1}c{c+1}")
            values[idx] = v
            cands[idx] = {v}
            newly_filled.append((idx, v))
            changed = True
        elif values[idx] != v:
            r, c = idx_to_rc(idx)
            raise ContradictionError(f"conflicting fill at r{r+1}c{c+1}")

    for idx, v in newly_filled:
        for p in PEERS[idx]:
            if values[p] == "." and v in cands[p]:
                cands[p].discard(v)
                if not cands[p]:
                    r, c = idx_to_rc(p)
                    raise ContradictionError(f"candidate wipeout at r{r+1}c{c+1}")

    validate_values(values)
    validate_candidates(values, cands)
    return changed


def is_solved(values: List[str]) -> bool:
    return all(v in DIGIT_SET for v in values)


def first_fill_note(idx: Idx, d: str) -> str:
    r, c = idx_to_rc(idx)
    return f"r{r+1}c{c+1}={d}"


def find_naked_single(values: List[str], cands: List[Set[str]]) -> Optional[Step]:
    for idx in range(81):
        if values[idx] == "." and len(cands[idx]) == 1:
            d = next(iter(cands[idx]))
            return Step("Naked Single", 1.0, fills=[(idx, d)], note=first_fill_note(idx, d))
    return None


def find_hidden_single(values: List[str], cands: List[Set[str]]) -> Optional[Step]:
    for unit in UNITS:
        for d in DIGITS:
            spots = [idx for idx in unit if values[idx] == "." and d in cands[idx]]
            if len(spots) == 1:
                idx = spots[0]
                return Step("Hidden Single", 1.2, fills=[(idx, d)], note=first_fill_note(idx, d))
    return None


def find_locked_candidates(values: List[str], cands: List[Set[str]]) -> Optional[Step]:
    # Pointing (box -> row/col)
    for b_idx, box in enumerate(BOX_UNITS):
        br, bc = divmod(b_idx, 3)
        for d in DIGITS:
            spots = [idx for idx in box if values[idx] == "." and d in cands[idx]]
            if len(spots) < 2:
                continue
            rows = {idx_to_rc(i)[0] for i in spots}
            cols = {idx_to_rc(i)[1] for i in spots}

            if len(rows) == 1:
                row = next(iter(rows))
                elims = []
                for idx in ROW_UNITS[row]:
                    if idx in box:
                        continue
                    if values[idx] == "." and d in cands[idx]:
                        elims.append((idx, d))
                if elims:
                    return Step("Locked Candidates", 2.0, eliminations=elims,
                                note=f"pointing row r{row+1} digit {d} in box {br+1},{bc+1}")

            if len(cols) == 1:
                col = next(iter(cols))
                elims = []
                for idx in COL_UNITS[col]:
                    if idx in box:
                        continue
                    if values[idx] == "." and d in cands[idx]:
                        elims.append((idx, d))
                if elims:
                    return Step("Locked Candidates", 2.0, eliminations=elims,
                                note=f"pointing col c{col+1} digit {d} in box {br+1},{bc+1}")

    # Claiming (row/col -> box)
    for r, row_unit in enumerate(ROW_UNITS):
        for d in DIGITS:
            spots = [idx for idx in row_unit if values[idx] == "." and d in cands[idx]]
            if len(spots) < 2:
                continue
            boxes = {((idx_to_rc(i)[0] // 3) * 3 + (idx_to_rc(i)[1] // 3)) for i in spots}
            if len(boxes) == 1:
                b = next(iter(boxes))
                elims = []
                for idx in BOX_UNITS[b]:
                    if idx in row_unit:
                        continue
                    if values[idx] == "." and d in cands[idx]:
                        elims.append((idx, d))
                if elims:
                    return Step("Locked Candidates", 2.0, eliminations=elims,
                                note=f"claiming row r{r+1} digit {d}")

    for c, col_unit in enumerate(COL_UNITS):
        for d in DIGITS:
            spots = [idx for idx in col_unit if values[idx] == "." and d in cands[idx]]
            if len(spots) < 2:
                continue
            boxes = {((idx_to_rc(i)[0] // 3) * 3 + (idx_to_rc(i)[1] // 3)) for i in spots}
            if len(boxes) == 1:
                b = next(iter(boxes))
                elims = []
                for idx in BOX_UNITS[b]:
                    if idx in col_unit:
                        continue
                    if values[idx] == "." and d in cands[idx]:
                        elims.append((idx, d))
                if elims:
                    return Step("Locked Candidates", 2.0, eliminations=elims,
                                note=f"claiming col c{c+1} digit {d}")
    return None


def find_naked_subset(values: List[str], cands: List[Set[str]], n: int, name: str, rating: float) -> Optional[Step]:
    for unit in UNITS:
        cells = [idx for idx in unit if values[idx] == "." and 2 <= len(cands[idx]) <= n]
        if len(cells) < n:
            continue
        for combo in itertools.combinations(cells, n):
            union = set().union(*(cands[idx] for idx in combo))
            if len(union) != n:
                continue
            subset_cells = [idx for idx in unit if values[idx] == "." and cands[idx].issubset(union)]
            if len(subset_cells) != n:
                continue
            elims: List[Tuple[Idx, str]] = []
            for idx in unit:
                if values[idx] != "." or idx in combo:
                    continue
                for d in sorted(union):
                    if d in cands[idx]:
                        elims.append((idx, d))
            if elims:
                return Step(name, rating, eliminations=elims)
    return None


def find_hidden_subset(values: List[str], cands: List[Set[str]], n: int, name: str, rating: float) -> Optional[Step]:
    for unit in UNITS:
        unsolved = [idx for idx in unit if values[idx] == "."]
        if len(unsolved) < n:
            continue
        for digits_combo in itertools.combinations(DIGITS, n):
            spot_sets = []
            valid = True
            for d in digits_combo:
                spots = {idx for idx in unsolved if d in cands[idx]}
                if not spots:
                    valid = False
                    break
                spot_sets.append(spots)
            if not valid:
                continue
            union = set().union(*spot_sets)
            if len(union) != n:
                continue

            allowed = set(digits_combo)
            elims: List[Tuple[Idx, str]] = []
            for idx in sorted(union):
                for d in sorted(cands[idx] - allowed):
                    elims.append((idx, d))
            if elims:
                return Step(name, rating, eliminations=elims)
    return None


def find_naked_pair(values: List[str], cands: List[Set[str]]) -> Optional[Step]:
    return find_naked_subset(values, cands, 2, "Naked Pair", 2.6)


def find_hidden_pair(values: List[str], cands: List[Set[str]]) -> Optional[Step]:
    return find_hidden_subset(values, cands, 2, "Hidden Pair", 2.8)


def find_naked_triple(values: List[str], cands: List[Set[str]]) -> Optional[Step]:
    return find_naked_subset(values, cands, 3, "Naked Triple", 3.2)


def find_hidden_triple(values: List[str], cands: List[Set[str]]) -> Optional[Step]:
    return find_hidden_subset(values, cands, 3, "Hidden Triple", 3.4)


def find_x_wing(values: List[str], cands: List[Set[str]]) -> Optional[Step]:
    for d in DIGITS:
        row_patterns: List[Tuple[int, Tuple[int, int]]] = []
        for r in range(9):
            cols = [c for c in range(9) if values[rc_to_idx(r, c)] == "." and d in cands[rc_to_idx(r, c)]]
            if len(cols) == 2:
                row_patterns.append((r, (cols[0], cols[1])))
        for (r1, cols1), (r2, cols2) in itertools.combinations(row_patterns, 2):
            if cols1 != cols2:
                continue
            c1, c2 = cols1
            elims = []
            for r in range(9):
                if r in (r1, r2):
                    continue
                for c in (c1, c2):
                    idx = rc_to_idx(r, c)
                    if values[idx] == "." and d in cands[idx]:
                        elims.append((idx, d))
            if elims:
                return Step("X-Wing", 4.2, eliminations=elims)

        col_patterns: List[Tuple[int, Tuple[int, int]]] = []
        for c in range(9):
            rows = [r for r in range(9) if values[rc_to_idx(r, c)] == "." and d in cands[rc_to_idx(r, c)]]
            if len(rows) == 2:
                col_patterns.append((c, (rows[0], rows[1])))
        for (c1, rows1), (c2, rows2) in itertools.combinations(col_patterns, 2):
            if rows1 != rows2:
                continue
            r1, r2 = rows1
            elims = []
            for c in range(9):
                if c in (c1, c2):
                    continue
                for r in (r1, r2):
                    idx = rc_to_idx(r, c)
                    if values[idx] == "." and d in cands[idx]:
                        elims.append((idx, d))
            if elims:
                return Step("X-Wing", 4.2, eliminations=elims)
    return None


def find_swordfish(values: List[str], cands: List[Set[str]]) -> Optional[Step]:
    for d in DIGITS:
        row_sets: List[Tuple[int, Set[int]]] = []
        for r in range(9):
            cols = {c for c in range(9) if values[rc_to_idx(r, c)] == "." and d in cands[rc_to_idx(r, c)]}
            if 2 <= len(cols) <= 3:
                row_sets.append((r, cols))
        for combo in itertools.combinations(row_sets, 3):
            rows = [r for r, _ in combo]
            union_cols = set().union(*(cols for _, cols in combo))
            if len(union_cols) != 3:
                continue
            elims = []
            for r in range(9):
                if r in rows:
                    continue
                for c in sorted(union_cols):
                    idx = rc_to_idx(r, c)
                    if values[idx] == "." and d in cands[idx]:
                        elims.append((idx, d))
            if elims:
                return Step("Swordfish", 4.8, eliminations=elims)

        col_sets: List[Tuple[int, Set[int]]] = []
        for c in range(9):
            rows = {r for r in range(9) if values[rc_to_idx(r, c)] == "." and d in cands[rc_to_idx(r, c)]}
            if 2 <= len(rows) <= 3:
                col_sets.append((c, rows))
        for combo in itertools.combinations(col_sets, 3):
            cols = [c for c, _ in combo]
            union_rows = set().union(*(rows for _, rows in combo))
            if len(union_rows) != 3:
                continue
            elims = []
            for c in range(9):
                if c in cols:
                    continue
                for r in sorted(union_rows):
                    idx = rc_to_idx(r, c)
                    if values[idx] == "." and d in cands[idx]:
                        elims.append((idx, d))
            if elims:
                return Step("Swordfish", 4.8, eliminations=elims)
    return None


def find_xy_wing(values: List[str], cands: List[Set[str]]) -> Optional[Step]:
    pivots = [idx for idx in range(81) if values[idx] == "." and len(cands[idx]) == 2]
    for pivot in pivots:
        x, y = sorted(cands[pivot])
        for wing1 in sorted(PEERS[pivot]):
            if values[wing1] != "." or len(cands[wing1]) != 2:
                continue
            s1 = cands[wing1]
            if x not in s1 or y in s1:
                continue
            z = next(iter(s1 - {x}))

            for wing2 in sorted(PEERS[pivot]):
                if wing2 == wing1:
                    continue
                if values[wing2] != "." or len(cands[wing2]) != 2:
                    continue
                s2 = cands[wing2]
                if y not in s2 or x in s2 or z not in s2:
                    continue

                targets = sorted((PEERS[wing1] & PEERS[wing2]) - {pivot, wing1, wing2})
                elims = [(idx, z) for idx in targets if values[idx] == "." and z in cands[idx]]
                if elims:
                    return Step("XY-Wing", 4.4, eliminations=elims)
    return None


TECHNIQUES: List[Technique] = [
    Technique("Naked Single", 1.0, find_naked_single),
    Technique("Hidden Single", 1.2, find_hidden_single),
    Technique("Locked Candidates", 2.0, find_locked_candidates),
    Technique("Naked Pair", 2.6, find_naked_pair),
    Technique("Hidden Pair", 2.8, find_hidden_pair),
    Technique("Naked Triple", 3.2, find_naked_triple),
    Technique("Hidden Triple", 3.4, find_hidden_triple),
    Technique("X-Wing", 4.2, find_x_wing),
    Technique("XY-Wing", 4.4, find_xy_wing),
    Technique("Swordfish", 4.8, find_swordfish),
]


LEVEL_SIMPLE = {"Naked Single", "Hidden Single"}
LEVEL_MEDIUM = {"Locked Candidates", "Naked Pair", "Hidden Pair", "Naked Triple", "Hidden Triple"}
LEVEL_HARD = {"X-Wing", "XY-Wing", "Swordfish"}


def map_difficulty(hardest: str, solved: bool) -> int:
    if not solved:
        return 4
    if hardest in LEVEL_SIMPLE:
        return 1
    if hardest in LEVEL_MEDIUM:
        return 2
    if hardest in LEVEL_HARD:
        return 3
    return 4


def solve_se_strict(givens: str, max_steps: int = 5000) -> SolveResult:
    try:
        values, cands = init_state(givens)
    except (ValueError, ContradictionError) as e:
        return SolveResult(
            solved=False,
            stuck=False,
            contradiction=True,
            final_grid="",
            steps=[],
            technique_counts={},
            hardest_technique="None",
            score=0.0,
            difficulty=4,
            error=str(e),
        )

    steps: List[Step] = []
    stuck = False

    while not is_solved(values) and len(steps) < max_steps:
        progressed = False
        for tech in TECHNIQUES:
            step = tech.finder(values, cands)
            if step is None:
                continue
            try:
                changed = apply_step(step, values, cands)
            except ContradictionError as e:
                return SolveResult(
                    solved=False,
                    stuck=False,
                    contradiction=True,
                    final_grid=grid_to_string(values),
                    steps=steps,
                    technique_counts={},
                    hardest_technique="None",
                    score=0.0,
                    difficulty=4,
                    error=str(e),
                )
            if changed:
                steps.append(step)
                progressed = True
                break
        if not progressed:
            stuck = True
            break

    counts: Dict[str, int] = {}
    hardest = "None"
    hardest_score = 0.0
    for st in steps:
        counts[st.technique] = counts.get(st.technique, 0) + 1
        if st.rating > hardest_score:
            hardest_score = st.rating
            hardest = st.technique

    solved = is_solved(values) and not stuck
    difficulty = map_difficulty(hardest, solved)

    return SolveResult(
        solved=solved,
        stuck=stuck,
        contradiction=False,
        final_grid=grid_to_string(values),
        steps=steps,
        technique_counts=counts,
        hardest_technique=hardest,
        score=round(hardest_score, 2),
        difficulty=difficulty,
        error="",
    )


def step_to_json(step: Step) -> Dict[str, object]:
    def pos(idx: Idx) -> str:
        r, c = idx_to_rc(idx)
        return f"r{r+1}c{c+1}"

    return {
        "technique": step.technique,
        "rating": step.rating,
        "fills": [{"cell": pos(idx), "digit": d} for idx, d in step.fills],
        "eliminations": [{"cell": pos(idx), "digit": d} for idx, d in step.eliminations],
        "note": step.note,
    }


def rate_file(path: str, write_path: str, keep_trace: bool = False) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    puzzles = data.get("puzzles", [])
    solved_count = 0
    stuck_count = 0
    contradiction_count = 0

    for puzzle in puzzles:
        givens = puzzle.get("givens", "")
        result = solve_se_strict(givens)

        if result.solved:
            solved_count += 1
        elif result.contradiction:
            contradiction_count += 1
        else:
            stuck_count += 1

        se_obj: Dict[str, object] = {
            "version": "SE-Strict-v1",
            "score": result.score,
            "hardestTechnique": result.hardest_technique,
            "steps": len(result.steps),
            "techniqueCounts": result.technique_counts,
            "solved": result.solved,
            "stuck": result.stuck,
            "contradiction": result.contradiction,
        }
        if result.error:
            se_obj["error"] = result.error

        if puzzle.get("solution") and result.solved:
            se_obj["matchesProvidedSolution"] = (puzzle["solution"] == result.final_grid)

        if keep_trace:
            se_obj["trace"] = [step_to_json(s) for s in result.steps]

        puzzle["seStrict"] = se_obj
        puzzle["difficulty"] = result.difficulty

    with open(write_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "total": len(puzzles),
        "solved": solved_count,
        "stuck": stuck_count,
        "contradiction": contradiction_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SE-Strict Sudoku rater")
    parser.add_argument("paths", nargs="+", help="JSON files with puzzles[]")
    parser.add_argument("--inplace", action="store_true", help="write back to source file")
    parser.add_argument("--keep-trace", action="store_true", help="store full step trace per puzzle")
    args = parser.parse_args()

    for path in args.paths:
        out_path = path if args.inplace else path.replace(".json", ".scored.json")
        summary = rate_file(path, out_path, keep_trace=args.keep_trace)
        print(
            f"{out_path}: total={summary['total']} solved={summary['solved']} "
            f"stuck={summary['stuck']} contradiction={summary['contradiction']}"
        )


if __name__ == "__main__":
    main()
