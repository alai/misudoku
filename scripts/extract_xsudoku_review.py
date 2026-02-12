#!/usr/bin/env python3
import argparse
import csv
import json
import io
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import List, Tuple, Iterable, Dict, Any, Set

import fitz
import cv2
import numpy as np
try:
    import pytesseract
except Exception:
    pytesseract = None

PDF_PATH = "/Users/alai/Projects/Codex/xtreme-suduku/X-sudoku.pdf"


def render_page(doc: fitz.Document, page_num: int, dpi: int = 300) -> np.ndarray:
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape((pix.height, pix.width, pix.n))
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    # stretch contrast
    adj = cv2.convertScaleAbs(cl, alpha=2.0, beta=-60)
    return cv2.cvtColor(adj, cv2.COLOR_GRAY2RGB)


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def find_board(img: np.ndarray, size: int = 900) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found")

    img_area = img.shape[0] * img.shape[1]
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.05:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area

    if best is None:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        best = box.astype(np.intp)

    pts = best.reshape(4, 2).astype("float32")
    rect = order_points(pts)

    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped, rect


def cluster_positions(values: List[float], max_gap: float) -> List[float]:
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if abs(v - clusters[-1][-1]) <= max_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def infer_edges(positions: List[float], length: int) -> Tuple[float, float]:
    positions = sorted(positions)
    if len(positions) < 2:
        raise RuntimeError("Not enough line positions to infer edges")
    diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    diffs = [d for d in diffs if d > 0.0]
    spacing = float(np.median(diffs)) if diffs else (positions[-1] - positions[0]) / max(1, len(positions) - 1)
    if spacing <= 0:
        spacing = max(1.0, (positions[-1] - positions[0]) / max(1, len(positions) - 1))

    if len(positions) >= 10:
        start = positions[0]
        end = positions[-1]
    else:
        start = positions[0] - spacing
        end = positions[-1] + spacing

    start = max(0.0, min(float(length - 1), start))
    end = max(0.0, min(float(length - 1), end))
    if end <= start:
        start = max(0.0, min(float(length - 2), start))
        end = start + 1.0
    return start, end


def find_board_by_lines(img: np.ndarray, size: int = 900) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 120)

    h, w = gray.shape
    min_len = int(0.25 * max(h, w))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=min_len, maxLineGap=40)
    if lines is None:
        raise RuntimeError("No Hough lines found")

    horizontal = []
    vertical = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle < 12:
            horizontal.append((y1 + y2) / 2.0)
        elif angle > 78:
            vertical.append((x1 + x2) / 2.0)

    max_gap_y = max(8.0, h * 0.012)
    max_gap_x = max(8.0, w * 0.012)
    hpos = cluster_positions(horizontal, max_gap_y)
    vpos = cluster_positions(vertical, max_gap_x)

    if len(hpos) < 4 or len(vpos) < 4:
        raise RuntimeError("Insufficient grid lines detected")

    top, bottom = infer_edges(hpos, h)
    left, right = infer_edges(vpos, w)

    rect = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype="float32")
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped, rect


def deskew_image(img: np.ndarray) -> Tuple[np.ndarray, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return img, 0.0
    angles = []
    for rho, theta in lines[:, 0]:
        # horizontal lines have theta around pi/2
        if abs(theta - (np.pi / 2)) < np.deg2rad(15):
            angles.append(theta - (np.pi / 2))
    if not angles:
        return img, 0.0
    angle = float(np.median(angles))
    if abs(angle) < np.deg2rad(0.5):
        return img, angle
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), np.degrees(angle), 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def select_best_lines(positions: List[float], target: int) -> List[float]:
    positions = sorted(positions)
    if len(positions) < target:
        raise RuntimeError("Not enough line positions")
    if len(positions) == target:
        return positions
    # choose consecutive window with most uniform spacing
    best = None
    best_score = None
    for i in range(0, len(positions) - target + 1):
        window = positions[i : i + target]
        diffs = [window[j + 1] - window[j] for j in range(len(window) - 1)]
        if not diffs:
            continue
        score = float(np.std(diffs))
        if best_score is None or score < best_score:
            best_score = score
            best = window
    return best if best is not None else positions[:target]


def find_board_by_gridlines(img: np.ndarray, size: int = 900) -> Tuple[np.ndarray, np.ndarray]:
    enhanced = enhance_contrast(img)
    deskewed, _ = deskew_image(enhanced)
    gray = cv2.cvtColor(deskewed, cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)

    h, w = bw.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 12), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 12)))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    h_proj = np.sum(h_lines > 0, axis=1)
    v_proj = np.sum(v_lines > 0, axis=0)
    h_thresh = max(10.0, float(np.max(h_proj)) * 0.45)
    v_thresh = max(10.0, float(np.max(v_proj)) * 0.45)

    h_positions = [i for i, v in enumerate(h_proj) if v >= h_thresh]
    v_positions = [i for i, v in enumerate(v_proj) if v >= v_thresh]

    h_clusters = cluster_positions(h_positions, max_gap=max(6.0, h * 0.008))
    v_clusters = cluster_positions(v_positions, max_gap=max(6.0, w * 0.008))

    if len(h_clusters) < 8 or len(v_clusters) < 8:
        raise RuntimeError("Insufficient grid lines from projection")

    # prefer 10 lines (with borders). if only 9, infer borders
    if len(h_clusters) >= 10:
        h_lines_sel = select_best_lines(h_clusters, 10)
        top, bottom = h_lines_sel[0], h_lines_sel[-1]
    else:
        top, bottom = infer_edges(h_clusters, h)

    if len(v_clusters) >= 10:
        v_lines_sel = select_best_lines(v_clusters, 10)
        left, right = v_lines_sel[0], v_lines_sel[-1]
    else:
        left, right = infer_edges(v_clusters, w)

    rect = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype="float32")
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(deskewed, M, (size, size))
    return warped, rect


def read_feedback_errors(path: Path) -> Set[int]:
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


def cell_ocr(cell_bw: np.ndarray) -> Tuple[str, float]:
    if pytesseract is not None:
        config = "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"
        data = pytesseract.image_to_data(cell_bw, config=config, output_type=pytesseract.Output.DICT)
        text = ""
        conf = 0.0
        for t, cf in zip(data.get("text", []), data.get("conf", [])):
            if t.strip():
                text = t.strip()
                try:
                    conf = float(cf)
                except Exception:
                    conf = 0.0
                break
        return text, conf

    # fallback to tesseract CLI
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, cell_bw)
        cmd = [
            "tesseract",
            tmp_path,
            "stdout",
            "--psm",
            "10",
            "-c",
            "tessedit_char_whitelist=123456789",
            "tsv",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            return "", 0.0
        lines = [line for line in res.stdout.splitlines() if line.strip()]
        if len(lines) < 2:
            return "", 0.0
        # TSV format: header then rows
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            text = parts[11].strip()
            conf_str = parts[10].strip()
            if not text:
                continue
            try:
                conf = float(conf_str)
            except Exception:
                conf = 0.0
            return text, conf
        return "", 0.0
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def grid_to_string(grid: List[List[str]]) -> str:
    return "".join(grid[r][c] for r in range(9) for c in range(9))


def find_conflicts(grid: List[List[str]]) -> List[Tuple[int, int]]:
    conflicts = set()

    def scan_unit(cells):
        seen = {}
        for r, c in cells:
            v = grid[r][c]
            if v == "0":
                continue
            if v in seen:
                conflicts.add((r, c))
                conflicts.add(seen[v])
            else:
                seen[v] = (r, c)

    for r in range(9):
        scan_unit([(r, c) for c in range(9)])
    for c in range(9):
        scan_unit([(r, c) for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            scan_unit([(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)])

    return sorted(list(conflicts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=164)
    parser.add_argument("--end", type=int, default=323)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--outdir", default="/Users/alai/Projects/Codex/Misudoku/review/xsudoku-164-323")
    parser.add_argument("--draft", default="/Users/alai/Projects/Codex/Misudoku/data/xsudoku-164-323-draft.json")
    parser.add_argument("--error-feedback", default="")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    cells_dir = outdir / "cells"
    boards_dir = outdir / "boards"
    outdir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)
    boards_dir.mkdir(parents=True, exist_ok=True)

    review_csv = outdir / "review.csv"
    summary_json = outdir / "summary.json"

    doc = fitz.open(PDF_PATH)

    pages_out = []
    review_rows = []

    pages_to_process: Iterable[int]
    error_pages: Set[int] = set()
    if args.error_feedback:
        error_pages = read_feedback_errors(Path(args.error_feedback))
        if not error_pages:
            print("No error pages found in feedback CSV.")
            return
        pages_to_process = sorted(error_pages)
    else:
        pages_to_process = range(args.start, args.end + 1)

    empty_threshold = 0.02
    conf_keep = 70.0
    conf_review = 80.0

    for page_num in pages_to_process:
        print(f"Processing page {page_num}...")
        img = render_page(doc, page_num, dpi=args.dpi)
        prefer_lines = page_num in error_pages
        board = None
        err = None
        if prefer_lines:
            try:
                board, _ = find_board_by_gridlines(img, size=900)
            except Exception as e:
                err = e
        if board is None:
            try:
                board, _ = find_board(img, size=900)
            except Exception as e:
                err = e
                try:
                    board, _ = find_board_by_gridlines(img, size=900)
                except Exception:
                    board = None
        if board is None:
            pages_out.append({
                "page": page_num,
                "status": "board_not_found",
                "error": str(err) if err else "board_not_found"
            })
            continue

        board_path = boards_dir / f"p{page_num}.png"
        cv2.imwrite(str(board_path), cv2.cvtColor(board, cv2.COLOR_RGB2BGR))

        gray = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)

        h, w = bw.shape
        cell_w = w // 9
        cell_h = h // 9
        margin = int(cell_w * 0.08)

        grid = [["0"] * 9 for _ in range(9)]
        confs = [[0.0] * 9 for _ in range(9)]
        review_flags = [[False] * 9 for _ in range(9)]

        for r in range(9):
            for c in range(9):
                x1 = c * cell_w + margin
                y1 = r * cell_h + margin
                x2 = (c + 1) * cell_w - margin
                y2 = (r + 1) * cell_h - margin
                cell_bw = bw[y1:y2, x1:x2]
                cell_bw = cv2.morphologyEx(cell_bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

                ink = cv2.countNonZero(cell_bw)
                ink_ratio = ink / float(cell_bw.size)

                if ink_ratio < empty_threshold:
                    grid[r][c] = "0"
                    confs[r][c] = 0.0
                else:
                    text, conf = cell_ocr(cell_bw)
                    if text in list("123456789"):
                        confs[r][c] = conf
                        if conf >= conf_keep:
                            grid[r][c] = text
                        else:
                            grid[r][c] = "0"
                        if conf < conf_review:
                            review_flags[r][c] = True
                    else:
                        grid[r][c] = "0"
                        confs[r][c] = conf
                        review_flags[r][c] = True

                # write review row if flagged
                if review_flags[r][c]:
                    cell_img = board[y1:y2, x1:x2]
                    cell_path = cells_dir / f"p{page_num}_r{r+1}c{c+1}.png"
                    cv2.imwrite(str(cell_path), cv2.cvtColor(cell_img, cv2.COLOR_RGB2BGR))
                    review_rows.append({
                        "page": page_num,
                        "row": r + 1,
                        "col": c + 1,
                        "ocr": grid[r][c] if grid[r][c] != "0" else "",
                        "conf": f"{confs[r][c]:.1f}",
                        "ink_ratio": f"{ink_ratio:.4f}",
                        "reason": "low_conf" if confs[r][c] < conf_review else "missing",
                        "image_path": str(cell_path),
                        "correct": ""
                    })

        conflicts = find_conflicts(grid)
        for (r, c) in conflicts:
            if grid[r][c] != "0":
                review_flags[r][c] = True
                # mark for review (conflict)
                cell_img = board[
                    r * cell_h + margin : (r + 1) * cell_h - margin,
                    c * cell_w + margin : (c + 1) * cell_w - margin,
                ]
                cell_path = cells_dir / f"p{page_num}_r{r+1}c{c+1}.png"
                if not cell_path.exists():
                    cv2.imwrite(str(cell_path), cv2.cvtColor(cell_img, cv2.COLOR_RGB2BGR))
                review_rows.append({
                    "page": page_num,
                    "row": r + 1,
                    "col": c + 1,
                    "ocr": grid[r][c],
                    "conf": f"{confs[r][c]:.1f}",
                    "ink_ratio": "",
                    "reason": "conflict",
                    "image_path": str(cell_path),
                    "correct": ""
                })
                grid[r][c] = "0"

        givens = grid_to_string(grid)
        givens_count = sum(1 for ch in givens if ch in "123456789")
        review_count = sum(1 for r in review_flags for v in r if v)

        pages_out.append({
            "page": page_num,
            "status": "ok",
            "givens": givens,
            "givens_count": givens_count,
            "review_count": review_count,
            "board_path": str(board_path)
        })

    if args.merge and review_csv.exists():
        existing_rows = []
        with review_csv.open(newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        existing_rows = [r for r in existing_rows if int(r.get("page", 0)) not in set(pages_to_process)]
        merged_rows = existing_rows + review_rows
    else:
        merged_rows = review_rows

    with review_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "page", "row", "col", "ocr", "conf", "ink_ratio", "reason", "image_path", "correct"
        ])
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    if args.merge and summary_json.exists():
        existing = json.loads(summary_json.read_text(encoding="utf-8"))
        by_page: Dict[int, Any] = {p["page"]: p for p in existing}
        for p in pages_out:
            by_page[p["page"]] = p
        summary_all = [by_page[k] for k in sorted(by_page.keys())]
    else:
        summary_all = pages_out
    summary_json.write_text(json.dumps(summary_all, indent=2, ensure_ascii=False), encoding="utf-8")

    draft_out = {
        "version": 1,
        "exportedAt": date.today().isoformat(),
        "source": "xsudoku-164-323",
        "puzzles": summary_all
    }
    Path(args.draft).write_text(json.dumps(draft_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Review CSV: {review_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Draft JSON: {args.draft}")


if __name__ == "__main__":
    main()
