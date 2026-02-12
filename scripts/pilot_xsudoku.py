#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import fitz
import cv2
import numpy as np
import pytesseract
from sklearn.neighbors import KNeighborsClassifier


PDF_PATH = "/Users/alai/Projects/Codex/xtreme-suduku/X-sudoku.pdf"


def render_page(doc: fitz.Document, page_num: int, dpi: int = 300) -> np.ndarray:
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape((pix.height, pix.width, pix.n))
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def find_board(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
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
        # fallback to minAreaRect of largest contour
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        best = np.int0(box)

    pts = best.reshape(4, 2).astype("float32")
    rect = order_points(pts)

    # target size
    size = 900
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped, rect


def preprocess_cell(cell_gray: np.ndarray) -> np.ndarray:
    # upscale to help OCR
    cell = cv2.resize(cell_gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    # local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    cell = clahe.apply(cell)
    cell = cv2.GaussianBlur(cell, (3, 3), 0)

    bw = cv2.adaptiveThreshold(
        cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )

    # remove thin grid lines inside the cell
    h, w = bw.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(2, w // 2), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, h // 2)))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    bw = cv2.subtract(bw, cv2.bitwise_or(h_lines, v_lines))

    # remove border artifacts more aggressively
    bw[:4, :] = 0
    bw[-4:, :] = 0
    bw[:, :4] = 0
    bw[:, -4:] = 0
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return bw


def select_digit_contour(cell_bw: np.ndarray):
    contours, _ = cv2.findContours(cell_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = cell_bw.shape[:2]
    cell_area = h * w
    cx = w / 2.0
    cy = h / 2.0
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cell_area * 0.006 or area > cell_area * 0.40:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / float(ch) if ch > 0 else 0
        if aspect < 0.2 or aspect > 1.6:
            continue
        extent = area / float(cw * ch) if cw * ch > 0 else 0
        if extent < 0.15:
            continue
        c_cx = x + cw / 2.0
        c_cy = y + ch / 2.0
        if abs(c_cx - cx) > w * 0.4 or abs(c_cy - cy) > h * 0.4:
            continue
        candidates.append((area, c))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def crop_digit(cell_bw: np.ndarray) -> Optional[np.ndarray]:
    contour = select_digit_contour(cell_bw)
    if contour is None:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    crop = cell_bw[y : y + h, x : x + w]
    if crop.size == 0:
        return None

    size = max(crop.shape[0], crop.shape[1])
    pad = 6
    canvas = np.zeros((size + pad * 2, size + pad * 2), dtype=np.uint8)
    y0 = (canvas.shape[0] - crop.shape[0]) // 2
    x0 = (canvas.shape[1] - crop.shape[1]) // 2
    canvas[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
    return canvas


def ocr_digit(digit_canvas: np.ndarray) -> Tuple[str, float]:
    # tesseract expects black text on white background
    ocr_img = 255 - digit_canvas
    ocr_img = cv2.resize(ocr_img, (48, 48), interpolation=cv2.INTER_CUBIC)

    configs = [
        "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789",
        "--psm 8 --oem 3 -c tessedit_char_whitelist=123456789",
    ]
    best_text = ""
    best_conf = -1.0
    for cfg in configs:
        data = pytesseract.image_to_data(ocr_img, config=cfg, output_type=pytesseract.Output.DICT)
        for t, cf in zip(data.get("text", []), data.get("conf", [])):
            if not t or not t.strip():
                continue
            text = t.strip()
            try:
                conf = float(cf)
            except Exception:
                conf = -1.0
            if conf > best_conf:
                best_conf = conf
                best_text = text
            break
    return best_text, best_conf


def digit_features(digit_canvas: np.ndarray) -> np.ndarray:
    # normalize to 20x20 binary and flatten
    resized = cv2.resize(digit_canvas, (20, 20), interpolation=cv2.INTER_AREA)
    feat = resized.astype(np.float32) / 255.0
    return feat.flatten()


def extract_digits(board: np.ndarray) -> Tuple[List[List[str]], List[List[float]], List[Tuple[int, int, int, float]]]:
    # Simple v2-style extraction (best recall in pilot so far).
    gray = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)

    h, w = bw.shape
    cell_w = w // 9
    cell_h = h // 9
    margin = int(cell_w * 0.08)

    grid = [["0"] * 9 for _ in range(9)]
    confs = [[0.0] * 9 for _ in range(9)]
    low_conf_cells = []

    for r in range(9):
        for c in range(9):
            x1 = c * cell_w + margin
            y1 = r * cell_h + margin
            x2 = (c + 1) * cell_w - margin
            y2 = (r + 1) * cell_h - margin
            cell = bw[y1:y2, x1:x2]

            cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

            nonzero = cv2.countNonZero(cell)
            if nonzero < (cell.size * 0.02):
                grid[r][c] = "0"
                confs[r][c] = 0.0
                continue

            config = "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"
            data = pytesseract.image_to_data(cell, config=config, output_type=pytesseract.Output.DICT)
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

            if text in list("123456789"):
                grid[r][c] = text
                confs[r][c] = conf
                if conf < 70:
                    low_conf_cells.append((r, c, int(text), conf))
            else:
                grid[r][c] = "0"
                confs[r][c] = conf
                low_conf_cells.append((r, c, 0, conf))

    return grid, confs, low_conf_cells


def grid_to_string(grid: List[List[str]]) -> str:
    return "".join(grid[r][c] for r in range(9) for c in range(9))


def is_valid_grid(grid: List[List[str]]) -> bool:
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


def resolve_conflicts(grid: List[List[str]], confs: List[List[float]], conf_threshold: float = 70.0) -> List[Tuple[int, int]]:
    # Remove low-confidence digits that cause conflicts in rows/cols/boxes.
    removed = []

    def handle_unit(cells):
        seen = {}
        for (r, c) in cells:
            v = grid[r][c]
            if v == "0":
                continue
            if v in seen:
                # resolve conflict by removing lower-confidence cell if possible
                r0, c0 = seen[v]
                c0_conf = confs[r0][c0]
                c1_conf = confs[r][c]
                if c0_conf < c1_conf and c0_conf < conf_threshold:
                    grid[r0][c0] = "0"
                    removed.append((r0, c0))
                    seen[v] = (r, c)
                elif c1_conf < conf_threshold:
                    grid[r][c] = "0"
                    removed.append((r, c))
                # else keep both; unresolved
            else:
                seen[v] = (r, c)

    for r in range(9):
        handle_unit([(r, c) for c in range(9)])
    for c in range(9):
        handle_unit([(r, c) for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            cells = [(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)]
            handle_unit(cells)

    return removed


def solve_grid(grid: List[List[str]]) -> Tuple[bool, List[List[str]]]:
    # backtracking solver
    def find_empty():
        for r in range(9):
            for c in range(9):
                if grid[r][c] == "0":
                    return r, c
        return None

    def is_valid(r, c, val):
        # row
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


def write_text_grid(grid: List[List[str]]) -> str:
    lines = []
    for r in range(9):
        lines.append(" ".join(grid[r]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", nargs="+", type=int, default=[164, 250, 323])
    parser.add_argument("--outdir", default="/tmp/xsudoku_review")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cells").mkdir(parents=True, exist_ok=True)

    doc = fitz.open(PDF_PATH)

    summary = []

    for p in args.pages:
        print(f"Processing page {p}...")
        img = render_page(doc, p, dpi=300)

        try:
            board, rect = find_board(img)
        except Exception as e:
            summary.append({"page": p, "status": "board_not_found", "error": str(e)})
            continue

        board_path = outdir / f"p{p}_board.png"
        cv2.imwrite(str(board_path), cv2.cvtColor(board, cv2.COLOR_RGB2BGR))

        grid, confs, low_cells = extract_digits(board)
        # attempt conflict cleanup using low-confidence cells
        removed = resolve_conflicts(grid, confs, conf_threshold=70.0)
        givens = grid_to_string(grid)

        # write text grid
        txt_path = outdir / f"p{p}_grid.txt"
        txt_path.write_text(write_text_grid(grid) + "\n", encoding="utf-8")

        valid = is_valid_grid(grid)
        solvable = False
        solution = None
        if valid:
            grid_copy = [row[:] for row in grid]
            solvable, solved = solve_grid(grid_copy)
            if solvable:
                solution = grid_to_string(solved)

        # save low confidence cells
        low_count = 0
        for (r, c, val, conf) in low_cells:
            # re-crop from board for saving
            h, w = board.shape[:2]
            cell_w = w // 9
            cell_h = h // 9
            margin = int(cell_w * 0.06)
            x1 = c * cell_w + margin
            y1 = r * cell_h + margin
            x2 = (c + 1) * cell_w - margin
            y2 = (r + 1) * cell_h - margin
            cell = board[y1:y2, x1:x2]
            cell_path = outdir / "cells" / f"p{p}_r{r+1}c{c+1}_v{val}_conf{int(conf)}.png"
            cv2.imwrite(str(cell_path), cv2.cvtColor(cell, cv2.COLOR_RGB2BGR))
            low_count += 1

        summary.append({
            "page": p,
            "status": "ok",
            "givens": givens,
            "valid": valid,
            "solvable": solvable,
            "low_conf_cells": low_count,
            "removed_conflicts": len(removed),
            "grid_path": str(txt_path),
            "board_path": str(board_path)
        })

    # write summary
    import json
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
