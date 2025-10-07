#!/usr/bin/env python3
"""OMR sheet processor for images captured from generated answer sheets.

This script rectifies a photographed OMR sheet, evaluates bubble fill levels,
infers the roll number and marked answers, and optionally saves debug images
and JSON summaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class Bubble:
    """Represents a single bubble on the OMR sheet."""

    kind: str  # "roll" or "question"
    center: Tuple[int, int]
    radius: int
    metadata: Dict[str, int | str] = field(default_factory=dict)
    score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a photographed OMR sheet")
    parser.add_argument("image", type=Path, help="Path to the captured OMR sheet image")
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Scaling factor for the rectified sheet relative to the PDF units",
    )
    parser.add_argument(
        "--fill-threshold",
        type=float,
        default=0.45,
        help="Fill ratio threshold for considering a bubble marked",
    )
    parser.add_argument(
        "--roll-threshold",
        type=float,
        default=0.35,
        help="Fill ratio threshold for selecting roll number digits",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the extracted data as JSON",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory to store intermediate debug images",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save an annotated image with detected selections",
    )
    return parser.parse_args()


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four corner points (top-left, top-right, bottom-right, bottom-left)."""

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def find_anchor_squares(image: np.ndarray) -> np.ndarray:
    """Locate the four corner anchor squares on the OMR sheet."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        5,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.002 * image.shape[0] * image.shape[1]
    squares = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        # Ensure the contour is roughly square
        pts = approx.reshape(4, 2)
        _, _, w, h = cv2.boundingRect(pts.astype(np.int32))
        ratio = w / float(h)
        if 0.6 < ratio < 1.4:
            squares.append((area, pts.astype("float32")))

    if len(squares) < 4:
        raise RuntimeError("Failed to locate the four anchor squares on the sheet")

    # Select the four largest squares (anchors)
    squares.sort(key=lambda x: x[0], reverse=True)
    anchors = np.array([sq[1] for sq in squares[:4]])
    return anchors


def compute_warp(
    image: np.ndarray, anchors: np.ndarray, output_size: Tuple[int, int]
) -> np.ndarray:
    """Warp the perspective using the detected anchors."""

    # Average the points from the four contours to mitigate contour ordering issues
    averaged_points = np.array([anchor.mean(axis=0) for anchor in anchors], dtype="float32")
    rect = order_points(averaged_points)

    (max_width, max_height) = output_size
    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    transform = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, transform, (max_width, max_height))
    return warped


def build_layout(scale: int = 4) -> Tuple[List[Bubble], Tuple[int, int]]:
    """Replicate the generator geometry and return a list of bubbles."""

    # Page and drawing parameters (in PDF points)
    page_width, page_height = 612, 792  # letter size in points
    square_size = 10
    margin = 50
    num_squares_width = 4

    bubble_radius = 8
    bubble_spacing_y = 25
    bubble_spacing_x = bubble_radius * 3

    available_width = page_width - (2 * margin)
    spacing_x = available_width / (num_squares_width - 1) if num_squares_width > 1 else 0
    spacing_y = spacing_x

    start_x = margin
    start_y = margin
    end_y = page_height - margin

    total_rows = int((page_height - 2 * margin) / spacing_y) + 1
    bubbles_start_y = start_y + (total_rows - 1) * spacing_y

    bubbles: List[Bubble] = []

    # Roll number configuration
    num_roll_rows = 10
    roll_label_row = 0
    roll_start_row = 1
    questions_gap_row = 11
    questions_label_row = 12
    questions_start_row = 13

    gap_center_x = start_x + (0 + 0.5) * spacing_x
    total_width = 3 * bubble_spacing_x
    gap_start_x = gap_center_x - total_width / 2

    # Roll number bubbles (3 digit columns)
    for digit_col in range(3):
        bubble_x = gap_start_x + digit_col * bubble_spacing_x
        for digit in range(num_roll_rows):
            bubble_y = bubbles_start_y - ((roll_start_row + digit) * bubble_spacing_y)
            bubbles.append(
                Bubble(
                    kind="roll",
                    center=(bubble_x, bubble_y),
                    radius=bubble_radius,
                    metadata={"column": digit_col, "digit": digit},
                )
            )

    # Question bubbles across all columns
    question_number = 1
    for col_gap in range(num_squares_width - 1):
        gap_center_x = start_x + (col_gap + 0.5) * spacing_x
        gap_start_x = gap_center_x - total_width / 2
        question_label_x = gap_start_x - 25  # unused here but kept for completeness

        if col_gap == 0:
            start_row = questions_start_row
        else:
            start_row = 0

        bubble_y = bubbles_start_y - (start_row * bubble_spacing_y)
        row_index = start_row

        while bubble_y >= start_y:
            for bubble_idx, option in enumerate("ABCD"):
                bubble_x = gap_start_x + bubble_idx * bubble_spacing_x
                bubbles.append(
                    Bubble(
                        kind="question",
                        center=(bubble_x, bubble_y),
                        radius=bubble_radius,
                        metadata={
                            "question": question_number,
                            "option": option,
                            "column": col_gap,
                            "row": row_index,
                        },
                    )
                )
            bubble_y -= bubble_spacing_y
            row_index += 1
            question_number += 1

    # Convert PDF coordinates to image coordinates (top-left origin) and scale
    scaled_bubbles: List[Bubble] = []
    for bubble in bubbles:
        x_pdf, y_pdf = bubble.center
        x_img = int(round(x_pdf * scale))
        y_img = int(round((page_height - y_pdf) * scale))
        radius_img = int(round(bubble.radius * scale))
        metadata = dict(bubble.metadata)
        scaled_bubbles.append(
            Bubble(
                kind=bubble.kind,
                center=(x_img, y_img),
                radius=radius_img,
                metadata=metadata,
            )
        )

    output_size = (int(page_width * scale), int(page_height * scale))
    return scaled_bubbles, output_size


def measure_bubble_scores(gray: np.ndarray, bubbles: List[Bubble]) -> None:
    """Compute fill scores for each bubble on the rectified sheet."""

    for bubble in bubbles:
        x, y = bubble.center
        radius = max(3, int(bubble.radius * 0.65))

        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(gray.shape[1], x + radius + 1)
        y1 = min(gray.shape[0], y + radius + 1)
        if x1 <= x0 or y1 <= y0:
            bubble.score = 0.0
            continue

        roi = gray[y0:y1, x0:x1]
        mask = np.zeros_like(roi, dtype=np.uint8)
        cv2.circle(mask, (x - x0, y - y0), radius, 1, -1)
        mask_area = int(mask.sum())
        if mask_area == 0:
            bubble.score = 0.0
            continue

        inverted = 255 - roi
        score = float((inverted * mask).sum()) / (255.0 * mask_area)
        bubble.score = score


def evaluate_roll_number(
    bubbles: List[Bubble], roll_threshold: float
) -> Dict[str, object]:
    """Determine the roll number digits based on the bubble scores."""

    columns: Dict[int, List[Bubble]] = {0: [], 1: [], 2: []}
    for bubble in bubbles:
        if bubble.kind == "roll":
            columns[bubble.metadata["column"]].append(bubble)

    roll_digits: List[str] = []
    column_scores: Dict[int, Dict[str, float]] = {}

    for column_idx, items in columns.items():
        items.sort(key=lambda b: b.metadata["digit"])  # ensure 0-9 order
        scores = {str(b.metadata["digit"]): b.score for b in items}
        column_scores[column_idx] = scores
        if not items:
            roll_digits.append("?")
            continue

        best = max(items, key=lambda b: b.score)
        if best.score >= roll_threshold:
            roll_digits.append(str(best.metadata["digit"]))
        else:
            roll_digits.append("?")

    return {
        "digits": roll_digits,
        "string": "".join(roll_digits),
        "columns": column_scores,
    }


def evaluate_questions(
    bubbles: List[Bubble], fill_threshold: float
) -> Dict[int, Dict[str, object]]:
    """Evaluate answer selections for each question."""

    grouped: Dict[int, List[Bubble]] = {}
    for bubble in bubbles:
        if bubble.kind != "question":
            continue
        q = int(bubble.metadata["question"])
        grouped.setdefault(q, []).append(bubble)

    results: Dict[int, Dict[str, object]] = {}

    for question, items in sorted(grouped.items()):
        items.sort(key=lambda b: b.metadata["option"])
        option_scores = {b.metadata["option"]: b.score for b in items}
        filled = [b for b in items if b.score >= fill_threshold]

        if not filled:
            status = "blank"
            selected = None
        elif len(filled) == 1:
            status = "single"
            selected = filled[0].metadata["option"]
        else:
            status = "multiple"
            selected = [b.metadata["option"] for b in filled]

        results[question] = {
            "question": question,
            "status": status,
            "selected": selected,
            "scores": option_scores,
        }

    return results


def visualize_results(
    warped: np.ndarray,
    bubbles: List[Bubble],
    roll_result: Dict[str, object],
    question_results: Dict[int, Dict[str, object]],
    debug_dir: Path,
) -> None:
    """Save an annotated overlay showing detected selections."""

    overlay = warped.copy()
    for bubble in bubbles:
        center = tuple(map(int, bubble.center))
        outer_radius = int(max(4, bubble.radius))
        inner_radius = int(max(2, bubble.radius * 0.6))

        color = (255, 165, 0)  # default orange for reference
        thickness = 2

        if bubble.kind == "question":
            q = int(bubble.metadata["question"])
            option = bubble.metadata["option"]
            result = question_results.get(q)
            if result:
                if result["status"] == "single" and result["selected"] == option:
                    color = (0, 255, 0)
                    thickness = -1
                elif result["status"] == "multiple" and option in result.get("selected", []):
                    color = (0, 255, 255)
                    thickness = -1
                elif result["status"] == "blank":
                    color = (0, 0, 255)
                    thickness = 2
        else:
            column = bubble.metadata.get("column")
            digit = str(bubble.metadata.get("digit"))
            selected_digit = roll_result["digits"][column]
            if selected_digit == digit:
                color = (255, 0, 255)
                thickness = -1

        cv2.circle(overlay, center, outer_radius, color, 2)
        if thickness < 0:
            cv2.circle(overlay, center, inner_radius, color, thickness)

    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "annotated_results.png"), overlay)


def save_debug_images(
    debug_dir: Path,
    original: np.ndarray,
    warped: np.ndarray,
    warped_gray: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "original.jpg"), original)
    cv2.imwrite(str(debug_dir / "warped.jpg"), warped)
    cv2.imwrite(str(debug_dir / "warped_gray.jpg"), warped_gray)


def process_sheet(args: argparse.Namespace) -> Dict[str, object]:
    bubbles_layout, output_size = build_layout(scale=args.scale)

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {args.image}")

    anchors = find_anchor_squares(image)
    warped = compute_warp(image, anchors, output_size)

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    measure_bubble_scores(blurred, bubbles_layout)

    roll_result = evaluate_roll_number(bubbles_layout, args.roll_threshold)
    question_results = evaluate_questions(bubbles_layout, args.fill_threshold)

    if args.debug_dir:
        save_debug_images(args.debug_dir, image, warped, warped_gray)
        if args.visualize:
            visualize_results(warped, bubbles_layout, roll_result, question_results, args.debug_dir)

    # Prepare a lightweight summary for JSON export
    summary = {
        "image": str(args.image),
        "roll_number": roll_result,
        "questions": question_results,
        "parameters": {
            "scale": args.scale,
            "fill_threshold": args.fill_threshold,
            "roll_threshold": args.roll_threshold,
        },
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    summary = process_sheet(args)

    print("Roll Number:", summary["roll_number"]["string"])
    print("Digits:", summary["roll_number"]["digits"])
    print("Questions evaluated:", len(summary["questions"]))

    for question, data in list(summary["questions"].items())[:10]:
        print(
            f"Q{question:02d}: status={data['status']}, selected={data['selected']}, scores={data['scores']}"
        )

    if len(summary["questions"]) > 10:
        print("... (truncated)")


if __name__ == "__main__":
    main()
