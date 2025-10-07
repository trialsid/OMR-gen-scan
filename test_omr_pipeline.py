#!/usr/bin/env python3
"""End-to-end test script for OMR sheet generation and processing."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from omr_sheet_generator import create_omr_sheet_pdf
from omr_processor import OMRProcessor
import omr_processor2


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
SQUARE_SIZE = 10
MARGIN = 50
NUM_SQUARES_WIDTH = 4
BUBBLE_RADIUS = 8
BUBBLE_SPACING_Y = 25
BUBBLE_SPACING_X = BUBBLE_RADIUS * 3
NUM_ROLL_ROWS = 10
ROLL_LABEL_ROW = 0
ROLL_START_ROW = 1
QUESTIONS_GAP_ROW = 11
QUESTIONS_LABEL_ROW = 12
QUESTIONS_START_ROW = 13


@dataclass
class SheetData:
    image_path: Path
    pdf_path: Path
    expected_roll_digits: List[int]
    expected_answers: Dict[int, str]


def pdf_point_to_image(x: float, y: float, scale: int) -> Tuple[int, int]:
    """Convert PDF coordinates (origin bottom-left) to image coords (origin top-left)."""

    return int(round(x * scale)), int(round((PAGE_HEIGHT - y) * scale))


def draw_rectangle(image: np.ndarray, bottom_left: Tuple[float, float], size: float, scale: int, color: Tuple[int, int, int], fill: bool) -> None:
    x, y = bottom_left
    top_left_img = pdf_point_to_image(x, y + size, scale)
    bottom_right_img = pdf_point_to_image(x + size, y, scale)
    thickness = -1 if fill else 2
    cv2.rectangle(image, top_left_img, bottom_right_img, color, thickness)


def generate_random_sheet(artifacts_dir: Path, scale: int = 4, seed: int | None = 1234) -> SheetData:
    """Use the sheet generator layout to create a randomly filled synthetic sheet image."""

    rng = random.Random(seed)

    pdf_path = artifacts_dir / "synthetic_sheet.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    create_omr_sheet_pdf(str(pdf_path))

    width_px = int(PAGE_WIDTH * scale)
    height_px = int(PAGE_HEIGHT * scale)
    image = np.full((height_px, width_px, 3), 255, dtype=np.uint8)

    available_width = PAGE_WIDTH - (2 * MARGIN)
    spacing_x = available_width / (NUM_SQUARES_WIDTH - 1) if NUM_SQUARES_WIDTH > 1 else 0
    spacing_y = spacing_x

    start_x = MARGIN
    start_y = MARGIN
    end_x = PAGE_WIDTH - MARGIN
    end_y = PAGE_HEIGHT - MARGIN

    # Draw positioning squares (anchors and grid markers)
    y = start_y
    row_count = 0
    last_row = int((end_y - start_y) / spacing_y)
    while y <= end_y:
        x = start_x
        square_count = 0
        while square_count < NUM_SQUARES_WIDTH:
            is_corner = (
                (row_count == 0 or row_count == last_row)
                and (square_count == 0 or square_count == NUM_SQUARES_WIDTH - 1)
            )
            current_size = SQUARE_SIZE * 2 if is_corner else SQUARE_SIZE
            half_size = current_size / 2
            bottom_left = (x - half_size, y - half_size)
            draw_rectangle(image, bottom_left, current_size, scale, (0, 0, 0), fill=True)

            if is_corner:
                inner_margin = current_size * 0.3
                inner_size = current_size - 2 * inner_margin
                if inner_size > 0:
                    inner_bottom_left = (
                        bottom_left[0] + inner_margin,
                        bottom_left[1] + inner_margin,
                    )
                    draw_rectangle(image, inner_bottom_left, inner_size, scale, (255, 255, 255), fill=True)

            x += spacing_x
            square_count += 1
        y += spacing_y
        row_count += 1

    total_rows = int((PAGE_HEIGHT - 2 * MARGIN) / spacing_y) + 1
    bubbles_start_y = start_y + (total_rows - 1) * spacing_y

    gap_center_x = start_x + (0 + 0.5) * spacing_x
    total_width = 3 * BUBBLE_SPACING_X
    gap_start_x = gap_center_x - total_width / 2

    # Draw headers
    roll_label_y = bubbles_start_y - (ROLL_LABEL_ROW * BUBBLE_SPACING_Y)
    roll_label_pos = pdf_point_to_image(gap_start_x, roll_label_y - 3, scale)
    cv2.putText(image, "Roll No.", roll_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale / 4, (0, 0, 0), 2)

    questions_label_y = bubbles_start_y - (QUESTIONS_LABEL_ROW * BUBBLE_SPACING_Y)
    questions_label_pos = pdf_point_to_image(gap_start_x, questions_label_y - 3, scale)
    cv2.putText(image, "Questions", questions_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale / 4, (0, 0, 0), 2)

    # Draw digit labels
    label_x = gap_start_x - 25
    for digit in range(NUM_ROLL_ROWS):
        bubble_y = bubbles_start_y - ((ROLL_START_ROW + digit) * BUBBLE_SPACING_Y)
        text_pos = pdf_point_to_image(label_x, bubble_y - 3, scale)
        cv2.putText(image, str(digit), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale / 4, (0, 0, 0), 1)

    roll_digit_positions: Dict[Tuple[int, int], Tuple[int, int]] = {}
    roll_digits: List[int] = []

    for digit_col in range(3):
        bubble_x = gap_start_x + digit_col * BUBBLE_SPACING_X
        for digit in range(NUM_ROLL_ROWS):
            bubble_y = bubbles_start_y - ((ROLL_START_ROW + digit) * BUBBLE_SPACING_Y)
            center = pdf_point_to_image(bubble_x, bubble_y, scale)
            radius = int(round(BUBBLE_RADIUS * scale))
            cv2.circle(image, center, radius, (0, 0, 0), 2)
            roll_digit_positions[(digit_col, digit)] = center

    # Draw question bubbles and labels while recording positions
    question_positions: Dict[int, Dict[str, Tuple[int, int]]] = {}
    question_number = 1

    for col_gap in range(NUM_SQUARES_WIDTH - 1):
        gap_center_x = start_x + (col_gap + 0.5) * spacing_x
        gap_start_x = gap_center_x - total_width / 2

        if col_gap == 0:
            start_row = QUESTIONS_START_ROW
        else:
            start_row = 0

        bubble_y = bubbles_start_y - (start_row * BUBBLE_SPACING_Y)
        row_index = start_row

        while bubble_y >= start_y:
            # Question number label
            question_label_x = gap_start_x - 25
            text_pos = pdf_point_to_image(question_label_x, bubble_y - 3, scale)
            cv2.putText(
                image,
                str(question_number),
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 * scale / 4,
                (0, 0, 0),
                1,
            )

            question_positions[question_number] = {}
            for idx, option in enumerate("ABCD"):
                bubble_x = gap_start_x + idx * BUBBLE_SPACING_X
                center = pdf_point_to_image(bubble_x, bubble_y, scale)
                radius = int(round(BUBBLE_RADIUS * scale))
                cv2.circle(image, center, radius, (0, 0, 0), 2)
                question_positions[question_number][option] = center

            bubble_y -= BUBBLE_SPACING_Y
            row_index += 1
            question_number += 1

            if bubble_y < start_y:
                break

    # Randomly select filled bubbles
    for digit_col in range(3):
        chosen_digit = rng.randint(0, 9)
        roll_digits.append(chosen_digit)
        center = roll_digit_positions[(digit_col, chosen_digit)]
        radius = int(round(BUBBLE_RADIUS * scale * 0.7))
        cv2.circle(image, center, radius, (30, 30, 30), -1)

    expected_answers: Dict[int, str] = {}
    for question, options in question_positions.items():
        chosen_option = rng.choice(list(options.keys()))
        expected_answers[question] = chosen_option
        center = options[chosen_option]
        radius = int(round(BUBBLE_RADIUS * scale * 0.7))
        cv2.circle(image, center, radius, (40, 40, 40), -1)

    image_path = artifacts_dir / "synthetic_sheet.png"
    cv2.imwrite(str(image_path), image)

    data_path = artifacts_dir / "expected_answers.json"
    payload = {
        "roll_digits": roll_digits,
        "answers": expected_answers,
    }
    with data_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return SheetData(
        image_path=image_path,
        pdf_path=pdf_path,
        expected_roll_digits=roll_digits,
        expected_answers=expected_answers,
    )


def extract_old_processor_results(processor: OMRProcessor, result: Dict[str, object]) -> Tuple[str, Dict[int, str]]:
    corrected = result["corrected"]
    bubble_data = result["bubble_data"]

    if corrected is None or bubble_data is None:
        raise RuntimeError("Old processor did not return corrected image or bubble data")

    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if corrected.ndim == 3 else corrected

    roll_bubbles = bubble_data["roll_numbers"]
    question_bubbles_by_column = bubble_data["questions"]

    roll_digits = ["?", "?", "?"]
    if roll_bubbles:
        roll_x_coords = [b["center"][0] for b in roll_bubbles]
        min_x, max_x = min(roll_x_coords), max(roll_x_coords)
        digit_columns = [
            min_x + (max_x - min_x) * 0.2,
            min_x + (max_x - min_x) * 0.5,
            min_x + (max_x - min_x) * 0.8,
        ]
        digit_tolerance = 15

        grouped_digits: List[List[Dict[str, object]]] = [[], [], []]
        for bubble in roll_bubbles:
            bx = bubble["center"][0]
            for idx, column_x in enumerate(digit_columns):
                if abs(bx - column_x) < digit_tolerance:
                    grouped_digits[idx].append(bubble)
                    break

        for column_idx, bubbles in enumerate(grouped_digits):
            bubbles.sort(key=lambda b: b["center"][1])
            best_digit = None
            lowest_intensity = float("inf")
            for row_idx, bubble in enumerate(bubbles):
                x, y = bubble["center"]
                region_size = 8
                x1, y1 = max(0, x - region_size), max(0, y - region_size)
                x2, y2 = min(gray.shape[1], x + region_size), min(gray.shape[0], y + region_size)
                roi = gray[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                mean_intensity = float(np.mean(roi))
                if mean_intensity < lowest_intensity:
                    lowest_intensity = mean_intensity
                    best_digit = row_idx
            if best_digit is not None and best_digit < 10:
                roll_digits[column_idx] = str(best_digit)

    fill_threshold = processor.calculate_fill_threshold(corrected)
    question_results: Dict[int, str] = {}

    question_counts = [len(col) // 4 for col in question_bubbles_by_column]

    for col_idx, col_bubbles in enumerate(question_bubbles_by_column):
        if not col_bubbles:
            continue
        bubbles_by_row: Dict[int, List[Dict[str, object]]] = {}
        question_tolerance = 15
        for bubble in col_bubbles:
            center_y = bubble["center"][1]
            assigned_row = None
            for existing_y in bubbles_by_row.keys():
                if abs(center_y - existing_y) < question_tolerance:
                    assigned_row = existing_y
                    break
            if assigned_row is None:
                assigned_row = center_y
                bubbles_by_row[assigned_row] = []
            bubbles_by_row[assigned_row].append(bubble)

        sorted_rows = sorted(bubbles_by_row.items(), key=lambda kv: kv[0])
        for row_idx, (_, row_bubbles) in enumerate(sorted_rows):
            row_bubbles.sort(key=lambda b: b["center"][0])
            question_number = sum(question_counts[:col_idx]) + row_idx + 1
            best_option = None
            best_score = float("inf")
            options = ["A", "B", "C", "D"]
            for option_idx, bubble in enumerate(row_bubbles[:4]):
                x, y = bubble["center"]
                region_size = 8
                x1, y1 = max(0, x - region_size), max(0, y - region_size)
                x2, y2 = min(gray.shape[1], x + region_size), min(gray.shape[0], y + region_size)
                roi = gray[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                mean_intensity = float(np.mean(roi))
                if mean_intensity < fill_threshold and mean_intensity < best_score:
                    best_score = mean_intensity
                    best_option = options[option_idx]
            if best_option:
                question_results[question_number] = best_option

    return "".join(roll_digits), question_results


def run_old_processor(image_path: Path, artifacts_dir: Path) -> Tuple[str, Dict[int, str]]:
    old_results_dir = artifacts_dir / "old_processor"
    old_results_dir.mkdir(parents=True, exist_ok=True)
    processor = OMRProcessor(scans_dir=str(image_path.parent), results_dir=str(old_results_dir))
    result = processor.process_image_stepwise(str(image_path))
    return extract_old_processor_results(processor, result)


def run_new_processor(image_path: Path, artifacts_dir: Path) -> Tuple[str, Dict[int, str]]:
    debug_dir = artifacts_dir / "new_processor"
    debug_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        image=image_path,
        scale=4,
        fill_threshold=0.45,
        roll_threshold=0.35,
        output_json=None,
        debug_dir=debug_dir,
        visualize=False,
    )
    summary = omr_processor2.process_sheet(args)
    roll_string = summary["roll_number"]["string"]
    question_results = {}
    for question, data in summary["questions"].items():
        selected = data["selected"]
        if isinstance(selected, list):
            if selected:
                question_results[question] = selected[0]
        elif isinstance(selected, str):
            question_results[question] = selected
    return roll_string, question_results


def compare_results(expected: SheetData, old_result: Tuple[str, Dict[int, str]], new_result: Tuple[str, Dict[int, str]]) -> None:
    expected_roll = "".join(str(d) for d in expected.expected_roll_digits)
    old_roll, old_answers = old_result
    new_roll, new_answers = new_result

    print("Expected roll number:", expected_roll)
    print("Old processor roll number:", old_roll)
    print("New processor roll number:", new_roll)

    total_questions = len(expected.expected_answers)
    old_matches = sum(1 for q, ans in expected.expected_answers.items() if old_answers.get(q) == ans)
    new_matches = sum(1 for q, ans in expected.expected_answers.items() if new_answers.get(q) == ans)

    print(f"Questions total: {total_questions}")
    print(f"Old processor correct answers: {old_matches}")
    print(f"New processor correct answers: {new_matches}")

    mismatches = []
    for question in sorted(expected.expected_answers.keys()):
        exp = expected.expected_answers[question]
        old_ans = old_answers.get(question)
        new_ans = new_answers.get(question)
        if old_ans != exp or new_ans != exp:
            mismatches.append({
                "question": question,
                "expected": exp,
                "old": old_ans,
                "new": new_ans,
            })

    comparison_path = expected.image_path.parent / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump({
            "expected_roll": expected_roll,
            "old_roll": old_roll,
            "new_roll": new_roll,
            "mismatches": mismatches,
        }, f, indent=2)

    print(f"Comparison saved to {comparison_path}")


def main() -> None:
    artifacts_dir = Path("test_artifacts")
    sheet_data = generate_random_sheet(artifacts_dir)
    old_result = run_old_processor(sheet_data.image_path, artifacts_dir)
    new_result = run_new_processor(sheet_data.image_path, artifacts_dir)
    compare_results(sheet_data, old_result, new_result)


if __name__ == "__main__":
    main()
