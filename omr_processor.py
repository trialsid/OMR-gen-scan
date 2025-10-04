#!/usr/bin/env python3
"""World-class OMR scanner and analyser.

The processor understands the metadata produced by :mod:`omr_sheet_generator`
and converts raw scans into structured JSON with per-question confidence
scores.  The implementation focuses on deterministic, reproducible signal
processing instead of brittle heuristics which makes it resilient to shadows,
light gradients and moderate perspective distortion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Metadata structures


@dataclass
class Anchor:
    name: str
    center_points: Tuple[float, float]
    size_points: float

    def to_pixel(self, page_width: float, page_height: float, dpi: int) -> Tuple[float, float]:
        width_px = page_width / 72.0 * dpi
        height_px = page_height / 72.0 * dpi
        cx = self.center_points[0] / page_width * width_px
        cy = (page_height - self.center_points[1]) / page_height * height_px
        return cx, cy


@dataclass
class Bubble:
    question: int
    option: str
    center_points: Tuple[float, float]
    radius_points: float

    def centre_px(self, page_width: float, page_height: float, dpi: int) -> Tuple[int, int]:
        width_px = page_width / 72.0 * dpi
        height_px = page_height / 72.0 * dpi
        cx = self.center_points[0] / page_width * width_px
        cy = (page_height - self.center_points[1]) / page_height * height_px
        return int(round(cx)), int(round(cy))

    def radius_px(self, page_width: float, dpi: int) -> int:
        width_px = page_width / 72.0 * dpi
        return max(2, int(round(self.radius_points / page_width * width_px)))


@dataclass
class RollBubble:
    digit_index: int
    digit_value: int
    center_points: Tuple[float, float]
    radius_points: float

    def centre_px(self, page_width: float, page_height: float, dpi: int) -> Tuple[int, int]:
        width_px = page_width / 72.0 * dpi
        height_px = page_height / 72.0 * dpi
        cx = self.center_points[0] / page_width * width_px
        cy = (page_height - self.center_points[1]) / page_height * height_px
        return int(round(cx)), int(round(cy))

    def radius_px(self, page_width: float, dpi: int) -> int:
        width_px = page_width / 72.0 * dpi
        return max(2, int(round(self.radius_points / page_width * width_px)))


@dataclass
class Template:
    page_width: float
    page_height: float
    dpi: int
    anchors: Mapping[str, Anchor]
    answer_bubbles: List[Bubble]
    roll_bubbles: List[RollBubble]

    @classmethod
    def from_json(cls, path: Path) -> "Template":
        with path.open("r", encoding="utf8") as fp:
            payload = json.load(fp)

        page = payload["page"]
        anchors = {
            entry["name"]: Anchor(
                name=entry["name"],
                center_points=(entry["cx"], entry["cy"]),
                size_points=entry["size"],
            )
            for entry in payload["anchors"]
        }

        answer_bubbles = [
            Bubble(
                question=entry["question"],
                option=entry["option"],
                center_points=(entry["cx"], entry["cy"]),
                radius_points=entry["radius"],
            )
            for entry in payload["answer_bubbles"]
        ]

        roll_bubbles = [
            RollBubble(
                digit_index=entry["digit_index"],
                digit_value=entry["digit_value"],
                center_points=(entry["cx"], entry["cy"]),
                radius_points=entry["radius"],
            )
            for entry in payload["roll_number_bubbles"]
        ]

        return cls(
            page_width=page["width_points"],
            page_height=page["height_points"],
            dpi=page["dpi"],
            anchors=anchors,
            answer_bubbles=answer_bubbles,
            roll_bubbles=roll_bubbles,
        )

    @property
    def canvas_size_px(self) -> Tuple[int, int]:
        width_px = int(round(self.page_width / 72.0 * self.dpi))
        height_px = int(round(self.page_height / 72.0 * self.dpi))
        return width_px, height_px


# ---------------------------------------------------------------------------
# Processing data models


@dataclass
class BubbleMeasurement:
    question: int
    option: str
    ratio: float
    centre: Tuple[int, int]
    radius: int


@dataclass
class QuestionResult:
    question: int
    selections: Dict[str, float]
    top_option: Optional[str]
    confidence: float


@dataclass
class ProcessedOMR:
    answers: List[QuestionResult]
    roll_number: Optional[str]
    debug_paths: Dict[str, Path]


# ---------------------------------------------------------------------------
# Image utilities


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image


def enhance_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def adaptive_binarise(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )
    cleaned = cv2.medianBlur(thresh, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def detect_anchor_candidates(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = gray.shape[0] * gray.shape[1]
    min_area = max(1500.0, image_area * 0.0002)

    candidates: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(h)
        if 0.6 <= aspect <= 1.4:
            candidates.append((x, y, w, h))

    candidates.sort(key=lambda box: box[2] * box[3], reverse=True)
    return candidates[:8]


def pick_anchor_points(
    candidates: Sequence[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int, int],
) -> Dict[str, Tuple[float, float]]:
    if not candidates:
        raise RuntimeError("No anchor candidates detected")

    height, width = image_shape[:2]
    anchors_by_quadrant: Dict[str, Tuple[int, int, int, int]] = {}

    for x, y, w, h in candidates:
        cx = x + w / 2
        cy = y + h / 2
        if cx < width / 2 and cy < height / 2:
            label = "top_left"
        elif cx >= width / 2 and cy < height / 2:
            label = "top_right"
        elif cx >= width / 2 and cy >= height / 2:
            label = "bottom_right"
        else:
            label = "bottom_left"

        if label not in anchors_by_quadrant or w * h > anchors_by_quadrant[label][2] * anchors_by_quadrant[label][3]:
            anchors_by_quadrant[label] = (x, y, w, h)

    if len(anchors_by_quadrant) < 4:
        raise RuntimeError("Unable to locate all four anchor markers")

    ordered = {
        "A1": anchors_by_quadrant["top_left"],
        "A2": anchors_by_quadrant["top_right"],
        "A3": anchors_by_quadrant["bottom_right"],
        "A4": anchors_by_quadrant["bottom_left"],
    }

    centres = {
        name: (box[0] + box[2] / 2, box[1] + box[3] / 2)
        for name, box in ordered.items()
    }
    return centres


# ---------------------------------------------------------------------------
# Core processing pipeline


class OMRProcessor:
    """Process scans using a sheet template."""

    def __init__(self, template: Template, results_dir: Path = Path("results")) -> None:
        self.template = template
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def _normalise(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Path]]:
        enhanced = enhance_image(image)
        candidate_boxes = detect_anchor_candidates(enhanced)
        detected = pick_anchor_points(candidate_boxes, image.shape)

        target_points = np.float32([
            self.template.anchors["A1"].to_pixel(self.template.page_width, self.template.page_height, self.template.dpi),
            self.template.anchors["A2"].to_pixel(self.template.page_width, self.template.page_height, self.template.dpi),
            self.template.anchors["A3"].to_pixel(self.template.page_width, self.template.page_height, self.template.dpi),
            self.template.anchors["A4"].to_pixel(self.template.page_width, self.template.page_height, self.template.dpi),
        ])

        source_points = np.float32([
            detected["A1"],
            detected["A2"],
            detected["A3"],
            detected["A4"],
        ])

        width_px, height_px = self.template.canvas_size_px
        matrix = cv2.getPerspectiveTransform(source_points, target_points)
        warped = cv2.warpPerspective(image, matrix, (width_px, height_px), flags=cv2.INTER_CUBIC)

        debug_paths = {}
        overlay = image.copy()
        for name, (cx, cy) in detected.items():
            cv2.circle(overlay, (int(round(cx)), int(round(cy))), 20, (0, 0, 255), 3)
            cv2.putText(overlay, name, (int(cx) - 10, int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        debug_anchor_path = self.results_dir / "debug_detected_anchors.jpg"
        cv2.imwrite(str(debug_anchor_path), overlay)
        debug_paths["anchors"] = debug_anchor_path

        debug_warped_path = self.results_dir / "debug_warped.jpg"
        cv2.imwrite(str(debug_warped_path), warped)
        debug_paths["warped"] = debug_warped_path

        return warped, debug_paths

    # ------------------------------------------------------------------

    def _measure_bubbles(self, warped: np.ndarray) -> Tuple[List[BubbleMeasurement], np.ndarray, Path]:
        enhanced = enhance_image(warped)
        binary = adaptive_binarise(enhanced)

        heatmap = np.zeros_like(binary, dtype=np.float32)
        measurements: List[BubbleMeasurement] = []

        for bubble in self.template.answer_bubbles:
            cx, cy = bubble.centre_px(self.template.page_width, self.template.page_height, self.template.dpi)
            radius = int(bubble.radius_px(self.template.page_width, self.template.dpi) * 1.15)
            mask = np.zeros_like(binary)
            cv2.circle(mask, (cx, cy), radius, 255, -1)

            bubble_pixels = cv2.countNonZero(cv2.bitwise_and(binary, binary, mask=mask))
            total_pixels = cv2.countNonZero(mask)
            ratio = bubble_pixels / max(total_pixels, 1)

            cv2.circle(heatmap, (cx, cy), radius, float(ratio), -1)

            measurements.append(
                BubbleMeasurement(
                    question=bubble.question,
                    option=bubble.option,
                    ratio=ratio,
                    centre=(cx, cy),
                    radius=radius,
                )
            )

        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)

        overlay_path = self.results_dir / "debug_bubble_heatmap.jpg"
        cv2.imwrite(str(overlay_path), heatmap_color)

        return measurements, binary, overlay_path

    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_answers(measurements: Sequence[BubbleMeasurement]) -> List[QuestionResult]:
        by_question: Dict[int, Dict[str, float]] = {}
        for measurement in measurements:
            by_question.setdefault(measurement.question, {})[measurement.option] = measurement.ratio

        results: List[QuestionResult] = []
        for question, options in sorted(by_question.items()):
            sorted_opts = sorted(options.items(), key=lambda item: item[1], reverse=True)
            top_option: Optional[str] = None
            confidence = 0.0
            if sorted_opts:
                top_option = sorted_opts[0][0] if sorted_opts[0][1] > 0.2 else None
                if len(sorted_opts) > 1:
                    confidence = sorted_opts[0][1] - sorted_opts[1][1]
                else:
                    confidence = sorted_opts[0][1]

            results.append(
                QuestionResult(
                    question=question,
                    selections=options,
                    top_option=top_option,
                    confidence=confidence,
                )
            )

        return results

    # ------------------------------------------------------------------

    def _decode_roll_number(self, binary: np.ndarray) -> Optional[str]:
        # Re-measure using the binary image to maintain parity with answers
        digits: Dict[int, Dict[int, float]] = {}
        for roll in self.template.roll_bubbles:
            cx, cy = roll.centre_px(self.template.page_width, self.template.page_height, self.template.dpi)
            radius = int(roll.radius_px(self.template.page_width, self.template.dpi) * 1.1)
            mask = np.zeros_like(binary)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            value = cv2.countNonZero(cv2.bitwise_and(binary, binary, mask=mask)) / max(cv2.countNonZero(mask), 1)
            digits.setdefault(roll.digit_index, {})[roll.digit_value] = value

        ordered = []
        for digit_idx in sorted(digits):
            sorted_digits = sorted(digits[digit_idx].items(), key=lambda item: item[1], reverse=True)
            if not sorted_digits:
                return None
            top_value, score = sorted_digits[0]
            if score < 0.2:
                return None
            ordered.append(str(top_value))

        return "".join(ordered) if ordered else None

    # ------------------------------------------------------------------

    def process(self, image_path: Path) -> ProcessedOMR:
        image = load_image(image_path)
        warped, debug_paths = self._normalise(image)
        measurements, binary, heatmap_path = self._measure_bubbles(warped)
        debug_paths["heatmap"] = heatmap_path

        answers = self._aggregate_answers(measurements)
        roll_number = self._decode_roll_number(binary)

        results_json = self.results_dir / "results.json"
        serialisable = {
            "answers": [
                {
                    "question": result.question,
                    "top_option": result.top_option,
                    "confidence": result.confidence,
                    "options": result.selections,
                }
                for result in answers
            ],
            "roll_number": roll_number,
        }
        with results_json.open("w", encoding="utf8") as fp:
            json.dump(serialisable, fp, indent=2)
        debug_paths["results_json"] = results_json

        return ProcessedOMR(answers=answers, roll_number=roll_number, debug_paths=debug_paths)


# ---------------------------------------------------------------------------
# Command line interface


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan an OMR sheet with unmatched accuracy")
    parser.add_argument("image", type=Path, help="Path to the scanned image")
    parser.add_argument("--metadata", type=Path, default=Path("sheets/omr_sheet.json"), help="Metadata JSON path")
    parser.add_argument("--results", type=Path, default=Path("results"), help="Directory for diagnostic outputs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    template = Template.from_json(args.metadata)
    processor = OMRProcessor(template=template, results_dir=args.results)
    result = processor.process(args.image)

    print("Roll number:", result.roll_number or "(undetected)")
    for answer in result.answers:
        formatted_options = ", ".join(f"{opt}={score:.2f}" for opt, score in sorted(answer.selections.items()))
        print(
            f"Q{answer.question:03d}: {answer.top_option or '-'} (confidence {answer.confidence:.2f}) | "
            f"{formatted_options}"
        )


if __name__ == "__main__":
    main()
