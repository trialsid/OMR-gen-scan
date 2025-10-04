#!/usr/bin/env python3
"""High fidelity OMR sheet generator.

This module produces professionally typeset OMR sheets alongside a machine
readable JSON specification that the scanner can consume for sub-pixel precise
decoding.  The layout is carefully designed so that the anchor squares,
question bubbles and candidate ID section are all described in a single place.

Usage example
-------------

.. code-block:: console

    $ python omr_sheet_generator.py \
        --questions 120 \
        --options ABCDE \
        --output sheets/world_class_omr.pdf

The command above will produce the PDF and a matching
``sheets/world_class_omr.json`` metadata file that the scanner understands.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


# ---------------------------------------------------------------------------
# Data models


@dataclass
class AnchorSpec:
    """Description of a fiducial anchor square."""

    name: str
    center: Tuple[float, float]
    size: float

    def to_dict(self) -> Dict[str, float]:
        return {"name": self.name, "cx": self.center[0], "cy": self.center[1], "size": self.size}


@dataclass
class BubbleSpec:
    """Description for a single answer bubble."""

    question: int
    option: str
    center: Tuple[float, float]
    radius: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "question": self.question,
            "option": self.option,
            "cx": self.center[0],
            "cy": self.center[1],
            "radius": self.radius,
        }


@dataclass
class RollNumberSpec:
    """Roll number bubble definition."""

    digit_index: int
    digit_value: int
    center: Tuple[float, float]
    radius: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "digit_index": self.digit_index,
            "digit_value": self.digit_value,
            "cx": self.center[0],
            "cy": self.center[1],
            "radius": self.radius,
        }


@dataclass
class SheetLayout:
    """Complete representation of an OMR sheet layout."""

    page_size: Tuple[float, float]
    dpi: int
    anchors: List[AnchorSpec] = field(default_factory=list)
    answer_bubbles: List[BubbleSpec] = field(default_factory=list)
    roll_number_bubbles: List[RollNumberSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "page": {
                "width_points": self.page_size[0],
                "height_points": self.page_size[1],
                "dpi": self.dpi,
            },
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "answer_bubbles": [bubble.to_dict() for bubble in self.answer_bubbles],
            "roll_number_bubbles": [bubble.to_dict() for bubble in self.roll_number_bubbles],
        }


# ---------------------------------------------------------------------------
# Generator implementation


def _linspace(start: float, stop: float, num: int) -> Iterable[float]:
    """Return evenly spaced values between start and stop inclusive."""

    if num == 1:
        yield start
        return

    step = (stop - start) / (num - 1)
    for idx in range(num):
        yield start + step * idx


class OMRSheetGenerator:
    """Builds professional quality OMR sheets with accompanying metadata."""

    def __init__(
        self,
        questions: int = 100,
        options: Sequence[str] = ("A", "B", "C", "D"),
        columns: int = 4,
        dpi: int = 300,
        page_size: Tuple[float, float] = letter,
        margin: float = 0.6 * inch,
        bubble_radius: float = 0.095 * inch,
        vertical_spacing: float = 0.33 * inch,
        anchor_size: float = 0.35 * inch,
    ) -> None:
        if questions <= 0:
            raise ValueError("questions must be a positive integer")

        if columns <= 0:
            raise ValueError("columns must be a positive integer")

        if not options:
            raise ValueError("options cannot be empty")

        self.questions = questions
        self.options = list(options)
        self.columns = columns
        self.dpi = dpi
        self.page_size = page_size
        self.margin = margin
        self.bubble_radius = bubble_radius
        self.vertical_spacing = vertical_spacing
        self.anchor_size = anchor_size

        self._canvas = None
        self.layout = SheetLayout(page_size=page_size, dpi=dpi)

    # ------------------------------------------------------------------
    # Layout helpers

    def _draw_anchor_squares(self, c: canvas.Canvas) -> None:
        width, height = self.page_size
        half = self.anchor_size / 2

        anchor_positions = {
            "A1": (self.margin, height - self.margin),
            "A2": (width - self.margin, height - self.margin),
            "A3": (width - self.margin, self.margin),
            "A4": (self.margin, self.margin),
        }

        for name, (cx, cy) in anchor_positions.items():
            x = cx - half
            y = cy - half
            c.rect(x, y, self.anchor_size, self.anchor_size, stroke=1, fill=1)
            self.layout.anchors.append(AnchorSpec(name=name, center=(cx, cy), size=self.anchor_size))

    def _draw_title(self, c: canvas.Canvas) -> None:
        width, height = self.page_size
        title = "Optical Mark Recognition Answer Sheet"
        c.setFont("Helvetica-Bold", 18)
        text_width = c.stringWidth(title, "Helvetica-Bold", 18)
        c.drawString((width - text_width) / 2, height - self.margin + 0.2 * inch, title)

        subtitle = "Align the four dark squares with the scanner guides"
        c.setFont("Helvetica", 10)
        text_width = c.stringWidth(subtitle, "Helvetica", 10)
        c.drawString((width - text_width) / 2, height - self.margin - 0.15 * inch, subtitle)

    def _draw_roll_number_region(self, c: canvas.Canvas, top_y: float) -> float:
        c.setFont("Helvetica-Bold", 12)
        label = "Candidate Roll Number"
        text_width = c.stringWidth(label, "Helvetica-Bold", 12)

        left = self.margin
        c.drawString(left, top_y, label)

        bubble_radius = self.bubble_radius * 0.9
        vertical_gap = bubble_radius * 2.3
        digit_gap = bubble_radius * 2.6

        start_x = left + text_width + 0.5 * inch
        start_y = top_y - bubble_radius

        c.setFont("Helvetica", 8)
        for digit_idx in range(4):
            cx = start_x + digit_idx * digit_gap
            c.drawString(cx - bubble_radius, start_y + bubble_radius + 6, f"Digit {digit_idx + 1}")

            for digit_value in range(10):
                cy = start_y - digit_value * vertical_gap
                c.circle(cx, cy, bubble_radius, stroke=1, fill=0)
                c.drawString(cx + bubble_radius + 4, cy - 3, str(digit_value))
                self.layout.roll_number_bubbles.append(
                    RollNumberSpec(digit_index=digit_idx, digit_value=digit_value, center=(cx, cy), radius=bubble_radius)
                )

        return start_y - 10 * vertical_gap - 0.4 * inch

    def _iter_question_centres(self, start_y: float) -> Iterable[Tuple[int, float, float]]:
        width, _ = self.page_size
        usable_width = width - 2 * self.margin
        column_gap = usable_width / self.columns

        rows_per_column = (self.questions + self.columns - 1) // self.columns

        question = 1
        for column in range(self.columns):
            cx = self.margin + column * column_gap + column_gap / 2
            cy = start_y
            for _ in range(rows_per_column):
                if question > self.questions:
                    break
                yield question, cx, cy
                question += 1
                cy -= self.vertical_spacing

    def _draw_questions(self, c: canvas.Canvas, start_y: float) -> None:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, start_y + 0.25 * inch, "Answer Bubbles")

        option_spacing = self.bubble_radius * 2.6

        c.setFont("Helvetica", 8)
        for question, base_x, base_y in self._iter_question_centres(start_y):
            c.drawString(base_x - 1.2 * inch, base_y - 3, f"{question:03d}")
            for idx, option in enumerate(self.options):
                cx = base_x + (idx - (len(self.options) - 1) / 2) * option_spacing
                cy = base_y
                c.circle(cx, cy, self.bubble_radius, stroke=1, fill=0)
                c.drawString(cx - 3, cy - self.bubble_radius - 8, option)
                self.layout.answer_bubbles.append(
                    BubbleSpec(question=question, option=option, center=(cx, cy), radius=self.bubble_radius)
                )

    # ------------------------------------------------------------------
    # Public API

    def build(self, output_pdf: Path) -> Path:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        c = canvas.Canvas(str(output_pdf), pagesize=self.page_size)
        self._canvas = c
        self.layout = SheetLayout(page_size=self.page_size, dpi=self.dpi)

        self._draw_anchor_squares(c)
        self._draw_title(c)

        roll_region_bottom = self._draw_roll_number_region(c, top_y=self.page_size[1] - self.margin - 0.5 * inch)
        self._draw_questions(c, start_y=roll_region_bottom)

        c.showPage()
        c.save()

        return output_pdf

    def export_metadata(self, metadata_path: Path) -> Path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf8") as fp:
            json.dump(self.layout.to_dict(), fp, indent=2)
        return metadata_path


# ---------------------------------------------------------------------------
# Command line interface


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a world-class OMR sheet and metadata")
    parser.add_argument("--questions", type=int, default=120, help="Total number of questions to display")
    parser.add_argument("--options", type=str, default="ABCD", help="Answer options to render (e.g. ABCD or ABCDE)")
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Number of answer columns. The generator automatically balances questions per column.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Virtual DPI used to describe the page to the scanner")
    parser.add_argument(
        "--page-size",
        choices=["letter", "a4"],
        default="letter",
        help="Output page size. The metadata embeds the correct width/height.",
    )
    parser.add_argument("--output", type=Path, default=Path("sheets/omr_sheet.pdf"), help="Destination PDF path")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata path. Defaults to the PDF path with .json extension.",
    )
    return parser.parse_args(argv)


def resolve_page_size(name: str) -> Tuple[float, float]:
    return letter if name == "letter" else A4


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    page_size = resolve_page_size(args.page_size)
    options = tuple(args.options.strip())

    generator = OMRSheetGenerator(
        questions=args.questions,
        options=options,
        columns=args.columns,
        dpi=args.dpi,
        page_size=page_size,
    )

    pdf_path = generator.build(args.output)

    metadata_path = args.metadata or pdf_path.with_suffix(".json")
    generator.export_metadata(metadata_path)

    print(f"PDF created: {pdf_path}")
    print(f"Metadata created: {metadata_path}")


if __name__ == "__main__":
    main()