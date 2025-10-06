#!/usr/bin/env python3
"""
Generate a PDF with a grid of small circles using reportlab.
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors

def create_omr_sheet_pdf(filename="sheets/omr_sheet.pdf", page_size=letter):
    """
    Create an OMR answer sheet PDF with positioning squares and answer bubbles.
    
    Args:
        filename (str): Output PDF filename
        page_size (tuple): Page size (width, height) in points
    """
    # Create canvas
    c = canvas.Canvas(filename, pagesize=page_size)
    width, height = page_size
    
    # Grid parameters
    square_size = 10  # side length in points
    margin = 50  # margin from page edges in points
    num_squares_width = 4
    
    # Calculate spacing to fit exactly 4 squares in width
    available_width = width - (2 * margin)
    spacing_x = available_width / (num_squares_width - 1) if num_squares_width > 1 else 0
    spacing_y = spacing_x  # use same spacing for height
    
    # Calculate grid dimensions
    start_x = margin
    start_y = margin
    end_x = width - margin
    end_y = height - margin
    
    # Calculate total rows
    total_rows = int((end_y - start_y) / spacing_y) + 1
    
    # Ensure drawing defaults
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)

    # Draw grid of squares
    y = start_y
    row_count = 0
    while y <= end_y:
        x = start_x
        square_count = 0
        while square_count < num_squares_width:
            # Check if this is a corner square
            is_corner = ((row_count == 0 or row_count == total_rows - 1) and
                        (square_count == 0 or square_count == num_squares_width - 1))

            # Use larger size for corner squares
            current_size = square_size * 2 if is_corner else square_size
            half_size = current_size / 2

            if is_corner:
                # Draw an anchor with a solid outer square and hollow center to
                # distinguish it from interior grid markers.
                outer_x = x - half_size
                outer_y = y - half_size
                c.setFillColor(colors.black)
                c.rect(outer_x, outer_y, current_size, current_size, stroke=1, fill=1)

                inner_margin = current_size * 0.3
                inner_size = current_size - 2 * inner_margin
                if inner_size > 0:
                    c.setFillColor(colors.white)
                    c.rect(outer_x + inner_margin, outer_y + inner_margin,
                           inner_size, inner_size, stroke=0, fill=1)

                # Reset fill color for subsequent shapes
                c.setFillColor(colors.black)
            else:
                c.rect(x - half_size, y - half_size, current_size, current_size, stroke=1, fill=1)

            x += spacing_x
            square_count += 1
        y += spacing_y
        row_count += 1
    
    # Add text headers and bubble configuration
    c.setFont("Helvetica-Bold", 12)
    bubble_radius = 8
    bubble_spacing_y = 25  # Vertical spacing between bubble rows
    bubble_spacing_x = bubble_radius * 3  # Horizontal spacing between bubbles

    # Grid row layout for column 0:
    # Row 0: "Roll No." label (reserved, no bubbles)
    # Rows 1-10: Roll number bubbles (digits 0-9, 10 rows)
    # Row 11: Gap
    # Row 12: "Questions" label (reserved)
    # Rows 13+: Question bubbles

    # Roll number section configuration
    num_roll_rows = 10  # Digits 0-9
    roll_label_row = 0   # Row position for "Roll No." label (reserved)
    roll_start_row = 1   # First row of roll bubbles (digit 0)
    questions_gap_row = 11     # Gap above Questions label
    questions_label_row = 12   # Row position for "Questions" label
    questions_start_row = 13   # First row of question bubbles in column 0

    # Calculate starting Y position with symmetric margins (no offset)
    bubbles_start_y = start_y + (total_rows - 1) * spacing_y

    # Configure column 0 bubble layout
    gap_center_x = start_x + (0 + 0.5) * spacing_x
    # Use 4-slot grid layout (same as questions) but only use first 3 slots
    total_width = 3 * bubble_spacing_x  # Width for 4 bubble slots
    gap_start_x = gap_center_x - total_width / 2

    # Draw "Roll No." header at reserved row position
    roll_no_text_width = c.stringWidth("Roll No.", "Helvetica-Bold", 12)
    roll_no_center_x = start_x + (0 + 0.5) * spacing_x - roll_no_text_width / 2
    roll_label_y = bubbles_start_y - (roll_label_row * bubble_spacing_y)
    c.drawString(roll_no_center_x, roll_label_y - 3, "Roll No.")

    # Draw digit labels (0-9) to the left of the first column
    c.setFont("Helvetica", 8)
    label_x = gap_start_x - 25
    for digit in range(num_roll_rows):
        bubble_y = bubbles_start_y - ((roll_start_row + digit) * bubble_spacing_y)
        c.drawString(label_x, bubble_y - 3, str(digit))

    # Draw roll number bubbles in first column gap (rows 0-9)
    # Draw 3 digit columns (hundreds, tens, units) in first 3 slots of 4-slot grid
    for digit_col in range(3):  # Use only first 3 of 4 available slots
        bubble_x = gap_start_x + digit_col * bubble_spacing_x

        # Draw bubbles for this digit position (0-9)
        for digit in range(num_roll_rows):
            bubble_y = bubbles_start_y - ((roll_start_row + digit) * bubble_spacing_y)
            c.circle(bubble_x, bubble_y, bubble_radius, stroke=1, fill=0)

    # Draw "Questions" header at reserved row position (below roll numbers)
    c.setFont("Helvetica-Bold", 12)
    questions_text_width = c.stringWidth("Questions", "Helvetica-Bold", 12)
    questions_center_x = start_x + (0 + 0.5) * spacing_x - questions_text_width / 2
    questions_label_y = bubbles_start_y - (questions_label_row * bubble_spacing_y)
    c.drawString(questions_center_x, questions_label_y - 3, "Questions")

    # Draw OMR answer bubbles in all column gaps
    question_number = 1  # Track question numbers across all columns

    for col_gap in range(num_squares_width - 1):
        # Calculate x position for this column gap
        gap_center_x = start_x + (col_gap + 0.5) * spacing_x
        # Center 4 bubbles within the gap using same 4-slot grid
        total_width = 3 * bubble_spacing_x  # Width for 4 bubble slots
        gap_start_x = gap_center_x - total_width / 2

        # Position for question number labels (to the left of bubbles)
        question_label_x = gap_start_x - 25

        # Determine starting row for this column
        if col_gap == 0:
            # Column 0: start questions at row 12 (after roll numbers and labels)
            start_row = questions_start_row
        else:
            # Columns 1-2: start questions at row 0 (aligned with roll bubbles)
            start_row = 0

        # Calculate starting Y position for this column
        bubble_y = bubbles_start_y - (start_row * bubble_spacing_y)
        row_index = start_row

        # Draw 4 answer bubbles (A, B, C, D) with question numbers
        while bubble_y >= start_y:
            # Draw question number label
            c.setFont("Helvetica", 8)
            c.drawString(question_label_x, bubble_y - 3, str(question_number))

            # Draw all 4 bubbles for this question
            for bubble_idx in range(4):
                bubble_x = gap_start_x + bubble_idx * bubble_spacing_x
                c.circle(bubble_x, bubble_y, bubble_radius, stroke=1, fill=0)

            bubble_y -= bubble_spacing_y
            row_index += 1
            question_number += 1
    
    # Save the PDF
    c.save()
    print(f"PDF created: {filename}")

def main():
    # Create PDF with default settings
    create_omr_sheet_pdf()
    
    # Create a custom PDF with A4 size and different parameters
    create_omr_sheet_pdf("sheets/omr_sheet_a4.pdf", A4)

if __name__ == "__main__":
    main()