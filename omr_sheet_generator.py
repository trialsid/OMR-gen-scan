#!/usr/bin/env python3
"""
Generate a PDF with a grid of small circles using reportlab.
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch

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
            c.rect(x - half_size, y - half_size, current_size, current_size, stroke=1, fill=1)
            
            x += spacing_x
            square_count += 1
        y += spacing_y
        row_count += 1
    
    # Draw OMR answer bubbles in column gaps - continuous layout
    bubble_radius = 8
    bubble_spacing_y = 25  # increased spacing between rows
    
    for col_gap in range(num_squares_width - 1):
        # Calculate x position for this column gap
        gap_center_x = start_x + (col_gap + 0.5) * spacing_x
        # Center the 4 bubbles within the gap with proper spacing
        bubble_spacing_x = bubble_radius * 3  # spacing between bubble centers
        total_width = 3 * bubble_spacing_x  # total width for 4 bubbles
        gap_start_x = gap_center_x - total_width / 2
        gap_spacing = bubble_spacing_x
        
        # Draw 4 answer bubbles (A, B, C, D) continuously within grid bounds
        # Find the actual grid bounds based on square positions
        first_square_y = start_y
        last_square_y = start_y + (total_rows - 1) * spacing_y
        
        bubble_y = first_square_y
        while bubble_y <= last_square_y:
            for bubble_idx in range(4):
                bubble_x = gap_start_x + bubble_idx * gap_spacing
                # Draw empty circles for OMR bubbles (stroke only, no fill)
                c.circle(bubble_x, bubble_y, bubble_radius, stroke=1, fill=0)
            
            bubble_y += bubble_spacing_y
    
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