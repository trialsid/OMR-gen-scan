#!/usr/bin/env python3
"""
OMR Sheet Image Processor
Processes scanned OMR answer sheets to extract filled bubbles.
"""

import cv2
import numpy as np
import os
from pathlib import Path

class OMRProcessor:
    def __init__(self, scans_dir="scans", results_dir="results"):
        self.scans_dir = Path(scans_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def step1_load_original(self, image_path):
        """Step 1: Load the original scanned image"""
        print("Step 1: Loading original image...")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get image info
        height, width, channels = image.shape
        print(f"  - Image size: {width}x{height}")
        print(f"  - Channels: {channels}")
        
        # Save step 1 result
        step1_path = self.results_dir / "step1_original.jpg"
        cv2.imwrite(str(step1_path), image)
        print(f"  - Saved: {step1_path}")
        
        return image
    
    def step2_convert_grayscale(self, image):
        """Step 2: Convert to grayscale"""
        print("Step 2: Converting to grayscale...")
        
        # Convert BGR to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        print(f"  - Converted to single channel")
        
        # Save step 2 result
        step2_path = self.results_dir / "step2_grayscale.jpg"
        cv2.imwrite(str(step2_path), gray)
        print(f"  - Saved: {step2_path}")
        
        return gray
    
    def step3_noise_reduction(self, gray_image):
        """Step 3: Apply noise reduction"""
        print("Step 3: Applying noise reduction...")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        print(f"  - Applied Gaussian blur (5x5 kernel)")
        
        # Save step 3 result
        step3_path = self.results_dir / "step3_blurred.jpg"
        cv2.imwrite(str(step3_path), blurred)
        print(f"  - Saved: {step3_path}")
        
        return blurred
    
    def step4_thresholding(self, blurred_image):
        """Step 4: Apply adaptive thresholding"""
        print("Step 4: Applying thresholding...")
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        print(f"  - Applied adaptive threshold (GAUSSIAN_C, 11, 2)")
        
        # Save step 4 result
        step4_path = self.results_dir / "step4_threshold.jpg"
        cv2.imwrite(str(step4_path), thresh)
        print(f"  - Saved: {step4_path}")
        
        return thresh
    
    def step5_detect_contours(self, thresh_image, original_image):
        """Step 5: Detect all contours"""
        print("Step 5: Detecting contours...")
        
        # Find contours
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  - Found {len(contours)} total contours")
        
        # Draw all contours on original image
        result_image = original_image.copy()
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 1)
        
        # Save step 5 result
        step5_path = self.results_dir / "step5_all_contours.jpg"
        cv2.imwrite(str(step5_path), result_image)
        print(f"  - Saved: {step5_path}")
        
        return contours
    
    def step6_detect_squares(self, contours, original_image):
        """Step 6: Detect and classify squares"""
        print("Step 6: Detecting squares...")
        
        all_squares = []
        debug_info = {'total_contours': 0, 'area_filtered': 0, 'polygon_filtered': 0, 'aspect_filtered': 0}
        
        for contour in contours:
            debug_info['total_contours'] += 1
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # More lenient area filtering for grid squares
            if area > 30 and area < 8000:
                debug_info['area_filtered'] += 1
                # Approximate contour to polygon - more lenient
                epsilon = 0.03 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly square (4 vertices, but allow some tolerance)
                if len(approx) >= 4 and len(approx) <= 6:
                    debug_info['polygon_filtered'] += 1
                    # Calculate bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # More lenient aspect ratio for squares (scanning can distort)
                    if 0.5 <= aspect_ratio <= 2.0:
                        debug_info['aspect_filtered'] += 1
                        all_squares.append((x, y, w, h, area))
        
        print(f"  - Debug: {debug_info['total_contours']} contours -> {debug_info['area_filtered']} area OK -> {debug_info['polygon_filtered']} polygon OK -> {debug_info['aspect_filtered']} aspect OK")
        
        # Sort squares by area
        all_squares.sort(key=lambda x: x[4], reverse=True)
        
        # Get image dimensions and calculate column positions
        img_height, img_width = original_image.shape[:2]
        
        # Define the 4 marker columns based on OMR sheet structure
        col1_x = img_width * 0.05   # Left edge (A1, G1-G3, A4)
        col2_x = img_width * 0.33   # Left-center (G4-G8) 
        col3_x = img_width * 0.67   # Right-center (G9-G13)
        col4_x = img_width * 0.95   # Right edge (A2, G14-G16, A3)
        
        tolerance = 80  # Increased tolerance for perspective distortion
        
        print(f"  - Image size: {img_width}x{img_height}")
        print(f"  - Expected marker columns at x: {col1_x:.0f}, {col2_x:.0f}, {col3_x:.0f}, {col4_x:.0f}")
        
        # Separate squares by their column position
        positioning_squares = []
        grid_squares = []
        
        if len(all_squares) > 0:
            # Show all detected squares
            print(f"  - All detected squares by area:")
            for i, (x, y, w, h, area) in enumerate(all_squares):
                print(f"    {i+1}: ({x}, {y}) {w}x{h} area={area:.0f}")
            
            # Classify squares by column and position
            for square in all_squares:
                x, y, w, h, area = square
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Determine which column this square belongs to
                column = None
                if abs(center_x - col1_x) < tolerance:
                    column = 1
                elif abs(center_x - col2_x) < tolerance:
                    column = 2
                elif abs(center_x - col3_x) < tolerance:
                    column = 3
                elif abs(center_x - col4_x) < tolerance:
                    column = 4
                
                if column is None:
                    print(f"    Rejected square at ({x}, {y}) - not in any marker column")
                    continue
                
                # Classify based on column and position
                is_top = center_y < img_height * 0.15
                is_bottom = center_y > img_height * 0.85
                
                if column == 1:  # Left edge: A1(top), G1-G3(middle), A4(bottom)
                    if is_top and area > 150:  # More tolerant for perspective distortion
                        positioning_squares.append(square)
                        print(f"    Found A1 (top-left) at ({x}, {y})")
                    elif is_bottom and area > 150:  # More tolerant for perspective distortion
                        positioning_squares.append(square)
                        print(f"    Found A4 (bottom-left) at ({x}, {y})")
                    elif not is_top and not is_bottom and area > 60:
                        grid_squares.append(square)
                        print(f"    Found G marker (col1-middle) at ({x}, {y})")
                        
                elif column == 2:  # Left-center: G4-G8 (5 markers full height)
                    if area > 60:
                        grid_squares.append(square)
                        print(f"    Found G marker (col2) at ({x}, {y})")
                        
                elif column == 3:  # Right-center: G9-G13 (5 markers full height)
                    if area > 60:
                        grid_squares.append(square)
                        print(f"    Found G marker (col3) at ({x}, {y})")
                        
                elif column == 4:  # Right edge: A2(top), G14-G16(middle), A3(bottom)
                    if is_top and area > 150:  # More tolerant for perspective distortion
                        positioning_squares.append(square)
                        print(f"    Found A2 (top-right) at ({x}, {y})")
                    elif is_bottom and area > 150:  # More tolerant for perspective distortion
                        positioning_squares.append(square)
                        print(f"    Found A3 (bottom-right) at ({x}, {y})")
                    elif not is_top and not is_bottom and area > 60:
                        grid_squares.append(square)
                        print(f"    Found G marker (col4-middle) at ({x}, {y})")
        
        print(f"  - Total squares found: {len(all_squares)}")
        print(f"  - Positioning squares: {len(positioning_squares)}")
        print(f"  - Grid squares: {len(grid_squares)}")
        
        # Sort positioning squares by position for proper A1-A4 labeling
        if len(positioning_squares) >= 4:
            # Identify corners by position
            img_height, img_width = original_image.shape[:2]
            corner_squares = {}
            
            for x, y, w, h, area in positioning_squares:
                center_x, center_y = x + w//2, y + h//2
                
                if center_x < img_width//2 and center_y < img_height//2:
                    corner_squares['A1'] = (x, y, w, h, area)  # Top-left
                elif center_x >= img_width//2 and center_y < img_height//2:
                    corner_squares['A2'] = (x, y, w, h, area)  # Top-right  
                elif center_x >= img_width//2 and center_y >= img_height//2:
                    corner_squares['A3'] = (x, y, w, h, area)  # Bottom-right
                elif center_x < img_width//2 and center_y >= img_height//2:
                    corner_squares['A4'] = (x, y, w, h, area)  # Bottom-left
            
            print(f"  - Anchor points:")
            for label in ['A1', 'A2', 'A3', 'A4']:
                if label in corner_squares:
                    x, y, w, h, area = corner_squares[label]
                    print(f"    {label}: ({x}, {y}) area={area:.0f}")
            
            # Create ordered list for visualization
            positioning_squares_sorted = [corner_squares.get(label, None) for label in ['A1', 'A2', 'A3', 'A4']]
            positioning_squares_sorted = [sq for sq in positioning_squares_sorted if sq is not None]
        
        # Create visualization
        result_image = original_image.copy()
        
        # Draw positioning squares (green) with anchor labels
        if len(positioning_squares) >= 4:
            positioning_squares_sorted = sorted(positioning_squares, key=lambda sq: (sq[1], sq[0]))
            labels = ['A1', 'A2', 'A3', 'A4']
            
            for i, (x, y, w, h, area) in enumerate(positioning_squares_sorted[:4]):
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Put label inside the square
                label_x = x + w // 2 - 15
                label_y = y + h // 2 + 5
                cv2.putText(result_image, labels[i], (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Put coordinates above the square
                coord_text = f"({x},{y})"
                cv2.putText(result_image, coord_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            # Fallback for less than 4 positioning squares
            for i, (x, y, w, h, area) in enumerate(positioning_squares):
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_image, f"P{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw grid squares (yellow) with numbering
        for i, (x, y, w, h, area) in enumerate(grid_squares):
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            # Add small number for grid squares
            cv2.putText(result_image, f"G{i+1}", (x + w//2 - 10, y + h//2 + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        # Save step 6 result
        step6_path = self.results_dir / "step6_squares.jpg"
        cv2.imwrite(str(step6_path), result_image)
        print(f"  - Saved: {step6_path}")
        
        return positioning_squares, grid_squares
    
    def step7_crop_and_correct_perspective(self, original_image, positioning_squares):
        """Step 7: Crop and correct perspective using anchor points"""
        print("Step 7: Cropping and correcting perspective...")
        
        if len(positioning_squares) < 4:
            print(f"  - Error: Need 4 positioning squares, found {len(positioning_squares)}")
            return original_image
        
        # Get image dimensions
        img_height, img_width = original_image.shape[:2]
        
        # Find the corner positions
        corner_points = {}
        for x, y, w, h, area in positioning_squares:
            center_x, center_y = x + w//2, y + h//2
            
            if center_x < img_width//2 and center_y < img_height//2:
                corner_points['A1'] = (center_x, center_y)  # Top-left
            elif center_x >= img_width//2 and center_y < img_height//2:
                corner_points['A2'] = (center_x, center_y)  # Top-right  
            elif center_x >= img_width//2 and center_y >= img_height//2:
                corner_points['A3'] = (center_x, center_y)  # Bottom-right
            elif center_x < img_width//2 and center_y >= img_height//2:
                corner_points['A4'] = (center_x, center_y)  # Bottom-left
        
        if len(corner_points) < 4:
            print(f"  - Error: Could not identify all 4 corners")
            return original_image
        
        print(f"  - Corner points:")
        for label in ['A1', 'A2', 'A3', 'A4']:
            if label in corner_points:
                x, y = corner_points[label]
                print(f"    {label}: ({x}, {y})")
        
        # Source points (actual detected corners)
        src_points = np.float32([
            corner_points['A1'],  # Top-left
            corner_points['A2'],  # Top-right
            corner_points['A3'],  # Bottom-right
            corner_points['A4']   # Bottom-left
        ])
        
        # Calculate the dimensions of the corrected rectangle
        # Use the maximum distances to preserve aspect ratio
        width_top = np.linalg.norm(np.array(corner_points['A2']) - np.array(corner_points['A1']))
        width_bottom = np.linalg.norm(np.array(corner_points['A3']) - np.array(corner_points['A4']))
        max_width = int(max(width_top, width_bottom))
        
        height_left = np.linalg.norm(np.array(corner_points['A4']) - np.array(corner_points['A1']))
        height_right = np.linalg.norm(np.array(corner_points['A3']) - np.array(corner_points['A2']))
        max_height = int(max(height_left, height_right))
        
        print(f"  - Calculated corrected dimensions: {max_width}x{max_height}")
        
        # Destination points (perfect rectangle)
        margin = 20  # Small margin around the OMR area
        dst_points = np.float32([
            [margin, margin],                           # Top-left
            [max_width - margin, margin],               # Top-right
            [max_width - margin, max_height - margin],  # Bottom-right
            [margin, max_height - margin]               # Bottom-left
        ])
        
        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective correction
        corrected = cv2.warpPerspective(original_image, matrix, (max_width, max_height))
        
        print(f"  - Applied perspective transformation")
        print(f"  - Cropped to OMR area: {max_width}x{max_height}")
        
        # Create visualization showing the transformation
        vis_image = original_image.copy()
        
        # Draw the source quadrilateral
        src_quad = src_points.astype(np.int32)
        cv2.polylines(vis_image, [src_quad], True, (0, 255, 0), 3)
        
        # Label the corners
        labels = ['A1', 'A2', 'A3', 'A4']
        for i, (x, y) in enumerate(src_points):
            cv2.circle(vis_image, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(vis_image, labels[i], (int(x) + 10, int(y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save step 7 results
        step7a_path = self.results_dir / "step7a_perspective_detection.jpg"
        cv2.imwrite(str(step7a_path), vis_image)
        print(f"  - Saved perspective detection: {step7a_path}")
        
        step7b_path = self.results_dir / "step7b_corrected_crop.jpg"
        cv2.imwrite(str(step7b_path), corrected)
        print(f"  - Saved corrected crop: {step7b_path}")
        
        return corrected
    
    def draw_dashed_circle(self, image, center, radius, color, thickness):
        """Draw a dashed circle to indicate inferred/missing bubbles"""
        import math
        
        # Draw dashed circle by drawing small arcs
        num_dashes = 12
        dash_length = 2 * math.pi / num_dashes / 2  # Half circumference divided by dashes
        
        for i in range(num_dashes):
            start_angle = i * 2 * math.pi / num_dashes
            end_angle = start_angle + dash_length
            
            # Convert to integer degrees for OpenCV
            start_deg = int(math.degrees(start_angle))
            end_deg = int(math.degrees(end_angle))
            
            # Draw arc (partial circle)
            cv2.ellipse(image, center, (radius, radius), 0, start_deg, end_deg, color, thickness)
    
    def process_roll_number_bubbles(self, roll_bubbles, result_image, color, img_width):
        """Process roll number bubbles (3 digits, 0-9 each)"""
        if not roll_bubbles:
            print("    No roll number bubbles detected")
            return
        
        # Expected grid: 3 columns of digits within the first column gap
        # Based on sheet generator: 3 digit columns with bubble_spacing_x spacing
        digit_col_spacing = 24  # bubble_radius * 3 from generator
        
        # Estimate the 3 digit column positions within the roll number area
        roll_bubbles_x = [b['center'][0] for b in roll_bubbles]
        min_x, max_x = min(roll_bubbles_x), max(roll_bubbles_x)
        
        # Define 3 digit columns
        digit_columns = [
            min_x + (max_x - min_x) * 0.2,   # Hundreds digit
            min_x + (max_x - min_x) * 0.5,   # Tens digit  
            min_x + (max_x - min_x) * 0.8    # Units digit
        ]
        
        print(f"    Expected digit columns at x: {[f'{x:.0f}' for x in digit_columns]}")
        
        # Group bubbles by digit column
        digit_tolerance = 15
        bubbles_by_digit = [[], [], []]
        
        for bubble in roll_bubbles:
            bx = bubble['center'][0]
            for i, col_x in enumerate(digit_columns):
                if abs(bx - col_x) < digit_tolerance:
                    bubbles_by_digit[i].append(bubble)
                    break
        
        # Process each digit column (0-9)
        for digit_idx, digit_bubbles in enumerate(bubbles_by_digit):
            digit_bubbles.sort(key=lambda b: b['center'][1])  # Sort by Y
            print(f"    Digit {digit_idx + 1}: {len(digit_bubbles)} bubbles")
            
            # Draw detected bubbles
            for bubble in digit_bubbles:
                center = bubble['center']
                radius = int(np.sqrt(bubble['area'] / np.pi))
                cv2.circle(result_image, center, radius, color, 2)
                cv2.circle(result_image, center, 2, color, -1)
        
        # Add roll number section label
        if roll_bubbles:
            label_x = min(roll_bubbles_x) - 50
            label_y = min(b['center'][1] for b in roll_bubbles) + 20
            cv2.putText(result_image, "Roll No.", (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def step8_detect_answer_bubbles(self, corrected_image):
        """Step 8: Detect answer bubbles in the corrected image"""
        print("Step 8: Detecting answer bubbles...")
        
        # Convert corrected image to grayscale for processing
        if len(corrected_image.shape) == 3:
            gray_corrected = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_corrected = corrected_image
        
        # Apply preprocessing for bubble detection
        blurred = cv2.GaussianBlur(gray_corrected, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 9, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  - Found {len(contours)} total contours in corrected image")
        
        # Get image dimensions
        img_height, img_width = corrected_image.shape[:2]
        
        # Define bubble area boundaries (exclude marker columns)
        # Based on our 4-column structure: markers at 5%, 33%, 67%, 95%
        # First gap contains roll numbers and questions, other gaps contain only questions
        roll_number_col_center = img_width * 0.19  # Between column 1 and 2 markers (roll numbers + questions)
        question_col2_center = img_width * 0.50  # Between column 2 and 3 markers  
        question_col3_center = img_width * 0.81  # Between column 3 and 4 markers
        
        all_columns = [roll_number_col_center, question_col2_center, question_col3_center]
        column_tolerance = img_width * 0.12  # 12% tolerance to capture all bubbles in each section
        
        print(f"  - Expected columns at x: {roll_number_col_center:.0f} (roll+questions), {question_col2_center:.0f}, {question_col3_center:.0f}")
        
        # Detect bubble candidates
        bubble_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (bubbles should be medium-sized circles)
            if 20 < area < 200:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Must be highly circular to be a bubble
                    if circularity > 0.7:
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Check if bubble is in one of the expected columns
                        in_column = False
                        column_index = -1
                        
                        for i, col_x in enumerate(all_columns):
                            if abs(center_x - col_x) < column_tolerance:
                                in_column = True
                                column_index = i
                                break
                        
                        if in_column:
                            # Include all bubbles in columns (no edge filtering)
                            # We'll rely on circularity and area to distinguish from markers
                                bubble_candidates.append({
                                    'center': (center_x, center_y),
                                    'area': area,
                                    'circularity': circularity,
                                    'column': column_index,
                                    'contour': contour
                                })
        
        print(f"  - Found {len(bubble_candidates)} bubble candidates")
        
        # Debug: show bubble count per column
        for i in range(3):
            col_bubbles = [b for b in bubble_candidates if b['column'] == i]
            print(f"    Column {i+1}: {len(col_bubbles)} bubbles")
        
        # Separate roll number bubbles from question bubbles in first column
        roll_number_bubbles = []
        question_bubbles_by_column = [[], [], []]
        
        # Determine the boundary between roll numbers and questions in first column
        # Roll numbers are in the top part, questions in the bottom part
        roll_question_boundary_y = img_height * 0.4  # Approximate boundary
        
        for bubble in bubble_candidates:
            col_idx = bubble['column']
            center_y = bubble['center'][1]
            
            if col_idx == 0:  # First column contains both roll numbers and questions
                if center_y < roll_question_boundary_y:
                    # Top part = roll numbers
                    roll_number_bubbles.append(bubble)
                else:
                    # Bottom part = questions
                    question_bubbles_by_column[col_idx].append(bubble)
            else:
                # Other columns only have questions
                question_bubbles_by_column[col_idx].append(bubble)
        
        # Sort roll number bubbles and each question column by Y position (top to bottom)
        roll_number_bubbles.sort(key=lambda b: b['center'][1])
        for col_bubbles in question_bubbles_by_column:
            col_bubbles.sort(key=lambda b: b['center'][1])
        
        print(f"  - Organized bubbles:")
        print(f"    Roll number bubbles: {len(roll_number_bubbles)}")
        for i, col_bubbles in enumerate(question_bubbles_by_column):
            print(f"    Question column {i+1}: {len(col_bubbles)} bubbles")
        
        # Create visualization with different colors for each section
        result_image = corrected_image.copy()
        
        # Define colors (BGR format)
        roll_number_color = (255, 255, 0)  # Cyan for roll numbers
        column_colors = [
            (0, 255, 0),    # Green for question column 1
            (255, 0, 0),    # Blue for question column 2  
            (0, 0, 255)     # Red for question column 3
        ]
        
        # Process roll number bubbles first
        print(f"  - Processing roll number section...")
        self.process_roll_number_bubbles(roll_number_bubbles, result_image, roll_number_color, img_width)
        
        # Group bubbles into questions and add numbering
        question_number = 1
        question_tolerance = 15  # pixels tolerance for horizontal alignment
        
        # First pass: Learn the grid structure from all questions in each column
        learned_grids = []
        for col_idx, col_bubbles in enumerate(question_bubbles_by_column):
            print(f"  - Learning grid structure for column {col_idx + 1}...")
            
            # Group bubbles by Y position to identify questions
            questions = []
            current_question_bubbles = []
            current_y = None
            
            for bubble in col_bubbles:
                bubble_y = bubble['center'][1]
                
                if current_y is None or abs(bubble_y - current_y) < question_tolerance:
                    current_question_bubbles.append(bubble)
                    if current_y is None:
                        current_y = bubble_y
                else:
                    if current_question_bubbles:
                        questions.append(current_question_bubbles)
                    current_question_bubbles = [bubble]
                    current_y = bubble_y
            
            if current_question_bubbles:
                questions.append(current_question_bubbles)
            
            # Analyze X positions from questions with 3+ detected bubbles (good examples)
            all_x_positions = []
            for q_bubbles in questions:
                if len(q_bubbles) >= 3:  # Good examples with most bubbles detected
                    q_bubbles.sort(key=lambda b: b['center'][0])
                    x_positions = [b['center'][0] for b in q_bubbles]
                    all_x_positions.extend(x_positions)
            
            if len(all_x_positions) >= 8:  # Need enough data points
                # Simple clustering approach to find 4 consistent grid positions
                all_x_positions.sort()
                
                # Use histogram-based approach to find 4 peaks
                # Create bins and count occurrences
                min_x = min(all_x_positions)
                max_x = max(all_x_positions)
                bin_width = 10  # 10 pixel bins
                bins = {}
                
                for x in all_x_positions:
                    bin_key = int(x // bin_width) * bin_width
                    bins[bin_key] = bins.get(bin_key, 0) + 1
                
                # Find the 4 bins with highest counts
                sorted_bins = sorted(bins.items(), key=lambda x: x[1], reverse=True)
                top_4_bins = sorted(sorted_bins[:4], key=lambda x: x[0])  # Sort by position
                
                if len(top_4_bins) == 4:
                    # Use bin centers as grid positions
                    grid_positions = [bin_pos + bin_width/2 for bin_pos, count in top_4_bins]
                else:
                    # Fallback: equal spacing
                    gap = (max_x - min_x) / 3
                    grid_positions = [min_x + i * gap for i in range(4)]
                
                print(f"    Learned grid positions: {[f'{x:.0f}' for x in grid_positions]}")
                learned_grids.append(grid_positions)
            else:
                # Fallback grid if not enough data
                learned_grids.append([50 + col_idx * 150 + i * 25 for i in range(4)])
        
        # Second pass: Apply learned grid to all questions
        question_number = 1
        for col_idx, col_bubbles in enumerate(question_bubbles_by_column):
            color = column_colors[col_idx]
            grid_positions = learned_grids[col_idx]
            
            # Group bubbles by Y position again
            current_question_bubbles = []
            current_y = None
            
            def process_question_with_learned_grid(bubbles, y_pos, q_num, grid):
                """Process question using actual circle detection guided by learned grid"""
                
                # Enhanced circle detection that prioritizes actual printed circles
                actual_circle_positions = []
                tolerance = 20  # Increased tolerance for better detection
                
                # First pass: detect all actual circles (both filled and empty) in the question row
                search_y_min = int(y_pos - 12)
                search_y_max = int(y_pos + 12)
                
                # Search across the entire width for this question row
                question_row_region = gray_corrected[search_y_min:search_y_max, :]
                
                if question_row_region.size > 0:
                    # Use Hough circles to detect all circles in this row
                    circles = cv2.HoughCircles(question_row_region, cv2.HOUGH_GRADIENT, 1, 15,
                                             param1=30, param2=15, minRadius=6, maxRadius=12)
                    
                    detected_circles = []
                    if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        for circle in circles:
                            circle_x = circle[0]
                            circle_y = circle[1] + search_y_min
                            detected_circles.append((circle_x, circle_y))
                    
                    # Also include filled bubbles detected by contour analysis
                    for bubble in bubbles:
                        bx, by = bubble['center']
                        if abs(by - y_pos) < 12:  # Within the question row
                            detected_circles.append((bx, by))
                    
                    # Remove duplicates (circles detected by both methods)
                    unique_circles = []
                    for cx, cy in detected_circles:
                        is_duplicate = False
                        for ux, uy in unique_circles:
                            if abs(cx - ux) < 10 and abs(cy - uy) < 10:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            unique_circles.append((cx, cy))
                    
                    print(f"    Q{q_num}: Detected {len(unique_circles)} actual circles")
                    
                    # Map detected circles to grid positions
                    bubble_to_grid_mapping = {}
                    for cx, cy in unique_circles:
                        best_grid_idx = None
                        best_distance = float('inf')
                        
                        for i, grid_x in enumerate(grid):
                            distance = abs(cx - grid_x)
                            if distance < tolerance and distance < best_distance:
                                best_grid_idx = i
                                best_distance = distance
                        
                        if best_grid_idx is not None:
                            if best_grid_idx not in bubble_to_grid_mapping:
                                bubble_to_grid_mapping[best_grid_idx] = (cx, cy)
                            else:
                                # Check if this circle is closer to the grid position
                                existing_x, existing_y = bubble_to_grid_mapping[best_grid_idx]
                                existing_distance = abs(existing_x - grid[best_grid_idx])
                                if best_distance < existing_distance:
                                    bubble_to_grid_mapping[best_grid_idx] = (cx, cy)
                    
                    # Build actual circle positions from mapping
                    for grid_idx, (circle_x, circle_y) in bubble_to_grid_mapping.items():
                        # Check if this is a filled bubble
                        is_filled = False
                        matched_bubble = None
                        for bubble in bubbles:
                            bx, by = bubble['center']
                            if abs(circle_x - bx) < 10 and abs(circle_y - by) < 10:
                                is_filled = True
                                matched_bubble = bubble
                                break
                        
                        circle_type = 'filled' if is_filled else 'empty'
                        actual_circle_positions.append((grid_idx, circle_x, circle_type, matched_bubble))
                    
                    print(f"    Q{q_num}: Mapped {len(actual_circle_positions)} circles to grid positions")
                
                # Calculate final positions using actual detected circles
                if len(actual_circle_positions) >= 2:
                    actual_circle_positions.sort(key=lambda x: x[0])  # Sort by grid position
                    
                    # Use actual detected positions to refine the grid
                    detected_positions = {}
                    for grid_idx, actual_x, circle_type, bubble_data in actual_circle_positions:
                        detected_positions[grid_idx] = actual_x
                    
                    # Calculate spacing from detected circles
                    if len(detected_positions) >= 2:
                        positions_list = sorted(detected_positions.items())
                        actual_gap = (positions_list[-1][1] - positions_list[0][1]) / (positions_list[-1][0] - positions_list[0][0])
                        actual_start = positions_list[0][1] - positions_list[0][0] * actual_gap
                        
                        # Generate refined grid using actual spacing
                        final_positions = [actual_start + i * actual_gap for i in range(4)]
                        
                        # Fine-tune positions where we have actual detections
                        for grid_idx, actual_x in detected_positions.items():
                            final_positions[grid_idx] = actual_x
                    else:
                        # Single detection - use it and estimate others
                        single_idx, single_x = list(detected_positions.items())[0]
                        estimated_gap = 20  # Default gap estimate
                        estimated_start = single_x - single_idx * estimated_gap
                        final_positions = [estimated_start + i * estimated_gap for i in range(4)]
                        final_positions[single_idx] = single_x
                else:
                    # Fallback to learned grid positions if no circles detected in this row
                    final_positions = grid
                
                # Draw all 4 positions using actual circle positions where available
                detected_positions_dict = {}
                for grid_idx, actual_x, circle_type, bubble_data in actual_circle_positions:
                    detected_positions_dict[grid_idx] = (actual_x, circle_type, bubble_data)
                
                for i, final_x in enumerate(final_positions):
                    final_center = (int(final_x), int(y_pos))
                    
                    if i in detected_positions_dict:
                        # Use actual detected position
                        actual_x, circle_type, bubble_data = detected_positions_dict[i]
                        actual_center = (int(actual_x), int(y_pos))
                        
                        if circle_type == 'filled' and bubble_data:
                            # Draw filled bubble at actual detected position
                            radius = int(np.sqrt(bubble_data['area'] / np.pi))
                            cv2.circle(result_image, actual_center, radius, color, 2)
                            cv2.circle(result_image, actual_center, 2, color, -1)
                        else:
                            # Draw empty circle at actual detected position
                            radius = 8
                            cv2.circle(result_image, actual_center, radius, color, 2)
                    else:
                        # No actual circle detected - draw inferred position with dashed circle
                        radius = 8
                        self.draw_dashed_circle(result_image, final_center, radius, color, 2)
                        cv2.putText(result_image, "?", (final_center[0] - 3, final_center[1] + 3), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
                # Add question number
                text_x = max(10, int(final_positions[0]) - 50)
                text_y = int(y_pos) + 5
                cv2.putText(result_image, f"Q{q_num}", (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                return q_num + 1
            
            for bubble in col_bubbles:
                bubble_y = bubble['center'][1]
                
                # Group bubbles by Y position for question numbering
                if current_y is None or abs(bubble_y - current_y) < question_tolerance:
                    current_question_bubbles.append(bubble)
                    if current_y is None:
                        current_y = bubble_y
                else:
                    # Process the previous question group
                    question_number = process_question_with_learned_grid(current_question_bubbles, current_y, question_number, grid_positions)
                    
                    # Start new group
                    current_question_bubbles = [bubble]
                    current_y = bubble_y
            
            # Don't forget the last group in this column
            if current_question_bubbles:
                question_number = process_question_with_learned_grid(current_question_bubbles, current_y, question_number, grid_positions)
        
        # Add legend for colors
        legend_y = 30
        cv2.putText(result_image, "Roll", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, roll_number_color, 2)
        cv2.putText(result_image, "Q1", (70, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, column_colors[0], 2)
        cv2.putText(result_image, "Q2", (110, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, column_colors[1], 2)
        cv2.putText(result_image, "Q3", (150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, column_colors[2], 2)
        
        # Save step 8 result
        step8_path = self.results_dir / "step8_bubble_detection.jpg"
        cv2.imwrite(str(step8_path), result_image)
        print(f"  - Saved: {step8_path}")
        
        return {'roll_numbers': roll_number_bubbles, 'questions': question_bubbles_by_column}, result_image
    
    def process_image_stepwise(self, image_path):
        """Process image through steps 1-8"""
        print(f"Starting stepwise processing of: {image_path}")
        print("=" * 50)
        
        # Step 1: Load original
        original = self.step1_load_original(image_path)
        
        # Step 2: Convert to grayscale
        gray = self.step2_convert_grayscale(original)
        
        # Step 3: Noise reduction
        blurred = self.step3_noise_reduction(gray)
        
        # Step 4: Thresholding
        thresh = self.step4_thresholding(blurred)
        
        # Step 5: Detect contours
        contours = self.step5_detect_contours(thresh, original)
        
        # Step 6: Detect squares
        positioning_squares, grid_squares = self.step6_detect_squares(contours, original)
        
        # Step 7: Crop and correct perspective
        corrected = self.step7_crop_and_correct_perspective(original, positioning_squares)
        
        # Step 8: Detect answer bubbles
        bubble_data, bubble_image = self.step8_detect_answer_bubbles(corrected)
        
        print("=" * 50)
        print("Processing complete! Check results directory for step images.")
        
        roll_bubbles = bubble_data['roll_numbers']
        question_bubbles = bubble_data['questions']
        total_question_bubbles = sum(len(col) for col in question_bubbles)
        print(f"Summary: Detected {len(roll_bubbles)} roll number bubbles and {total_question_bubbles} question bubbles")
        
        return {
            'original': original,
            'gray': gray,
            'blurred': blurred,
            'thresh': thresh,
            'contours': contours,
            'positioning_squares': positioning_squares,
            'grid_squares': grid_squares,
            'corrected': corrected,
            'bubble_data': bubble_data,
            'bubble_image': bubble_image
        }

def main():
    import sys
    
    processor = OMRProcessor()
    
    # Get image filename from command line argument or default to 1.jpg
    if len(sys.argv) > 1:
        image_filename = sys.argv[1]
        # Add .jpg extension if not provided
        if not image_filename.endswith('.jpg'):
            image_filename += '.jpg'
    else:
        image_filename = "1.jpg"
    
    image_path = f"scans/{image_filename}"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print(f"Please place your scanned OMR sheet as '{image_path}'")
        return
    
    # Process the image stepwise (steps 1-6)
    result = processor.process_image_stepwise(image_path)

if __name__ == "__main__":
    main()