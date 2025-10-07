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
    
    def calculate_fill_threshold(self, image):
        """Calculate adaptive threshold for filled bubble detection based on image characteristics"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate image statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Adaptive threshold based on image brightness
        # For darker images (like 4roll.jpg), use a higher threshold
        # For brighter images (like 4rollpersp.jpg), use a lower threshold
        if mean_intensity > 180:  # Bright image
            threshold = 200
        elif mean_intensity > 150:  # Medium brightness
            threshold = 220
        else:  # Dark image
            threshold = 240
            
        print(f"  - Image mean intensity: {mean_intensity:.1f}, using fill threshold: {threshold}")
        return threshold
    
    def infer_missing_grid_positions(self, detected_circles, question_num):
        """Infer missing bubble positions based on detected circles in a perfect grid"""
        if len(detected_circles) < 2:
            return detected_circles
            
        # Sort circles by X position (left to right)
        circles_by_x = sorted(detected_circles, key=lambda c: c[0])
        
        # Calculate the expected spacing between bubbles
        if len(circles_by_x) >= 2:
            # Use the spacing between detected circles to infer grid
            spacing = (circles_by_x[-1][0] - circles_by_x[0][0]) / (len(circles_by_x) - 1)
            
            # Standard bubble spacing for OMR sheets (approximately 20 pixels)
            expected_spacing = 20
            
            # If spacing is too large, we might be missing bubbles in between
            if spacing > expected_spacing * 1.5:
                # Estimate how many bubbles we're missing
                missing_count = int(spacing / expected_spacing) - 1
                
                # Generate missing positions
                inferred_circles = list(detected_circles)
                
                for i in range(len(circles_by_x) - 1):
                    current_x, current_y = circles_by_x[i]
                    next_x, next_y = circles_by_x[i + 1]
                    
                    gap = next_x - current_x
                    if gap > expected_spacing * 1.5:
                        # Calculate number of missing bubbles in this gap
                        gap_missing = int(gap / expected_spacing) - 1
                        
                        # Add missing bubbles with interpolated positions
                        for j in range(1, gap_missing + 1):
                            ratio = j / (gap_missing + 1)
                            missing_x = current_x + ratio * gap
                            missing_y = current_y + ratio * (next_y - current_y)  # Interpolate Y too
                            inferred_circles.append((missing_x, missing_y))
                            print(f"    Q{question_num}: Inferred bubble at ({missing_x:.0f}, {missing_y:.0f})")
                
                # Sort final circles by X position
                inferred_circles.sort(key=lambda c: c[0])
                
                # Ensure we have exactly 4 bubbles (A, B, C, D)
                if len(inferred_circles) > 4:
                    # Keep the 4 most evenly spaced ones
                    step = len(inferred_circles) / 4
                    final_circles = []
                    for i in range(4):
                        idx = int(i * step)
                        final_circles.append(inferred_circles[idx])
                    return final_circles
                elif len(inferred_circles) == 4:
                    return inferred_circles
                else:
                    # Still missing some, add more based on pattern
                    while len(inferred_circles) < 4:
                        # Add at the end with consistent spacing
                        last_x, last_y = inferred_circles[-1]
                        new_x = last_x + expected_spacing
                        inferred_circles.append((new_x, last_y))
                    return inferred_circles[:4]  # Limit to 4
            else:
                # Spacing looks normal, just extend the pattern to 4 bubbles
                inferred_circles = list(detected_circles)
                
                # Calculate consistent spacing
                if len(circles_by_x) >= 2:
                    avg_spacing = spacing
                    avg_y = sum(c[1] for c in circles_by_x) / len(circles_by_x)
                    
                    # Extend pattern to left and right as needed
                    leftmost_x = circles_by_x[0][0]
                    rightmost_x = circles_by_x[-1][0]
                    
                    # Add bubbles to the left if needed
                    while leftmost_x > 0 and len(inferred_circles) < 4:
                        leftmost_x -= avg_spacing
                        inferred_circles.append((leftmost_x, avg_y))
                    
                    # Add bubbles to the right if needed
                    while len(inferred_circles) < 4:
                        rightmost_x += avg_spacing
                        inferred_circles.append((rightmost_x, avg_y))
                    
                    # Sort and return exactly 4 bubbles
                    inferred_circles.sort(key=lambda c: c[0])
                    return inferred_circles[:4]
        
        return detected_circles
    
    def filter_to_best_grid_circles(self, detected_circles, question_num):
        """Filter detected circles to the best 4 that match expected grid positions"""
        if len(detected_circles) <= 4:
            return detected_circles
            
        # Sort circles by X position (left to right)
        circles_by_x = sorted(detected_circles, key=lambda c: c[0])
        
        # Expected spacing for OMR bubbles (approximately 20 pixels)
        expected_spacing = 20
        
        # Method 1: Try to find 4 evenly spaced circles
        best_set = []
        best_score = float('inf')
        
        # Try different combinations of 4 circles
        from itertools import combinations
        for combo in combinations(circles_by_x, 4):
            # Calculate spacing consistency score
            combo_sorted = sorted(combo, key=lambda c: c[0])
            spacings = []
            for i in range(3):
                spacing = combo_sorted[i+1][0] - combo_sorted[i][0]
                spacings.append(spacing)
            
            # Score based on how close spacings are to expected and to each other
            avg_spacing = sum(spacings) / len(spacings)
            spacing_variance = sum((s - avg_spacing) ** 2 for s in spacings) / len(spacings)
            
            # Prefer spacings close to expected_spacing and low variance
            score = abs(avg_spacing - expected_spacing) + spacing_variance
            
            if score < best_score:
                best_score = score
                best_set = list(combo_sorted)
        
        if best_set:
            print(f"    Q{question_num}: Selected 4 circles with score {best_score:.1f}")
            return best_set
        
        # Fallback: just take the 4 most evenly distributed
        if len(circles_by_x) >= 4:
            step = len(circles_by_x) / 4
            result = []
            for i in range(4):
                idx = int(i * step)
                result.append(circles_by_x[idx])
            return result
            
        return detected_circles
    
    def create_standard_4_bubble_grid(self, existing_circles, grid_positions, question_num):
        """Create exactly 4 bubble positions using grid positions"""
        if len(grid_positions) == 4:
            # Use the learned grid positions
            avg_y = 0
            if existing_circles:
                avg_y = sum(c[1] for c in existing_circles) / len(existing_circles)
            else:
                # Estimate Y position based on question number (rough estimate)
                avg_y = 300 + (question_num - 1) * 23  # Approximate row spacing
            
            # Create 4 circles at grid X positions
            standard_circles = [(int(x), int(avg_y)) for x in grid_positions]
            print(f"    Q{question_num}: Created standard 4-bubble grid at Y={avg_y:.0f}")
            return standard_circles
        else:
            # Fallback if no proper grid learned
            if existing_circles:
                # Use existing circles to estimate spacing
                existing_circles.sort(key=lambda c: c[0])
                min_x = existing_circles[0][0]
                max_x = existing_circles[-1][0]
                avg_y = sum(c[1] for c in existing_circles) / len(existing_circles)
                
                # Create 4 evenly spaced positions
                gap = (max_x - min_x) / 3 if len(existing_circles) > 1 else 20
                standard_circles = [(int(min_x + i * gap), int(avg_y)) for i in range(4)]
            else:
                # Complete fallback - use estimated positions
                base_x = 65  # Default starting X
                avg_y = 300 + (question_num - 1) * 23
                standard_circles = [(base_x + i * 20, int(avg_y)) for i in range(4)]
            
            print(f"    Q{question_num}: Created fallback 4-bubble grid")
            return standard_circles
    
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
        
        # Find contours - use RETR_TREE so we can inspect hierarchy (anchors have holes)
        contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print(f"  - Found {len(contours)} total contours")

        # Draw all contours on original image
        result_image = original_image.copy()
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 1)
        
        # Save step 5 result
        step5_path = self.results_dir / "step5_all_contours.jpg"
        cv2.imwrite(str(step5_path), result_image)
        print(f"  - Saved: {step5_path}")
        
        return contours, hierarchy
    
    def step6_detect_squares(self, contours, hierarchy, original_image):
        """Step 6: Detect and classify squares"""
        print("Step 6: Detecting squares...")

        def square_to_tuple(square):
            return (square['x'], square['y'], square['w'], square['h'], square['area'])

        all_squares = []
        debug_info = {'total_contours': 0, 'area_filtered': 0, 'polygon_filtered': 0, 'aspect_filtered': 0}
        hierarchy_data = hierarchy[0] if hierarchy is not None else None

        for idx, contour in enumerate(contours):
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
                        has_hole = False

                        if hierarchy_data is not None:
                            child_idx = hierarchy_data[idx][2]
                            while child_idx != -1:
                                child_contour = contours[child_idx]
                                child_area = cv2.contourArea(child_contour)
                                if child_area > 10:
                                    has_hole = True
                                    break
                                child_idx = hierarchy_data[child_idx][0]

                        all_squares.append({
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                            'area': area,
                            'center_x': x + w // 2,
                            'center_y': y + h // 2,
                            'has_hole': has_hole
                        })

        print(f"  - Debug: {debug_info['total_contours']} contours -> {debug_info['area_filtered']} area OK -> {debug_info['polygon_filtered']} polygon OK -> {debug_info['aspect_filtered']} aspect OK")

        # Sort squares by area
        all_squares.sort(key=lambda s: s['area'], reverse=True)

        # Get image dimensions and calculate column positions
        img_height, img_width = original_image.shape[:2]

        # Define the 4 marker columns based on OMR sheet structure
        col1_x = img_width * 0.05   # Left edge (A1, G1-G3, A4)
        col2_x = img_width * 0.33   # Left-center (G4-G8) 
        col3_x = img_width * 0.67   # Right-center (G9-G13)
        col4_x = img_width * 0.95   # Right edge (A2, G14-G16, A3)
        
        tolerance = 120  # Further increased tolerance for perspective distortion
        
        print(f"  - Image size: {img_width}x{img_height}")
        print(f"  - Expected marker columns at x: {col1_x:.0f}, {col2_x:.0f}, {col3_x:.0f}, {col4_x:.0f}")
        
        # Separate squares by their column position
        positioning_squares = []
        grid_squares = []
        
        if len(all_squares) > 0:
            # Show all detected squares
            print(f"  - All detected squares by area:")
            for i, square in enumerate(all_squares):
                print(f"    {i+1}: ({square['x']}, {square['y']}) {square['w']}x{square['h']} area={square['area']:.0f} hole={square['has_hole']}")

            # Classify squares by column and position
            for square in all_squares:
                center_x = square['center_x']
                center_y = square['center_y']
                area = square['area']

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
                    print(f"    Rejected square at ({square['x']}, {square['y']}) - not in any marker column")
                    continue

                # Classify based on column and position
                is_top = center_y < img_height * 0.25
                is_bottom = center_y > img_height * 0.75

                if square['has_hole']:
                    if column == 1:
                        if is_top:
                            positioning_squares.append(square_to_tuple(square))
                            print(f"    Found A1 (top-left anchor with hollow center) at ({square['x']}, {square['y']})")
                        elif is_bottom:
                            positioning_squares.append(square_to_tuple(square))
                            print(f"    Found A4 (bottom-left anchor with hollow center) at ({square['x']}, {square['y']})")
                        else:
                            print(f"    Ignored hollow square in column 1 at ({square['x']}, {square['y']}) - unexpected vertical position")
                    elif column == 4:
                        if is_top:
                            positioning_squares.append(square_to_tuple(square))
                            print(f"    Found A2 (top-right anchor with hollow center) at ({square['x']}, {square['y']})")
                        elif is_bottom:
                            positioning_squares.append(square_to_tuple(square))
                            print(f"    Found A3 (bottom-right anchor with hollow center) at ({square['x']}, {square['y']})")
                        else:
                            print(f"    Ignored hollow square in column 4 at ({square['x']}, {square['y']}) - unexpected vertical position")
                    else:
                        print(f"    Ignored hollow square at ({square['x']}, {square['y']}) - anchors expected only in outer columns")
                else:
                    if column == 1:  # Left edge: G markers between anchors
                        if not is_top and not is_bottom and area > 40:
                            grid_squares.append(square_to_tuple(square))
                            print(f"    Found G marker (col1-middle) at ({square['x']}, {square['y']})")
                    elif column == 2:  # Left-center: G4-G8 (5 markers full height)
                        if area > 40:
                            grid_squares.append(square_to_tuple(square))
                            print(f"    Found G marker (col2) at ({square['x']}, {square['y']})")
                    elif column == 3:  # Right-center: G9-G13 (5 markers full height)
                        if area > 40:
                            grid_squares.append(square_to_tuple(square))
                            print(f"    Found G marker (col3) at ({square['x']}, {square['y']})")
                    elif column == 4:  # Right edge: G markers between anchors
                        if not is_top and not is_bottom and area > 40:
                            grid_squares.append(square_to_tuple(square))
                            print(f"    Found G marker (col4-middle) at ({square['x']}, {square['y']})")

        print(f"  - Total squares found: {len(all_squares)}")
        print(f"  - Positioning squares: {len(positioning_squares)}")
        print(f"  - Grid squares: {len(grid_squares)}")

        # If we don't have 4 positioning squares, try alternative detection
        if len(positioning_squares) < 4:
            print(f"  - Attempting alternative anchor detection...")
            fallback_anchors = self.detect_anchors_by_position(all_squares, img_width, img_height)
            positioning_squares = [square_to_tuple(square) for square in fallback_anchors]
            print(f"  - Alternative detection found: {len(positioning_squares)} positioning squares")
        
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
    
    def detect_anchors_by_position(self, all_squares, img_width, img_height):
        """Alternative anchor detection based on corner positions and size"""
        print("    - Looking for large squares in corner regions...")
        print(f"    - Image dimensions: {img_width}x{img_height}")

        # Define corner regions more liberally for perspective distortion
        corner_regions = {
            'top_left': {'x_min': 0, 'x_max': img_width * 0.25, 'y_min': 0, 'y_max': img_height * 0.25},
            'top_right': {'x_min': img_width * 0.75, 'x_max': img_width, 'y_min': 0, 'y_max': img_height * 0.25},
            'bottom_left': {'x_min': 0, 'x_max': img_width * 0.25, 'y_min': img_height * 0.75, 'y_max': img_height},
            'bottom_right': {'x_min': img_width * 0.75, 'x_max': img_width, 'y_min': img_height * 0.75, 'y_max': img_height}
        }
        
        # Debug: print region boundaries
        for region_name, region in corner_regions.items():
            print(f"    - {region_name} region: x={region['x_min']:.0f}-{region['x_max']:.0f}, y={region['y_min']:.0f}-{region['y_max']:.0f}")
        
        corner_anchors = {}

        for region_name, region in corner_regions.items():
            # Find largest square in this corner region
            candidates = []
            for square in all_squares:
                if not square.get('has_hole'):
                    continue

                x = square['x']
                y = square['y']
                w = square['w']
                h = square['h']
                area = square['area']
                center_x = square['center_x']
                center_y = square['center_y']

                # Debug: check large squares
                if area > 500:
                    print(f"    - Checking large square at ({x}, {y}) center=({center_x}, {center_y}) area={area:.0f} for region {region_name}")

                # Check if square is in this corner region and large enough
                if (region['x_min'] <= center_x <= region['x_max'] and
                    region['y_min'] <= center_y <= region['y_max'] and
                    area > 200):  # Large enough to be an anchor
                    candidates.append(square)
                    print(f"    - Added candidate for {region_name}: ({x}, {y}) area={area:.0f}")

            if candidates:
                # Pick the largest one in this region
                largest = max(candidates, key=lambda s: s['area'])  # Sort by area
                corner_anchors[region_name] = largest
                x = largest['x']
                y = largest['y']
                w = largest['w']
                h = largest['h']
                area = largest['area']
                print(f"    - Found {region_name} anchor at ({x}, {y}) area={area:.0f}")

        # Convert to positioning squares list
        positioning_squares = []
        for region_name in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            if region_name in corner_anchors:
                positioning_squares.append(corner_anchors[region_name])

        return positioning_squares
    
    def step7_crop_and_correct_perspective(self, original_image, positioning_squares):
        """Step 7: Crop and correct perspective using anchor points"""
        print("Step 7: Cropping and correcting perspective...")
        
        if len(positioning_squares) < 4:
            print(f"  - Warning: Need 4 positioning squares, found {len(positioning_squares)}")
            print("  - Attempting fallback cropping method...")
            return self.step7_fallback_crop(original_image, positioning_squares)
        
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
    
    def step7_fallback_crop(self, original_image, positioning_squares):
        """Fallback cropping method when perspective correction fails"""
        print("  - Using fallback cropping method...")
        
        img_height, img_width = original_image.shape[:2]
        
        if len(positioning_squares) >= 2:
            # Use available positioning squares to estimate crop area
            squares_x = [x for x, y, w, h, area in positioning_squares]
            squares_y = [y for x, y, w, h, area in positioning_squares]
            
            # Estimate margins based on detected squares
            left_margin = max(10, min(squares_x) - 50)
            top_margin = max(10, min(squares_y) - 50)
            right_margin = min(img_width - 10, max(squares_x) + 100)
            bottom_margin = min(img_height - 10, max(squares_y) + 100)
            
            print(f"  - Crop area: ({left_margin}, {top_margin}) to ({right_margin}, {bottom_margin})")
        else:
            # Conservative crop with standard margins
            margin_x = int(img_width * 0.05)
            margin_y = int(img_height * 0.05)
            left_margin = margin_x
            top_margin = margin_y
            right_margin = img_width - margin_x
            bottom_margin = img_height - margin_y
            
            print(f"  - Using conservative crop with 5% margins")
        
        # Crop the image
        cropped = original_image[top_margin:bottom_margin, left_margin:right_margin]
        
        # Save step 7 fallback results
        step7a_path = self.results_dir / "step7a_perspective_detection.jpg"
        vis_image = original_image.copy()
        cv2.rectangle(vis_image, (left_margin, top_margin), (right_margin, bottom_margin), (0, 255, 0), 3)
        cv2.putText(vis_image, "Fallback Crop Area", (left_margin, top_margin - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(str(step7a_path), vis_image)
        print(f"  - Saved fallback detection: {step7a_path}")
        
        step7b_path = self.results_dir / "step7b_corrected_crop.jpg"
        cv2.imwrite(str(step7b_path), cropped)
        print(f"  - Saved fallback crop: {step7b_path}")
        
        return cropped
    
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
        
    
    def step8_detect_answer_bubbles(self, corrected_image):
        """Step 8: Detect answer bubbles using the layout defined in the sheet generator"""
        print("Step 8: Detecting answer bubbles...")

        if len(corrected_image.shape) == 3:
            gray_corrected = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_corrected = corrected_image

        blurred = cv2.GaussianBlur(gray_corrected, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  - Found {len(contours)} contours in corrected image")

        img_height, img_width = corrected_image.shape[:2]

        def measure_mean_intensity(center, radius):
            x, y = center
            sample_radius = max(4, int(radius * 0.7))
            x1 = max(0, x - sample_radius)
            y1 = max(0, y - sample_radius)
            x2 = min(img_width, x + sample_radius)
            y2 = min(img_height, y + sample_radius)

            region = gray_corrected[y1:y2, x1:x2]
            if region.size == 0:
                return 255.0
            return float(np.mean(region))

        # Detect circular contours that correspond to bubbles
        bubble_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 700:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.55:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center = (int(x + w / 2), int(y + h / 2))
            radius = (w + h) / 4.0

            duplicate = False
            for existing in bubble_candidates:
                if abs(existing['center'][0] - center[0]) < 6 and abs(existing['center'][1] - center[1]) < 6:
                    duplicate = True
                    break
            if duplicate:
                continue

            mean_intensity = measure_mean_intensity(center, radius)

            bubble_candidates.append({
                'center': center,
                'radius': radius,
                'area': area,
                'mean_intensity': mean_intensity
            })

        print(f"  - Bubble candidates after filtering: {len(bubble_candidates)}")

        if not bubble_candidates:
            step8_path = self.results_dir / "step8_bubble_detection.jpg"
            cv2.imwrite(str(step8_path), corrected_image)
            print(f"  - Saved: {step8_path}")
            return {'roll_numbers': {'bubbles': [], 'detected_digits': [None, None, None]}, 'questions': []}, corrected_image

        # Cluster bubbles into the three answer regions (roll/questions column, middle, right)
        x_samples = np.float32([[b['center'][0]] for b in bubble_candidates])
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
            _, _, centers = cv2.kmeans(x_samples, 3, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
            column_centers = sorted([float(c[0]) for c in centers])
        except cv2.error:
            column_centers = [img_width * 0.19, img_width * 0.50, img_width * 0.81]

        print(f"  - Column centers: {[f'{c:.1f}' for c in column_centers]}")

        for bubble in bubble_candidates:
            distances = [abs(bubble['center'][0] - c) for c in column_centers]
            bubble['column_index'] = int(np.argmin(distances))

        def estimate_row_spacing(values):
            if len(values) < 2:
                return 25.0
            values = sorted(values)
            diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            diffs = [d for d in diffs if 5 < d < 80]
            if not diffs:
                return 25.0
            return float(np.median(diffs))

        def group_rows(bubbles, tolerance):
            rows = []
            for bubble in sorted(bubbles, key=lambda b: b['center'][1]):
                assigned = False
                for row in rows:
                    if abs(bubble['center'][1] - row['y']) <= tolerance:
                        row['bubbles'].append(bubble)
                        row['y_values'].append(bubble['center'][1])
                        row['y'] = float(np.mean(row['y_values']))
                        assigned = True
                        break
                if not assigned:
                    rows.append({'y': float(bubble['center'][1]), 'y_values': [bubble['center'][1]], 'bubbles': [bubble]})
            return rows

        first_column_bubbles = [b for b in bubble_candidates if b['column_index'] == 0]
        base_spacing = estimate_row_spacing([b['center'][1] for b in first_column_bubbles])
        row_tolerance = max(8.0, base_spacing * 0.45)

        print(f"  - Estimated row spacing: {base_spacing:.2f} (tolerance {row_tolerance:.2f})")

        fill_threshold = self.calculate_fill_threshold(gray_corrected)
        print(f"  - Using fill threshold: {fill_threshold}")

        roll_data = {'bubbles': [], 'detected_digits': [None, None, None]}
        questions = []

        column_colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255)
        ]
        roll_color = (255, 255, 0)
        debug_image = corrected_image.copy()

        question_number = 1

        for column_index in range(3):
            column_bubbles = [b for b in bubble_candidates if b['column_index'] == column_index]
            if not column_bubbles:
                continue

            rows = group_rows(column_bubbles, row_tolerance)

            if column_index == 0:
                roll_rows = rows[:min(10, len(rows))]
                question_rows = rows[min(10, len(rows)):]

                for digit_value, row in enumerate(roll_rows):
                    row_bubbles = sorted(row['bubbles'], key=lambda b: b['center'][0])[:3]
                    for digit_index, bubble in enumerate(row_bubbles):
                        filled = bubble['mean_intensity'] < fill_threshold
                        roll_data['bubbles'].append({
                            'center': bubble['center'],
                            'radius': bubble['radius'],
                            'digit_index': digit_index,
                            'digit_value': digit_value,
                            'filled': filled,
                            'mean_intensity': bubble['mean_intensity']
                        })

                        draw_color = roll_color if filled else (200, 200, 0)
                        cv2.circle(debug_image, bubble['center'], int(max(6, bubble['radius'])), draw_color, 2)
                        if filled:
                            cv2.circle(debug_image, bubble['center'], 3, draw_color, -1)

                        if filled and roll_data['detected_digits'][digit_index] is None:
                            roll_data['detected_digits'][digit_index] = digit_value

                rows_to_process = question_rows
            else:
                rows_to_process = rows

            for row in rows_to_process:
                row_bubbles = sorted(row['bubbles'], key=lambda b: b['center'][0])
                if not row_bubbles:
                    continue

                choices = []
                for choice_idx, bubble in enumerate(row_bubbles[:4]):
                    filled = bubble['mean_intensity'] < fill_threshold
                    choice_label = chr(ord('A') + choice_idx)
                    choices.append({
                        'choice': choice_label,
                        'center': bubble['center'],
                        'radius': bubble['radius'],
                        'filled': filled,
                        'mean_intensity': bubble['mean_intensity']
                    })

                    draw_color = column_colors[column_index]
                    cv2.circle(debug_image, bubble['center'], int(max(6, bubble['radius'])), draw_color, 2)
                    if filled:
                        cv2.circle(debug_image, bubble['center'], 3, draw_color, -1)

                if not choices:
                    continue

                filled_choices = [c['choice'] for c in choices if c['filled']]
                if filled_choices:
                    label_text = f"Q{question_number}:" + "/".join(filled_choices)
                else:
                    label_text = f"Q{question_number}"

                text_position = (int(choices[0]['center'][0] - 30), int(row['y'] + 5))
                cv2.putText(debug_image, label_text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, column_colors[column_index], 1, cv2.LINE_AA)

                questions.append({
                    'question_number': question_number,
                    'column_index': column_index,
                    'center_y': row['y'],
                    'choices': choices
                })

                question_number += 1

        detected_roll = ''.join(str(d) if d is not None else '?' for d in roll_data['detected_digits'])
        print(f"  - Detected roll digits: {detected_roll}")
        print(f"  - Total questions detected: {len(questions)}")

        step8_path = self.results_dir / "step8_bubble_detection.jpg"
        cv2.imwrite(str(step8_path), debug_image)
        print(f"  - Saved: {step8_path}")

        return {'roll_numbers': roll_data, 'questions': questions}, debug_image

    def step9_draw_section_rectangles(self, corrected_image, bubble_data):
        """Step 9: Draw rectangles around roll number and question sections"""
        print("Step 9: Drawing section rectangles...")

        result_image = corrected_image.copy()
        img_height, img_width = corrected_image.shape[:2]

        roll_data = bubble_data['roll_numbers']
        questions = bubble_data['questions']

        roll_color = (255, 255, 0)
        column_colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255)
        ]
        column_names = [
            "Questions Col 1",
            "Questions Col 2",
            "Questions Col 3"
        ]

        roll_bubbles = roll_data['bubbles']
        if roll_bubbles:
            roll_x = [b['center'][0] for b in roll_bubbles]
            roll_y = [b['center'][1] for b in roll_bubbles]
            margin = 20
            left = max(0, min(roll_x) - margin)
            right = min(img_width, max(roll_x) + margin)
            top = max(0, min(roll_y) - margin)
            bottom = min(img_height, max(roll_y) + margin)

            cv2.rectangle(result_image, (left, top), (right, bottom), roll_color, 3)
            cv2.putText(result_image, "Roll Number", (left, max(15, top) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, roll_color, 2)

            for bubble in roll_bubbles:
                radius = int(max(6, bubble['radius']))
                cv2.circle(result_image, bubble['center'], radius, roll_color, 2)

            detected_roll = ''.join(str(d) if d is not None else '?' for d in roll_data['detected_digits'])
            print(f"  - Roll number bounding box: ({left}, {top}) to ({right}, {bottom})")
            print(f"  - Detected roll digits: {detected_roll}")
        else:
            print("  - No roll number bubbles detected")

        column_points = {0: [], 1: [], 2: []}
        for question in questions:
            column_index = question['column_index']
            for choice in question['choices']:
                column_points[column_index].append(choice['center'])

        for col_index in range(3):
            points = column_points[col_index]
            if not points:
                print(f"  - {column_names[col_index]}: no bubbles detected")
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            margin = 25
            left = max(0, min(xs) - margin)
            right = min(img_width, max(xs) + margin)
            top = max(0, min(ys) - margin)
            bottom = min(img_height, max(ys) + margin)

            color = column_colors[col_index]
            cv2.rectangle(result_image, (left, top), (right, bottom), color, 3)
            cv2.putText(result_image, column_names[col_index], (left, max(15, top) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            for question in [q for q in questions if q['column_index'] == col_index]:
                for choice in question['choices']:
                    radius = int(max(6, choice['radius']))
                    cv2.circle(result_image, choice['center'], radius, color, 2)

            print(f"  - {column_names[col_index]}: ({left}, {top}) to ({right}, {bottom})")

        step9_path = self.results_dir / "step9_section_rectangles.jpg"
        cv2.imwrite(str(step9_path), result_image)
        print(f"  - Saved: {step9_path}")

        return result_image

    def step10_show_filled_bubbles_only(self, corrected_image, bubble_data):
        """Step 10: Highlight only filled bubbles"""
        print("Step 10: Showing only filled bubbles...")

        result_image = corrected_image.copy()
        if len(corrected_image.shape) == 3:
            gray_corrected = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_corrected = corrected_image

        img_height, img_width = corrected_image.shape[:2]

        roll_data = bubble_data['roll_numbers']
        questions = bubble_data['questions']

        roll_color = (255, 255, 0)
        column_colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255)
        ]

        roll_bubbles = roll_data['bubbles']
        filled_roll = 0
        if roll_bubbles:
            roll_x = [b['center'][0] for b in roll_bubbles]
            roll_y = [b['center'][1] for b in roll_bubbles]
            margin = 20
            left = max(0, min(roll_x) - margin)
            right = min(img_width, max(roll_x) + margin)
            top = max(0, min(roll_y) - margin)
            bottom = min(img_height, max(roll_y) + margin)

            cv2.rectangle(result_image, (left, top), (right, bottom), roll_color, 3)
            cv2.putText(result_image, "Roll Number", (left, max(15, top) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, roll_color, 2)

            for bubble in roll_bubbles:
                if bubble['filled']:
                    radius = int(max(6, bubble['radius']))
                    cv2.circle(result_image, bubble['center'], radius, roll_color, 2)
                    cv2.circle(result_image, bubble['center'], 3, roll_color, -1)
                    filled_roll += 1

            detected_roll = ''.join(str(d) if d is not None else '?' for d in roll_data['detected_digits'])
            print(f"  - Roll number section: {filled_roll} filled bubbles")
            print(f"  - Detected roll number: {detected_roll}")
        else:
            print("  - No roll number bubbles detected")

        filled_answers = {}
        column_points = {0: [], 1: [], 2: []}

        for question in questions:
            column_index = question['column_index']
            color = column_colors[column_index]
            filled_choices = [choice for choice in question['choices'] if choice['filled']]

            for choice in question['choices']:
                column_points[column_index].append(choice['center'])

            if filled_choices:
                filled_answers[question['question_number']] = [c['choice'] for c in filled_choices]
                for choice in filled_choices:
                    radius = int(max(6, choice['radius']))
                    cv2.circle(result_image, choice['center'], radius, color, 2)
                    cv2.circle(result_image, choice['center'], 3, color, -1)
                    label = f"Q{question['question_number']}:{choice['choice']}"
                    text_pos = (choice['center'][0] + radius + 5, choice['center'][1] + 5)
                    cv2.putText(result_image, label, text_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        for col_index in range(3):
            points = column_points[col_index]
            if not points:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            margin = 25
            left = max(0, min(xs) - margin)
            right = min(img_width, max(xs) + margin)
            top = max(0, min(ys) - margin)
            bottom = min(img_height, max(ys) + margin)

            color = column_colors[col_index]
            cv2.rectangle(result_image, (left, top), (right, bottom), color, 3)
            cv2.putText(result_image, column_names[col_index], (left, max(15, top) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if filled_answers:
            print("  - Detected answers:")
            for q_num in sorted(filled_answers.keys()):
                choices = ', '.join(filled_answers[q_num])
                print(f"    Question {q_num}: {choices}")
        else:
            print("  - No filled answers detected")

        step10_path = self.results_dir / "step10_filled_bubbles_only.jpg"
        cv2.imwrite(str(step10_path), result_image)
        print(f"  - Saved: {step10_path}")

        return result_image

    def process_image_stepwise(self, image_path):
        """Process image through steps 1-10"""
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
        contours, hierarchy = self.step5_detect_contours(thresh, original)

        # Step 6: Detect squares
        positioning_squares, grid_squares = self.step6_detect_squares(contours, hierarchy, original)
        
        # Step 7: Crop and correct perspective
        corrected = self.step7_crop_and_correct_perspective(original, positioning_squares)
        
        # Step 8: Detect answer bubbles
        bubble_data, bubble_image = self.step8_detect_answer_bubbles(corrected)
        
        # Step 9: Draw section rectangles
        section_image = self.step9_draw_section_rectangles(corrected, bubble_data)
        
        # Step 10: Show only filled bubbles
        filled_image = self.step10_show_filled_bubbles_only(corrected, bubble_data)
        
        print("=" * 50)
        print("Processing complete! Check results directory for step images.")
        
        roll_bubble_count = len(bubble_data['roll_numbers']['bubbles'])
        question_choice_count = sum(len(q['choices']) for q in bubble_data['questions'])
        question_count = len(bubble_data['questions'])
        print(f"Summary: Detected {roll_bubble_count} roll number bubbles and {question_choice_count} answer bubbles across {question_count} questions")
        
        return {
            'original': original,
            'gray': gray,
            'blurred': blurred,
            'thresh': thresh,
            'contours': contours,
            'hierarchy': hierarchy,
            'positioning_squares': positioning_squares,
            'grid_squares': grid_squares,
            'corrected': corrected,
            'bubble_data': bubble_data,
            'bubble_image': bubble_image,
            'section_image': section_image,
            'filled_image': filled_image
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
        image_filename = "4roll.jpg"
    
    image_path = f"scans/{image_filename}"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print(f"Please place your scanned OMR sheet as '{image_path}'")
        return
    
    # Process the image stepwise (steps 1-6)
    result = processor.process_image_stepwise(image_path)

if __name__ == "__main__":
    main()