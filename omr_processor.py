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
        
        # Find contours - use RETR_LIST to detect all individual shapes
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
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
        
        tolerance = 120  # Further increased tolerance for perspective distortion
        
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
                is_top = center_y < img_height * 0.25
                is_bottom = center_y > img_height * 0.75
                
                if column == 1:  # Left edge: A1(top), G1-G3(middle), A4(bottom)
                    if is_top and area > 120:  # More tolerant for perspective distortion
                        positioning_squares.append(square)
                        print(f"    Found A1 (top-left) at ({x}, {y})")
                    elif is_bottom and area > 120:  # More tolerant for perspective distortion
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
                    if is_top and area > 120:  # More tolerant for perspective distortion
                        positioning_squares.append(square)
                        print(f"    Found A2 (top-right) at ({x}, {y})")
                    elif is_bottom and area > 120:  # More tolerant for perspective distortion
                        positioning_squares.append(square)
                        print(f"    Found A3 (bottom-right) at ({x}, {y})")
                    elif not is_top and not is_bottom and area > 60:
                        grid_squares.append(square)
                        print(f"    Found G marker (col4-middle) at ({x}, {y})")
        
        print(f"  - Total squares found: {len(all_squares)}")
        print(f"  - Positioning squares: {len(positioning_squares)}")
        print(f"  - Grid squares: {len(grid_squares)}")
        
        # If we don't have 4 positioning squares, try alternative detection
        if len(positioning_squares) < 4:
            print(f"  - Attempting alternative anchor detection...")
            positioning_squares = self.detect_anchors_by_position(all_squares, img_width, img_height)
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
                x, y, w, h, area = square
                center_x = x + w // 2
                center_y = y + h // 2
                
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
                largest = max(candidates, key=lambda s: s[4])  # Sort by area
                corner_anchors[region_name] = largest
                x, y, w, h, area = largest
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
        """Step 8: Detect answer bubbles in the corrected image (improved)"""
        print("Step 8: Detecting answer bubbles...")
        
        # Convert corrected image to grayscale for processing
        if len(corrected_image.shape) == 3:
            gray_corrected = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_corrected = corrected_image
        
        # Enhanced preprocessing for better bubble detection
        # Use multiple techniques to ensure robust circle detection
        blurred = cv2.GaussianBlur(gray_corrected, (5, 5), 0)
        
        # Method 1: Contour-based detection (existing)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours - use RETR_LIST to detect all individual shapes
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Method 2: HoughCircles detection with multiple parameter sets for better coverage
        all_circles = []
        
        # Detect if this might be a screen capture by checking for high circle density
        test_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 15,
                                       param1=40, param2=20, minRadius=7, maxRadius=12)
        is_screen_capture = test_circles is not None and len(test_circles[0]) > 200
        
        if test_circles is not None:
            print(f"  - Initial circle detection found {len(test_circles[0])} circles")
            if is_screen_capture:
                print(f"  - High circle count detected - likely screen capture")
        
        if is_screen_capture:
            print("  - Detected screen capture - using adaptive parameters")
            # Moderately restrictive parameters for screen captures to balance noise reduction with bubble detection
            circles1 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 18,
                                       param1=50, param2=25, minRadius=7, maxRadius=12)
            if circles1 is not None:
                all_circles.extend(circles1[0])
            
            # Additional parameter set for screen captures with different sensitivities
            circles2 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 16,
                                       param1=45, param2=22, minRadius=8, maxRadius=13)
            if circles2 is not None:
                all_circles.extend(circles2[0])
        else:
            # Original parameters for regular scans
            # Parameter set 1: For well-defined empty circles
            circles1 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 15,
                                       param1=40, param2=20, minRadius=7, maxRadius=12)
            if circles1 is not None:
                all_circles.extend(circles1[0])
            
            # Parameter set 2: For filled/darker circles (more sensitive)
            circles2 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 15,
                                       param1=30, param2=15, minRadius=6, maxRadius=13)
            if circles2 is not None:
                all_circles.extend(circles2[0])
            
            # Parameter set 3: Very sensitive for missed filled circles
            circles3 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 12,
                                       param1=25, param2=12, minRadius=5, maxRadius=14)
            if circles3 is not None:
                all_circles.extend(circles3[0])
        
        # Convert back to the expected format
        if all_circles:
            circles = np.array([all_circles])
        else:
            circles = None
        
        print(f"  - Found {len(contours)} total contours in corrected image")
        if circles is not None:
            print(f"  - Found {len(circles[0])} circles via HoughCircles")
        
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
        
        # Combined bubble detection using both contours and HoughCircles
        bubble_candidates = []
        detected_positions = set()  # Track positions to avoid duplicates
        
        # Method 1: Process contours
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Adjust constraints based on whether this is a screen capture
            if is_screen_capture:
                # Much stricter constraints for screen captures to avoid noise
                area_min, area_max = 180, 350
                circularity_min = 0.80
            else:
                # Relaxed constraints for regular scans
                area_min, area_max = 150, 400
                circularity_min = 0.65
            
            if area_min < area < area_max:
                # Check circularity - bubbles should be reasonably circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > circularity_min:
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
                            pos_key = (center_x // 5, center_y // 5)  # Group nearby positions
                            if pos_key not in detected_positions:
                                detected_positions.add(pos_key)
                                bubble_candidates.append({
                                    'center': (center_x, center_y),
                                    'area': area,
                                    'circularity': circularity,
                                    'column': column_index,
                                    'contour': contour,
                                    'method': 'contour'
                                })
        
        # Method 2: Process HoughCircles (these are often more accurate for perfect circles)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (center_x, center_y, radius) in circles:
                # Calculate area from radius
                area = np.pi * radius * radius
                
                # Adjust area filtering based on screen capture detection
                if is_screen_capture:
                    area_valid = 180 < area < 350  # Stricter for screen captures
                else:
                    area_valid = 150 < area < 400  # Relaxed for regular scans
                    
                if area_valid:
                    # Check if circle is in one of the expected columns
                    in_column = False
                    column_index = -1
                    
                    for i, col_x in enumerate(all_columns):
                        if abs(center_x - col_x) < column_tolerance:
                            in_column = True
                            column_index = i
                            break
                    
                    if in_column:
                        pos_key = (center_x // 5, center_y // 5)  # Group nearby positions
                        if pos_key not in detected_positions:
                            detected_positions.add(pos_key)
                            bubble_candidates.append({
                                'center': (center_x, center_y),
                                'area': area,
                                'circularity': 1.0,  # HoughCircles are perfect circles
                                'column': column_index,
                                'contour': None,
                                'method': 'hough'
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
            
            # Analyze X positions from questions - focus on well-formed questions
            all_x_positions = []
            for q_bubbles in questions:
                if 3 <= len(q_bubbles) <= 6:  # Well-formed questions (not too few, not too many)
                    q_bubbles.sort(key=lambda b: b['center'][0])
                    # Take only the 4 most evenly spaced bubbles if more than 4
                    if len(q_bubbles) > 4:
                        # Select 4 evenly distributed bubbles
                        step = (len(q_bubbles) - 1) / 3
                        selected_bubbles = [q_bubbles[int(i * step)] for i in range(4)]
                        x_positions = [b['center'][0] for b in selected_bubbles]
                    else:
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
                
                # Use bubbles already detected in step 8 for this question row
                detected_circles = []
                for bubble in bubbles:
                    bx, by = bubble['center']
                    if abs(by - y_pos) < 12:  # Within the question row
                        detected_circles.append((bx, by))
                
                # Remove duplicates
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
                
                # Enforce exactly 4 bubbles per question (A, B, C, D)
                if len(unique_circles) != 4:
                    if len(unique_circles) < 4:
                        # Too few circles - infer missing positions based on grid
                        if len(unique_circles) >= 2:
                            inferred_circles = self.infer_missing_grid_positions(unique_circles, q_num)
                            if inferred_circles and len(inferred_circles) == 4:
                                print(f"    Q{q_num}: Inferred {len(inferred_circles) - len(unique_circles)} missing circles from grid")
                                unique_circles = inferred_circles
                            else:
                                # Fallback: create 4 evenly spaced circles based on detected ones
                                unique_circles = self.create_standard_4_bubble_grid(unique_circles, grid, q_num)
                        else:
                            # Too few circles to infer - create standard grid
                            unique_circles = self.create_standard_4_bubble_grid([], grid, q_num)
                    else:
                        # Too many circles - always select exactly the best 4
                        filtered_circles = self.filter_to_best_grid_circles(unique_circles, q_num)
                        if filtered_circles and len(filtered_circles) == 4:
                            print(f"    Q{q_num}: Filtered {len(unique_circles)} circles down to exactly 4 best grid matches")
                            unique_circles = filtered_circles
                        else:
                            # Force to 4 by taking the 4 most evenly distributed
                            unique_circles.sort(key=lambda c: c[0])  # Sort by X
                            if len(unique_circles) > 4:
                                step = (len(unique_circles) - 1) / 3
                                unique_circles = [unique_circles[int(i * step)] for i in range(4)]
                                print(f"    Q{q_num}: Forced to 4 evenly distributed circles")
                
                # Ensure we always have exactly 4 circles
                if len(unique_circles) != 4:
                    print(f"    Q{q_num}: Warning - Still don't have 4 circles, creating standard grid")
                    unique_circles = self.create_standard_4_bubble_grid(unique_circles, grid, q_num)
                
                # Map detected circles to grid positions
                bubble_to_grid_mapping = {}
                for cx, cy in unique_circles:
                    best_grid_idx = None
                    best_distance = float('inf')
                    
                    for i, grid_x in enumerate(grid):
                        distance = abs(cx - grid_x)
                        # Use adaptive tolerance - stricter for well-detected grids, looser for sparse ones
                        adaptive_tolerance = max(tolerance, 25) if len(unique_circles) < 4 else tolerance
                        if distance < adaptive_tolerance and distance < best_distance:
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
                actual_circle_positions = []
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
                if len(actual_circle_positions) == 0:
                    print(f"    Q{q_num}: Warning - No circles mapped to grid! Grid: {[f'{g:.0f}' for g in grid]}, Circles: {[(int(cx), int(cy)) for cx, cy in unique_circles]}, Tolerance: {tolerance:.0f}")
                
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
                
                # Only draw circles that were actually detected
                for i in detected_positions_dict:
                    actual_x, circle_type, bubble_data = detected_positions_dict[i]
                    actual_center = (int(actual_x), int(y_pos))
                    
                    # Use consistent radius for all bubbles (8 pixels)
                    radius = 8
                    
                    if circle_type == 'filled' and bubble_data:
                        # Draw filled bubble with filled center
                        cv2.circle(result_image, actual_center, radius, color, 2)
                        cv2.circle(result_image, actual_center, 3, color, -1)  # Slightly smaller filled center
                    else:
                        # Draw empty circle
                        cv2.circle(result_image, actual_center, radius, color, 2)
                
                # Add question number text next to the bubble set
                if detected_positions_dict:
                    # Find the leftmost bubble position to place the question number
                    min_x = min(actual_x for actual_x, _, _ in detected_positions_dict.values())
                    text_position = (int(min_x - 25), int(y_pos + 5))  # Place text to the left of bubbles
                    cv2.putText(result_image, f"Q{q_num}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, color, 1, cv2.LINE_AA)
                
                
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
        
        
        # Save step 8 result
        step8_path = self.results_dir / "step8_bubble_detection.jpg"
        cv2.imwrite(str(step8_path), result_image)
        print(f"  - Saved: {step8_path}")
        
        return {'roll_numbers': roll_number_bubbles, 'questions': question_bubbles_by_column}, result_image
    
    def step9_draw_section_rectangles(self, corrected_image, bubble_data):
        """Step 9: Draw rectangles around each section (roll number, question columns)"""
        print("Step 9: Drawing section rectangles...")
        
        result_image = corrected_image.copy()
        img_height, img_width = corrected_image.shape[:2]
        
        roll_bubbles = bubble_data['roll_numbers']
        question_bubbles_by_column = bubble_data['questions']
        
        # Define section colors
        roll_color = (255, 255, 0)      # Cyan for roll number section
        col1_color = (0, 255, 0)        # Green for questions column 1
        col2_color = (255, 0, 0)        # Blue for questions column 2  
        col3_color = (0, 0, 255)        # Red for questions column 3
        
        # Draw roll number section rectangle and bubbles
        if roll_bubbles:
            roll_x_coords = [b['center'][0] for b in roll_bubbles]
            roll_y_coords = [b['center'][1] for b in roll_bubbles]
            
            # Add margin around roll number bubbles
            margin = 20
            roll_left = max(0, min(roll_x_coords) - margin)
            roll_right = min(img_width, max(roll_x_coords) + margin)
            roll_top = max(0, min(roll_y_coords) - margin)
            roll_bottom = min(img_height, max(roll_y_coords) + margin)
            
            cv2.rectangle(result_image, (roll_left, roll_top), (roll_right, roll_bottom), roll_color, 3)
            cv2.putText(result_image, "Roll Number", (roll_left, roll_top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, roll_color, 2)
            
            # Draw roll number bubbles with consistent size
            for bubble in roll_bubbles:
                center = bubble['center']
                radius = 8  # Consistent radius for all bubbles
                cv2.circle(result_image, center, radius, roll_color, 2)
            
            print(f"  - Roll number section: ({roll_left}, {roll_top}) to ({roll_right}, {roll_bottom})")
        
        # Draw question column rectangles
        column_colors = [col1_color, col2_color, col3_color]
        column_names = ["Questions Col 1", "Questions Col 2", "Questions Col 3"]
        
        for col_idx, (col_bubbles, color, name) in enumerate(zip(question_bubbles_by_column, column_colors, column_names)):
            if col_bubbles:
                col_x_coords = [b['center'][0] for b in col_bubbles]
                col_y_coords = [b['center'][1] for b in col_bubbles]
                
                # Add margin around question bubbles
                margin = 25
                col_left = max(0, min(col_x_coords) - margin)
                col_right = min(img_width, max(col_x_coords) + margin)
                col_top = max(0, min(col_y_coords) - margin)
                col_bottom = min(img_height, max(col_y_coords) + margin)
                
                cv2.rectangle(result_image, (col_left, col_top), (col_right, col_bottom), color, 3)
                cv2.putText(result_image, name, (col_left, col_top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw question bubbles with consistent size
                for bubble in col_bubbles:
                    center = bubble['center']
                    radius = 8  # Consistent radius for all bubbles
                    cv2.circle(result_image, center, radius, color, 2)
                
                print(f"  - {name}: ({col_left}, {col_top}) to ({col_right}, {col_bottom})")
        
        # Save step 9 result
        step9_path = self.results_dir / "step9_section_rectangles.jpg"
        cv2.imwrite(str(step9_path), result_image)
        print(f"  - Saved: {step9_path}")
        
        return result_image
    
    def step10_show_filled_bubbles_only(self, corrected_image, bubble_data):
        """Step 10: Show only filled bubbles with section rectangles"""
        print("Step 10: Showing only filled bubbles...")
        
        result_image = corrected_image.copy()
        img_height, img_width = corrected_image.shape[:2]
        
        roll_bubbles = bubble_data['roll_numbers']
        question_bubbles_by_column = bubble_data['questions']
        
        # Define section colors (same as step 9)
        roll_color = (255, 255, 0)      # Cyan for roll number section
        col1_color = (0, 255, 0)        # Green for questions column 1
        col2_color = (255, 0, 0)        # Blue for questions column 2  
        col3_color = (0, 0, 255)        # Red for questions column 3
        
        # Draw roll number section rectangle and only filled bubbles
        filled_roll_count = 0
        detected_roll_digits = [None, None, None]  # For 3 digit positions
        if roll_bubbles:
            roll_x_coords = [b['center'][0] for b in roll_bubbles]
            roll_y_coords = [b['center'][1] for b in roll_bubbles]
            
            # Add margin around roll number bubbles
            margin = 20
            roll_left = max(0, min(roll_x_coords) - margin)
            roll_right = min(img_width, max(roll_x_coords) + margin)
            roll_top = max(0, min(roll_y_coords) - margin)
            roll_bottom = min(img_height, max(roll_y_coords) + margin)
            
            cv2.rectangle(result_image, (roll_left, roll_top), (roll_right, roll_bottom), roll_color, 3)
            cv2.putText(result_image, "Roll Number", (roll_left, roll_top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, roll_color, 2)
            
            # Analyze roll number bubbles to extract digits
            # First, organize bubbles by their digit column and row position
            roll_bubbles_by_digit = [[], [], []]  # 3 digit columns
            
            # Determine digit column positions (based on step 8 logic)
            roll_bubbles_x = [b['center'][0] for b in roll_bubbles]
            min_x, max_x = min(roll_bubbles_x), max(roll_bubbles_x)
            digit_columns = [
                min_x + (max_x - min_x) * 0.2,   # Hundreds digit
                min_x + (max_x - min_x) * 0.5,   # Tens digit  
                min_x + (max_x - min_x) * 0.8    # Units digit
            ]
            
            # Group bubbles by digit column
            digit_tolerance = 15
            for bubble in roll_bubbles:
                bx = bubble['center'][0]
                for i, col_x in enumerate(digit_columns):
                    if abs(bx - col_x) < digit_tolerance:
                        roll_bubbles_by_digit[i].append(bubble)
                        break
            
            # Process each digit column to find filled bubbles
            for digit_idx, digit_bubbles in enumerate(roll_bubbles_by_digit):
                digit_bubbles.sort(key=lambda b: b['center'][1])  # Sort by Y position (top to bottom)
                
                for row_idx, bubble in enumerate(digit_bubbles):
                    center = bubble['center']
                    area = bubble['area']
                    
                    # Check if this bubble is filled
                    x, y = center
                    region_size = 8
                    x1, y1 = max(0, x - region_size), max(0, y - region_size)
                    x2, y2 = min(img_width, x + region_size), min(img_height, y + region_size)
                    
                    if len(corrected_image.shape) == 3:
                        gray_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_image = corrected_image
                    
                    bubble_region = gray_image[y1:y2, x1:x2]
                    if bubble_region.size > 0:
                        mean_intensity = np.mean(bubble_region)
                        # Filled bubbles are darker (lower intensity)
                        if mean_intensity < 200:  # Threshold for filled bubbles
                            radius = 8  # Consistent radius for all bubbles
                            cv2.circle(result_image, center, radius, roll_color, 2)
                            cv2.circle(result_image, center, 3, roll_color, -1)  # Filled center
                            filled_roll_count += 1
                            
                            # Record the digit (row_idx corresponds to digit 0-9)
                            if row_idx < 10:  # Only digits 0-9
                                detected_roll_digits[digit_idx] = row_idx
                                
                                # Add digit label next to filled bubble
                                digit_names = ["Hundreds", "Tens", "Units"]
                                label_text = f"{digit_names[digit_idx]}:{row_idx}"
                                label_x = center[0] + radius + 5
                                label_y = center[1] + 5
                                cv2.putText(result_image, label_text, (label_x, label_y), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, roll_color, 1)
            
            # Construct roll number from detected digits
            roll_number = ""
            for digit in detected_roll_digits:
                if digit is not None:
                    roll_number += str(digit)
                else:
                    roll_number += "?"
            
            print(f"  - Roll number section: {filled_roll_count} filled bubbles")
            print(f"  - Detected roll number: {roll_number}")
        
        # Draw question column rectangles and only filled bubbles
        column_colors = [col1_color, col2_color, col3_color]
        column_names = ["Questions Col 1", "Questions Col 2", "Questions Col 3"]
        detected_answers = {}  # Store detected answers by question number
        
        for col_idx, (col_bubbles, color, name) in enumerate(zip(question_bubbles_by_column, column_colors, column_names)):
            filled_col_count = 0
            if col_bubbles:
                col_x_coords = [b['center'][0] for b in col_bubbles]
                col_y_coords = [b['center'][1] for b in col_bubbles]
                
                # Add margin around question bubbles
                margin = 25
                col_left = max(0, min(col_x_coords) - margin)
                col_right = min(img_width, max(col_x_coords) + margin)
                col_top = max(0, min(col_y_coords) - margin)
                col_bottom = min(img_height, max(col_y_coords) + margin)
                
                cv2.rectangle(result_image, (col_left, col_top), (col_right, col_bottom), color, 3)
                cv2.putText(result_image, name, (col_left, col_top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Organize bubbles by question row (Y position) and choice column (X position)
                # Group bubbles by Y position (each row is a question)
                bubbles_by_row = {}
                question_tolerance = 15  # pixels tolerance for horizontal alignment
                
                for bubble in col_bubbles:
                    center_y = bubble['center'][1]
                    # Find existing row or create new one
                    assigned_row = None
                    for existing_y in bubbles_by_row.keys():
                        if abs(center_y - existing_y) < question_tolerance:
                            assigned_row = existing_y
                            break
                    
                    if assigned_row is None:
                        bubbles_by_row[center_y] = []
                        assigned_row = center_y
                    
                    bubbles_by_row[assigned_row].append(bubble)
                
                # Sort rows by Y position (top to bottom)
                sorted_rows = sorted(bubbles_by_row.items(), key=lambda x: x[0])
                
                # Process each question row
                for row_idx, (row_y, row_bubbles) in enumerate(sorted_rows):
                    # Sort bubbles in this row by X position (left to right = A, B, C, D)
                    row_bubbles.sort(key=lambda b: b['center'][0])
                    
                    # Calculate question number dynamically based on layout
                    # Each column continues from where the previous ended
                    questions_per_column = [len(question_bubbles_by_column[i]) // 4 for i in range(len(question_bubbles_by_column))]
                    
                    if col_idx == 0:
                        question_num = row_idx + 1
                    elif col_idx == 1:
                        # Start after all questions from column 0
                        question_num = questions_per_column[0] + row_idx + 1
                    elif col_idx == 2:
                        # Start after all questions from columns 0 and 1
                        question_num = questions_per_column[0] + questions_per_column[1] + row_idx + 1
                    else:
                        # For additional columns, continue the sequence
                        prev_questions = sum(questions_per_column[:col_idx])
                        question_num = prev_questions + row_idx + 1
                    
                    filled_choices = []
                    
                    # Check each bubble in this row for filled status
                    for choice_idx, bubble in enumerate(row_bubbles):
                        center = bubble['center']
                        area = bubble['area']
                        
                        # Check if this bubble is filled
                        x, y = center
                        region_size = 8
                        x1, y1 = max(0, x - region_size), max(0, y - region_size)
                        x2, y2 = min(img_width, x + region_size), min(img_height, y + region_size)
                        
                        if len(corrected_image.shape) == 3:
                            gray_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
                        else:
                            gray_image = corrected_image
                        
                        bubble_region = gray_image[y1:y2, x1:x2]
                        if bubble_region.size > 0:
                            mean_intensity = np.mean(bubble_region)
                            # Filled bubbles are darker (lower intensity)
                            # Adaptive threshold based on image characteristics
                            threshold = self.calculate_fill_threshold(corrected_image)
                            if mean_intensity < threshold:
                                radius = 8  # Consistent radius for all bubbles
                                cv2.circle(result_image, center, radius, color, 2)
                                cv2.circle(result_image, center, 3, color, -1)  # Filled center
                                filled_col_count += 1
                                
                                # Record the choice (A=0, B=1, C=2, D=3)
                                choice_letter = chr(ord('A') + choice_idx) if choice_idx < 4 else str(choice_idx)
                                filled_choices.append(choice_letter)
                                
                                # Add question number and choice label next to filled bubble
                                label_text = f"Q{question_num}:{choice_letter}"
                                label_x = center[0] + radius + 5
                                label_y = center[1] + 5
                                cv2.putText(result_image, label_text, (label_x, label_y), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Store detected answer for this question
                    if filled_choices:
                        detected_answers[question_num] = filled_choices
                
                print(f"  - {name}: {filled_col_count} filled bubbles")
                
                # Log detected answers for this column  
                if col_idx == 0:  # Questions Col 1: 1-25
                    column_questions = [q for q in detected_answers.keys() if 1 <= q <= 25]
                elif col_idx == 1:  # Questions Col 2: 16-41
                    column_questions = [q for q in detected_answers.keys() if 16 <= q <= 41] 
                elif col_idx == 2:  # Questions Col 3: 42-67
                    column_questions = [q for q in detected_answers.keys() if 42 <= q <= 67]
                else:
                    column_questions = []
                if column_questions:
                    print(f"  - Detected answers in {name}:")
                    for q_num in sorted(column_questions):
                        choices = detected_answers[q_num]
                        if len(choices) == 1:
                            print(f"    Question {q_num}: {choices[0]}")
                        else:
                            print(f"    Question {q_num}: {', '.join(choices)} (multiple answers)")
                else:
                    print(f"  - No answers detected in {name}")
        
        # Save step 10 result
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
        contours = self.step5_detect_contours(thresh, original)
        
        # Step 6: Detect squares
        positioning_squares, grid_squares = self.step6_detect_squares(contours, original)
        
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