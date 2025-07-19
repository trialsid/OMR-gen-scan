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
        
        tolerance = 40  # pixels tolerance for column alignment
        
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
                    if is_top and area > 200:
                        positioning_squares.append(square)
                        print(f"    Found A1 (top-left) at ({x}, {y})")
                    elif is_bottom and area > 200:
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
                    if is_top and area > 200:
                        positioning_squares.append(square)
                        print(f"    Found A2 (top-right) at ({x}, {y})")
                    elif is_bottom and area > 200:
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
    
    def process_image_stepwise(self, image_path):
        """Process image through steps 1-7"""
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
        
        print("=" * 50)
        print("Processing complete! Check results directory for step images.")
        
        return {
            'original': original,
            'gray': gray,
            'blurred': blurred,
            'thresh': thresh,
            'contours': contours,
            'positioning_squares': positioning_squares,
            'grid_squares': grid_squares,
            'corrected': corrected
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