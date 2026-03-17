import cv2
from ultralytics import YOLO
import numpy as np
import yaml
import os
import time
import logging
import random
from datetime import datetime
from utils import roi as roi_utils, mapping
from utils.db_utils import init_db, store_billet, get_last_billet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BilletSizingApp")

def main():
    """Main function for billet sizing application"""
    start_time = time.time()
    logger.info("="*60)
    logger.info(f"Starting Billet Sizing Application at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    try:
        # ======================
        # 1. CONFIGURATION LOADING
        # ======================
        logger.info("Loading configuration files...")
        config_path = 'config/config.yaml'
        pixel_mapping_path = 'config/pixel_mapping.yaml'
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not os.path.exists(pixel_mapping_path):
            raise FileNotFoundError(f"Pixel mapping file not found: {pixel_mapping_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        with open(pixel_mapping_path, 'r') as f:
            pixel_mapping = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded successfully from {config_path}")
        logger.info(f"Pixel mapping loaded successfully from {pixel_mapping_path}")
        
        # Validate ROI configuration
        roi = config.get('roi', {})
        line_x = config.get('line_x', None)
        
        if not roi:
            raise ValueError("ROI configuration is missing in config.yaml")
        
        roi_x_min = roi.get('x_min', 0)
        roi_y_min = roi.get('y_min', 0)
        roi_x_max = roi.get('x_max', 3840)
        roi_y_max = roi.get('y_max', 2160)
        
        logger.info(f"ROI Configuration: x_min={roi_x_min}, y_min={roi_y_min}, x_max={roi_x_max}, y_max={roi_y_max}")
        
        # Validate line_x
        if line_x is None:
            line_x = roi_x_min + (roi_x_max - roi_x_min) // 2
            logger.warning(f"Line X not specified in config. Using default: {line_x}")
        else:
            logger.info(f"Line X configured: {line_x}")
        
        # ======================
        # 2. MODEL LOADING
        # ======================
        logger.info("Loading models...")
        
        # Load Detection Model (for dimensions/bounding boxes)
        det_model_path = 'models/det/best.pt'
        if not os.path.exists(det_model_path):
            raise FileNotFoundError(f"Detection model not found: {det_model_path}")
        det_model = YOLO(det_model_path)
        logger.info(f"Detection model loaded: {det_model_path}")
        
        # Load Pose Model (for length/keypoints)
        pose_model_path = 'models/pose/best.pt'
        if not os.path.exists(pose_model_path):
            raise FileNotFoundError(f"Pose model not found: {pose_model_path}")
        pose_model = YOLO(pose_model_path)
        logger.info(f"Pose model loaded: {pose_model_path}")
        
        # ======================
        # 3. DATABASE INITIALIZATION
        # ======================
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully")
        
        # ======================
        # 4. IMAGE PROCESSING
        # ======================
        input_image_path = '2.jpg'
        output_image_path = 'output.jpg'
        
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Input image not found: {input_image_path}")
        
        logger.info(f"Processing image: {input_image_path}")
        
        # Read and validate image
        frame = cv2.imread(input_image_path)
        if frame is None:
            raise ValueError(f"Failed to read image: {input_image_path}")
        
        height, width = frame.shape[:2]
        logger.info(f"Image dimensions: {width}x{height} pixels")
        
        # Validate ROI against image dimensions
        if roi_x_min < 0 or roi_y_min < 0 or roi_x_max > width or roi_y_max > height:
            raise ValueError(f"ROI exceeds image boundaries. Image: {width}x{height}, ROI: {roi_x_min}-{roi_x_max}x{roi_y_min}-{roi_y_max}")
        
        # ======================
        # 5. DETECTION MODEL INFERENCE (Dimensions)
        # ======================
        # We use the detection model first to find the billets and their precise dimensions
        logger.info("Running detection model on FULL FRAME...")
        det_results = det_model(frame, verbose=False)
        
        billet_id = 1
        detected_billets = []
        total_detections = 0
        
        # --- Text Overlay Configuration ---
        text_start_x = 20  # Padding from left
        text_start_y = 40  # Padding from top
        line_height = 30   # Space between lines
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        
        # Iterate over detections provided by the DETECTION MODEL
        if det_results[0].boxes is not None:
            total_detections = len(det_results[0].boxes)
            logger.info(f"Detection model detected {total_detections} objects.")
            
            for i in range(len(det_results[0].boxes)):
                # --- Initialize variables for this billet ---
                keypoints = None
                length_mm = None
                distance_px = 0.0
                
                # Get bounding box from Detection Model
                box = det_results[0].boxes.xyxy[i].cpu().numpy()
                x_min, y_min, x_max, y_max = box
                
                # Check if billet is within ROI
                if not (roi_x_min <= x_min < roi_x_max and roi_y_min <= y_min < roi_y_max):
                    logger.debug(f"Skipping billet outside ROI: {x_min}, {y_min}")
                    continue
                
                # Check if billet passed the line
                center_x = (x_min + x_max) / 2
                passed_line = center_x >= line_x
                
                if not passed_line:
                    logger.debug(f"Billet {billet_id} not passed line check (center_x={center_x}, line_x={line_x})")
                    continue
                
                # --- Calculate Dimensions from Detection Model ---
                width_px = x_max - x_min
                height_px = y_max - y_min
                dim_mm = mapping.get_dimension_mm(width_px, height_px, pixel_mapping)
                
                # ======================
                # 6. POSE MODEL INFERENCE (Length)
                # ======================
                # Extract ROI for Pose Model
                billet_roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                h_roi, w_roi = billet_roi.shape[:2]
                
                # Resize ROI to match training size of Pose Model (3840x2160)
                billet_roi_resized = cv2.resize(billet_roi, (3840, 2160))
                billet_roi_rgb = cv2.cvtColor(billet_roi_resized, cv2.COLOR_BGR2RGB)
                
                # Run Pose Model
                pose_results = pose_model(billet_roi_rgb, conf=0.2, verbose=False)
                
                # Process Pose Results
                if len(pose_results) > 0 and len(pose_results[0].keypoints.xy) > 0:
                    raw_keypoints = pose_results[0].keypoints.xy.cpu().numpy()[0]
                    
                    if len(raw_keypoints) >= 2:
                        # Get first two keypoints (assumed endpoints)
                        x1_scaled, y1_scaled = raw_keypoints[0]
                        x2_scaled, y2_scaled = raw_keypoints[1]
                        
                        # Map coordinates back to original ROI
                        x1_roi = (x1_scaled / 3840) * w_roi
                        y1_roi = (y1_scaled / 2160) * h_roi
                        x2_roi = (x2_scaled / 3840) * w_roi
                        y2_roi = (y2_scaled / 2160) * h_roi
                        
                        # Map coordinates to original Frame
                        x1_orig = x_min + x1_roi
                        y1_orig = y_min + y1_roi
                        x2_orig = x_min + x2_roi
                        y2_orig = y_min + y2_roi
                        
                        # Calculate Euclidean distance (pixel distance)
                        distance_px = np.sqrt((x2_orig - x1_orig)**2 + (y2_orig - y1_orig)**2)
                        
                        # Convert to mm
                        length_mm = mapping.get_length_mm(distance_px, pixel_mapping)
                        
                        # Store keypoints for visualization (in original frame coordinates)
                        keypoints = np.array([[x1_orig, y1_orig], [x2_orig, y2_orig]])
                        
                        logger.info(f"Billet {billet_id}: Dim={dim_mm}, Length={length_mm} mm")
                    else:
                        logger.warning(f"Billet {billet_id} has insufficient keypoints")
                else:
                    logger.warning(f"Billet {billet_id} detected but no keypoints found")
                
                # ======================
                # 7. VISUALIZATION AND LOGGING
                # ======================
                
                # --- Generate a unique random color for this billet ---
                color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                
                # Draw bounding box with the unique color
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 3)
                
                # --- Draw ID on the bounding box ---
                label_text = f"{billet_id}"
                (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                cv2.putText(frame, label_text, (int(x_min), int(y_min) - 10), font, font_scale, color, font_thickness)
                
                # Draw keypoints and lines with the same color
                if keypoints is not None and len(keypoints) > 0:
                    # Draw line between keypoints
                    p1 = (int(keypoints[0][0]), int(keypoints[0][1]))
                    p2 = (int(keypoints[1][0]), int(keypoints[1][1]))
                    cv2.line(frame, p1, p2, color, 2)
                    
                    # Draw keypoints
                    cv2.circle(frame, p1, 5, color, -1)
                    cv2.circle(frame, p2, 5, color, -1)
                
                # --- Left side text info ---
                dim_text = f"{dim_mm}" if dim_mm else f"{int(width_px)}x{int(height_px)} px"
                len_text = f"{length_mm}" if length_mm is not None else f"{int(distance_px)} px"
                
                info_line_1 = f"Billet ID: {billet_id} | Dim: {dim_text} | Length: {len_text}"
                current_y = text_start_y + (len(detected_billets)) * (line_height * 4) + 10
                cv2.putText(frame, info_line_1, (text_start_x, current_y), font, font_scale, color, font_thickness)
                
                # Store for database
                detected_billets.append({
                    'id': billet_id,
                    'width_px': width_px,
                    'height_px': height_px,
                    'length_px': distance_px,
                    'length_mm': length_mm,
                    'keypoints': keypoints,
                    'box': (x_min, y_min, x_max, y_max)
                })
                
                billet_id += 1
        else:
            logger.warning("No objects detected by detection model.")
        
        # ======================
        # 8. FINAL PROCESSING AND OUTPUT
        # ======================
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)
        logger.info(f"Line size drawn at position: {line_x}")
        
        cv2.imwrite(output_image_path, frame)
        logger.info(f"Output image saved to: {output_image_path}")
        
        if detected_billets:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for billet in detected_billets:
                store_billet(
                    timestamp,
                    f"{billet['width_px']}x{billet['height_px']}",
                    f"{billet['width_px']}x{billet['height_px']}",
                    billet['length_mm'] if billet['length_mm'] else f"{billet['length_px']:.2f}"
                )
            logger.info(f"Stored {len(detected_billets)} billets in database")
        else:
            logger.warning("No billets detected for database storage")
        
        # ======================
        # 9. FINAL REPORT
        # ======================
        logger.info("="*60)
        logger.info(f"Processing completed successfully in {time.time() - start_time:.2f} seconds")
        logger.info(f"Total billets detected: {len(detected_billets)}")
        logger.info(f"Total detections: {total_detections}")
        logger.info("="*60)
        
        print("\n" + "="*60)
        print(f"BILLET SIZING REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print(f"Total billets processed: {len(detected_billets)}")
        print(f"Total detections: {total_detections}")
        print(f"Output image saved to: {output_image_path}")
        print(f"Database updated with {len(detected_billets)} entries")
        print("="*60)
        
        if detected_billets:
            print("\nDETAILED BILLET INFORMATION:")
            print("-"*60)
            print(f"{'ID':<5} {'Width (px)':<15} {'Height (px)':<15} {'Length (px)':<15} {'Length (mm)':<15}")
            print("-"*60)
            for billet in detected_billets:
                print(f"{billet['id']:<5} {billet['width_px']:<15.2f} {billet['height_px']:<15.2f} {billet['length_px']:<15.2f} {billet['length_mm'] if billet['length_mm'] else 'N/A':<15}")
            print("-"*60)
        
        return True
    
    except Exception as e:
        logger.exception(f"Critical error occurred: {str(e)}")
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print("Please check main.log for detailed error information")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Billet sizing process completed successfully!")
    else:
        print("\n❌ Billet sizing process failed. Check logs for details.")