import cv2
from ultralytics import YOLO
import numpy as np
import yaml
import os
import random
import logging
from datetime import datetime

# فرض بر این است که فایل‌های utils در مسیر درست موجود هستند
from utils import roi as roi_utils, mapping
from utils.db_utils import init_db, store_billet

class BilletSizingSystem:
    def __init__(self, config_path='config/config.yaml', pixel_mapping_path='config/pixel_mapping.yaml'):
        """
        سازنده کلاس: تنظیمات، مدل‌ها و دیتابیس را مقداردهی اولیه می‌کند.
        """
        # تنظیم لاگر
        self.logger = logging.getLogger("BilletSizingSystem")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("system.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger.info("="*60)
        self.logger.info("Initializing BilletSizingSystem (Det + Pose)...")
        
        # ======================
        # 1. CONFIGURATION LOADING
        # ======================
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not os.path.exists(pixel_mapping_path):
            raise FileNotFoundError(f"Pixel mapping file not found: {pixel_mapping_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        with open(pixel_mapping_path, 'r') as f:
            self.pixel_mapping = yaml.safe_load(f)
        
        self.logger.info("Configuration files loaded successfully.")
        
        # ======================
        # 2. ROI & LINE CONFIGURATION
        # ======================
        roi = self.config.get('roi', {})
        self.roi_x_min = roi.get('x_min', 0)
        self.roi_y_min = roi.get('y_min', 0)
        self.roi_x_max = roi.get('x_max', 3840)
        self.roi_y_max = roi.get('y_max', 2160)
        
        # پارامترهای ROI برای توابع utils
        self.roi_x = self.roi_x_min
        self.roi_y = self.roi_y_min
        self.roi_w = self.roi_x_max - self.roi_x_min
        self.roi_h = self.roi_y_max - self.roi_y_min
        
        # تنظیمات Line X
        self.line_x_global = self.config.get('line_x', self.roi_x_min + (self.roi_w // 2))
        
        if self.line_x_global < self.roi_x_min or self.line_x_global > self.roi_x_max:
            self.logger.warning(f"Line X is out of ROI bounds. Using default center.")
            self.line_x_global = self.roi_x_min + (self.roi_w // 2)
            
        self.logger.info(f"ROI: x={self.roi_x_min}, y={self.roi_y_min}, w={self.roi_w}, h={self.roi_h}")
        self.logger.info(f"Line X Global: {self.line_x_global}")
        
        # ======================
        # 3. MODELS LOADING
        # ======================
        # مدل تشخیص (Det)
        det_model_path = 'models/det/best.pt'
        if not os.path.exists(det_model_path):
            raise FileNotFoundError(f"Detection model not found: {det_model_path}")
        self.det_model = YOLO(det_model_path)
        self.logger.info(f"Detection model loaded: {det_model_path}")
        
        # مدل ژست (Pose)
        pose_model_path = 'models/pose/best.pt'
        if not os.path.exists(pose_model_path):
            raise FileNotFoundError(f"Pose model not found: {pose_model_path}")
        self.pose_model = YOLO(pose_model_path)
        self.logger.info(f"Pose model loaded: {pose_model_path}")
        
        # ======================
        # 4. DATABASE INITIALIZATION
        # ======================
        init_db()
        self.logger.info("Database initialized successfully")

    def _get_unique_color(self):
        """تولید یک رنگ تصادفی روشن برای ترسیم"""
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def process_image(self, input_image_path, output_image_path='final_output.jpg', save_db=True):
        """
        پردازش اصلی تصویر: تشخیص با Det، اندازه‌گیری با Pose و ترسیم نتایج.
        """
        start_time = datetime.now()
        
        self.logger.info(f"Processing image: {input_image_path}")
        
        # 1. خواندن تصویر
        frame = cv2.imread(input_image_path)
        if frame is None:
            raise ValueError(f"Failed to read image: {input_image_path}")
        
        if os.path.exists(output_image_path):
            os.remove(output_image_path)
            
        # 2. تشخیص اولیه با مدل Det روی ROI
        roi_frame = roi_utils.apply_roi(frame, (self.roi_x, self.roi_y, self.roi_w, self.roi_h))
        det_results = self.det_model(roi_frame, verbose=False)
        
        detected_billets = []
        billet_id = 1
        total_detections = 0
        
        # تنظیمات فونت
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # 3. پردازش هر باکس تشخیص داده شده
        for result in det_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            total_detections += len(boxes)
            
            for box in boxes:
                # مختصات در ROI
                x_min_roi, y_min_roi, x_max_roi, y_max_roi = box
                
                # تبدیل به مختصات اصلی تصویر
                x_min, y_min, x_max, y_max = roi_utils.get_original_coords(
                    (self.roi_x, self.roi_y, self.roi_w, self.roi_h), box
                )
                
                # فیلتر ROI
                if not (self.roi_x_min <= x_min < self.roi_x_max and self.roi_y_min <= y_min < self.roi_y_max):
                    continue
                
                # فیلتر خط عبور
                center_x_roi = (x_min_roi + x_max_roi) / 2
                line_x_roi_relative = self.line_x_global - self.roi_x
                
                if center_x_roi >= line_x_roi_relative:
                    # --- داده‌های اولیه ---
                    width_px = x_max_roi - x_min_roi
                    height_px = y_max_roi - y_min_roi
                    dim_mm = mapping.get_dimension_mm(width_px, height_px, self.pixel_mapping)
                    
                    # --- رنگ یکتا برای این شمش ---
                    color = self._get_unique_color()
                    
                    # --- استخراج ناحیه شمش برای مدل Pose ---
                    # اطمینان از اینکه مختصات صحیح هستند
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    billet_roi = frame[y_min:y_max, x_min:x_max]
                    
                    length_mm = None
                    length_px = None # تغییر مقدار پیش‌فرض به None
                    
                    if billet_roi.size == 0:
                        self.logger.warning(f"Billet {billet_id}: ROI is empty.")
                    else:
                        h_roi, w_roi = billet_roi.shape[:2]
                        
                        # تغییر سایز برای مدل Pose
                        try:
                            # نکته: سایز ورودی باید با سایزی که مدل با آن آموزش دیده همخوانی داشته باشد
                            billet_roi_resized = cv2.resize(billet_roi, (3840, 2160))
                        except cv2.error:
                            self.logger.warning(f"Billet {billet_id}: Could not resize ROI.")
                        else:
                            billet_roi_rgb = cv2.cvtColor(billet_roi_resized, cv2.COLOR_BGR2RGB)
                            
                            # اجرای Pose با آستانه اطمینان پایین‌تر (0.01) برای تشخیص بهتر
                            pose_results = self.pose_model(billet_roi_rgb, conf=0.01, verbose=False)
                            
                            # پردازش نتایج Pose
                            if len(pose_results) > 0 and len(pose_results[0].keypoints.xy) > 0:
                                keypoints = pose_results[0].keypoints.xy.cpu().numpy()[0]
                                
                                if len(keypoints) >= 2:
                                    # تبدیل مختصات keypoints به تصویر اصلی
                                    # نقطه 0
                                    x1_scaled, y1_scaled = keypoints[0]
                                    x1_orig = x_min + (x1_scaled / 3840) * w_roi
                                    y1_orig = y_min + (y1_scaled / 2160) * h_roi
                                    
                                    # نقطه 1
                                    x2_scaled, y2_scaled = keypoints[1]
                                    x2_orig = x_min + (x2_scaled / 3840) * w_roi
                                    y2_orig = y_min + (y2_scaled / 2160) * h_roi
                                    
                                    # محاسبه طول
                                    distance_px = np.sqrt((x2_orig - x1_orig)**2 + (y2_orig - y1_orig)**2)
                                    length_px = distance_px
                                    length_mm = mapping.get_length_mm(distance_px, self.pixel_mapping)
                                    
                                    # --- ترسیم خطوط Pose ---
                                    cv2.line(frame, (int(x1_orig), int(y1_orig)), (int(x2_orig), int(y2_orig)), color, 3)
                                    cv2.circle(frame, (int(x1_orig), int(y1_orig)), 6, (255, 0, 255), -1)
                                    cv2.circle(frame, (int(x2_orig), int(y2_orig)), 6, (255, 0, 255), -1)
                                else:
                                    self.logger.warning(f"Billet {billet_id}: Not enough keypoints detected.")
                            else:
                                self.logger.warning(f"Billet {billet_id}: No keypoints found by Pose model.")

                    # --- ترسیم باکس تشخیص (Det) ---
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # --- نوشتن آیدی روی باکس ---
                    cv2.putText(frame, f"ID: {billet_id}", (x_min, y_min - 10), 
                                font, font_scale, color, font_thickness)
                    
                    # ذخیره اطلاعات
                    detected_billets.append({
                        'id': billet_id,
                        'color': color,
                        'width_px': width_px,
                        'height_px': height_px,
                        'dim_mm': dim_mm,
                        'length_px': length_px, # حالا می‌تواند None باشد
                        'length_mm': length_mm,
                        'keypoints': None # ساده‌سازی برای ذخیره
                    })
                    
                    billet_id += 1

        # 4. رسم خط شمارش
        cv2.line(frame, (self.line_x_global, 0), (self.line_x_global, frame.shape[0]), (0, 0, 255), 2)
        
        # 5. رسم اطلاعات در سمت چپ (Left Side Info)
        text_start_x = 20
        text_start_y = 40
        line_height = 30
        
        for i, billet in enumerate(detected_billets):
            info_text = ""
            
            # متن ابعاد سطح مقطع
            dim_str = billet['dim_mm'] if billet['dim_mm'] else f"{int(billet['width_px'])}x{int(billet['height_px'])} px"
            
            # متن طول
            if billet['length_mm']:
                len_str = f"{billet['length_mm']} mm"
            elif billet['length_px'] is not None and billet['length_px'] > 0:
                len_str = f"{int(billet['length_px'])} px"
            else:
                len_str = "N/A" # اگر هیچ طولی محاسبه نشد
            
            info_text = f"ID: {billet['id']} | Dim: {dim_str} | Len: {len_str}"
            
            current_y = text_start_y + (i * line_height)
            cv2.putText(frame, info_text, (text_start_x, current_y), font, font_scale, billet['color'], font_thickness)

        # 6. ذخیره خروجی
        cv2.imwrite(output_image_path, frame)
        self.logger.info(f"Output saved to: {output_image_path}")
        
        # 7. ذخیره در دیتابیس
        if save_db and detected_billets:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for billet in detected_billets:
                # اگر میلی‌متر نبود، پیکسل را ذخیره کن (اگر پیکسل هم None بود، N/A)
                if billet['length_mm']:
                    val_to_store = billet['length_mm']
                elif billet['length_px'] is not None:
                    val_to_store = f"{billet['length_px']:.2f}"
                else:
                    val_to_store = "N/A"
                    
                store_billet(
                    timestamp,
                    f"{billet['width_px']}x{billet['height_px']}",
                    f"{billet['width_px']}x{billet['height_px']}",
                    val_to_store
                )
            self.logger.info(f"Stored {len(detected_billets)} records in database.")
            
        processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return {
            'output_image_path': output_image_path,
            'detected_billets': detected_billets,
            'total_detections': total_detections,
            'processing_time': processing_time
        }

# ============================
# نحوه استفاده (Example Usage)
# ============================
if __name__ == "__main__":
    try:
        # ۱. راه‌اندازی سیستم (بارگذاری مدل‌ها)
        system = BilletSizingSystem()
        
        # ۲. پردازش تصویر
        result = system.process_image(input_image_path='2.jpg', output_image_path='final_output.jpg')
        
        # ۳. گزارش نهایی
        if result:
            print("\n" + "="*60)
            print("FINAL REPORT")
            print("="*60)
            print(f"Total Detections: {result['total_detections']}")
            print(f"Processed Billets: {len(result['detected_billets'])}")
            print(f"Time: {result['processing_time']:.2f} s")
            print("-"*60)
            for b in result['detected_billets']:
                l_str = f"{b['length_mm']} mm" if b['length_mm'] else \
                        (f"{b['length_px']:.1f} px" if b['length_px'] is not None else "N/A")
                d_str = b['dim_mm'] if b['dim_mm'] else f"{int(b['width_px'])}x{int(b['height_px'])}"
                print(f"ID: {b['id']} | Dim: {d_str} | Length: {l_str}")
            print("="*60)
            
    except Exception as e:
        print(f"Critical Error: {e}")