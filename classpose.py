import cv2
from ultralytics import YOLO
import numpy as np
import yaml
import os
import time
import logging
import random
from datetime import datetime

# فرض بر این است که فایل‌های utils در مسیر درست موجود هستند
from utils import roi as roi_utils, mapping
from utils.db_utils import init_db, store_billet, get_last_billet

class BilletSizingApp:
    def __init__(self, config_path='config/config.yaml', pixel_mapping_path='config/pixel_mapping.yaml'):
        """
        سازنده کلاس: مدل را بارگذاری کرده و تنظیمات را مقداردهی اولیه می‌کند.
        """
        # تنظیم لاگر
        self.logger = logging.getLogger("BilletSizingApp")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("pose_main.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger.info("="*60)
        self.logger.info("Initializing BilletSizingApp...")
        
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
        
        self.line_x = self.config.get('line_x', None)
        if self.line_x is None:
            self.line_x = self.roi_x_min + (self.roi_x_max - self.roi_x_min) // 2
            self.logger.warning(f"Line X not specified. Using default: {self.line_x}")
        
        # ======================
        # 3. MODEL LOADING
        # ======================
        pose_model_path = 'models/pose/best.pt'
        if not os.path.exists(pose_model_path):
            raise FileNotFoundError(f"Pose model not found: {pose_model_path}")
        
        self.pose_model = YOLO(pose_model_path)
        self.logger.info(f"Pose model loaded successfully from {pose_model_path}")
        
        # ======================
        # 4. DATABASE INITIALIZATION
        # ======================
        init_db()
        self.logger.info("Database initialized successfully")

    def process_image(self, input_image_path, output_image_path='output.jpg', save_db=True):
        """
        تصویر ورودی را پردازش کرده و نتایج را برمی‌گرداند.
        
        Args:
            input_image_path (str): مسیر تصویر ورودی.
            output_image_path (str): مسیر ذخیره تصویر خروجی (پیش‌فرض: output.jpg).
            save_db (bool): آیا نتایج در دیتابیس ذخیره شود؟ (پیش‌فرض: True).
            
        Returns:
            dict: شامل اطلاعات پردازش شده (تصویر خروجی، لیست بیلت‌ها، و گزارش).
        """
        start_time = time.time()
        self.logger.info(f"Processing image: {input_image_path}")
        
        # خواندن تصویر
        frame = cv2.imread(input_image_path)
        if frame is None:
            raise ValueError(f"Failed to read image: {input_image_path}")
        
        height, width = frame.shape[:2]
        
        # اعتبارسنجی ROI
        if self.roi_x_min < 0 or self.roi_y_min < 0 or self.roi_x_max > width or self.roi_y_max > height:
            self.logger.error(f"ROI exceeds image boundaries.")
            return None

        # ======================
        # 5. POSE INFERENCE
        # ======================
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_model(img_rgb, verbose=False)
        r = results[0]
        
        detected_billets = []
        total_detections = 0
        
        # تنظیمات متن روی تصویر
        text_start_x = 20
        text_start_y = 40
        line_height = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        
        if r.boxes is not None:
            total_detections = len(r.boxes)
            self.logger.info(f"Detected {total_detections} objects.")
            
            billet_id = 1
            
            for i in range(len(r.boxes)):
                keypoints = None
                length_mm = None
                distance_px = 0.0
                
                # دریافت Bounding Box
                box = r.boxes.xyxy[i].cpu().numpy()
                x_min, y_min, x_max, y_max = box
                
                # بررسی ROI
                if not (self.roi_x_min <= x_min < self.roi_x_max and self.roi_y_min <= y_min < self.roi_y_max):
                    continue
                
                # بررسی خط عبور
                center_x = (x_min + x_max) / 2
                if center_x < self.line_x:
                    continue
                
                # ======================
                # 6. KEYPOINT EXTRACTION
                # # ======================
                if r.keypoints is not None:
                    raw_keypoints = r.keypoints.data[i].cpu().numpy()
                    keypoints = raw_keypoints.squeeze()
                    
                    if keypoints.shape[0] >= 2:
                        # استخراج دو نقطه اول (فرض: انتهای تسمه)
                        pt1 = (int(keypoints[0][0]), int(keypoints[0][1]))
                        pt2 = (int(keypoints[1][0]), int(keypoints[1][1]))
                        
                        # محاسبه فاصله اقلیدسی
                        distance_px = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                        
                        # تبدیل به میلی‌متر
                        length_mm = mapping.get_length_mm(distance_px, self.pixel_mapping)
                        
                        # --- اصلاحیه: اگر میلی‌متر محاسبه نشد، لاگ بزن ---
                        if length_mm is None:
                            self.logger.warning(f"Billet {billet_id}: Could not calculate mm. Using pixel value: {distance_px:.2f} px")
                        
                        # ذخیره اطلاعات بیلت
                        detected_billets.append({
                            'id': billet_id,
                            'width_px': x_max - x_min,
                            'height_px': y_max - y_min,
                            'length_px': distance_px,
                            'length_mm': length_mm, # اینجا می‌تواند None باشد
                            'keypoints': keypoints,
                            'box': (x_min, y_min, x_max, y_max)
                        })
                        
                        # ======================
                        # 7. VISUALIZATION
                        # ======================
                        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                        
                        # رسم باکس
                        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 3)
                        cv2.putText(frame, str(billet_id), (int(x_min), int(y_min) - 10), font, font_scale, color, font_thickness)
                        
                        # رسم خطوط و نقاط کلیدی
                        if keypoints is not None and len(keypoints) > 0:
                            for k in range(len(keypoints) - 1):
                                p1 = (int(keypoints[k][0]), int(keypoints[k][1]))
                                p2 = (int(keypoints[k+1][0]), int(keypoints[k+1][1]))
                                cv2.line(frame, p1, p2, color, 2)
                            
                            for idx in range(len(keypoints)):
                                x = int(keypoints[idx][0])
                                y = int(keypoints[idx][1])
                                cv2.circle(frame, (x, y), 5, color, -1)
                        
                        # نوشتن اطلاعات متنی روی تصویر
                        width_px = x_max - x_min
                        height_px = y_max - y_min
                        dim_mm = mapping.get_dimension_mm(width_px, height_px, self.pixel_mapping)
                        
                        # --- اصلاحیه: نمایش هوشمند روی تصویر ---
                        dim_text = f"{dim_mm}" if dim_mm else f"{int(width_px)}x{int(height_px)} px"
                        
                        # اگر میلی‌متر نبود، پیکسل را نشان بده
                        if length_mm is not None:
                            len_text = f"{length_mm} mm"
                        else:
                            len_text = f"{int(distance_px)} px"
                            
                        info_line_1 = f"Billet ID: {billet_id} | Dim: {dim_text} | Length: {len_text}"
                        current_y = text_start_y + (len(detected_billets) - 1) * (line_height * 4) + 10
                        cv2.putText(frame, info_line_1, (text_start_x, current_y), font, font_scale, color, font_thickness)
                        
                        billet_id += 1
        
        # رسم خط عبور
        cv2.line(frame, (self.line_x, 0), (self.line_x, frame.shape[0]), (0, 0, 255), 2)
        
        # ذخیره تصویر خروجی
        cv2.imwrite(output_image_path, frame)
        self.logger.info(f"Output image saved to: {output_image_path}")
        
        # ذخیره در دیتابیس
        if save_db and detected_billets:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for billet in detected_billets:
                # --- اصلاحیه: ذخیره پیکسل اگر میلی‌متر موجود نبود ---
                val_to_store = billet['length_mm'] if billet['length_mm'] is not None else f"{billet['length_px']:.2f}"
                
                store_billet(
                    timestamp,
                    f"{billet['width_px']}x{billet['height_px']}",
                    f"{billet['width_px']}x{billet['height_px']}",
                    val_to_store
                )
            self.logger.info(f"Stored {len(detected_billets)} billets in database")
        
        # آمار نهایی
        processing_time = time.time() - start_time
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
        # ۱. نمونه‌سازی از کلاس (مدل یک بار بارگذاری می‌شود)
        app = BilletSizingApp()
        
        # ۲. پردازش یک تصویر خاص
        result = app.process_image(input_image_path='2.jpg', output_image_path='output_class_2.jpg')
        
        # ۳. دسترسی به نتایج
        if result:
            print("\n" + "="*60)
            print(f"Total billets processed: {len(result['detected_billets'])}")
            print(f"Processing time: {result['processing_time']:.2f} s")
            print("="*60)
            
            for billet in result['detected_billets']:
                # --- اصلاحیه: چاپ هوشمند در کنسول ---
                length_str = f"{billet['length_mm']} mm" if billet['length_mm'] is not None else f"{billet['length_px']:.2f} px"
                print(f"ID: {billet['id']}, Length: {length_str}")
                
    except Exception as e:
        print(f"Error: {e}")