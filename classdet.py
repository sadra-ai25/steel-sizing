import cv2
from ultralytics import YOLO
import numpy as np
import yaml
import os
import random
import logging

# فرض بر این است که فایل‌های utils در مسیر درست موجود هستند
from utils import roi as roi_utils, mapping


class BilletDetectionApp:
    def __init__(self, config_path='config/config.yaml', pixel_mapping_path='config/pixel_mapping.yaml'):
        """
        سازنده کلاس: مدل تشخیص و تنظیمات را بارگذاری می‌کند.
        """
        # تنظیم لاگر
        self.logger = logging.getLogger("BilletDetectionApp")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("det_main.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger.info("="*60)
        self.logger.info("Initializing BilletDetectionApp...")
        
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
        
        # محاسبه پارامترهای ROI برای تابع apply_roi
        self.roi_x = self.roi_x_min
        self.roi_y = self.roi_y_min
        self.roi_w = self.roi_x_max - self.roi_x_min
        self.roi_h = self.roi_y_max - self.roi_y_min
        
        # تنظیمات Line X
        self.line_x_global = self.config.get('line_x', self.roi_x_min + (self.roi_w // 2))
        
        # بررسی اعتبار خط
        if self.line_x_global < self.roi_x_min or self.line_x_global > self.roi_x_max:
            self.logger.warning(f"Line X is out of ROI bounds. Using default center.")
            self.line_x_global = self.roi_x_min + (self.roi_w // 2)
            
        self.logger.info(f"ROI Config: x={self.roi_x_min}, y={self.roi_y_min}, w={self.roi_w}, h={self.roi_h}")
        self.logger.info(f"Line X Global: {self.line_x_global}")
        
        # ======================
        # 3. MODEL LOADING
        # ======================
        det_model_path = 'models/det/best.pt'
        if not os.path.exists(det_model_path):
            raise FileNotFoundError(f"Detection model not found: {det_model_path}")
        
        self.detection_model = YOLO(det_model_path)
        self.logger.info(f"Detection model loaded successfully from {det_model_path}")

    def process_image(self, input_image_path, output_image_path='2_main_output.jpg'):
        """
        تصویر ورودی را پردازش کرده و نتایج را برمی‌گرداند.
        
        Args:
            input_image_path (str): مسیر تصویر ورودی.
            output_image_path (str): مسیر ذخیره تصویر خروجی.
            
        Returns:
            dict: شامل لیست بیلت‌های شناسایی شده و آمار.
        """
        self.logger.info(f"Processing image: {input_image_path}")
        
        # خواندن تصویر
        frame = cv2.imread(input_image_path)
        if frame is None:
            raise ValueError(f"Failed to read image: {input_image_path}")
        
        # پاک کردن فایل خروجی قبلی اگر وجود داشته باشد
        if os.path.exists(output_image_path):
            os.remove(output_image_path)
        
        # اعمال ROI برای پردازش مدل
        roi_frame = roi_utils.apply_roi(frame, (self.roi_x, self.roi_y, self.roi_w, self.roi_h))
        
        # اجرای مدل تشخیص
        detection_results = self.detection_model(roi_frame, verbose=False)
        
        detected_billets = []
        billet_id = 1
        total_boxes = 0
        
        # ======================
        # رسم اطلاعات در سمت چپ (Left Side Info)
        # ======================
        text_start_x = 20  # فاصله از لبه چپ
        text_start_y = 40  # فاصله از بالا
        line_height = 30   # فاصله بین خطوط

        # تنظیمات متن روی تصویر
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        
        # دیکشنری برای نگهداری رنگ هر آیدی
        id_colors = {}
                
        for result in detection_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            total_boxes += len(boxes)
            
            for box in boxes:
                # مختصات باکس در سیستم مختصات ROI
                x_min_roi, y_min_roi, x_max_roi, y_max_roi = box
                
                # تبدیل مختصات به فریم اصلی
                x_min, y_min, x_max, y_max = roi_utils.get_original_coords(
                    (self.roi_x, self.roi_y, self.roi_w, self.roi_h), box
                )
                
                # بررسی اینکه آیا شمش در ROI است
                if not (self.roi_x_min <= x_min < self.roi_x_max and self.roi_y_min <= y_min < self.roi_y_max):
                    self.logger.debug(f"Ignored (Billet outside ROI)")
                    continue
                
                # محاسبه مرکز باکس
                center_x_roi = (x_min_roi + x_max_roi) / 2
                
                # بررسی عبور از خط مرجع (تبدیل خط جهانی به مختصات نسبی ROI)
                line_x_roi_relative = self.line_x_global - self.roi_x
                
                if center_x_roi >= line_x_roi_relative:
                    self.logger.info(f"Billet {billet_id} passed line check.")
                    
                    # --- تولید رنگ تصادفی برای این باکس ---
                    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    id_colors[billet_id] = color # ذخیره رنگ برای استفاده در متن سمت چپ
                    
                    # --- محاسبه ابعاد ---
                    width_px = x_max_roi - x_min_roi
                    height_px = y_max_roi - y_min_roi
                    dim_mm = mapping.get_dimension_mm(width_px, height_px, self.pixel_mapping)
                    
                    # --- ترسیم باکس ---
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                    
                    # --- نوشتن آیدی باکس (فقط آیدی، بدون متن اضافه روی باکس) ---
                    # متن کمی بالاتر از باکس نوشته می‌شود تا روی باکس نباشد
                    cv2.putText(frame, f"ID: {billet_id}", (int(x_min), int(y_min) - 10), 
                                font, font_scale, color, font_thickness)
                    
                    # ذخیره اطلاعات
                    detected_billets.append({
                        'id': billet_id,
                        'width_px': width_px,
                        'height_px': height_px,
                        'dim_mm': dim_mm,
                        'box': (x_min, y_min, x_max, y_max)
                    })
                    
                    billet_id += 1
                else:
                    self.logger.debug(f"Ignored (Center is left of the line)")
        
        # رسم خط شمارش (Line X)
        cv2.line(frame, (self.line_x_global, 0), (self.line_x_global, frame.shape[0]), (0, 0, 255), 2)
                      
        for i, billet in enumerate(detected_billets):
            # دریافت رنگ اختصاصی این شمش
            billet_color = id_colors.get(billet['id'], (255, 255, 255))
            
            # ساخت متن نمایشی
            dim_text = billet['dim_mm'] if billet['dim_mm'] else f"{int(billet['width_px'])}x{int(billet['height_px'])} px"
            info_text = f"Billet ID: {billet['id']} | Dim: {dim_text}"
            
            # محاسبه موقعیت Y برای متن
            current_y = text_start_y + (i * line_height * 3)
            
            # رسم متن در سمت چپ با رنگ مربوط به همان شمش
            cv2.putText(frame, info_text, (text_start_x, current_y), font, font_scale, billet_color, font_thickness)
        
        # ذخیره تصویر خروجی
        cv2.imwrite(output_image_path, frame)
        self.logger.info(f"Output image saved to: {output_image_path}")
        
        self.logger.info(f"Total Detections: {total_boxes}")
        self.logger.info(f"Counted Billets: {len(detected_billets)}")
        
        return {
            'output_image_path': output_image_path,
            'detected_billets': detected_billets,
            'total_detections': total_boxes
        }

# ============================
# نحوه استفاده (Example Usage)
# ============================
if __name__ == "__main__":
    try:
        # ۱. نمونه‌سازی از کلاس
        app = BilletDetectionApp()
        
        # ۲. پردازش تصویر
        result = app.process_image(input_image_path='2.jpg', output_image_path='output_class_1.jpg')
        
        # ۳. نمایش نتایج
        if result:
            print("\n" + "="*60)
            print(f"Total Detections: {result['total_detections']}")
            print(f"Counted Billets: {len(result['detected_billets'])}")
            print("="*60)
            for billet in result['detected_billets']:
                print(f"ID: {billet['id']}, Dim: {billet['dim_mm'] if billet['dim_mm'] else 'N/A'}")
                
    except Exception as e:
        print(f"Error: {e}")