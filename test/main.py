import cv2
from ultralytics import YOLO
import numpy as np
import yaml
from utils import roi as roi_utils, mapping
import os

# بارگذاری تنظیمات
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# بارگذاری مدل‌ها
detection_model = YOLO('models/det/best.pt')
pose_model = YOLO('models/pose/best.pt')
# Add this verification code
print("Checking model type...")
print("Model type:", pose_model.model.yaml.get('task', 'Not a pose model'))
print("Keypoints shape:", pose_model.model.yaml.get('kpt_shape', 'Not a pose model'))

# تنظیمات مپینگ
with open('config/pixel_mapping.yaml', 'r') as f:
    pixel_mapping = yaml.safe_load(f)

# مسیر تصویر ورودی و خروجی
input_image_path = '2.jpg'
output_image_path = '2_main_output.jpg'

# پاک کردن فایل خروجی قبلی
if os.path.exists(output_image_path):
    os.remove(output_image_path)

# خواندن تصویر ورودی
if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"❌ تصویر ورودی {input_image_path} یافت نشد!")

frame = cv2.imread(input_image_path)
if frame is None:
    raise ValueError(f"❌ خطا در خواندن تصویر {input_image_path}")

print(f"🔍 پردازش تصویر: {input_image_path}")

# تنظیمات ROI (با x_min, x_max, y_min, y_max)
roi_x_min = config['roi']['x_min']
roi_y_min = config['roi']['y_min']
roi_x_max = config['roi']['x_max']
roi_y_max = config['roi']['y_max']

# بررسی معتبر بودن ROI
if roi_x_min < 0 or roi_y_min < 0 or roi_x_max <= roi_x_min or roi_y_max <= roi_y_min:
    raise ValueError(f"❌ ROI نامعتبر: x_min={roi_x_min}, x_max={roi_x_max}, y_min={roi_y_min}, y_max={roi_y_max}")

# محاسبه مختصات برای تابع apply_roi
roi_w = roi_x_max - roi_x_min
roi_h = roi_y_max - roi_y_min
roi_x = roi_x_min
roi_y = roi_y_min

# تنظیمات Line X
line_x_global = config.get('line_x', roi_x_min + (roi_x_max - roi_x_min) // 2)

# بررسی خط در محدوده ROI
if line_x_global < roi_x_min or line_x_global > roi_x_max:
    print(f"⚠️ هشدار: خط شمارش (line_x={line_x_global}) خارج از ROI است. مقدار پیش‌فرض استفاده می‌شود.")
    line_x_global = roi_x_min + (roi_x_max - roi_x_min) // 2

print(f"🔧 CONFIGURATION:")
print(f"   ROI (x_min, y_min, x_max, y_max): {roi_x_min}, {roi_y_min}, {roi_x_max}, {roi_y_max}")
print(f"   Line X (Global): {line_x_global}")
print(f"--------------------------------------------------")

# اعمال ROI برای پردازش
roi_frame = roi_utils.apply_roi(frame, (roi_x, roi_y, roi_w, roi_h))

# تشخیص سطح مقطع شمش
detection_results = detection_model(roi_frame, verbose=False)

billet_id = 1
total_boxes = 0

for result in detection_results:
    boxes = result.boxes.xyxy.cpu().numpy()
    total_boxes += len(boxes)
    
    for box in boxes:
        # مختصات باکس در سیستم مختصات ROI
        x_min_roi, y_min_roi, x_max_roi, y_max_roi = box
        
        # تبدیل مختصات به فریم اصلی
        x_min, y_min, x_max, y_max = roi_utils.get_original_coords((roi_x, roi_y, roi_w, roi_h), box)
        
        # بررسی اینکه آیا شمش در ROI است
        if not (roi_x_min <= x_min < roi_x_max and roi_y_min <= y_min < roi_y_max):
            print(f"   ❌ IGNORED (Billet outside ROI: x_min={x_min}, y_min={y_min})")
            continue
        
        # محاسبه مرکز باکس
        center_x_roi = (x_min_roi + x_max_roi) / 2
        
        # بررسی عبور از خط مرجع (تبدیل خط جهانی به مختصات نسبی ROI)
        line_x_roi_relative = line_x_global - roi_x
        
        if center_x_roi >= line_x_roi_relative:
            print(f"   ✅ PASSED Line Check. Processing...")
            
            # --- محاسبه ابعاد ---
            width_px = x_max_roi - x_min_roi
            height_px = y_max_roi - y_min_roi
            dim_mm = mapping.get_dimension_mm(width_px, height_px, pixel_mapping)
            
            # --- تشخیص طول (با تغییر اندازه تصویر به 1024x1024) ---
            billet_roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            h_roi, w_roi = billet_roi.shape[:2]

            # ⚠️ تغییر حیاتی: تغییر اندازه ROI به 1024x1024 (همان اندازه آموزش)
            billet_roi_resized = cv2.resize(billet_roi, (3840, 2160))

            # تبدیل BGR به RGB
            billet_roi_rgb = cv2.cvtColor(billet_roi_resized, cv2.COLOR_BGR2RGB)

            # اجرای مدل با تنظیم conf=0.2 (کاهش آستانه اطمینان)
            pose_results = pose_model(billet_roi_rgb, conf=0.2, verbose=False)

            length_mm = "N/A"
            distance_px = 0

            if len(pose_results) > 0 and len(pose_results[0].keypoints.xy) > 0:
                keypoints = pose_results[0].keypoints.xy.cpu().numpy()[0]
                
                # تبدیل مختصات از تصویر ریسایز شده به ROI اصلی
                x1_scaled, y1_scaled = keypoints[0]
                x2_scaled, y2_scaled = keypoints[1]
                
                # محاسبه مختصات در ROI اصلی (با توجه به نسبت ریسایز)
                x1_roi = (x1_scaled / 3840) * w_roi
                y1_roi = (y1_scaled / 2160) * h_roi
                x2_roi = (x2_scaled / 3840) * w_roi
                y2_roi = (y2_scaled / 2160) * h_roi
                
                # تبدیل مختصات به تصویر اصلی (Frame)
                x1_orig = x_min + x1_roi
                y1_orig = y_min + y1_roi
                x2_orig = x_min + x2_roi
                y2_orig = y_min + y2_roi
                
                # محاسبه فاصله
                distance_px = np.sqrt((x2_orig - x1_orig)**2 + (y2_orig - y1_orig)**2)
                length_mm = mapping.get_length_mm(distance_px, pixel_mapping)
                
                # رسم خط و نقاط
                cv2.line(frame, (int(x1_orig), int(y1_orig)), (int(x2_orig), int(y2_orig)), (255, 255, 0), 3)
                cv2.circle(frame, (int(x1_orig), int(y1_orig)), 6, (255, 0, 255), -1)
                cv2.circle(frame, (int(x2_orig), int(y2_orig)), 6, (255, 0, 255), -1)
            else:
                print(f"   ❌ No keypoints found by pose model!")

            # --- ترسیم و نمایش اطلاعات ---
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            
            cv2.putText(frame, f"ID: {billet_id}", (int(x_min), int(y_min) - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            dim_text = f"Dim: {dim_mm}" if dim_mm else f"Dim: {int(width_px)}x{int(height_px)} px"
            cv2.putText(frame, dim_text, (int(x_min), int(y_min) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            len_text = f"Len: {length_mm}" if length_mm != "N/A" else f"Len: {int(distance_px)} px"
            cv2.putText(frame, len_text, (int(x_min), int(y_min) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            billet_id += 1
        else:
            print(f"   ❌ IGNORED (Center is left of the line)")

print(f"--------------------------------------------------")
print(f"✅ Total Detections: {total_boxes}")
print(f"✅ Counted Billets: {billet_id - 1}")

# رسم خط شمارش
cv2.line(frame, (line_x_global, 0), (line_x_global, frame.shape[0]), (0, 0, 255), 2)

# ذخیره تصویر خروجی
cv2.imwrite(output_image_path, frame)
print(f"✅ image saved at: {output_image_path}")