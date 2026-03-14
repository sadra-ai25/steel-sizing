import cv2
import yaml
import os
import numpy as np

# مسیر تصویر ورودی
input_image_path = '1.jpg'
config_path = 'config/config.yaml'

# بررسی وجود تصویر ورودی
if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"تصویر {input_image_path} یافت نشد!")

# خواندن تصویر
img = cv2.imread(input_image_path)
if img is None:
    raise ValueError(f"خطا در خواندن تصویر {input_image_path}")

# ایجاد کپی برای رسم
img_copy = img.copy()

# تنظیمات اولیه ROI (کل تصویر)
roi = {
    'x': 0,
    'y': 0,
    'width': img.shape[1],
    'height': img.shape[0]
}
line_x = None  # مختصات X خط شمارش

# متغیرهای مربوط به رسم
start_point = None
end_point = None
drawing = False
mode = 'roi'  # حالت‌ها: 'roi' یا 'line'

# تابعی برای تغییر اندازه تصویر برای نمایش
def resize_image(image, max_width=1200):
    """تغییر اندازه تصویر برای جا شدن در صفحه نمایش"""
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        # استفاده از cv2.resize به جای تابع فرضی
        return cv2.resize(image, (max_width, new_h)), ratio
    return image, 1.0

# آماده‌سازی برای نمایش
display_img, scale_ratio = resize_image(img_copy)

def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, img_copy, display_img, line_x, mode
    
    # تبدیل مختصات ماوس به مختصات تصویر اصلی
    real_x = int(x / scale_ratio)
    real_y = int(y / scale_ratio)

    if mode == 'roi':
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (real_x, real_y)
            drawing = True
            img_copy = img.copy()
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (real_x, real_y)
                img_copy = img.copy()
                cv2.rectangle(img_copy, start_point, end_point, (0, 255, 0), 2)
                # به‌روزرسانی تصویر نمایشی
                display_img, _ = resize_image(img_copy)
                cv2.imshow('Select ROI', display_img)
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (real_x, real_y)
            img_copy = img.copy()
            cv2.rectangle(img_copy, start_point, end_point, (0, 255, 0), 2)
            display_img, _ = resize_image(img_copy)
            cv2.imshow('Select ROI', display_img)

    elif mode == 'line':
        if event == cv2.EVENT_LBUTTONDOWN:
            line_x = real_x
            img_copy = img.copy()
            # رسم خط عمودی قرمز
            cv2.line(img_copy, (line_x, 0), (line_x, img.shape[0]), (0, 0, 255), 2)
            display_img, _ = resize_image(img_copy)
            cv2.imshow('Select ROI', display_img)

# ایجاد پنجره و تنظیم callback
cv2.namedWindow('Select ROI')
cv2.setMouseCallback('Select ROI', draw_rectangle)

print("✅ تصویر بارگذاری شد.")
print("✅ برای رسم مستطیل ROI، کلید 'r' را فشار دهید (پیش‌فرض).")
print("✅ برای انتخاب نقطه خط شمارش (Line X)، کلید 'l' را فشار دهید.")
print("✅ برای ذخیره تنظیمات، کلید 's' را فشار دهید.")
print("✅ برای انصراف، کلید 'q' را فشار دهید.")

while True:
    cv2.imshow('Select ROI', display_img)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r'):  # تغییر حالت به ROI
        mode = 'roi'
        print("🔲 حالت انتخاب ROI فعال شد.")
        img_copy = img.copy()
        display_img, _ = resize_image(img_copy)
    
    elif key == ord('l'):  # تغییر حالت به Line
        mode = 'line'
        print("📍 حالت انتخاب نقطه خط (Line X) فعال شد.")
        img_copy = img.copy()
        display_img, _ = resize_image(img_copy)
    
    elif key == ord('s'):  # ذخیره تنظیمات
        # محاسبه نهایی ROI
        if start_point and end_point:
            x1, y1 = start_point
            x2, y2 = end_point
            roi['x'] = min(x1, x2)
            roi['y'] = min(y1, y2)
            roi['width'] = abs(x2 - x1)
            roi['height'] = abs(y2 - y1)
        
        # بررسی وجود line_x
        if line_x is None:
            print("⚠️ هشدار: نقطه خط (Line X) انتخاب نشده است. مقدار پیش‌فرض استفاده می‌شود.")
            line_x = roi['x'] + roi['width'] // 2
        
        # ذخیره در config.yaml
        if not os.path.exists('config'):
            os.makedirs('config')
        
        config_data = {
            'roi': roi,
            'line_x': line_x
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        print(f"\n✅ تنظیمات ذخیره شد:")
        print(f"  ROI -> x: {roi['x']}, y: {roi['y']}, w: {roi['width']}, h: {roi['height']}")
        print(f"  Line X -> {line_x}")
        print(f"  فایل config.yaml به‌روزرسانی شد.")
        break
    
    elif key == ord('q'):  # انصراف
        print("\n❌ عملیات لغو شد.")
        break

cv2.destroyAllWindows()