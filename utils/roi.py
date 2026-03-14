import cv2

def apply_roi(frame, roi):
    """
    اعمال ROI بر روی فریم بدون تغییر اندازه فریم اصلی
    :param frame: فریم اصلی
    :param roi: [x, y, width, height]
    :return: تصویر برش‌خورده (ROI)
    """
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]

def get_original_coords(roi, bbox_roi):
    """
    تبدیل مختصات Bounding Box از ROI به فریم اصلی
    :param roi: [x, y, width, height]
    :param bbox_roi: [x_min, y_min, x_max, y_max] در ROI
    :return: مختصات در فریم اصلی
    """
    x_roi, y_roi, _, _ = roi
    x_min, y_min, x_max, y_max = bbox_roi
    return (
        x_min + x_roi,
        y_min + y_roi,
        x_max + x_roi,
        y_max + y_roi
    )