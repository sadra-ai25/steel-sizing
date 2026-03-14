def get_dimension_mm(width_px, height_px, mapping_config):
    """
    تبدیل ابعاد پیکسلی به میلی‌متر با استفاده از آستانه
    :param width_px: عرض در پیکسل
    :param height_px: ارتفاع در پیکسل
    :param mapping_config: تنظیمات مپینگ
    :return: رشته ابعاد (مثلاً "15x15") یا None
    """
    for dim, ranges in mapping_config['dimensions'].items():
        if (ranges['min'] <= width_px <= ranges['max'] and 
            ranges['min'] <= height_px <= ranges['max']):
            return dim
    return None

def get_length_mm(distance_px, mapping_config):
    """
    تبدیل طول پیکسلی به میلی‌متر
    :param distance_px: فاصله بین دو نقطه در پیکسل
    :param mapping_config: تنظیمات مپینگ
    :return: رشته طول (مثلاً "12000") یا None
    """
    for length, ranges in mapping_config['length'].items():
        if ranges['min'] <= distance_px <= ranges['max']:
            return length
    return None