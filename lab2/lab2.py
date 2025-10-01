import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_BUILDING_AREA_FINAL = 1000
MAX_BUILDING_AREA_FINAL = 400000
MIN_ASPECT_RATIO = 0.15
MAX_ASPECT_RATIO = 6.0

# Завантажуємо зображення
image = cv2.imread('bing.png')
output_image = image.copy()

# ВИДАЛЕННЯ РОСЛИННОСТІ ---
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_green = np.array([56, 0, 20])
upper_green = np.array([255, 255, 255])
no_green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Видаляємо зелені зони з оригінального зображення
image_no_vegetation = cv2.bitwise_and(image, image, mask=no_green_mask)
gray_image = cv2.cvtColor(image_no_vegetation, cv2.COLOR_BGR2GRAY)
plt.imshow(image_no_vegetation)
plt.show()

_, building_mask = cv2.threshold(gray_image, 37, 255, cv2.THRESH_BINARY)
plt.imshow(building_mask)
plt.show()

# Спочатку закриття (заповнюємо дірки), потім відкриття (видаляємо шум)
cleaned_mask = cv2.morphologyEx(building_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8)))
closed_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12)))
plt.imshow(cleaned_mask)
plt.show()
plt.imshow(closed_mask)
plt.show()

# ПОШУК ТА ФІЛЬТРАЦІЯ КОНТУРІВ
contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

building_count = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if MIN_BUILDING_AREA_FINAL < area:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        rect = cv2.minAreaRect(contour)
        (cx, cy), (width, height), angle = rect
        aspect_ratio = max(width / height, height / width) # беремо більшу/меншу сторону — щоб AR не залежав від повороту

        if MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(output_image, [box], -1, (0, 255, 0), 2)
            building_count += 1

print(f"Знайдено будівель: {building_count}")
# Додаємо текст на зображення
cv2.putText(output_image, f"Found buildings: {building_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
plt.imshow(output_image)
plt.show()