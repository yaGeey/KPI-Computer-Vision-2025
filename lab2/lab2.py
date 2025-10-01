import cv2
import numpy as np

# --- ПАРАМЕТРИ ---
# Параметри для морфологічної очистки
# Ядро для видалення дрібного шуму (цяток)
OPEN_KERNEL_SIZE = 3
# Ядро для заповнення дірок у контурах будівель
CLOSE_KERNEL_SIZE = 5

# Параметри для фінальної фільтрації контурів
MIN_BUILDING_AREA_FINAL = 1500
MAX_BUILDING_AREA_FINAL = 400000
MIN_ASPECT_RATIO = 0.15
MAX_ASPECT_RATIO = 6.0

# --- КОД ---

# 1. Завантажуємо зображення
image = cv2.imread('bing.png')
if image is None:
    print("Помилка: не вдалося завантажити зображення 'bing.png'")
    exit()

output_image = image.copy()

# 2. Перетворюємо зображення в відтінки сірого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Бінаризація зображення
# Використовуємо адаптивний поріг, він краще працює з нерівномірним освітленням
# ніж фіксований поріг cv2.threshold.
binary_mask = cv2.adaptiveThreshold(
    gray_image, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV, # Інвертуємо, бо будівлі часто світліші за тіні
    11, # Розмір блоку для аналізу
    5   # Константа, що віднімається від середнього
)

# 4. Морфологічна очистка маски
# Спочатку видаляємо дрібні білі цятки (шум)
open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE))
mask_no_noise = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

# Потім заповнюємо дірки всередині контурів будівель
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
cleaned_mask = cv2.morphologyEx(mask_no_noise, cv2.MORPH_CLOSE, close_kernel, iterations=1)

# 5. Пошук та фільтрація контурів
contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

building_count = 0
for contour in contours:
    area = cv2.contourArea(contour)
    # Фільтруємо контури за площею
    if MIN_BUILDING_AREA_FINAL < area < MAX_BUILDING_AREA_FINAL:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)

        if h == 0: continue
        aspect_ratio = float(w) / h

        # Фільтруємо за співвідношенням сторін і кількістю вершин
        if len(approx) >= 4 and MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
            cv2.drawContours(output_image, [approx], -1, (0, 255, 0), 2)
            building_count += 1

print(f"Знайдено будівель: {building_count}")

# 6. Вивід результатів
cv2.putText(output_image, f"Found buildings: {building_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

# Показуємо очищену маску та фінальний результат
cv2.imshow('Cleaned Mask', cleaned_mask)
cv2.imshow('Final Detected Buildings', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()