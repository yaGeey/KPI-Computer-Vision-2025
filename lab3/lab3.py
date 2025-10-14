import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_object_features(image):
    # базова фільтрація і маскування
    blurred_image = cv2.GaussianBlur(image, (11, 11), 0)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 50, 30])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv_image, lower, upper)

    # очищення маски
    kernel = np.ones((2, 2), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=3)

    # виділення об'єкта
    object_only = cv2.bitwise_and(image, image, mask=mask_closed)

    # колірна гістограма
    hist = cv2.calcHist([hsv_image], [0], mask_closed, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    visualizations = {
        'original': cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        'mask': mask_closed,
        'object_only': cv2.cvtColor(object_only, cv2.COLOR_BGR2RGB)
    }
    return hist, visualizations

image1 = cv2.imread('1.jpg')
image2 = cv2.imread('2.jpg')

hist1, vis1 = extract_object_features(image1)
hist2, vis2 = extract_object_features(image2)

similarity_same_object = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

# візуалізація
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f"Схожість за гістограмою: {similarity_same_object:.4f}\n(0 = ідеально схожі, 1 = абсолютно різні)")

axes[0, 0].imshow(vis1['original'])
axes[0, 0].set_title('Зображення 1: Оригінал')
axes[0, 0].axis('off')

axes[0, 1].imshow(vis1['mask'], cmap='gray')
axes[0, 1].set_title('Зображення 1: Маска кольору')
axes[0, 1].axis('off')

axes[0, 2].imshow(vis1['object_only'])
axes[0, 2].set_title('Зображення 1: Виділений об\'єкт')
axes[0, 2].axis('off')

axes[1, 0].imshow(vis2['original'])
axes[1, 0].set_title('Зображення 2: Оригінал')
axes[1, 0].axis('off')

axes[1, 1].imshow(vis2['mask'], cmap='gray')
axes[1, 1].set_title('Зображення 2: Маска кольору')
axes[1, 1].axis('off')

axes[1, 2].imshow(vis2['object_only'])
axes[1, 2].set_title('Зображення 2: Виділений об\'єкт')
axes[1, 2].axis('off')

plt.show()