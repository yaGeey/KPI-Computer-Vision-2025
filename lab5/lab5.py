import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_color_histogram(image_path, bins=(8, 8, 8)):
    try:
        # Завантажуємо зображення
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не вдалося завантажити: {image_path}")
            return None

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Розраховуємо 3D гістограму для всіх трьох каналів (Hue, Saturation, Value)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist) # Нормалізуємо
        return hist.flatten() # Повертаємо сплющену гістограму як 1D вектор ознак
    except Exception as e:
        print(f"{image_path}: {e}")
        return None


def cluster_images(image_folder, num_clusters=4):
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Вилучення ознак з {len(image_paths)} зображень...")

    features_list = []
    valid_paths = []
    # Вилучаємо ознаки (гістограми) для кожного зображення
    for path in image_paths:
        features = extract_color_histogram(path)
        if features is not None:
            features_list.append(features)
            valid_paths.append(path)

    if not features_list:
        print("Не вдалося вилучити ознаки з жодного зображення.")
        return

    features_array = np.array(features_list)

    # Створюємо та навчаємо модель K-Means
    print("Кластеризація K-Means...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    kmeans.fit(features_array)
    labels = kmeans.labels_

    # Створюємо словник для зберігання зображень по кластерах
    clusters = {i: [] for i in range(num_clusters)}
    for i, path in enumerate(valid_paths):
        clusters[labels[i]].append(path)

    # Візуалізація
    print("Візуалізація результатів...")
    fig, axes = plt.subplots(num_clusters, 10, figsize=(20, 8))
    fig.suptitle(f'Результати кластеризації на {num_clusters} групи', fontsize=16)

    for i in range(num_clusters):
        # Заголовок для кожного кластера
        axes[i, 0].set_ylabel(f'Кластер {i + 1}', rotation=0, size='large', labelpad=60)

        cluster_images = clusters[i]
        for j in range(len(axes[i])):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if j < len(cluster_images):
                img = cv2.imread(cluster_images[j])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i, j].imshow(img)
            else:
                # Зайві комірки
                axes[i, j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    cluster_images('data', 4)