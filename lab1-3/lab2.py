import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = r"C:\Users\andre\Desktop\lab1-3\Monro.jpg"

def apply_canny(image, t1=100, t2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, t1, t2)
    return gray, edges

def compare_thresholds(gray):
    thresholds = [(50, 150), (100, 200), (150, 250), (200, 300)]
    results = []
    for t1, t2 in thresholds:
        edges = cv2.Canny(gray, t1, t2)
        results.append((edges, t1, t2))
    return results

def run_lab2():
    print("ЛР №2: Фильтр Canny")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("Ошибка загрузки")
        return
    
    gray, edges = apply_canny(image, 100, 200)
    threshold_results = compare_thresholds(gray)
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.title('Оригинал')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title('Canny (100, 200)')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    for idx, (result_edges, t1, t2) in enumerate(threshold_results, 3):
        plt.subplot(2, 3, idx)
        plt.title(f'({t1}, {t2})')
        plt.imshow(result_edges, cmap='gray')
        plt.axis('off')
        if idx >= 6:
            break
    
    plt.tight_layout()
    plt.show()
    print("Готово!")

if __name__ == "__main__":
    run_lab2()
