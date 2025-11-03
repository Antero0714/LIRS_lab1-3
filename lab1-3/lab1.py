import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = r"C:\Users\andre\Desktop\lab1-3\Monro.jpg"

def blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)

def unsharp_mask(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

def edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    edge = cv2.convertScaleAbs(edge)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

def emboss(img):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)

def combine(blurred, edge, sharpened):
    b = np.uint8(blurred)
    e = np.uint8(edge)
    s = np.uint8(np.clip(sharpened, 0, 255))
    
    result = cv2.addWeighted(b, 0.5, e, 0.5, 0)
    result = cv2.addWeighted(result, 0.5, s, 0.5, 0)
    return result

img = cv2.imread(IMAGE_PATH)
if img is None:
    print("Ошибка загрузки")
    exit()

print("Обработка...")

blurred = blur(img)
sharpened = unsharp_mask(img)
edge = edges(img)
embossed = emboss(img)
combined = combine(blurred, edge, sharpened)

print("Показ результатов...")

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Оригинал')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
plt.title('Размытие')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(np.uint8(np.clip(sharpened, 0, 255)), cv2.COLOR_BGR2RGB))
plt.title('Резкость')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(edge, cv2.COLOR_BGR2RGB))
plt.title('Границы')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(np.uint8(np.clip(embossed, 0, 255)), cv2.COLOR_BGR2RGB))
plt.title('Тиснение')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
plt.title('Комбо')
plt.axis('off')

plt.tight_layout()
plt.show()

print("Готово!")
