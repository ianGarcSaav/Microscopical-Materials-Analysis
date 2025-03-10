
import cv2
import numpy as np
import os
from config import IMG_PATH

def load_image():
    if not os.path.exists(IMG_PATH):
        # Crear imagen dummy
        dummy = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(dummy, (100, 100), 50, 255, -1)
        cv2.imwrite(IMG_PATH, dummy)
        print(f"Imagen dummy creada en {IMG_PATH}")
    
    img = cv2.imread(IMG_PATH, 0)
    if img is None:
        raise FileNotFoundError(f"Error: No se pudo cargar la imagen en {IMG_PATH}.")
    return img

def preprocess_image(img):
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    return dilated == 255