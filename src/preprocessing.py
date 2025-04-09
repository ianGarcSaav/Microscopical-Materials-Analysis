import cv2
import numpy as np

def create_dummy_image(img_path):
    dummy = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(dummy, (100, 100), 50, 255, -1)
    cv2.imwrite(img_path, dummy)
    print(f"Imagen dummy creada en {img_path}")

def read_image(img_path):
    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {img_path}. Verifica la ruta.")
        exit()
    return img

def preprocess_image(img, threshold_value=127):
    # Convertir a escala de grises si no lo está
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un umbral global simple
    _, binary_mask = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    return binary_mask == 255
