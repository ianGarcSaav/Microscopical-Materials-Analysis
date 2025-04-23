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

def preprocess_image(img, median_blur_kernel_size=5, adaptive_thresh_block_size=15, adaptive_thresh_C=5):
    # Convertir a escala de grises si no lo está
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque mediano para reducir ruido
    blurred = cv2.medianBlur(img, median_blur_kernel_size)
    
    # Aplicar umbral adaptativo para segmentar las figuras
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, adaptive_thresh_block_size, adaptive_thresh_C)
    
    # Realizar operaciones morfológicas para limpiar la máscara
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return cleaned == 255
