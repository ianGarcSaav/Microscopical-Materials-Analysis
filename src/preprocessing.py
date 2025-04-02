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

def preprocess_image(img, resize_dim=(512, 512), blur_ksize=3, threshold_blocksize=11, threshold_C=2):
    # Convertir a escala de grises (si no lo está)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Cambiar tamaño de la imagen
    img = cv2.resize(img, resize_dim)
    
    # Aplicar desenfoque mediano
    blurred = cv2.medianBlur(img, blur_ksize)
    
    # Aplicar CLAHE para mejorar el contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(blurred)
    
    # Aplicar umbral adaptativo
    thresh = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, threshold_blocksize, threshold_C)
    
    # Operaciones morfológicas
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    return dilated == 255
