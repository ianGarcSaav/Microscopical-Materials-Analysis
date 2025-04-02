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

def preprocess_image(img):
    # Convert to grayscale (if not already)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur to reduce noise while preserving edges
    blurred = cv2.medianBlur(img, 3)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(blurred)
    
    # Apply adaptive thresholding for better edge detection
    thresh = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Perform morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    return dilated == 255
