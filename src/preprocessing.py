import cv2
import numpy as np

def read_image(img_path):
    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {img_path}. Verifica la ruta.")
        exit()
    return img

def preprocess_image(img, median_blur_kernel_size=3, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
                       adaptive_thresh_block_size=11, adaptive_thresh_C=2,
                       morph_kernel_size=3, morph_iterations=1):
    # Convert to grayscale (if not already)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur to reduce noise while preserving edges
    blurred = cv2.medianBlur(img, median_blur_kernel_size)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    contrast_enhanced = clahe.apply(blurred)
    
    # Apply adaptive thresholding for better edge detection
    thresh = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, adaptive_thresh_block_size, adaptive_thresh_C)
    
    # Perform morphological operations to clean up the image
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=morph_iterations)
    dilated = cv2.dilate(eroded, kernel, iterations=morph_iterations)
    
    return dilated == 255