import numpy as np
import cv2
# import matplotlib.pyplot as plt # Ya no es necesario para guardar directamente
from PIL import Image
from scipy.ndimage import gaussian_filter
import os
import shutil # Añadido para la gestión de directorios, aunque se usará más en main.py
import glob # Añadido para listar archivos, aunque se usará más en main.py
from pathlib import Path # Añadido para manejar rutas, aunque se usará más en main.py
from src.config import ensure_directory_exists

def sobel_filters(img):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    G = np.hypot(Ix, Iy)
    theta = np.arctan2(Iy, Ix)
    return G, theta

def non_max_suppression(G, theta):
    angle = np.rad2deg(theta) % 180
    Z = np.zeros_like(G, dtype=np.float32)

    # Shifted copies for comparison
    for direction in [0, 45, 90, 135]:
        mask = np.zeros_like(G, dtype=bool)

        if direction == 0:
            mask = ((0 <= angle) & (angle < 22.5)) | ((157.5 <= angle) & (angle <= 180))
            before = np.roll(G, 1, axis=1)
            after = np.roll(G, -1, axis=1)
        elif direction == 45:
            mask = (22.5 <= angle) & (angle < 67.5)
            before = np.roll(np.roll(G, 1, axis=0), -1, axis=1)
            after = np.roll(np.roll(G, -1, axis=0), 1, axis=1)
        elif direction == 90:
            mask = (67.5 <= angle) & (angle < 112.5)
            before = np.roll(G, 1, axis=0)
            after = np.roll(G, -1, axis=0)
        elif direction == 135:
            mask = (112.5 <= angle) & (angle < 157.5)
            before = np.roll(np.roll(G, 1, axis=0), 1, axis=1)
            after = np.roll(np.roll(G, -1, axis=0), -1, axis=1)

        local_max = (G >= before) & (G >= after)
        Z[mask & local_max] = G[mask & local_max]

    return Z

def double_threshold(img, low_ratio=0.05, high_ratio=0.15):
    high = img.max() * high_ratio
    low = high * low_ratio

    strong = 255
    weak = 75

    res = np.zeros_like(img, dtype=np.uint8)
    strong_pixels = img >= high
    weak_pixels = (img >= low) & (img < high)

    res[strong_pixels] = strong
    res[weak_pixels] = weak

    return res, weak, strong

def hysteresis(img, weak=75, strong=255):
    from scipy.ndimage import binary_dilation

    strong_mask = (img == strong)
    weak_mask = (img == weak)

    dilated_strong = binary_dilation(strong_mask, structure=np.ones((3,3)))
    connected = dilated_strong & weak_mask

    result = np.copy(img)
    result[weak_mask] = 0
    result[connected] = strong

    return result

def canny_edge_detector(img, sigma=1.2, low_ratio=0.05, high_ratio=0.15):
    blurred = gaussian_filter(img, sigma=sigma)
    G, theta = sobel_filters(blurred)
    suppressed = non_max_suppression(G, theta)
    thresholded, weak, strong = double_threshold(suppressed, low_ratio, high_ratio)
    edges = hysteresis(thresholded, weak, strong)
    return edges.astype(np.uint8)

# Nueva función para procesar una sola imagen y guardar los bordes Canny
def process_image_with_canny(input_image_path, output_directory):
    name = Path(input_image_path).stem
    
    image = cv2.imread(str(input_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Advertencia: No se pudo cargar la imagen {input_image_path}. Saltando...")
        return None

    edges = canny_edge_detector(image)
    
    # Asegurar que el directorio de salida existe
    ensure_directory_exists(output_directory)
    
    # Guardar la imagen de bordes
    output_file_path = output_directory / f"{name}_canny.jpg"
    cv2.imwrite(str(output_file_path), edges)
    print(f"   ✅ Bordes Canny guardados en: {output_file_path}")
    return output_file_path

# Se elimina el bloque if __name__ == '__main__': de este archivo
