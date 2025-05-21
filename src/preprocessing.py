import cv2
import numpy as np

def read_image(img_path):
    """Reads an image from the specified path as grayscale."""
    # Read as grayscale (0 flag)
    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {img_path}. Verifica la ruta.")
        # Consider raising an exception instead of exiting
        # raise FileNotFoundError(f"Error: No se pudo cargar la imagen en {img_path}")
        exit() # Keep exit for now based on original code
    return img