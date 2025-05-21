import cv2
import numpy as np
from skimage import measure

def label_components(binary_mask):
    """
    Etiqueta los componentes conectados en una máscara binaria.
    
    Args:
        binary_mask (numpy.ndarray): Máscara binaria de entrada
        
    Returns:
        tuple: (labeled_mask, num_labels) donde labeled_mask es la máscara etiquetada
               y num_labels es el número de etiquetas encontradas
    """
    labeled_mask = measure.label(binary_mask, connectivity=2)
    num_labels = labeled_mask.max()
    return labeled_mask, num_labels