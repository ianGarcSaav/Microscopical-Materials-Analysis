import numpy as np
from skimage import measure, segmentation
from skimage.color import label2rgb
import cv2

def label_components(binary_mask, min_size=50):
    """
    Etiqueta los componentes conectados en una máscara binaria usando scikit-image.
    
    Args:
        binary_mask (numpy.ndarray): Máscara binaria de entrada
        min_size (int): Tamaño mínimo de los componentes a considerar
        
    Returns:
        tuple: (labeled_mask, num_labels) donde labeled_mask es la máscara etiquetada
               y num_labels es el número de etiquetas encontradas
    """
    # Asegurar que la máscara sea binaria
    binary_mask = binary_mask > 0
    
    # Etiquetar componentes conectados
    labeled_mask = measure.label(binary_mask, connectivity=2, background=0)
    
    # Obtener propiedades de las regiones
    regions = measure.regionprops(labeled_mask)
    
    # Filtrar por tamaño mínimo y crear máscara
    mask = np.zeros_like(labeled_mask, dtype=bool)
    for region in regions:
        if region.area >= min_size:
            mask[labeled_mask == region.label] = True
    
    # Re-etiquetar solo los componentes que pasaron el filtro
    labeled_mask = measure.label(mask, connectivity=2, background=0)
    num_labels = len(np.unique(labeled_mask)) - 1  # Restar 1 para no contar el fondo
    
    # Si no se encontraron componentes, retornar None
    if num_labels == 0:
        return None, 0
        
    # Crear visualización con colores aleatorios
    label_image = label2rgb(labeled_mask, bg_label=0, bg_color=(0, 0, 0))
    
    # Convertir a uint8 para OpenCV
    label_image = (label_image * 255).astype(np.uint8)
    
    return label_image, num_labels 