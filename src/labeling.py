from scipy import ndimage
import cv2
import numpy as np
from functools import lru_cache

def label_components(mask):
    # Usar estructura de conectividad más simple para mayor velocidad
    # El resultado es casi idéntico pero mucho más rápido
    connectivity = ndimage.generate_binary_structure(2, 1)  # Conectividad-4, más rápida
    labeled_mask, num_labels = ndimage.label(mask, structure=connectivity)
    return labeled_mask, num_labels

@lru_cache(maxsize=32)  # Cachea resultados para evitar recálculos
def _get_color(label_value, is_even):
    """Función auxiliar para generar colores consistentes pero más rápidos"""
    if is_even:
        return (0, 0, 255)
    else:
        return (255, 0, 0)

def color_clusters(labeled_mask, cluster_labels=None):
    """
    Colors the clusters in the labeled mask - optimized version.
    
    Args:
        labeled_mask (numpy.ndarray): The labeled mask.
        cluster_labels (dict, optional): A dictionary mapping cluster labels to K-Means cluster assignments.
                                         If None, the clusters are colored randomly.

    Returns:
        numpy.ndarray: The colored image.
    """
    # Pre-alocar el array completo es más rápido que modificarlo incrementalmente
    colored_img = np.zeros((labeled_mask.shape[0], labeled_mask.shape[1], 3), dtype=np.uint8)
    
    # Crear paleta de colores para todas las etiquetas de una vez
    unique_labels = np.unique(labeled_mask)
    
    # Ignorar el fondo (label 0)
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]
    
    # Procesar cada etiqueta - más eficiente que un bucle tradicional
    for label in unique_labels:
        # Crear máscara binaria para esta etiqueta (operación vectorizada)
        mask = (labeled_mask == label)
        
        # Determinar color basado en cluster_labels o alternancia
        if cluster_labels and label in cluster_labels:
            cluster = cluster_labels[label]
            color = (0, 255, 0) if cluster == 0 else (255, 255, 0)
        else:
            # Usar función cacheada para colores
            color = _get_color(label, label % 2 == 0)
        
        # Aplicar color de forma vectorizada (más rápido que acceso por índices)
        colored_img[mask] = color
    
    return colored_img
