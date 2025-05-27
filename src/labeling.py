import numpy as np
from skimage import measure, segmentation
from skimage.color import label2rgb
import cv2

def label_components(binary_mask, min_size=25):
    """
    Etiqueta los componentes conectados en una máscara binaria usando scikit-image.
    Optimizado para trabajar con regiones pequeñas y detalles finos.
    
    Args:
        binary_mask (numpy.ndarray): Máscara binaria de entrada
        min_size (int): Tamaño mínimo de los componentes a considerar
        
    Returns:
        tuple: (labeled_mask, num_labels) donde labeled_mask es la máscara etiquetada
               y num_labels es el número de etiquetas encontradas
    """
    # Asegurar que la máscara sea binaria
    binary_mask = binary_mask > 0
    
    # Etiquetar componentes conectados con conectividad-8 para mejor detección de detalles
    labeled_mask = measure.label(binary_mask, connectivity=2, background=0)
    
    # Obtener propiedades de las regiones
    regions = measure.regionprops(labeled_mask)
    
    # Filtrar por tamaño mínimo y crear máscara
    # Usando dtype=np.uint8 para mejor compatibilidad con OpenCV
    mask = np.zeros_like(labeled_mask, dtype=np.uint8)
    valid_regions = 0
    
    for region in regions:
        if region.area >= min_size:
            mask[labeled_mask == region.label] = 255
            valid_regions += 1
    
    if valid_regions == 0:
        return None, 0
    
    # Re-etiquetar solo los componentes que pasaron el filtro
    labeled_mask = measure.label(mask > 0, connectivity=2, background=0)
    
    # Crear visualización con colores aleatorios
    # Usando un mapa de colores más contrastado para regiones pequeñas
    label_image = label2rgb(labeled_mask, bg_label=0, bg_color=(0, 0, 0), 
                          colors=[(1,0,0), (0,1,0), (0,0,1), (1,1,0), 
                                 (1,0,1), (0,1,1), (0.5,0.5,0.5)])
    
    # Convertir a uint8 para OpenCV y asegurar contraste
    label_image = (label_image * 255).astype(np.uint8)
    
    # Aplicar un pequeño realce de contraste para mejor visualización
    for channel in range(3):
        label_image[:,:,channel] = cv2.equalizeHist(label_image[:,:,channel])
    
    return label_image, valid_regions 