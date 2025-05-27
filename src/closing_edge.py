import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class MorphologyParams:
    """Parámetros para operaciones morfológicas."""
    close_kernel_size: Tuple[int, int] = (3, 3)
    min_area: int = 150
    contour_thickness: int = 2

def validate_input(img: np.ndarray, operation_name: str) -> bool:
    """
    Valida la entrada de imagen para las operaciones.
    
    Args:
        img (np.ndarray): Imagen a validar
        operation_name (str): Nombre de la operación para el mensaje de error
        
    Returns:
        bool: True si la imagen es válida
    
    Raises:
        ValueError: Si la imagen no es válida
    """
    if img is None:
        raise ValueError(f"{operation_name}: La imagen de entrada es None")
    if img.size == 0:
        raise ValueError(f"{operation_name}: La imagen está vacía")
    if not isinstance(img, np.ndarray):
        raise ValueError(f"{operation_name}: La entrada debe ser un array NumPy")
    return True

def close_edges(edges: np.ndarray, params: Optional[MorphologyParams] = None) -> np.ndarray:
    """
    Cierra los bordes de la imagen usando una única operación morfológica.
    
    Args:
        edges (np.ndarray): Imagen binaria con bordes (1.edges)
        params (MorphologyParams, optional): Parámetros de morfología
        
    Returns:
        np.ndarray: Imagen con bordes cerrados (2.closed_edges)
    """
    validate_input(edges, "close_edges")
    if params is None:
        params = MorphologyParams()
    
    # Convertir a binario y uint8
    edges = (edges > 0).astype(np.uint8) * 255
    
    # Aplicar cierre morfológico con kernel (3,3)
    kernel = np.ones(params.close_kernel_size, np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

def fill_regions(closed_edges: np.ndarray, params: Optional[MorphologyParams] = None) -> np.ndarray:
    """
    Rellena las regiones cerradas.
    
    Args:
        closed_edges (np.ndarray): Imagen con bordes cerrados (2.closed_edges)
        params (MorphologyParams, optional): Parámetros de morfología
        
    Returns:
        np.ndarray: Imagen con regiones rellenas (3.filled_regions)
    """
    validate_input(closed_edges, "fill_regions")
    if params is None:
        params = MorphologyParams()
    
    # Preparar imagen y máscara para flood fill
    h, w = closed_edges.shape[:2]
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear imagen para el relleno
    filled = np.zeros_like(closed_edges)
    
    # Rellenar cada contorno válido
    for contour in contours:
        if cv2.contourArea(contour) >= params.min_area:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.floodFill(filled, fill_mask, (cx, cy), 255)
    
    return filled

def extract_boundaries(filled_regions: np.ndarray, params: Optional[MorphologyParams] = None) -> np.ndarray:
    """
    Extrae las fronteras de las regiones rellenas.
    
    Args:
        filled_regions (np.ndarray): Imagen con regiones rellenas (3.filled_regions)
        params (MorphologyParams, optional): Parámetros de morfología
        
    Returns:
        np.ndarray: Imagen con fronteras (4.boundaries)
    """
    validate_input(filled_regions, "extract_boundaries")
    if params is None:
        params = MorphologyParams()
    
    # Encontrar contornos en las regiones rellenas
    contours, _ = cv2.findContours(filled_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear imagen para las fronteras
    boundaries = np.zeros_like(filled_regions)
    
    # Dibujar solo los contornos válidos
    for contour in contours:
        if cv2.contourArea(contour) >= params.min_area:
            cv2.drawContours(boundaries, [contour], -1, 255, params.contour_thickness)
    
    return boundaries

def process_closed_edges(edges: np.ndarray, params: Optional[MorphologyParams] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Procesa los bordes siguiendo un flujo lineal.
    
    Args:
        edges (np.ndarray): Imagen original con bordes (1.edges)
        params (MorphologyParams, optional): Parámetros de morfología
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - closed_edges: Bordes cerrados (2.closed_edges)
            - filled_regions: Regiones rellenas (3.filled_regions)
            - boundaries: Fronteras (4.boundaries)
            - enhanced: Imagen final mejorada (5.enhanced)
    """
    validate_input(edges, "process_closed_edges")
    if params is None:
        params = MorphologyParams()
    
    try:
        # 2. Cerrar bordes
        closed_edges = close_edges(edges, params)
        
        # 3. Rellenar regiones
        filled_regions = fill_regions(closed_edges, params)
        
        # 4. Extraer fronteras
        boundaries = extract_boundaries(filled_regions, params)
        
        # 5. Imagen mejorada (usando directamente filled_regions)
        enhanced = filled_regions
        
        return closed_edges, filled_regions, boundaries, enhanced
        
    except Exception as e:
        raise RuntimeError(f"Error en el procesamiento: {str(e)}")

def improve_edge_detection(img: np.ndarray) -> np.ndarray:
    """
    Mejora la detección de bordes usando técnicas avanzadas de OpenCV.
    """
    # Reducción de ruido
    denoised = cv2.fastNlMeansDenoising(img)
    
    # Mejora de contraste
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def improve_contour_closing(edges: np.ndarray) -> np.ndarray:
    """
    Mejora el cierre de contornos usando técnicas morfológicas avanzadas.
    """
    # Kernel adaptativo basado en el tamaño de la imagen
    kernel_size = max(3, min(edges.shape) // 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Cierre morfológico adaptativo
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Eliminación de ruido pequeño
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, 
                            cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
    
    return closed

def optimize_array_operations(img: np.ndarray) -> np.ndarray:
    """
    Optimiza operaciones usando funciones vectorizadas de NumPy.
    """
    # Usar operaciones vectorizadas en lugar de bucles
    normalized = np.clip((img - img.mean()) / img.std(), -3, 3)
    return normalized

# Parámetros optimizados
OPTIMAL_PARAMS = MorphologyParams(
    close_kernel_size=(3, 3),    # Kernel (3,3) para cierre morfológico
    min_area=150,               # Área mínima para filtrar regiones pequeñas
    contour_thickness=2         # Grosor de contornos
) 