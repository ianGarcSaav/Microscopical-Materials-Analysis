import cv2
import numpy as np

def read_image(img_path, resize_factor=None):
    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {img_path}. Verifica la ruta.")
        exit()
    
    # Redimensionar para procesar más rápido si se especifica
    if resize_factor and resize_factor < 1.0 and resize_factor > 0:
        width = int(img.shape[1] * resize_factor)
        height = int(img.shape[0] * resize_factor)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    
    return img

def preprocess_image(img, sigma=0.8, low_threshold=30, high_threshold=90):
    """
    Preprocesa una imagen utilizando detección de bordes Canny con suavizado menos agresivo.
    
    Args:
        img: Imagen de entrada
        sigma: Sigma para filtro Gaussiano (reducido a 0.8 para suavizado menos agresivo)
        low_threshold: Umbral bajo para Canny
        high_threshold: Umbral alto para Canny
        
    Returns:
        Máscara binaria procesada
    """
    # Convertir a escala de grises si no lo está
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Opcional: Mejorar contraste para detectar mejor los bordes tenues
    
    # a) Suavizado con filtro Gaussiano (sigma reducido a 0.8)
    kernel_size = int(2 * round(3 * sigma) + 1)  # Tamaño apropiado para sigma dado
    smoothed = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    # b,c,d,e) Aplicar Canny con umbrales menos agresivos
    edges = cv2.Canny(smoothed, low_threshold, high_threshold, L2gradient=True)
    
    # Mejorar la conectividad de los bordes detectados para evitar contornos fragmentados
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Enfoque alternativo para extraer regiones de la imagen:
    # En lugar de usar floodFill, podemos usar técnicas morfológicas
    # para cerrar contornos y extraer regiones
    
    # 1. Cerrar pequeñas brechas en los bordes
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 2. Encontrar contornos externos
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Dibujar contornos rellenados en una nueva imagen
    filled = np.zeros_like(closed_edges)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    
    # 4. Combinar con los bordes originales para no perder detalles
    result = cv2.bitwise_or(filled, edges)
    
    return result > 0
