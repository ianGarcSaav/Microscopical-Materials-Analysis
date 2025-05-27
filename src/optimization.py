import os
import psutil
from multiprocessing import Pool
import cv2
from typing import List, Tuple, Dict, Any

def get_safe_process_count() -> int:
    """
    Determina un número seguro de procesos basado en los recursos del sistema.
    
    Returns:
        int: Número óptimo de procesos a utilizar
    """
    # Obtener información del sistema
    cpu_count = psutil.cpu_count(logical=False) or 2  # Solo CPUs físicas, default 2
    available_memory = psutil.virtual_memory().available
    total_memory = psutil.virtual_memory().total
    
    # Si hay menos de 2GB de memoria disponible, usar solo 2 procesos
    if available_memory < 2 * 1024 * 1024 * 1024:
        return min(2, cpu_count)
    
    # Si hay menos del 25% de memoria disponible, usar la mitad de CPUs
    if available_memory < total_memory * 0.25:
        return max(1, cpu_count // 2)
    
    # En otros casos, usar CPU count - 1 (máximo 4)
    return min(4, max(1, cpu_count - 1))

def verify_system_resources() -> bool:
    """
    Verifica que haya recursos suficientes para procesar imágenes.
    
    Returns:
        bool: True si hay recursos suficientes, False en caso contrario
    """
    min_memory = 2 * 1024 * 1024 * 1024  # 2GB
    available_memory = psutil.virtual_memory().available
    
    if available_memory < min_memory:
        print(f"Advertencia: Memoria disponible baja ({available_memory / 1024 / 1024 / 1024:.1f}GB)")
        return False
    return True

def process_images_parallel(image_files: List[str], 
                          img_folder: str, 
                          output_folders: Dict[str, str],
                          process_single_image_func: Any) -> bool:
    """
    Procesa las imágenes en paralelo con manejo de recursos y errores.
    
    Args:
        image_files: Lista de nombres de archivos a procesar
        img_folder: Carpeta que contiene las imágenes
        output_folders: Diccionario con las rutas de las carpetas de salida
        process_single_image_func: Función que procesa una única imagen
        
    Returns:
        bool: True si al menos una imagen se procesó exitosamente
    """
    if not verify_system_resources():
        print("Recursos del sistema insuficientes para continuar")
        return False

    num_processes = get_safe_process_count()
    print(f"Utilizando {num_processes} procesos para el procesamiento")
    
    results = []
    failed_images = []
    
    # Preparar argumentos
    args = [(img_file, img_folder, output_folders) for img_file in image_files]
    
    # Procesar imágenes en paralelo
    with Pool(num_processes) as pool:
        for result in pool.imap_unordered(process_single_image_func, args):
            success, img_file, message = result
            if not success:
                failed_images.append((img_file, message))
            results.append(result)
    
    # Reportar resultados
    total = len(image_files)
    successful = len([r for r in results if r[0]])
    failed = len(failed_images)
    
    print(f"\nResumen del procesamiento:")
    print(f"Total de imágenes: {total}")
    print(f"Procesadas exitosamente: {successful}")
    print(f"Fallidas: {failed}")
    
    if failed > 0:
        print("\nImágenes con errores:")
        for img_file, error in failed_images:
            print(f"- {img_file}: {error}")
    
    return successful > 0 