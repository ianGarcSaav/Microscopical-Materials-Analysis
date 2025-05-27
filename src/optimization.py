import os
import psutil
from multiprocessing import Pool
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

def get_safe_process_count() -> int:
    """
    Determina un número seguro de procesos basado en los recursos del sistema.
    Optimizado para procesamiento de imágenes con OpenCV.
    """
    # Obtener información del sistema
    cpu_count = psutil.cpu_count(logical=False) or 2
    available_memory = psutil.virtual_memory().available
    total_memory = psutil.virtual_memory().total
    
    # Calcular memoria por proceso considerando el tamaño típico de imagen
    # Asumiendo imágenes de 2048x2048 en escala de grises (4MB)
    mem_per_process = 50 * 1024 * 1024  # 50MB por proceso como margen seguro
    
    # Calcular procesos basados en memoria disponible
    mem_based_processes = max(1, (available_memory // 2) // mem_per_process)
    
    # Si hay menos del 30% de memoria disponible, ser más conservador
    if available_memory < total_memory * 0.3:
        return max(1, min(2, cpu_count, mem_based_processes))
    
    # En otros casos, usar el mínimo entre CPU count y procesos basados en memoria
    return min(4, cpu_count - 1, mem_based_processes)

def verify_system_resources() -> bool:
    """
    Verifica que haya recursos suficientes para procesar imágenes.
    Optimizado para procesamiento de imágenes con OpenCV.
    """
    min_memory = 2 * 1024 * 1024 * 1024  # 2GB mínimo
    available_memory = psutil.virtual_memory().available
    
    # Verificar memoria y CPU
    if available_memory < min_memory:
        print(f"Advertencia: Memoria disponible baja ({available_memory / 1024 / 1024 / 1024:.1f}GB)")
        return False
    
    # Verificar espacio en disco para resultados
    try:
        disk = psutil.disk_usage(os.getcwd())
        if disk.free < min_memory:
            print(f"Advertencia: Espacio en disco bajo ({disk.free / 1024 / 1024 / 1024:.1f}GB)")
            return False
    except:
        pass
    
    return True

def process_images_parallel(image_files: List[str], 
                          img_folder: str, 
                          output_folders: Dict[str, str],
                          process_single_image_func: Any) -> bool:
    """
    Procesa las imágenes en paralelo con manejo optimizado de recursos.
    Implementa las mejores prácticas de OpenCV para procesamiento paralelo.
    """
    if not verify_system_resources():
        print("Recursos del sistema insuficientes para continuar")
        return False

    num_processes = get_safe_process_count()
    print(f"Utilizando {num_processes} procesos para el procesamiento")
    
    # Dividir imágenes en lotes para mejor manejo de memoria
    batch_size = max(1, len(image_files) // num_processes)
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    
    results = []
    failed_images = []
    
    # Procesar por lotes
    for batch in batches:
        args = [(img_file, img_folder, output_folders) for img_file in batch]
        
        # Procesar lote en paralelo
        with Pool(num_processes) as pool:
            batch_results = list(pool.imap_unordered(process_single_image_func, args))
            
            # Procesar resultados del lote
            for result in batch_results:
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