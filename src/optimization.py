import os
import psutil
from multiprocessing import Pool
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

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
    Procesa en batches ordenados para mantener el orden de las imágenes.
    """
    if not verify_system_resources():
        print("Recursos del sistema insuficientes para continuar")
        return False

    num_processes = min(4, get_safe_process_count())
    print(f"\nUtilizando {num_processes} procesos para el procesamiento")
    
    # Procesar todas las imágenes en orden, en batches de 2
    batch_size = 2  # Tamaño fijo de batch
    total_images = len(image_files)
    results = []
    failed_images = []
    
    # Crear barra de progreso principal
    with tqdm(total=total_images, 
              desc="Progreso total", 
              unit="img",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        # Procesar por batches ordenados
        for start_idx in range(0, total_images, batch_size):
            end_idx = min(start_idx + batch_size, total_images)
            current_batch = image_files[start_idx:end_idx]
            
            # Mostrar información del batch actual
            batch_info = f"Batch {start_idx//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}"
            tqdm.write(f"\n{batch_info}")
            tqdm.write(f"Procesando: {', '.join(current_batch)}")
            
            args = [(img_file, img_folder, output_folders) for img_file in current_batch]
            
            # Procesar batch actual en paralelo
            with Pool(min(len(current_batch), num_processes)) as pool:
                batch_results = pool.map(process_single_image_func, args)
                
                # Procesar resultados del batch actual (manteniendo orden)
                for result in batch_results:
                    success, img_file, message = result
                    if success:
                        tqdm.write(f"✓ {img_file}: Procesada exitosamente")
                    else:
                        tqdm.write(f"✗ {img_file}: {message}")
                        failed_images.append((img_file, message))
                    results.append(result)
                    pbar.update(1)  # Actualizar barra de progreso
    
    # Reportar resultados finales
    successful = len([r for r in results if r[0]])
    failed = len(failed_images)
    
    print(f"\nResumen del procesamiento:")
    print(f"Total de imágenes: {total_images}")
    print(f"Procesadas exitosamente: {successful}")
    print(f"Fallidas: {failed}")
    
    if failed > 0:
        print("\nImágenes con errores:")
        for img_file, error in failed_images:
            print(f"- {img_file}: {error}")
    
    return successful > 0 