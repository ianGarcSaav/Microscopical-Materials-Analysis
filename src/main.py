import os
import numpy as np
import cv2
import psutil
from multiprocessing import Pool
from config import img_folder, clusters_folder
from preprocessing import read_image
from labeling import label_components
from edge_detection import canny_edge_detector

def get_safe_process_count():
    """
    Determina un número seguro de procesos basado en los recursos del sistema.
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

def create_output_folders():
    """
    Crea las carpetas necesarias para almacenar los resultados
    """
    folders = {
        'edges': os.path.join(clusters_folder, 'edges'),
        'closed_edges': os.path.join(clusters_folder, 'closed_edges'),
        'labeled_masks': os.path.join(clusters_folder, 'labeled_masks')
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders

def verify_system_resources():
    """
    Verifica que haya recursos suficientes para procesar imágenes.
    """
    min_memory = 2 * 1024 * 1024 * 1024  # 2GB
    available_memory = psutil.virtual_memory().available
    
    if available_memory < min_memory:
        print(f"Advertencia: Memoria disponible baja ({available_memory / 1024 / 1024 / 1024:.1f}GB)")
        return False
    return True

def process_single_image(image_data):
    """
    Procesa una única imagen con verificaciones explícitas.
    """
    image_file, img_folder, output_folders = image_data
    img_path = os.path.join(img_folder, image_file)
    print(f"Procesando: {image_file}")

    # Extract output folder paths
    edges_folder = output_folders['edges']
    closed_edges_folder = output_folders['closed_edges']
    labeled_masks_folder = output_folders['labeled_masks']

    # --- Step 1: Read and preprocess image ---
    # _original.jpg
    img = read_image(img_path)
    if img is None or img.size == 0:
        print(f"Error al leer la imagen: {image_file}")
        return False, image_file, "Error al leer la imagen"

    # --- Step 2: Canny Edge Detection ---
    # _edges.jpg
    # Convert to float32 once and reuse
    edges = canny_edge_detector(img)
    if edges is None or edges.size == 0:
        print(f"Error en detección de bordes para: {image_file}")
        return False, image_file, "Error en detección de bordes"

    edges_output_path = os.path.join(edges_folder, f"{os.path.splitext(image_file)[0]}_edges.jpg")
    if not cv2.imwrite(edges_output_path, edges):
        print(f"Error al guardar bordes para: {image_file}")
        return False, image_file, "Error al guardar bordes"

    # --- Step 3: Close edges --- 
    # _closed_edges.jpg
    kernel = np.ones((3,3), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    if closed_edges is None or closed_edges.size == 0:
        print(f"Error en cierre de bordes para: {image_file}")
        return False, image_file, "Error en cierre de bordes"

    closed_edges_output_path = os.path.join(closed_edges_folder, f"{os.path.splitext(image_file)[0]}_closed_edges.jpg")
    if not cv2.imwrite(closed_edges_output_path, closed_edges):
        print(f"Error al guardar bordes cerrados para: {image_file}")
        return False, image_file, "Error al guardar bordes cerrados"

    # --- Step 4: Label components ---
    # _labeled_mask.jpg
    labeled_image, num_labels = label_components(closed_edges)
    if labeled_image is None:
        print(f"Error en etiquetado para: {image_file}")
        return False, image_file, "Error en etiquetado"

    print(f"Número de componentes encontrados en {image_file}: {num_labels}")

    # Guardar imagen etiquetada
    labeled_output_path = os.path.join(labeled_masks_folder, f"{os.path.splitext(image_file)[0]}_labeledMask.jpg")
    if not cv2.imwrite(labeled_output_path, cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR)):
        print(f"Error al guardar máscara etiquetada para: {image_file}")
        return False, image_file, "Error al guardar máscara etiquetada"

    return True, image_file, "Procesamiento exitoso"

def process_images_parallel(image_files, img_folder, output_folders):
    """
    Procesa las imágenes en paralelo con manejo de recursos y errores.
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
        for result in pool.imap_unordered(process_single_image, args):
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

def main():
    # Crear las carpetas de salida
    output_folders = create_output_folders()
    
    # Obtener lista de imágenes
    image_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    if not image_files:
        print("No se encontraron imágenes para procesar")
        return
    
    # Procesar imágenes en paralelo
    if not process_images_parallel(image_files, img_folder, output_folders):
        print("Error en el procesamiento paralelo")
        return

if __name__ == "__main__":
    main()
