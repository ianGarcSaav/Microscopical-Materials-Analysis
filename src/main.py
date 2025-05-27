import os
import numpy as np
import cv2
from config import img_folder, clusters_folder
from preprocessing import read_image
from labeling import label_components
from edge_detection import canny_edge_detector
from closing_edge import process_closed_edges, MorphologyParams, OPTIMAL_PARAMS
from optimization import process_images_parallel

def create_output_folders():
    """
    Crea las carpetas necesarias para almacenar los resultados
    """
    folders = {
        'edges': os.path.join(clusters_folder, '1.edges'),
        'closed_edges': os.path.join(clusters_folder, '2.closed_edges'),
        'filled_regions': os.path.join(clusters_folder, '3.filled_regions'),
        'boundaries': os.path.join(clusters_folder, '4.boundaries'),
        'labeled_masks': os.path.join(clusters_folder, '5.labeled_masks')
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders

def process_single_image(image_data):
    """
    Procesa una única imagen con verificaciones explícitas.
    """
    image_file, img_folder, output_folders = image_data
    img_path = os.path.join(img_folder, image_file)
    print(f"Procesando: {image_file}")

    try:
        # Extract output folder paths
        edges_folder = output_folders['edges']
        closed_edges_folder = output_folders['closed_edges']
        filled_regions_folder = output_folders['filled_regions']
        boundaries_folder = output_folders['boundaries']
        labeled_masks_folder = output_folders['labeled_masks']

        # --- Step 1: Read and preprocess image ---
        img = read_image(img_path)
        if img is None or img.size == 0:
            return False, image_file, "Error al leer la imagen"

        # --- Step 2: Canny Edge Detection ---
        edges = canny_edge_detector(img)
        if edges is None or edges.size == 0:
            return False, image_file, "Error en detección de bordes"

        edges_output_path = os.path.join(edges_folder, f"{os.path.splitext(image_file)[0]}_edges.jpg")
        if not cv2.imwrite(edges_output_path, edges):
            return False, image_file, "Error al guardar bordes"

        # --- Step 3: Process Closed Edges ---
        try:
            closed, filled, boundaries = process_closed_edges(edges, OPTIMAL_PARAMS)
        except Exception as e:
            return False, image_file, f"Error en procesamiento de bordes: {str(e)}"
        
        # Guardar resultados del procesamiento de bordes
        base_name = os.path.splitext(image_file)[0]
        
        # Guardar bordes cerrados
        closed_edges_path = os.path.join(closed_edges_folder, f"{base_name}_closed_edges.jpg")
        if not cv2.imwrite(closed_edges_path, closed):
            return False, image_file, "Error al guardar bordes cerrados"
        
        # Guardar regiones rellenas
        filled_path = os.path.join(filled_regions_folder, f"{base_name}_filled.jpg")
        if not cv2.imwrite(filled_path, filled):
            return False, image_file, "Error al guardar regiones rellenas"
        
        # Guardar fronteras
        boundaries_path = os.path.join(boundaries_folder, f"{base_name}_boundaries.jpg")
        if not cv2.imwrite(boundaries_path, boundaries):
            return False, image_file, "Error al guardar fronteras"

        # --- Step 4: Label components ---
        try:
            labeled_image, num_labels = label_components(filled)
            if labeled_image is None:
                return False, image_file, "Error en etiquetado"

            print(f"Número de componentes encontrados en {image_file}: {num_labels}")

            # Guardar imagen etiquetada
            labeled_output_path = os.path.join(labeled_masks_folder, f"{base_name}_labeledMask.jpg")
            if not cv2.imwrite(labeled_output_path, labeled_image):
                return False, image_file, "Error al guardar máscara etiquetada"

        except Exception as e:
            return False, image_file, f"Error en etiquetado: {str(e)}"

        return True, image_file, "Procesamiento exitoso"
        
    except Exception as e:
        return False, image_file, f"Error inesperado: {str(e)}"

def main():
    # Crear las carpetas de salida
    output_folders = create_output_folders()
    
    # Obtener lista de imágenes
    image_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    if not image_files:
        print("No se encontraron imágenes para procesar")
        return
    
    # Procesar imágenes en paralelo
    if not process_images_parallel(image_files, img_folder, output_folders, process_single_image):
        print("Error en el procesamiento paralelo")
        return

if __name__ == "__main__":
    main()
