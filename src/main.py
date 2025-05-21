import os
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count
from config import img_folder, clusters_folder
from preprocessing import read_image
from labeling import label_components
from edge_detection import canny_edge_detector

def process_single_image(image_data):
    image_file, img_folder, output_folders = image_data
    img_path = os.path.join(img_folder, image_file)
    print(f"Procesando: {image_file}")

    # Extract output folder paths
    edges_folder, closed_edges_folder, filled_mask_folder, labeled_mask_folder = output_folders

    # --- Step 1: Read and preprocess image ---
    # _original.jpg
    img = read_image(img_path)
    if img is None:
        print(f"Error al leer la imagen: {image_file}")
        return

    # --- Step 2: Canny Edge Detection ---
    # _edges.jpg
    # Convert to float32 once and reuse
    img_float = img.astype(np.float32) / 255.0
    edges = canny_edge_detector(img_float, sigma=1.5, low_ratio=0.4, high_ratio=0.10)
    edges_uint8 = edges.astype(np.uint8)
    edges_filename = f"{os.path.splitext(image_file)[0]}_edges.jpg"
    cv2.imwrite(os.path.join(edges_folder, edges_filename), edges_uint8)

    # --- Step 3: Post-process Edges ---
    # _closed_edges.jpg
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    closed_edges = cv2.morphologyEx(edges_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed_edges_filename = f"{os.path.splitext(image_file)[0]}_closed_edges.jpg"
    cv2.imwrite(os.path.join(closed_edges_folder, closed_edges_filename), closed_edges)

    # --- Step 4: Segmentation ---
    # _filled_mask.jpg
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(closed_edges)
    cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)
    filled_mask_filename = f"{os.path.splitext(image_file)[0]}_filled_mask.jpg"
    cv2.imwrite(os.path.join(filled_mask_folder, filled_mask_filename), filled_mask)

    # --- Step 5: Labeling ---
    # _labeledMask.jpg
    labeled_mask, num_labels = label_components(filled_mask)
    print(f"  Encontrados {num_labels-1} componentes en {image_file}")
    
    if labeled_mask.max() > 0:
        labeled_mask_display = (labeled_mask / labeled_mask.max() * 255).astype(np.uint8)
    else:
        labeled_mask_display = labeled_mask.astype(np.uint8)
    
    labeled_mask_colored = cv2.applyColorMap(labeled_mask_display, cv2.COLORMAP_JET)
    labeled_output_path = os.path.join(labeled_mask_folder, f"{os.path.splitext(image_file)[0]}_labeledMask.jpg")
    cv2.imwrite(labeled_output_path, labeled_mask_colored)

def main():
    print("Procesando todas las imagenes en:", img_folder)

    # --- Setup Output Folders ---
    edges_folder = os.path.join(clusters_folder, "Edges")
    closed_edges_folder = os.path.join(clusters_folder, "ClosedEdges")
    filled_mask_folder = os.path.join(clusters_folder, "FilledMask")
    labeled_mask_folder = os.path.join(clusters_folder, "LabeledMask")

    output_folders = [edges_folder, closed_edges_folder, filled_mask_folder, labeled_mask_folder]
    
    # Create output folders
    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)

    # --- Get list of images ---
    image_files = [f for f in os.listdir(img_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print("No se encontraron imágenes en", img_folder)
        return

    # --- Process images in parallel ---
    num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    print(f"Utilizando {num_processes} procesos para el procesamiento paralelo")
    
    # Prepare data for parallel processing
    process_data = [(img_file, img_folder, output_folders) for img_file in image_files]
    
    # Process images in parallel
    with Pool(processes=num_processes) as pool:
        pool.map(process_single_image, process_data)

if __name__ == "__main__":
    print("Programa principal iniciado.")
    main()
    print("Programa principal finalizado.")