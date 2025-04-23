import os
import numpy as np
import cv2  # Importar para guardar imágenes intermedias
from config import img_folder, clusters_folder, csv_folder, histogram_folder, pixels_to_um
from preprocessing import read_image, preprocess_image
from labeling import color_clusters  # Keep color_clusters for now, but check its behavior
from measurement import measure_properties, save_measurements_to_csv
from visualization import save_colored_clusters, generate_histograms
from edge_detection import canny_edge_detector

def main():
    print("Procesando todas las imagenes en:", img_folder)

    # Crear carpetas para coloredClusters, Mask, y Edges (Removed LabeledMask)
    colored_clusters_folder = os.path.join(clusters_folder, "coloredClusters")
    mask_folder = os.path.join(clusters_folder, "Mask")
    edges_folder = os.path.join(clusters_folder, "Edges")
    os.makedirs(colored_clusters_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(edges_folder, exist_ok=True)

    # Listar solo archivos de imagen
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No se encontraron imágenes en", img_folder)
        return

    for image_file in image_files:
        img_path = os.path.join(img_folder, image_file)
        # print("Procesando:", image_file)

        # Paso 1: Leer imagen
        img = read_image(img_path)

        # Paso 2: Preprocesamiento (Using original image 'img')
        mask = preprocess_image(img)
        # Guardar máscara preprocesada en Mask (Optional, kept for debugging/comparison)
        mask_output_path = os.path.join(mask_folder, f"{os.path.splitext(image_file)[0]}_mask.jpg")
        cv2.imwrite(mask_output_path, (mask * 255).astype(np.uint8))

        # Ensure image is grayscale float32 for Canny detector
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img  # Assume already grayscale if not 3 channels
        # Apply Canny to the original grayscale image, not the preprocessed mask
        img_gray_float = img_gray.astype(np.float32) / 255.0

        edges = canny_edge_detector(img_gray_float)  # Apply Canny
        # Convert edges to uint8 for saving and potential use
        edges_uint8 = edges.astype(np.uint8)
        # Save edge map
        edges_output_path = os.path.join(edges_folder, f"{os.path.splitext(image_file)[0]}_edges.jpg")
        cv2.imwrite(edges_output_path, edges_uint8)
        # --- End of Edge Detection Step ---

        # Visualización de clusters coloreados (Now using edges_uint8 as input)
        # WARNING: Check if color_clusters handles binary edge input correctly.
        img2 = color_clusters(edges_uint8)  # <-- Changed input from labeled_mask
        colored_filename = f"{os.path.splitext(image_file)[0]}_coloredClusters.jpg"
        colored_output_path = os.path.join(colored_clusters_folder, colored_filename)
        save_colored_clusters(img2, colored_output_path)

        # Paso 4: Medición de propiedades y guardado en CSV (Now using edges_uint8 as input)
        # WARNING: measure_properties likely expects labeled regions, not binary edges.
        # Results might be incorrect or measure properties of the entire edge set as one object.
        measurements = measure_properties(edges_uint8, img, pixels_to_um)  # <-- Changed input from labeled_mask
        csv_filename = f"{os.path.splitext(image_file)[0]}.csv"
        csv_output_path = os.path.join(csv_folder, csv_filename)
        save_measurements_to_csv(measurements, csv_output_path)

        # Paso 5: Generar histogramas (Based on potentially incorrect measurements from edges)
        areas = [row[1] for row in measurements]
        perimeters = [row[5] for row in measurements]
        equivalent_diameters = [row[2] for row in measurements]
        histogram_filename = f"{os.path.splitext(image_file)[0]}_histogram.jpg"
        histogram_output_path = os.path.join(histogram_folder, histogram_filename)
        generate_histograms(areas, perimeters, equivalent_diameters, histogram_output_path)

        print(f"Procesado: {image_file}")

if __name__ == "__main__":

    print("Programa principal iniciado.")
    main()