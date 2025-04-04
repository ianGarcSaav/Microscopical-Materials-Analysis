import os
import numpy as np
import cv2  # Importar para guardar imágenes intermedias
from config import img_folder, clusters_folder, csv_folder, histogram_folder, pixels_to_um
from preprocessing import read_image, preprocess_image
from labeling import label_components, color_clusters
from measurement import measure_properties, save_measurements_to_csv
from visualization import save_colored_clusters, generate_histograms
from initialize_results import reset_results_folder

def main():
    print("Procesando todas las imagenes en:", img_folder)
    
    # Crear carpetas para coloredClusters, LabeledMask y Mask
    colored_clusters_folder = os.path.join(clusters_folder, "coloredClusters")
    labeled_mask_folder = os.path.join(clusters_folder, "LabeledMask")
    mask_folder = os.path.join(clusters_folder, "Mask")
    os.makedirs(colored_clusters_folder, exist_ok=True)
    os.makedirs(labeled_mask_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    # Listar solo archivos de imagen
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No se encontraron imágenes en", img_folder)
        return

    for image_file in image_files:
        img_path = os.path.join(img_folder, image_file)
        print("Procesando:", image_file)

        # Paso 1: Leer imagen
        img = read_image(img_path)

        # Paso 2: Preprocesamiento
        mask = preprocess_image(img)
        # Guardar máscara preprocesada en Mask
        mask_output_path = os.path.join(mask_folder, f"{os.path.splitext(image_file)[0]}_mask.jpg")
        cv2.imwrite(mask_output_path, (mask * 255).astype(np.uint8))

        # Paso 3: Etiquetado de componentes
        labeled_mask, _ = label_components(mask)
        # Guardar máscara etiquetada en LabeledMask
        labeled_output_path = os.path.join(labeled_mask_folder, f"{os.path.splitext(image_file)[0]}_labeledMask.jpg")
        cv2.imwrite(labeled_output_path, (labeled_mask / labeled_mask.max() * 255).astype(np.uint8))

        # Visualización de clusters coloreados
        img2 = color_clusters(labeled_mask)
        colored_filename = f"{os.path.splitext(image_file)[0]}_coloredClusters.jpg"
        colored_output_path = os.path.join(colored_clusters_folder, colored_filename)
        save_colored_clusters(img2, colored_output_path)

        # Paso 4: Medición de propiedades y guardado en CSV
        measurements = measure_properties(labeled_mask, img, pixels_to_um)
        csv_filename = f"{os.path.splitext(image_file)[0]}.csv"
        csv_output_path = os.path.join(csv_folder, csv_filename)
        save_measurements_to_csv(measurements, csv_output_path)

        # Paso 5: Generar histogramas
        areas = [row[1] for row in measurements]
        perimeters = [row[5] for row in measurements]
        equivalent_diameters = [row[2] for row in measurements]
        histogram_filename = f"{os.path.splitext(image_file)[0]}_histogram.jpg"
        histogram_output_path = os.path.join(histogram_folder, histogram_filename)
        generate_histograms(areas, perimeters, equivalent_diameters, histogram_output_path)
        print(f"Procesado: {image_file}")

if __name__ == "__main__":
    # Reiniciar la carpeta de resultados al inicio
    reset_results_folder()
    
    print("Programa principal iniciado.")
    main()
