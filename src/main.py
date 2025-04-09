import os
import numpy as np
import cv2
from config import img_folder, clusters_folder, csv_folder, histogram_folder, pixels_to_um
from preprocessing import read_image, preprocess_image
from labeling import label_components, color_clusters
from measurement import measure_properties, save_measurements_to_csv
from visualization import save_colored_clusters, generate_histograms
from clustering import perform_kmeans_clustering, save_clustered_data  # Import clustering functions
import pandas as pd

def main():
    print("Procesando todas las imágenes en:", img_folder)
    
    # Crear carpetas para coloredClusters, LabeledMask, Mask, imageCutting y detectedShapes
    colored_clusters_folder = os.path.join(clusters_folder, "coloredClusters")
    labeled_mask_folder = os.path.join(clusters_folder, "LabeledMask")
    mask_folder = os.path.join(clusters_folder, "Mask")
    detected_shapes_folder = os.path.join(clusters_folder, "detectedShapes")
    os.makedirs(colored_clusters_folder, exist_ok=True)
    os.makedirs(labeled_mask_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(detected_shapes_folder, exist_ok=True)

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

        # Paso 4: Medición de propiedades y guardado en CSV
        measurements = measure_properties(labeled_mask, img, pixels_to_um)
        csv_filename = f"{os.path.splitext(image_file)[0]}.csv"
        csv_output_path = os.path.join(csv_folder, csv_filename)
        save_measurements_to_csv(measurements, csv_output_path)

        # Paso 5: Perform K-Means clustering
        cluster_labels = perform_kmeans_clustering(csv_output_path, n_clusters=3)  # You can adjust the number of clusters

        # Visualización de clusters coloreados
        img2 = color_clusters(labeled_mask, cluster_labels)
        colored_filename = f"{os.path.splitext(image_file)[0]}_coloredClusters.jpg"
        colored_output_path = os.path.join(colored_clusters_folder, colored_filename)
        save_colored_clusters(img2, colored_output_path)

        if cluster_labels is not None:
            # Read the original CSV data
            original_df = pd.read_csv(csv_output_path)

            # Add the cluster labels to the DataFrame
            original_df['Cluster'] = original_df['Label'].map(cluster_labels)

            clustered_csv_filename = f"{os.path.splitext(image_file)[0]}_clustered.csv"
            clustered_csv_output_path = os.path.join(csv_folder, clustered_csv_filename)
            save_clustered_data(original_df, clustered_csv_output_path)

        # Paso 6: Generar histogramas
        areas = [row[1] for row in measurements]
        perimeters = [row[5] for row in measurements]
        equivalent_diameters = [row[2] for row in measurements]
        histogram_filename = f"{os.path.splitext(image_file)[0]}_histogram.jpg"
        histogram_output_path = os.path.join(histogram_folder, histogram_filename)
        generate_histograms(areas, perimeters, equivalent_diameters, histogram_output_path)

        # Paso 7: Detectar y clasificar figuras geométricas
        image = cv2.imread(img_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            if i == 0:
                continue

            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)

            x, y, w, h = cv2.boundingRect(approx)
            x_mid = int(x + w / 2)
            y_mid = int(y + h / 2)

            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Quadrilateral"
            elif len(approx) == 5:
                shape = "Pentagon"
            elif len(approx) == 6:
                shape = "Hexagon"
            else:
                shape = "Circle"

            cv2.putText(image, shape, (x_mid - 50, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        output_path = os.path.join(detected_shapes_folder, f"{os.path.splitext(image_file)[0]}_shapes.jpg")
        cv2.imwrite(output_path, image)
        print(f"Figuras detectadas y guardadas en: {output_path}")

        print(f"Procesado: {image_file}")

if __name__ == "__main__":
    print("Programa principal iniciado.")
    main()
