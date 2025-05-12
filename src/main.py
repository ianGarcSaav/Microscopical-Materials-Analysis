import os
import numpy as np
import cv2
from config import img_folder, clusters_folder, csv_folder, histogram_folder, pixels_to_um
from preprocessing import read_image
from labeling import label_components, color_clusters
from measurement import measure_properties, save_measurements_to_csv
from visualization import save_colored_clusters, generate_histograms
from edge_detection import canny_edge_detector

def main():
    print("Procesando todas las imagenes en:", img_folder)

    # --- Setup Output Folders ---
    edges_folder = os.path.join(clusters_folder, "Edges")
    closed_edges_folder = os.path.join(clusters_folder, "ClosedEdges")
    filled_mask_folder = os.path.join(clusters_folder, "FilledMask")
    labeled_mask_folder = os.path.join(clusters_folder, "LabeledMask")
    colored_clusters_folder = os.path.join(clusters_folder, "coloredClusters")

    os.makedirs(edges_folder, exist_ok=True)
    os.makedirs(closed_edges_folder, exist_ok=True)
    os.makedirs(filled_mask_folder, exist_ok=True)
    os.makedirs(labeled_mask_folder, exist_ok=True)
    os.makedirs(colored_clusters_folder, exist_ok=True)

    # --- Process Images ---
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No se encontraron imágenes en", img_folder)
        return

    for image_file in image_files:
        img_path = os.path.join(img_folder, image_file)
        print(f"Procesando: {image_file}")

        # --- Step 1: Read Image ---
        img = read_image(img_path)

        # --- Step 2: Canny Edge Detection ---
        img_float = img.astype(np.float32) / 255.0
        edges = canny_edge_detector(img_float, sigma=1.4, low_ratio=0.05, high_ratio=0.15)
        edges_uint8 = edges.astype(np.uint8)
        edges_filename = f"{os.path.splitext(image_file)[0]}_edges.jpg"
        cv2.imwrite(os.path.join(edges_folder, edges_filename), edges_uint8)

        # --- Step 3: Post-process Edges (Morphological Closing) ---
        kernel_size = 7
        iterations = 2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed_edges = cv2.morphologyEx(edges_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        closed_edges_filename = f"{os.path.splitext(image_file)[0]}_closed_edges.jpg"
        cv2.imwrite(os.path.join(closed_edges_folder, closed_edges_filename), closed_edges)

        # --- Step 4: Segmentation by Filling Contours ---
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(closed_edges)
        cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)
        filled_mask_filename = f"{os.path.splitext(image_file)[0]}_filled_mask.jpg"
        cv2.imwrite(os.path.join(filled_mask_folder, filled_mask_filename), filled_mask)

        # --- Step 5: Labeling Components ---
        labeled_mask, num_labels = label_components(filled_mask)
        print(f"  Encontrados {num_labels-1} componentes.")
        labeled_output_path = os.path.join(labeled_mask_folder, f"{os.path.splitext(image_file)[0]}_labeledMask.jpg")
        if labeled_mask.max() > 0:
             labeled_mask_display = (labeled_mask / labeled_mask.max() * 255).astype(np.uint8)
        else:
             labeled_mask_display = labeled_mask.astype(np.uint8)
        labeled_mask_colored = cv2.applyColorMap(labeled_mask_display, cv2.COLORMAP_JET)
        cv2.imwrite(labeled_output_path, labeled_mask_colored)

        # --- Step 6: Visualization of Clusters ---
        img_colored_clusters = color_clusters(labeled_mask)
        colored_filename = f"{os.path.splitext(image_file)[0]}_coloredClusters.jpg"
        colored_output_path = os.path.join(colored_clusters_folder, colored_filename)
        save_colored_clusters(img_colored_clusters, colored_output_path)

        # --- Step 7: Measurement ---
        measurements = measure_properties(labeled_mask, img, pixels_to_um)
        csv_filename = f"{os.path.splitext(image_file)[0]}.csv"
        csv_output_path = os.path.join(csv_folder, csv_filename)
        save_measurements_to_csv(measurements, csv_output_path)

        # --- Step 8: Histograms ---
        if measurements:
            areas = [row[1] for row in measurements]
            perimeters = [row[5] for row in measurements]
            equivalent_diameters = [row[2] for row in measurements]
            histogram_filename = f"{os.path.splitext(image_file)[0]}_histogram.jpg"
            histogram_output_path = os.path.join(histogram_folder, histogram_filename)
            generate_histograms(areas, perimeters, equivalent_diameters, histogram_output_path)
        else:
            print("  No se encontraron componentes para medir.")

if __name__ == "__main__":
    print("Programa principal iniciado.")
    main()
    print("Programa principal finalizado.")