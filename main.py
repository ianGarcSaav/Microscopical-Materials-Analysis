import os
import numpy as np
from config import img_path, clusters_folder, output_csv, histogram_path, pixels_to_um
from preprocessing import create_dummy_image, read_image, preprocess_image
from labeling import label_components, color_clusters
from measurement import measure_properties, save_measurements_to_csv
from visualization import save_colored_clusters, show_image, generate_histograms

def main():
    print("Iniciando código numeros 2...")

    # Si la imagen no existe, se crea una imagen dummy para probar el pipeline
    if not os.path.exists(img_path):
        create_dummy_image(img_path)

    # Paso 1: Leer imagen
    img = read_image(img_path)

    # Paso 2: Preprocesamiento
    mask = preprocess_image(img)

    # Paso 3: Etiquetado de componentes
    labeled_mask, num_labels = label_components(mask)

    # Visualización de clusters coloreados
    img2 = color_clusters(labeled_mask)
    save_colored_clusters(img2, clusters_folder)
    # show_image(img2_bgr)  # Uncomment to display the image

    # Paso 4: Medición de propiedades
    measurements = measure_properties(labeled_mask, img, pixels_to_um)
    save_measurements_to_csv(measurements, output_csv)

    # Paso 6: Generar histogramas y calcular estadísticas para Área, Perímetro y Diámetro Equivalente
    areas = [row[1] for row in measurements]
    perimeters = [row[5] for row in measurements]
    equivalent_diameters = [row[2] for row in measurements]

    if areas and perimeters and equivalent_diameters:
        area_mean = np.mean(areas)
        area_std = np.std(areas)
        perimeter_mean = np.mean(perimeters)
        perimeter_std = np.std(perimeters)
        diameter_mean = np.mean(equivalent_diameters)
        diameter_std = np.std(equivalent_diameters)

        print("Estadísticas:")
        print(f"Área: Promedio = {area_mean:.4f}, Desviación estándar = {area_std:.4f}")
        print(f"Perímetro: Promedio = {perimeter_mean:.4f}, Desviación estándar = {perimeter_std:.4f}")
        print(f"Diámetro equivalente: Promedio = {diameter_mean:.4f}, Desviación estándar = {diameter_std:.4f}")
    else:
        print("No se detectaron clusters para calcular estadísticas.")

    generate_histograms(areas, perimeters, equivalent_diameters, histogram_path)
    print(f"Histograma guardado en {histogram_path}")

if __name__ == "__main__":
    main()
