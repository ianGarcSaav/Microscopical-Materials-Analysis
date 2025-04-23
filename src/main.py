import os
import numpy as np
import cv2
from config import img_folder, clusters_folder, csv_folder, histogram_folder, pixels_to_um
from preprocessing import read_image, preprocess_image
from labeling import label_components, color_clusters
from measurement import measure_properties, save_measurements_to_csv
from visualization import save_colored_clusters, generate_histograms, visualize_edge_detection_steps
import concurrent.futures
import time

def process_single_image(image_file):
    """Procesa una sola imagen - para uso con procesamiento paralelo"""
    start_time = time.time()
    img_path = os.path.join(img_folder, image_file)
    
    # Directorios de salida
    colored_clusters_folder = os.path.join(clusters_folder, "coloredClusters")
    labeled_mask_folder = os.path.join(clusters_folder, "LabeledMask")
    mask_folder = os.path.join(clusters_folder, "Mask")
    
    # Nombres de archivos de salida
    base_name = os.path.splitext(image_file)[0]
    mask_output_path = os.path.join(mask_folder, f"{base_name}_mask.jpg")
    labeled_output_path = os.path.join(labeled_mask_folder, f"{base_name}_labeledMask.jpg")
    csv_output_path = os.path.join(csv_folder, f"{base_name}.csv")
    colored_output_path = os.path.join(colored_clusters_folder, f"{base_name}_coloredClusters.jpg")
    histogram_output_path = os.path.join(histogram_folder, f"{base_name}_histogram.jpg")
    
    try:
        # Paso 1: Leer imagen
        img = read_image(img_path)
        
        # Paso 2: Preprocesamiento con detección de bordes Canny
        # Parámetros de Canny modificados para ser menos agresivos
        sigma = 0.8  # Reducido de 1.4 para un suavizado menos agresivo
        
        # Umbrales más bajos para detectar más bordes (menos agresivos)
        low_threshold = 30   # Mantenemos el umbral bajo actual
        high_threshold = 90  # Mantenemos el umbral alto actual
        
        # Almacenar pasos intermedios para visualización
        orig_img = img.copy()
        
        # Mejorar contraste antes del procesamiento para detectar mejor los bordes
        # Esto ayuda a que Canny encuentre bordes en imágenes de bajo contraste
        enhanced_img = cv2.equalizeHist(img)  # Ecualización de histograma
        
        # Realizar preprocesamiento usando Canny con la imagen mejorada
        # Pasamos el nuevo valor de sigma para un suavizado menos agresivo
        mask = preprocess_image(enhanced_img, sigma=sigma, 
                              low_threshold=low_threshold,
                              high_threshold=high_threshold)
        
        # Generar visualización de los pasos intermedios
        # Calculamos las imágenes intermedias para la visualización
        # Usando el nuevo valor de sigma para un suavizado menos agresivo
        kernel_size = int(2 * round(3 * sigma) + 1)  # Kernel más pequeño según el sigma reducido
        smoothed = cv2.GaussianBlur(enhanced_img, (kernel_size, kernel_size), sigma)
        edges = cv2.Canny(smoothed, low_threshold, high_threshold, L2gradient=True)
        
        # Crear nombre de archivo para la visualización
        steps_output_path = os.path.join(histogram_folder, f"{base_name}_edge_steps.jpg")
        
        # Visualizar pasos intermedios (ahora incluye la imagen mejorada)
        visualize_edge_detection_steps(
            original=orig_img,
            smoothed=smoothed,
            edges=edges,
            filled=mask.astype(np.uint8)*255,
            save_path=steps_output_path
        )
        
        # Añade depuración visual directa para ver exactamente qué sucede en cada paso
        debug_folder = os.path.join(clusters_folder, "debug")
        os.makedirs(debug_folder, exist_ok=True)
        
        # Guarda cada paso para inspección
        cv2.imwrite(os.path.join(debug_folder, f"{base_name}_1_original.png"), orig_img)
        cv2.imwrite(os.path.join(debug_folder, f"{base_name}_2_smoothed.png"), smoothed)
        cv2.imwrite(os.path.join(debug_folder, f"{base_name}_3_edges.png"), edges)
        
        # Guarda la imagen de la máscara directamente antes del etiquetado
        cv2.imwrite(os.path.join(debug_folder, f"{base_name}_4_mask.png"), (mask.astype(np.uint8)*255))
        
        # Paso 3: Etiquetado de componentes - optimizado
        labeled_mask, num_labels = label_components(mask)
        
        # Paso 4: Medición de propiedades - optimizado
        measurements = measure_properties(labeled_mask, img, pixels_to_um)
        
        # Paso 5: Colorear clusters - optimizado
        img2 = color_clusters(labeled_mask)
        
        # Guardar resultados - solo al final para minimizar I/O
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder, exist_ok=True)
        if not os.path.exists(labeled_mask_folder):
            os.makedirs(labeled_mask_folder, exist_ok=True)
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder, exist_ok=True)
        if not os.path.exists(colored_clusters_folder):
            os.makedirs(colored_clusters_folder, exist_ok=True)
        if not os.path.exists(histogram_folder):
            os.makedirs(histogram_folder, exist_ok=True)
            
        # Guardar máscara preprocesada
        cv2.imwrite(mask_output_path, (mask * 255).astype(np.uint8))
        
        # Guardar máscara etiquetada con compresión para ahorrar espacio
        cv2.imwrite(labeled_output_path, (labeled_mask / labeled_mask.max() * 255).astype(np.uint8),
                   [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Guardar mediciones en CSV
        save_measurements_to_csv(measurements, csv_output_path)
        
        # Guardar clusters coloreados
        save_colored_clusters(img2, colored_output_path)
        
        # Extraer datos para histogramas
        areas = [row[1] for row in measurements]
        perimeters = [row[5] for row in measurements]
        equivalent_diameters = [row[2] for row in measurements]
        
        # Generar histogramas
        generate_histograms(areas, perimeters, equivalent_diameters, histogram_output_path)
        
        elapsed_time = time.time() - start_time
        return f"Procesado: {image_file} en {elapsed_time:.2f} segundos ({num_labels} componentes)"
    
    except Exception as e:
        return f"Error procesando {image_file}: {str(e)}"

def main():
    print(f"Procesando imágenes en: {img_folder}")
    
    # Crear carpetas si no existen
    os.makedirs(os.path.join(clusters_folder, "coloredClusters"), exist_ok=True)
    os.makedirs(os.path.join(clusters_folder, "LabeledMask"), exist_ok=True)
    os.makedirs(os.path.join(clusters_folder, "Mask"), exist_ok=True)
    
    # Listar solo archivos de imagen
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"No se encontraron imágenes en {img_folder}")
        return
    
    print(f"Encontradas {len(image_files)} imágenes para procesar")
    
    # Determinar el número óptimo de workers basado en CPU
    max_workers = min(os.cpu_count() or 4, len(image_files))
    
    # Procesamiento paralelo
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, image_file): image_file 
                  for image_file in image_files}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            print(f"[{i+1}/{len(image_files)}] {result}")
    
    total_time = time.time() - start_time
    print(f"Procesamiento completado en {total_time:.2f} segundos")

if __name__ == "__main__":
    print("Programa principal iniciado.")
    main()
