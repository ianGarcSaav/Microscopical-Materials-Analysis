import glob
from pathlib import Path
import cv2
import sys

# Añadir el directorio raíz del proyecto al path de Python
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Importar las funciones de los otros módulos
from src.edges.cannyEdge import process_image_with_canny
from src.segmentation.semanticSegmentation import generate_border
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    IMAGE_MEASUREMENTS_DIR,
    SCATTERS_DIR,
    HISTOGRAMS_DIR,
    ensure_directory_exists,
    verify_path,
    get_files_in_directory,
    RAW_DATA_PATHS
)

if __name__ == "__main__":
    print("\n=== Iniciando proceso de segmentación de imágenes ===")
    
    # Rutas de entrada y salida usando config.py
    ruta_imagenes_originales = RAW_DATA_PATHS['high_res']  # Usar la ruta correcta del diccionario
    ruta_canny_salida = PROCESSED_DATA_DIR / 'canny_edges'
    ruta_border_salida = PROCESSED_DATA_DIR / 'borderMasks'

    # Verificar que las rutas existen
    print("\nVerificando rutas de entrada y salida...")
    if not verify_path(ruta_imagenes_originales):
        print(f"❌ Error: No se encontró el directorio de imágenes originales: {ruta_imagenes_originales}")
        print("Por favor, asegúrate de que las imágenes estén en el directorio correcto.")
        sys.exit(1)

    # Crear y limpiar directorios de salida
    print("\nPreparando directorios de salida...")
    for path in [ruta_canny_salida, ruta_border_salida]:
        ensure_directory_exists(path)

    # Verificar que hay imágenes para procesar
    print("\nBuscando imágenes para procesar...")
    image_files = get_files_in_directory(ruta_imagenes_originales, "*.jpg")
    if not image_files:
        print("❌ Error: No se encontraron imágenes .jpg en el directorio de entrada")
        print(f"Directorio buscado: {ruta_imagenes_originales}")
        sys.exit(1)
    
    print(f"\nSe encontraron {len(image_files)} imágenes para procesar")

    # Procesar cada imagen
    for file_path in image_files:
        name = Path(file_path).stem
        print(f"\n--- Procesando imagen: {name} ---")

        # Paso 1: Generar bordes Canny
        print("Generando bordes Canny...")
        canny_output_path = process_image_with_canny(file_path, ruta_canny_salida)
        
        if canny_output_path is None:
            print(f"    🚫 No se pudo generar bordes Canny para {name}. Saltando máscara de borde.")
            continue
        
        # Cargar la imagen Canny generada para pasarla a generate_border
        print("Cargando imagen Canny para procesamiento...")
        img_canny = cv2.imread(str(canny_output_path), cv2.IMREAD_GRAYSCALE)
        if img_canny is None:
            print(f"    🚫 Error al cargar la imagen Canny guardada: {canny_output_path}. Saltando máscara de borde.")
            continue

        # Asegurarse de que la imagen Canny sea binaria para generate_border
        print("Aplicando umbral binario...")
        _, binary_canny_for_border = cv2.threshold(img_canny, 127, 255, cv2.THRESH_BINARY) 

        # Paso 2: Generar máscara con borde a partir de los bordes Canny
        print("Generando máscara con borde...")
        border_mask = generate_border(binary_canny_for_border, border_size=5, n_erosions=1)
        
        # Guardar la máscara con borde
        border_output_file_path = ruta_border_salida / f"{name}_border.jpg"
        cv2.imwrite(str(border_output_file_path), border_mask)
        print(f"    ✅ Máscara con borde guardada en: {border_output_file_path}")

    print("\n=== Proceso principal completado ===")
    print("Revisa los resultados en las carpetas de processed_data.")
