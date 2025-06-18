import numpy as np
from matplotlib import pyplot as plt
import pyclesperanto_prototype as cle  # type: ignore
from skimage import io, measure, color # Agregamos measure y color
import stackview  # type: ignore
import cv2  # Añadimos OpenCV
from skimage.filters import median  # Añadimos el filtro de mediana de scikit-image
import pandas as pd # Agregamos pandas
import itertools # Agregamos itertools para combinaciones
from pathlib import Path
from src.config import (
    PROCESSED_DATA_DIR,
    IMAGE_MEASUREMENTS_DIR,
    SCATTERS_DIR,
    HISTOGRAMS_DIR,
    ensure_directory_exists
)

# print("python version is: ", sys.version)

# Crear directorio para guardar resultados
output_dir = PROCESSED_DATA_DIR / "vonoriSegmentation"
ensure_directory_exists(output_dir)
print(f"Directorio limpiado/creado: {output_dir}")

# Contador para el ID de las imágenes
image_counter = 0

# Cargar imagen y convertir a tipo uint8 para evitar advertencias de imshow
img = io.imread(str(PROCESSED_DATA_DIR / "borderMasks/FT_500X_A_4_border.jpg"))
if img.dtype != np.uint8:
    img = img.astype(np.uint8)

# Asegurar que la imagen esté en escala de grises para las mediciones de intensidad (para regionprops)
if img.ndim == 3 and img.shape[2] == 3: # Si es una imagen RGB
    original_grayscale_img_np = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
else: # Ya está en escala de grises o es un solo canal
    original_grayscale_img_np = img.copy() # Usar .copy() para evitar modificaciones inesperadas

plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.title("Imagen original")
plt.axis("off")
plt.savefig(str(output_dir / f"{image_counter}.original.jpg"))
plt.close()
image_counter += 1

# Mostrar dispositivos disponibles y seleccionar GPU NVIDIA
print("Available OpenCL devices:" + str(cle.available_device_names()))
device = cle.select_device("NVIDIA")
print("Used GPU: ", device)

# Subir imagen a la GPU
img_gpu = cle.push(img)
print("Image size in GPU:", img_gpu.shape)
# stackview.imshow(img_gpu, colormap='gray')

# Confirmar dimensión
print("size:", img_gpu.shape[0])

# Paso 1: Aplicar blur fuerte y detectar máximos
img_gaussian = cle.gaussian_blur(img_gpu, sigma_x=20, sigma_y=20, sigma_z=40)
img_maxima_locations = cle.detect_maxima_box(img_gaussian, radius_x=0, radius_y=0, radius_z=0)

number_of_maxima_locations = cle.sum_of_all_pixels(img_maxima_locations)
print("number of detected maxima locations:", number_of_maxima_locations)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
stackview.imshow(img_gaussian, plot=axs[0], colormap='gray')
axs[0].set_title("Gaussian blur (sigma=9)")
stackview.imshow(img_maxima_locations, plot=axs[1], colormap='gray')
axs[1].set_title("Maxima locations")
plt.savefig(str(output_dir / f"{image_counter}.gaussian_maxima.jpg"))
plt.close()
image_counter += 1

# Paso 2: Aplicar blur suave y umbral de Otsu
img_gaussian2 = cle.gaussian_blur(img_gpu, sigma_x=10, sigma_y=10, sigma_z=10)
img_thresh = cle.threshold_otsu(img_gaussian2)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
stackview.imshow(img_gaussian2, plot=axs[0], colormap='gray')
axs[0].set_title("Gaussian blur (sigma=1)")
stackview.imshow(img_thresh, plot=axs[1], colormap='gray')
axs[1].set_title("Threshold (Otsu)")
plt.savefig(str(output_dir / f"{image_counter}.gaussian_otsu.jpg"))
plt.close()
image_counter += 1

# Paso 3: Filtrar máximos relevantes
img_relevant_maxima = cle.binary_and(img_thresh, img_maxima_locations)
number_of_relevant_maxima_locations = cle.sum_of_all_pixels(img_relevant_maxima)
print("number of relevant maxima locations:", number_of_relevant_maxima_locations)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
stackview.imshow(img_maxima_locations, plot=axs[0], colormap='gray')
axs[0].set_title("Maxima")
stackview.imshow(img_thresh, plot=axs[1], colormap='gray')
axs[1].set_title("Threshold")
stackview.imshow(img_relevant_maxima, plot=axs[2], colormap='gray')
axs[2].set_title("Relevant Maxima")
plt.savefig(str(output_dir / f"{image_counter}.relevant_maxima.jpg"))
plt.close()
image_counter += 1

# Paso 4: Voronoi con máscara
voronoi_separation = cle.masked_voronoi_labeling(img_relevant_maxima, img_thresh)

# Invertir los valores de la segmentación
voronoi_inverted = cle.binary_not(voronoi_separation)

# Crear figura solo para Voronoi
fig, ax = plt.subplots(figsize=(10, 8))
stackview.imshow(voronoi_inverted, plot = ax, labels=True)
plt.title("Segmentación Voronoi (Invertida)")
plt.axis('off')

# Guardar la imagen
plt.savefig(str(output_dir / f"{image_counter}.voronoi_segmentation.jpg"))
plt.close()
image_counter += 1  

# Procesamiento con medianBlur
# Convertir la imagen de GPU a numpy array
voronoi_np = cle.pull(voronoi_inverted)

# Aplicar filtro de mediana de scikit-image
# Usamos un footprint circular de radio 5
from skimage.morphology import disk
footprint = disk(30)
median_filtered = median(voronoi_np.astype(np.uint8), footprint)

# Visualizar y guardar el resultado
fig, ax = plt.subplots(figsize=(10, 8))
stackview.imshow(median_filtered, plot=ax, labels=True)
plt.title("Segmentación Voronoi con Median Filter (scikit-image)")
plt.axis('off')
plt.savefig(str(output_dir / f"{image_counter}.voronoi_median_skimage.jpg"))
plt.close()
image_counter += 1

# Subir imagen filtrada nuevamente a GPU
img_median_gpu = cle.push(median_filtered)

# Asegúrate de que sea binaria (por seguridad)
img_binary = cle.threshold_otsu(img_median_gpu)

# Etiquetado con conectividad en caja (26-neighbors en 3D, 8-neighbors en 2D)
img_labels = cle.connected_components_labeling_box(img_binary)

# Visualizar y guardar el resultado del etiquetado
fig, ax = plt.subplots(figsize=(10, 8))
stackview.imshow(img_labels, plot=ax, colormap='glasbey', labels=True)
plt.title("Segmentación Voronoi con Median Filter y Etiquetado")
plt.axis('off')
plt.savefig(str(output_dir / f"{image_counter}.voronoi_median_skimage_labeled.jpg"))
plt.close()
image_counter += 1

# Si deseas obtenerlo como array NumPy
img_labels_np = cle.pull(img_labels).astype(int)

###############################################################################
# Extract statistics and plot using seaborn / matplotlib

# Crear directorio para guardar estadísticas
statistics_dir = IMAGE_MEASUREMENTS_DIR
ensure_directory_exists(statistics_dir)
print(f"Directorio de estadísticas limpiado/creado: {statistics_dir}")

# Definir pixels_to_um
# ¡IMPORTANTE! Ajusta este valor si tu imagen tiene una conversión de píxeles a micrómetros diferente.
pixels_to_um = 1.0 # Por ejemplo, 0.5 si 1 píxel = 500 nm

# Extraer propiedades de las regiones detectadas
regions = measure.regionprops(img_labels_np, intensity_image=original_grayscale_img_np)

propList = ['Area',
            'equivalent_diameter',
            'orientation',
            'MajorAxisLength',
            'MinorAxisLength',
            'Perimeter',
            'MinIntensity',
            'MeanIntensity',
            'MaxIntensity']    

data = []
for region_props in regions:
    row = {'Label': region_props.label}
    for prop in propList:
        if(prop == 'Area'): 
            to_print = region_props[prop]*pixels_to_um**2   #Convertir píxeles cuadrados a micrómetros cuadrados
        elif(prop == 'orientation'): 
            to_print = region_props[prop]*57.2958  #Convertir de radianes a grados
        elif('Intensity' not in prop):          # Cualquier propiedad sin 'Intensity' en su nombre
            to_print = region_props[prop]*pixels_to_um
        else: 
            to_print = region_props[prop]     # Propiedades restantes, las que tienen 'Intensity' en su nombre
        row[prop] = to_print
    data.append(row)

stats_df = pd.DataFrame(data)

# Guardar estadísticas en CSV
output_file_path = statistics_dir / 'image_measurements.csv'
stats_df.to_csv(str(output_file_path), index=False)
print(f"Estadísticas guardadas en: {output_file_path}")

###############################################################################
# Generación de gráficos (Histogramas y Gráficos de Dispersión)

# Crear subdirectorios para los gráficos
ensure_directory_exists(HISTOGRAMS_DIR)
ensure_directory_exists(SCATTERS_DIR)
print(f"Directorios de gráficos creados: {HISTOGRAMS_DIR} y {SCATTERS_DIR}")

# Generar todos los histogramas
print("\nGenerando histogramas...")
for prop in propList:
    plt.figure(figsize=(10, 6))
    plt.hist(stats_df[prop], bins=25, color='skyblue', edgecolor='black')
    plt.title(f'Histograma de {prop}')
    plt.xlabel(f'{prop} (unidades en micrómetros o intensidad)')
    plt.ylabel('Frecuencia de conteo')
    plt.grid(axis='y', alpha=0.75)
    histogram_file_path = HISTOGRAMS_DIR / f'histogram_{prop}.jpg'
    plt.savefig(str(histogram_file_path))
    plt.close()
    print(f"  - Histograma de {prop} guardado en: {histogram_file_path}")

# Generar todos los gráficos de dispersión
print("\nGenerando gráficos de dispersión...")
for prop1, prop2 in itertools.combinations(propList, 2):
    plt.figure(figsize=(10, 6))
    plt.scatter(stats_df[prop1], stats_df[prop2], alpha=0.6, color='blue')
    plt.title(f'{prop1} vs {prop2}')
    plt.xlabel(f'{prop1} (unidades en micrómetros o intensidad)')
    plt.ylabel(f'{prop2} (unidades en micrómetros o intensidad)')
    plt.grid(True)
    scatter_file_path = SCATTERS_DIR / f'scatter_{prop1}_vs_{prop2}.jpg'
    plt.savefig(str(scatter_file_path))
    plt.close()
    print(f"  - Gráfico de dispersión {prop1} vs {prop2} guardado en: {scatter_file_path}")

