import os
import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, color, io
from matplotlib import pyplot as plt

print("Iniciando código numeros 2...")

# Configuración inicial
pixels_to_um = 0.5  # 1 px = 500 nm
img_path = "grainImages/grain.jpg"

# Si la imagen no existe, se crea una imagen dummy para probar el pipeline
if not os.path.exists(img_path):
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    # Crear imagen dummy: fondo negro con un círculo blanco (simula un grano)
    dummy = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(dummy, (100, 100), 50, 255, -1)
    cv2.imwrite(img_path, dummy)
    print(f"Imagen dummy creada en {img_path}")

# Paso 1: Leer imagen
img = cv2.imread(img_path, 0)
if img is None:
    print(f"Error: No se pudo cargar la imagen en {img_path}. Verifica la ruta.")
    exit()

# Paso 2: Preprocesamiento
ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)

# Operaciones morfológicas
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(thresh, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)
mask = dilated == 255

# Paso 3: Etiquetado de componentes
connectivity = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
labeled_mask, num_labels = ndimage.label(mask, structure=connectivity)

# Visualización de clusters coloreados
img2 = color.label2rgb(labeled_mask, bg_label=0)

# Guardar la imagen con clusters coloreados en la carpeta "imageClusters"
clusters_folder = 'imageClusters'
if not os.path.exists(clusters_folder):
    os.makedirs(clusters_folder)
# Convertir la imagen a formato adecuado para guardar con cv2
img2_uint8 = (img2 * 255).astype(np.uint8)
img2_bgr = cv2.cvtColor(img2_uint8, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(clusters_folder, 'colored_clusters.jpg'), img2_bgr)

# Mostrar la imagen (opcional)
cv2.imshow('Colored Grains', img2_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Paso 4: Medición de propiedades
clusters = measure.regionprops(labeled_mask, intensity_image=img)

# Lista de propiedades a exportar
prop_list = [
    'Area',
    'equivalent_diameter',
    'orientation',
    'MajorAxisLength',
    'MinorAxisLength',
    'Perimeter',
    'MinIntensity',
    'MeanIntensity',
    'MaxIntensity'
]

# Paso 5: Exportar a CSV
output_csv = 'imageMeasurements/image_measurements.csv'
csv_dir = os.path.dirname(output_csv)
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

with open(output_csv, 'w') as output_file:
    headers = ['Label'] + prop_list
    output_file.write(','.join(headers) + '\n')
    
    for cluster in clusters:
        row = [str(cluster.label)]
        for prop in prop_list:
            # Conversión de unidades y uso de atributos correctos
            if prop == 'Area':
                value = cluster.area * (pixels_to_um ** 2)
            elif prop == 'orientation':
                value = np.degrees(cluster.orientation)
            elif prop == 'Perimeter':
                value = cluster.perimeter * pixels_to_um
            elif prop == 'equivalent_diameter':
                value = cluster.equivalent_diameter * pixels_to_um
            elif prop == 'MajorAxisLength':
                value = cluster.major_axis_length * pixels_to_um
            elif prop == 'MinorAxisLength':
                value = cluster.minor_axis_length * pixels_to_um
            elif prop == 'MinIntensity':
                value = cluster.min_intensity
            elif prop == 'MeanIntensity':
                value = cluster.mean_intensity
            elif prop == 'MaxIntensity':
                value = cluster.max_intensity
            row.append(f"{value:.4f}")
        output_file.write(','.join(row) + '\n')

print(f"Análisis completado. Resultados guardados en {output_csv}")

# Paso 6: Generar histogramas y calcular estadísticas para Área, Perímetro y Diámetro Equivalente
areas = []
perimeters = []
equivalent_diameters = []

for cluster in clusters:
    areas.append(cluster.area * (pixels_to_um ** 2))
    perimeters.append(cluster.perimeter * pixels_to_um)
    equivalent_diameters.append(cluster.equivalent_diameter * pixels_to_um)

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

# Crear histogramas
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(areas, bins=10, color='blue', edgecolor='black')
plt.title('Histograma del Área')
plt.xlabel('Área (um²)')
plt.ylabel('Frecuencia')

plt.subplot(1, 3, 2)
plt.hist(perimeters, bins=10, color='green', edgecolor='black')
plt.title('Histograma del Perímetro')
plt.xlabel('Perímetro (um)')
plt.ylabel('Frecuencia')

plt.subplot(1, 3, 3)
plt.hist(equivalent_diameters, bins=10, color='red', edgecolor='black')
plt.title('Histograma del Diámetro Equivalente')
plt.xlabel('Diámetro (um)')
plt.ylabel('Frecuencia')

plt.tight_layout()

# Guardar el histograma en la misma carpeta donde se guarda el CSV
histogram_path = os.path.join(csv_dir, 'histograms.png')
plt.savefig(histogram_path)
#plt.show()

print(f"Histograma guardado en {histogram_path}")
