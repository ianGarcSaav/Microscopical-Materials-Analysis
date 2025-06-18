import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters, morphology, exposure, feature
from scipy import ndimage as ndi
import os
import shutil

# Crear directorio de resultados si no existe

ruta_results = 'results/'

if os.path.exists(ruta_results):
    shutil.rmtree(ruta_results)
os.makedirs(ruta_results)

# === CONFIGURACIÓN ===
ruta_imagen = 'articulo/imagen.jpg'   # Cambia por el nombre real
nombre_salida_csv = 'results/filtros_imagen.csv'
nombre_salida_imagen = 'results/comparacion_filtros.png'

# === LEER IMAGEN (8-bit grayscale) ===
original = cv2.imread(ruta_imagen)
img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
if img is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")

imagenes = [img]
nombres = ['original']

# === FILTROS (parámetros default razonables) ===

# OpenCV filters
imagenes.append(cv2.GaussianBlur(img, (5, 5), 0))
nombres.append('gaussian_blur')

imagenes.append(cv2.medianBlur(img, 5))
nombres.append('median_blur')

imagenes.append(cv2.bilateralFilter(img, 9, 75, 75))
nombres.append('bilateral_filter')

_, thres = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
imagenes.append(thres)
nombres.append('threshold')

adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
imagenes.append(adaptive)
nombres.append('adaptive_threshold')

equalized = cv2.equalizeHist(img)
imagenes.append(equalized)
nombres.append('equalize_hist')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
imagenes.append(morph)
nombres.append('morph_open')

# skimage filters
imagenes.append((filters.gaussian(img, sigma=1)*255).astype(np.uint8))
nombres.append('skimage_gaussian')

imagenes.append(filters.sobel(img).astype(np.uint8)*255)
nombres.append('sobel')

imagenes.append(filters.scharr(img).astype(np.uint8)*255)
nombres.append('scharr')

imagenes.append((filters.median(img)).astype(np.uint8))
nombres.append('skimage_median')

imagenes.append((res := exposure.rescale_intensity(img)).astype(np.uint8))
nombres.append('rescale_intensity')

imagenes.append((feature.canny(img, sigma=1)*255).astype(np.uint8))
nombres.append('canny')

# skimage morphology
binary = img > 127
imagenes.append((morphology.remove_small_objects(binary, 30)*255).astype(np.uint8))
nombres.append('remove_small_objects')

imagenes.append((morphology.binary_closing(binary, morphology.disk(3))*255).astype(np.uint8))
nombres.append('binary_closing')

# scipy.ndimage filters
imagenes.append(ndi.gaussian_filter(img, sigma=1).astype(np.uint8))
nombres.append('ndi_gaussian')

imagenes.append(ndi.median_filter(img, size=3).astype(np.uint8))
nombres.append('ndi_median')

imagenes.append(ndi.maximum_filter(img, size=3).astype(np.uint8))
nombres.append('ndi_maximum')

imagenes.append((ndi.binary_fill_holes(binary)*255).astype(np.uint8))
nombres.append('binary_fill_holes')

# === CREAR DATAFRAME ===
df = pd.DataFrame()
df['filtro'] = nombres
df['imagen_array'] = [img.flatten().tolist() for img in imagenes]
df.to_csv(nombre_salida_csv, index=False)
print(f'✅ CSV guardado en: {nombre_salida_csv}')

# === GUARDAR IMÁGENES FILTRADAS INDIVIDUALMENTE ===
# Guardar cada imagen filtrada como un archivo .tif con el nombre del filtro
for i, img_filtrada in enumerate(imagenes):
    nombre_archivo_tif = os.path.join(ruta_results, f'{nombres[i]}.tif')
    cv2.imwrite(nombre_archivo_tif, img_filtrada)
    print(f'💾 Imagen "{nombres[i]}" guardada en: {nombre_archivo_tif}')

# === CREAR IMAGEN COMPARATIVA (en mosaico horizontal o grid) ===
# Ajustar tamaño
imagenes_resized = [cv2.resize(img, (256, 256)) for img in imagenes]

# Calcular layout para mostrar como grid
cols = 5
rows = int(np.ceil(len(imagenes_resized) / cols))
mosaico_filas = []

for r in range(rows):
    fila_imagenes = imagenes_resized[r*cols:(r+1)*cols]
    fila_nombres = nombres[r*cols:(r+1)*cols]

    # Crear una fila de imágenes con sus nombres
    fila_con_texto = []
    for i, img in enumerate(fila_imagenes):
        # Asegurarse de que la imagen sea de 3 canales para poder dibujar texto en color
        if len(img.shape) == 2:
            img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_display = img.copy()

        # Añadir texto (nombre del filtro)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (0, 255, 0) # Verde
        text_size = cv2.getTextSize(fila_nombres[i], font, font_scale, font_thickness)[0]
        text_x = 5
        text_y = text_size[1] + 5 # Un pequeño margen desde arriba

        cv2.putText(img_display, fila_nombres[i], (text_x, text_y), font,
                    font_scale, text_color, font_thickness, cv2.LINE_AA)
        fila_con_texto.append(img_display)

    # Rellenar la fila si es necesario con imágenes en blanco para mantener el grid
    while len(fila_con_texto) < cols:
        blank_img = np.zeros_like(fila_con_texto[0])
        fila_con_texto.append(blank_img)
        
    mosaico_filas.append(np.hstack(fila_con_texto))

img_final = np.vstack(mosaico_filas)
cv2.imshow('Comparacion de Filtros', img_final) # Mostrar la imagen en una ventana emergente
cv2.waitKey(0) # Esperar indefinidamente hasta que se presione una tecla
cv2.destroyAllWindows() # Cerrar todas las ventanas de OpenCV
