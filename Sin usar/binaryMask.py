import cv2
import numpy as np
import os
import shutil

# === CONFIGURACIÓN DE RUTAS ===
ruta_imagenes_originales = 'images/original/'
ruta_mascaras_salida = 'images/mask/'

# Crear/limpiar directorio de salida
if os.path.exists(ruta_mascaras_salida):
    shutil.rmtree(ruta_mascaras_salida)
os.makedirs(ruta_mascaras_salida)

# === 1. Procesar imágenes una por una ===
for nombre_archivo in os.listdir(ruta_imagenes_originales):
    # Asegurarse de que sea un archivo de imagen (puedes añadir más extensiones si es necesario)
    if nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
        ruta_imagen_completa = os.path.join(ruta_imagenes_originales, nombre_archivo)
        
        # === 2. Cargar imagen original ===
        imagen = cv2.imread(ruta_imagen_completa)
        if imagen is None:
            print(f"Advertencia: No se pudo cargar la imagen {nombre_archivo}. Saltando...")
            continue

        # === 3. Convertir a escala de grises ===
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # === 4. Aplicar umbral binario ===
        # Puedes ajustar el valor 120 si es muy bajo o alto
        _, mascara_binaria = cv2.threshold(gris, 120, 255, cv2.THRESH_BINARY) # Valor 120, ajuste si es necesario

        # === 5. Guardar máscara binaria ===
        # Cambiar la extensión a .tif
        nombre_base, _ = os.path.splitext(nombre_archivo)
        nombre_salida_mascara = f'{nombre_base}.tif'
        ruta_salida_completa = os.path.join(ruta_mascaras_salida, nombre_salida_mascara)
        cv2.imwrite(ruta_salida_completa, mascara_binaria)
        print(f'✅ Máscara binaria guardada para {nombre_archivo} en: {ruta_salida_completa}')

print("Proceso de generación de máscaras completado.")
