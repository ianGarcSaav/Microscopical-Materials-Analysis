# https://youtu.be/65qPtD6khzg
"""
Create border pixels from binary masks. 
We can include these border pixels as another class to train a multiclass semantic segmenter
What is the advantage?
We can use border pixels to perform watershed and achieve 'instance' segmentation. 

"""

import cv2
import numpy as np
# import matplotlib.pyplot as plt # Ya no es necesario para guardar directamente
from pathlib import Path
from src.config import (
    PROCESSED_DATA_DIR,
    ensure_directory_exists
)

# === CONFIGURACIÓN DE RUTAS ===
ruta_imagenes_canny = PROCESSED_DATA_DIR / 'canny_edges'
ruta_mascara_salida = PROCESSED_DATA_DIR / 'borderMasks'

# Crear/limpiar directorio de salida
ensure_directory_exists(ruta_mascara_salida)

#Function to define border. 
#Just erode some pixels into objects and dilate tooutside the objects. 
#This region would be the border. Replace border pixel value to something other than 255. 
def generate_border(image, border_size=5, n_erosions=1):

    erosion_kernel = np.ones((1,1), np.uint8)      ## Start by eroding edge pixels
    eroded_image = cv2.erode(image, erosion_kernel, iterations=n_erosions)  
 
    ## Define the kernel size for dilation based on the desired border size (Add 1 to keep it odd)
    kernel_size = 2*border_size + 1 
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)   #Kernel to be used for dilation
    dilated  = cv2.dilate(eroded_image, dilation_kernel, iterations = 1)
    #plt.imshow(dilated, cmap='gray')
    
    ## Replace 255 values to 127 for all pixels. Eventually we will only define border pixels with this value
    dilated_127 = np.where(dilated == 255, 127, dilated) 	
    
    #In the above dilated image, convert the eroded object parts to pixel value 255
    #What's remaining with a value of 127 would be the boundary pixels. 
    original_with_border = np.where(eroded_image > 127, 255, dilated_127)
    
    #plt.imshow(original_with_border,cmap='gray')
    
    return original_with_border
 
# Nueva función para procesar una sola imagen Canny y guardar la máscara con borde
def process_canny_for_border_mask(input_canny_path, output_directory):
    name = Path(input_canny_path).stem  # Extract name without extension
    
    img = cv2.imread(str(input_canny_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Advertencia: No se pudo cargar la imagen {input_canny_path}. Saltando...")
        return None

    # Asegurarse de que la imagen de entrada sea binaria para generate_border
    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
    
    processed_image = generate_border(binary_image, border_size=5, n_erosions=1)
    
    # Guardar imágenes con el mismo nombre que la imagen Canny original
    output_file_path = output_directory / f'{name}_border.jpg'
    cv2.imwrite(str(output_file_path), processed_image)
    print(f"   ✅ Máscara con borde guardada en: {output_file_path}")
    return output_file_path
