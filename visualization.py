
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from config import CLUSTERS_FOLDER, HISTOGRAM_PATH

def save_labeled_image(labeled_mask):
    img2 = color.label2rgb(labeled_mask, bg_label=0)
    img2_uint8 = (img2 * 255).astype(np.uint8)
    img2_bgr = cv2.cvtColor(img2_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{CLUSTERS_FOLDER}/colored_clusters.jpg", img2_bgr)
    print(f"Imagen de clusters guardada en {CLUSTERS_FOLDER}")

def generate_histograms(areas, perimeters, equivalent_diameters):
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
    plt.savefig(HISTOGRAM_PATH)
    print(f"Histograma guardado en {HISTOGRAM_PATH}")