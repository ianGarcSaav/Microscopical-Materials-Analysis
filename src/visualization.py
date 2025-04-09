import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_colored_clusters(img2, output_path):
    # Verificar si la imagen ya está en formato BGR
    if img2.dtype == np.uint8 and img2.shape[-1] == 3:
        img2_bgr = img2  # La imagen ya está en BGR
    else:
        img2_uint8 = (img2 * 255).astype(np.uint8)
        img2_bgr = cv2.cvtColor(img2_uint8, cv2.COLOR_RGB2BGR)

    # Guardar la imagen
    cv2.imwrite(output_path, img2_bgr)

def show_image(img2_bgr):
    cv2.imshow('Colored Grains', img2_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_histograms(areas, perimeters, equivalent_diameters, histogram_path,
                        num_bins=10, area_color='blue', perimeter_color='green', diameter_color='red',
                        area_title='Histograma del Área', perimeter_title='Histograma del Perímetro',
                        diameter_title='Histograma del Diámetro Equivalente', show=False):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(areas, bins=num_bins, color=area_color, edgecolor='black')
    plt.title(area_title)
    plt.xlabel('Área (um²)')
    plt.ylabel('Frecuencia')

    plt.subplot(1, 3, 2)
    plt.hist(perimeters, bins=num_bins, color=perimeter_color, edgecolor='black')
    plt.title(perimeter_title)
    plt.xlabel('Perímetro (um)')
    plt.ylabel('Frecuencia')

    plt.subplot(1, 3, 3)
    plt.hist(equivalent_diameters, bins=num_bins, color=diameter_color, edgecolor='black')
    plt.title(diameter_title)
    plt.xlabel('Diámetro (um)')
    plt.ylabel('Frecuencia')

    plt.tight_layout()
    plt.savefig(histogram_path)
    if show:
        plt.show()
