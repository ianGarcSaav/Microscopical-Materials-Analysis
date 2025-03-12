import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_colored_clusters(img2, output_path):
    img2_uint8 = (img2 * 255).astype(np.uint8)
    img2_bgr = cv2.cvtColor(img2_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img2_bgr)

def show_image(img2_bgr):
    cv2.imshow('Colored Grains', img2_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_histograms(areas, perimeters, equivalent_diameters, histogram_path):
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
    plt.savefig(histogram_path)
    # plt.show()
