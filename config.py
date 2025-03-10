import os

# Parámetros globales
PIXELS_TO_UM = 0.5  # 1 px = 500 nm
IMG_PATH = "images/grain.jpg"
CLUSTERS_FOLDER = "results/imageClusters"
CSV_OUTPUT = "results/image_measurements.csv"
HISTOGRAM_PATH = "results/histograms.png"

# Crear directorios necesarios
os.makedirs(os.path.dirname(IMG_PATH), exist_ok=True)
os.makedirs(CLUSTERS_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
