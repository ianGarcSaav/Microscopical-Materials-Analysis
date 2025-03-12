import os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
img_folder = os.path.join(base_path, "images")  # images folder is outside src
clusters_folder = os.path.join(base_path, "results", "imageClusters")
csv_folder = os.path.join(base_path, "results", "imageMeasurements", "csv")
histogram_folder = os.path.join(base_path, "results", "imageMeasurements", "histograms")
pixels_to_um = 0.5  # 1 px = 500 nm

# Ensure directories exist
os.makedirs(img_folder, exist_ok=True)
os.makedirs(clusters_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)
os.makedirs(histogram_folder, exist_ok=True)
