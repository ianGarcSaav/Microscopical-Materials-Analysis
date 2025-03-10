import os

# Global configuration
pixels_to_um = 0.5  # 1 px = 500 nm
img_path = "images/grain.jpg"
clusters_folder = 'results/imageClusters'
output_csv = 'results/imageMeasurements/image_measurements.csv'
histogram_path = 'results/imageMeasurements/histograms.png'

# Ensure directories exist
os.makedirs(os.path.dirname(img_path), exist_ok=True)
os.makedirs(clusters_folder, exist_ok=True)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
