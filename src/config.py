import os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
img_folder = os.path.join(base_path, "images/grains/12 Mpx 4608x3456px a 500X")  # images folder is outside src
clusters_folder = os.path.join(base_path, "results", "imageClusters")

# Ensure directories exist
os.makedirs(img_folder, exist_ok=True)
os.makedirs(clusters_folder, exist_ok=True)