
from skimage import measure
import numpy as np
import csv
from config import PIXELS_TO_UM, CSV_OUTPUT

def measure_clusters(labeled_mask, img):
    clusters = measure.regionprops(labeled_mask, intensity_image=img)
    
    prop_list = ['Area', 'equivalent_diameter', 'orientation', 'MajorAxisLength', 'MinorAxisLength', 'Perimeter', 'MinIntensity', 'MeanIntensity', 'MaxIntensity']
    
    with open(CSV_OUTPUT, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Label'] + prop_list)
        for cluster in clusters:
            row = [cluster.label]
            row.append(cluster.area * (PIXELS_TO_UM ** 2))
            row.append(np.degrees(cluster.orientation))
            row.append(cluster.major_axis_length * PIXELS_TO_UM)
            row.append(cluster.minor_axis_length * PIXELS_TO_UM)
            row.append(cluster.perimeter * PIXELS_TO_UM)
            row.append(cluster.min_intensity)
            row.append(cluster.mean_intensity)
            row.append(cluster.max_intensity)
            writer.writerow(row)
    print(f"Mediciones guardadas en {CSV_OUTPUT}")
    return clusters