from skimage import measure
import numpy as np
import cv2

prop_list = [
    'Area',
    'equivalent_diameter',
    'orientation',
    'MajorAxisLength',
    'MinorAxisLength',
    'Perimeter',
    'MinIntensity',
    'MeanIntensity',
    'MaxIntensity'
]

def measure_properties(labeled_mask, img, pixels_to_um):
    # Resize img to match the dimensions of labeled_mask
    img_resized = cv2.resize(img, (labeled_mask.shape[1], labeled_mask.shape[0]))
    
    clusters = measure.regionprops(labeled_mask, intensity_image=img_resized)
    measurements = []
    for cluster in clusters:
        row = [cluster.label]
        for prop in prop_list:
            if prop == 'Area':
                value = cluster.area * (pixels_to_um ** 2)
            elif prop == 'orientation':
                value = np.degrees(cluster.orientation)
            elif prop == 'Perimeter':
                value = cluster.perimeter * pixels_to_um
            elif prop == 'equivalent_diameter':
                value = cluster.equivalent_diameter * pixels_to_um
            elif prop == 'MajorAxisLength':
                value = cluster.major_axis_length * pixels_to_um
            elif prop == 'MinorAxisLength':
                value = cluster.minor_axis_length * pixels_to_um
            elif prop == 'MinIntensity':
                value = cluster.min_intensity
            elif prop == 'MeanIntensity':
                value = cluster.mean_intensity
            elif prop == 'MaxIntensity':
                value = cluster.max_intensity
            row.append(value)
        measurements.append(row)
    return measurements

def save_measurements_to_csv(measurements, output_csv):
    with open(output_csv, 'w') as output_file:
        headers = ['Label'] + prop_list
        output_file.write(','.join(headers) + '\n')
        for row in measurements:
            output_file.write(','.join(map(str, row)) + '\n')
