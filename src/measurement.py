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
        
        prop_calculations = {
            'Area': cluster.area * (pixels_to_um ** 2),
            'orientation': np.degrees(cluster.orientation),
            'Perimeter': cluster.perimeter * pixels_to_um,
            'equivalent_diameter': cluster.equivalent_diameter * pixels_to_um,
            'MajorAxisLength': cluster.major_axis_length * pixels_to_um,
            'MinorAxisLength': cluster.minor_axis_length * pixels_to_um,
            'MinIntensity': cluster.min_intensity,
            'MeanIntensity': cluster.mean_intensity,
            'MaxIntensity': cluster.max_intensity
        }
        
        for prop in prop_list:
            value = prop_calculations.get(prop)
            row.append(value)
        measurements.append(row)
    return measurements

def save_measurements_to_csv(measurements, output_csv):
    with open(output_csv, 'w') as output_file:
        headers = ['Label'] + prop_list
        output_file.write(','.join(headers) + '\n')
        for row in measurements:
            output_file.write(','.join(map(str, row)) + '\n')