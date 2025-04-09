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
    'MaxIntensity',
    'NumSides'  # Add NumSides to the property list
]

def calculate_num_sides(mask):
    """
    Calculates the number of sides (edges) of a region in a binary mask.

    Args:
        mask (numpy.ndarray): A binary mask containing a single region.

    Returns:
        int: The number of sides (edges) of the region.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return len(contours[0])
    else:
        return 0

def measure_properties(labeled_mask, img, pixels_to_um):
    # Resize img to match the dimensions of labeled_mask
    img_resized = cv2.resize(img, (labeled_mask.shape[1], labeled_mask.shape[0]))
    
    clusters = measure.regionprops(labeled_mask, intensity_image=img_resized)
    measurements = []
    for cluster in clusters:
        row = [cluster.label]
        
        prop_calculations = {
            'Area': cluster.area * (pixels_to_um ** 2),
            'equivalent_diameter': cluster.equivalent_diameter * pixels_to_um,
            'orientation': np.degrees(cluster.orientation),
            'MajorAxisLength': cluster.major_axis_length * pixels_to_um,
            'MinorAxisLength': cluster.minor_axis_length * pixels_to_um,
            'Perimeter': cluster.perimeter * pixels_to_um,
            'MinIntensity': cluster.min_intensity,
            'MeanIntensity': cluster.mean_intensity,
            'MaxIntensity': cluster.max_intensity,
            'NumSides': calculate_num_sides(labeled_mask == cluster.label)  # Calculate NumSides
        }
        
        for prop in prop_list:
            value = prop_calculations.get(prop)
            row.append(value)
        measurements.append(row)
    return measurements

def save_measurements_to_csv(measurements, output_csv):
    with open(output_csv, 'w') as output_file:
        headers = ['Label'] + [
            'Area (um^2)',
            'equivalent_diameter (um)',
            'orientation (degrees)',
            'MajorAxisLength (um)',
            'MinorAxisLength (um)',
            'Perimeter (um)',
            'MinIntensity',
            'MeanIntensity',
            'MaxIntensity',
            'NumSides'  # Add NumSides to the header
        ]
        output_file.write(','.join(headers) + '\n')
        for row in measurements:
            output_file.write(','.join(map(str, row)) + '\n')
