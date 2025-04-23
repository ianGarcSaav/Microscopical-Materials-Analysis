from skimage import measure
import numpy as np
import cv2
import csv

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
]

def measure_properties(labeled_mask, img, pixels_to_um):
    # Evitar redimensionamiento si no es necesario
    if img.shape[:2] == labeled_mask.shape:
        img_resized = img
    else:
        # Usar INTER_NEAREST para máscaras - mucho más rápido y adecuado para etiquetas
        img_resized = cv2.resize(img, (labeled_mask.shape[1], labeled_mask.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    # Optimización: convertir a uint8 si es necesario para ahorrar memoria
    if labeled_mask.dtype != np.uint16 and np.max(labeled_mask) < 65535:
        labeled_mask = labeled_mask.astype(np.uint16)
    
    # Calcular propiedades de todas las regiones de una vez (más rápido)
    clusters = measure.regionprops(labeled_mask, intensity_image=img_resized)
    
    # Pre-calcular cuadrado de pixels_to_um para evitar cálculos repetidos
    pixels_to_um_sq = pixels_to_um ** 2
    
    # Reservar espacio para resultados mejora velocidad
    measurements = []
    measurements.append(['Label'] + [
        'Area (um^2)',
        'equivalent_diameter (um)',
        'orientation (degrees)',
        'MajorAxisLength (um)',
        'MinorAxisLength (um)',
        'Perimeter (um)',
        'MinIntensity',
        'MeanIntensity',
        'MaxIntensity',
    ])
    
    for cluster in clusters:
        row = [cluster.label]
        
        # Cálculo de propiedades optimizado
        row.extend([
            cluster.area * pixels_to_um_sq,
            cluster.equivalent_diameter * pixels_to_um,
            np.degrees(cluster.orientation),
            cluster.major_axis_length * pixels_to_um,
            cluster.minor_axis_length * pixels_to_um,
            cluster.perimeter * pixels_to_um,
            cluster.min_intensity,
            cluster.mean_intensity,
            cluster.max_intensity,
        ])
        
        measurements.append(row)
    
    return measurements[1:]  # Retornar sin header

def save_measurements_to_csv(measurements, output_csv):
    # Usar csv.writer es más eficiente que manipular strings manualmente
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
    ]
    
    with open(output_csv, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(headers)
        writer.writerows(measurements)
