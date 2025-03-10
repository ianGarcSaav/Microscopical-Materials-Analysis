import preprocessing
import labeling
import measurement
import visualization

print("Iniciando análisis de imagen...")

# Cargar imagen y preprocesar
img = preprocessing.load_image()
mask = preprocessing.preprocess_image(img)

# Etiquetar clusters
labeled_mask, num_labels = labeling.label_clusters(mask)
print(f"Se detectaron {num_labels} clusters.")

# Medir propiedades
clusters = measurement.measure_clusters(labeled_mask, img)

# Guardar imagen segmentada
visualization.save_labeled_image(labeled_mask)

# Extraer datos para histogramas
areas = [c.area * (0.5 ** 2) for c in clusters]
perimeters = [c.perimeter * 0.5 for c in clusters]
equivalent_diameters = [c.equivalent_diameter * 0.5 for c in clusters]

if areas:
    visualization.generate_histograms(areas, perimeters, equivalent_diameters)
    print("Análisis completado con éxito.")
else:
    print("No se detectaron clusters para generar histogramas.")
