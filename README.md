# Estructura del Proyecto

Este proyecto está organizado en diferentes directorios para mantener una estructura clara y modular. A continuación se describe la organización de las carpetas:

## Directorios Principales

### `/data`
Almacena los datos utilizados en el proyecto:
- `/raw`: Contiene los datos originales sin procesar
- `/processed`: Contiene los datos procesados y listos para su uso

### `/src`
Contiene el código fuente del proyecto, organizado en diferentes módulos:
- `/labeling`: Módulo para el etiquetado de imágenes
- `/segmentation`: Módulo para la segmentación de imágenes
- `/edges`: Módulo para el procesamiento de bordes
- `main.py`: Archivo principal del proyecto


### `/outputs`
Contiene los resultados generados por el proyecto:
- `/image_measurements`: Mediciones y análisis de imágenes
- `/scatters`: Gráficos de dispersión
- `/histograms`: Histogramas generados

## Archivos de Configuración
- `requirements.txt`: Lista de dependencias del proyecto
- `.gitignore`: Configuración de archivos ignorados por Git

## Uso
Para comenzar a trabajar con este proyecto:
1. Instalar las dependencias: `pip install -r requirements.txt`
2. Ejecutar el script principal: `python src/main.py`

config.py (configuración)
    ↓
main.py (punto de entrada)
    ↓
cannyEdge.py (detecta bordes)
    ↓
semanticSegmentation.py (genera máscaras)
    ↓
objectSegmentationVoronoiOtsu.py (segmenta objetos)
    ↓
voronoiOtsuLabeling.py (análisis alternativo)