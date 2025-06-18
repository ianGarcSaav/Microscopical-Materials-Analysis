import os
from pathlib import Path
from typing import Union, List, Optional
import glob

# Obtener la ruta base del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
print(f"Directorio raíz del proyecto: {PROJECT_ROOT}")

# Rutas principales del proyecto
SRC_DIR = PROJECT_ROOT / 'src'
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

# Rutas de datos
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Rutas de salida
IMAGE_MEASUREMENTS_DIR = OUTPUTS_DIR / 'image_measurements'
SCATTERS_DIR = OUTPUTS_DIR / 'scatters'
HISTOGRAMS_DIR = OUTPUTS_DIR / 'histograms'

# Rutas de código fuente
LABELING_DIR = SRC_DIR / 'labeling'
SEGMENTATION_DIR = SRC_DIR / 'segmentation'
EDGES_DIR = SRC_DIR / 'edges'

# Rutas configurables para datos raw
# Estas rutas pueden ser modificadas por el técnico encargado
RAW_DATA_PATHS = {
    'extras': RAW_DATA_DIR / 'extras',
    'internet': RAW_DATA_DIR / 'internet',
    'high_res': RAW_DATA_DIR / '12 Mpx 4608x3456px a 500X'
}

def update_raw_data_path(key: str, new_path: str) -> None:
    """
    Actualiza una ruta específica en RAW_DATA_PATHS.
    
    Args:
        key (str): Clave de la ruta a actualizar
        new_path (str): Nueva ruta a establecer
    """
    if key in RAW_DATA_PATHS:
        RAW_DATA_PATHS[key] = Path(new_path)
    else:
        raise KeyError(f"La clave '{key}' no existe en RAW_DATA_PATHS")

def create_directories():
    """Crea todos los directorios necesarios si no existen."""
    print("\nCreando directorios del proyecto...")
    
    directories = [
        SRC_DIR,
        DATA_DIR,
        OUTPUTS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        IMAGE_MEASUREMENTS_DIR,
        SCATTERS_DIR,
        HISTOGRAMS_DIR,
        LABELING_DIR,
        SEGMENTATION_DIR,
        EDGES_DIR
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Directorio creado/verificado: {directory}")
        except Exception as e:
            print(f"✗ Error al crear directorio {directory}: {str(e)}")
    
    # Crear directorios de datos raw configurables
    print("\nCreando directorios de datos raw...")
    for key, path in RAW_DATA_PATHS.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Directorio raw creado/verificado: {key} -> {path}")
        except Exception as e:
            print(f"✗ Error al crear directorio raw {key}: {str(e)}")

def verify_path(path: Union[str, Path]) -> bool:
    """
    Verifica si una ruta existe.
    
    Args:
        path (Union[str, Path]): Ruta a verificar
        
    Returns:
        bool: True si la ruta existe, False en caso contrario
    """
    path = Path(path)
    exists = path.exists()
    print(f"Verificando ruta: {path} -> {'✓ Existe' if exists else '✗ No existe'}")
    return exists

def get_files_in_directory(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    Obtiene una lista de archivos en un directorio que coinciden con un patrón.
    
    Args:
        directory (Union[str, Path]): Directorio a buscar
        pattern (str): Patrón de búsqueda (ej: "*.jpg", "*.png")
        
    Returns:
        List[Path]: Lista de rutas a los archivos encontrados
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"✗ El directorio no existe: {directory}")
        return []
    
    files = list(directory.glob(pattern))
    print(f"Encontrados {len(files)} archivos en {directory} con patrón {pattern}")
    return files

def get_raw_data_files(category: str, pattern: str = "*") -> List[Path]:
    """
    Obtiene archivos de una categoría específica en los datos raw.
    
    Args:
        category (str): Categoría de datos raw ('extras', 'internet', 'high_res')
        pattern (str): Patrón de búsqueda de archivos
        
    Returns:
        List[Path]: Lista de rutas a los archivos encontrados
    """
    if category not in RAW_DATA_PATHS:
        raise KeyError(f"Categoría '{category}' no encontrada en RAW_DATA_PATHS")
    return get_files_in_directory(RAW_DATA_PATHS[category], pattern)

def get_output_path(category: str, filename: str) -> Path:
    """
    Obtiene la ruta completa para un archivo de salida.
    
    Args:
        category (str): Categoría de salida ('image_measurements', 'scatters', 'histograms')
        filename (str): Nombre del archivo
        
    Returns:
        Path: Ruta completa al archivo de salida
    """
    output_dirs = {
        'image_measurements': IMAGE_MEASUREMENTS_DIR,
        'scatters': SCATTERS_DIR,
        'histograms': HISTOGRAMS_DIR
    }
    
    if category not in output_dirs:
        raise KeyError(f"Categoría '{category}' no válida para salidas")
    
    return output_dirs[category] / filename

def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Asegura que un directorio existe, creándolo si es necesario.
    
    Args:
        path (Union[str, Path]): Ruta del directorio
        
    Returns:
        Path: Ruta al directorio (creado o existente)
    """
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directorio asegurado: {path}")
    except Exception as e:
        print(f"✗ Error al crear directorio {path}: {str(e)}")
    return path

# Crear directorios al importar el módulo
print("\n=== Inicializando configuración del proyecto ===")
create_directories()
print("=== Configuración completada ===\n")
