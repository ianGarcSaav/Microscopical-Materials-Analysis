import os
import shutil

def reset_results_folder(results_path="results"):
    # Eliminar la carpeta de resultados si existe
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
        print(f"Carpeta '{results_path}' eliminada.")
    
    # Crear la carpeta de resultados nuevamente
    os.makedirs(results_path)
    print(f"Carpeta '{results_path}' creada.")

if __name__ == "__main__":
    reset_results_folder()
