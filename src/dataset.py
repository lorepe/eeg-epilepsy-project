import os
import glob
import numpy as np
import pandas as pd

FS = 173.61  # Frecuencia de muestreo en Hz

def cargar_segmento_eeg(ruta_archivo):
    """Carga un archivo .txt individual del dataset de Bonn."""
    return np.loadtxt(ruta_archivo)

def cargar_dataset_completo(ruta_base='data/raw/'):
    """
    Carga las 5 carpetas originales del dataset de Bonn: Z, O, N, F, S.
    Mapeo clínico oficial:
    - Z: Sano, ojos abiertos (Equivalente a Clase A)
    - O: Sano, ojos cerrados (Equivalente a Clase B)
    - N: Interictal, hemisferio opuesto (Equivalente a Clase C)
    - F: Interictal, zona epileptógena (Equivalente a Clase D)
    - S: Ictal / Actividad convulsiva (Equivalente a Clase E)
    """
    carpetas = ['Z', 'O', 'N', 'F', 'S']
    mapeo_clases = {
        'Z': 'Sano_Abierto (A)',
        'O': 'Sano_Cerrado (B)',
        'N': 'Interictal_Externo (C)',
        'F': 'Interictal_Foco (D)',
        'S': 'Ictal_Crisis (E)'
    }
    
    datos = []
    
    for carpeta in carpetas:
        ruta_clase = os.path.join(ruta_base, carpeta, '*.txt')
        archivos = glob.glob(ruta_clase)
        
        for archivo in archivos:
            senal = cargar_segmento_eeg(archivo)
            datos.append({
                'archivo': os.path.basename(archivo),
                'carpeta_original': carpeta,
                'clase': mapeo_clases[carpeta],
                'muestras': len(senal),
                'senal': senal
            })
            
    df = pd.DataFrame(datos)
    return df

def verificar_calidad_datos(df):
    """Valida valores faltantes y el balance de clases para la Fase 2."""
    nulos = df['senal'].apply(lambda x: np.isnan(x).sum()).sum()
    balance = df['clase'].value_counts()
    
    print(f"Total de valores nulos o NaN encontrados: {nulos}")
    print("\nDistribución de segmentos por categoría:")
    print(balance)
    return nulos == 0