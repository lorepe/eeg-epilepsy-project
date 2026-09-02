import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import zscore

def pipeline_preprocesamiento(senal, fs=173.61):
    """Aplica filtros Notch y Butterworth, y normaliza con Z-score."""
    # 1. Filtro Notch (50 Hz)
    b_n, a_n = iirnotch(50.0, 30.0, fs)
    senal_limpia = filtfilt(b_n, a_n, senal)

    # 2. Filtro Pasa-banda (0.5 Hz - 60 Hz)
    nyq = 0.5 * fs
    b_b, a_b = butter(4, [0.5/nyq, 60.0/nyq], btype='band')
    senal_limpia = filtfilt(b_b, a_b, senal_limpia)

    # 3. Normalización Z-score
    return zscore(senal_limpia)


def segmentar_senal(senal, fs=173.61, duracion_ventana=4.0, solapamiento=0.50):
    """
    Divide una señal 1D en ventanas superpuestas.
    Por defecto: ventanas de 4 segundos con 50% de solapamiento.
    """
    muestras_ventana = int(duracion_ventana * fs)
    paso = int(muestras_ventana * (1 - solapamiento))
    
    segmentos = []
    inicio = 0
    
    while inicio + muestras_ventana <= len(senal):
        segmento = senal[inicio:inicio + muestras_ventana]
        segmentos.append(segmento)
        inicio += paso
        
    return np.array(segmentos)