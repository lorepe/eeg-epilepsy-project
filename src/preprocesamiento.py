from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import zscore

def pipeline_preprocesamiento(senal, fs=173.61):
    # 1. Filtro Notch (50 Hz)
    b_n, a_n = iirnotch(50.0, 30.0, fs)
    senal_limpia = filtfilt(b_n, a_n, senal)

    # 2. Filtro Pasa-banda (0.5 Hz - 60 Hz)
    nyq = 0.5 * fs
    b_b, a_b = butter(4, [0.5/nyq, 60.0/nyq], btype='band')
    senal_limpia = filtfilt(b_b, a_b, senal_limpia)

    # 3. Normalización Z-score
    return zscore(senal_limpia)