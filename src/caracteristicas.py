import numpy as np
from scipy.signal import welch, stft
import pywt

FS = 173.61

BANDAS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 40)
}

def calcular_psd(senal, fs=FS):
    """Calcula la Densidad Espectral de Potencia usando el método de Welch."""
    nperseg = min(256, len(senal))
    frecuencias, psd = welch(senal, fs=fs, nperseg=nperseg)
    return frecuencias, psd

def calcular_stft(senal, fs=FS):
    """Calcula la Transformada de Fourier de Tiempo Corto (STFT)."""
    frecuencias, tiempos, Zxx = stft(senal, fs=fs, window="hann", nperseg=128, noverlap=64)
    potencia = np.abs(Zxx) ** 2
    return frecuencias, tiempos, potencia

def calcular_wavelet(senal, fs=FS):
    """Calcula la transformada continua con Wavelet de Morlet."""
    frecuencias_deseadas = np.linspace(1, 40, 80)
    wavelet = "cmor1.5-1.0"
    frecuencia_central = pywt.central_frequency(wavelet)
    escalas = frecuencia_central * fs / frecuencias_deseadas
    
    coeficientes, frecuencias = pywt.cwt(senal, escalas, wavelet, sampling_period=1/fs)
    potencia = np.abs(coeficientes) ** 2
    return frecuencias, potencia

def potencia_por_bandas(senal, fs=FS):
    """Extrae la energía absoluta en las bandas de frecuencia clásicas del EEG."""
    frecuencias, psd = calcular_psd(senal, fs)
    resultados = {}
    
    for nombre, (f_min, f_max) in BANDAS.items():
        indices = (frecuencias >= f_min) & (frecuencias < f_max)
        if np.any(indices):
            potencia = np.trapezoid(psd[indices], frecuencias[indices])
        else:
            potencia = 0.0
        resultados[nombre] = potencia
        
    return resultados