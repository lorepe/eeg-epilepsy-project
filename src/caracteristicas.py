import numpy as np
import matplotlib.pyplot as plt
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

# ============================================================
# FUNCIONES DE VISUALIZACIÓN (Fase 4 - Análisis Visual)
# ============================================================

def plot_espectro_frecuencias(senal, fs=FS, titulo_extra=""):
    """Compara FFT y PSD (Welch) de una señal de forma visual."""
    N = len(senal)
    
    # 1. FFT
    fft_resultado = np.fft.rfft(senal)
    frecuencias_fft = np.fft.rfftfreq(N, d=1/fs)
    magnitud_fft = np.abs(fft_resultado)
    
    # 2. PSD (Welch)
    frec_psd, pot_psd = welch(senal, fs=fs, nperseg=1024)
    
    # Gráficas
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].plot(frecuencias_fft, magnitud_fft, color='darkorange')
    axes[0].set_title(f"FFT {titulo_extra}")
    axes[0].set_xlabel("Frecuencia (Hz)")
    axes[0].set_ylabel("Magnitud")
    axes[0].set_xlim(0, 60)
    axes[0].grid(True)
    
    axes[1].semilogy(frec_psd, pot_psd, color='darkgreen')
    axes[1].set_title(f"PSD (Welch) {titulo_extra}")
    axes[1].set_xlabel("Frecuencia (Hz)")
    axes[1].set_ylabel("Potencia/Hz")
    axes[1].set_xlim(0, 60)
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_tiempo_frecuencia(senal, fs=FS, titulo_extra=""):
    """Compara STFT y Transformada Wavelet Continua visualmente."""
    N = len(senal)
    tiempo = np.arange(N) / fs
    
    # 1. STFT
    frec_stft, t_stft, Zxx = stft(senal, fs=fs, nperseg=256, noverlap=128)
    mag_stft = np.abs(Zxx)
    
    # 2. Wavelet (Morlet Clásica para visualización)
    escalas = np.arange(1, 128)
    coeficientes, frec_wavelet = pywt.cwt(senal, escalas, "morl", sampling_period=1/fs)
    orden = np.argsort(frec_wavelet)
    frec_wavelet = frec_wavelet[orden]
    mag_wavelet = np.abs(coeficientes[orden, :])
    
    # Gráficas
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    pcm = axes[0].pcolormesh(t_stft, frec_stft, mag_stft, shading="auto", cmap='viridis')
    axes[0].set_title(f"STFT {titulo_extra}")
    axes[0].set_ylabel("Frecuencia (Hz)")
    axes[0].set_ylim(0, 60)
    fig.colorbar(pcm, ax=axes[0], label="Magnitud")
    
    im = axes[1].imshow(mag_wavelet, extent=[tiempo[0], tiempo[-1], frec_wavelet[0], frec_wavelet[-1]],
                        aspect="auto", origin="lower", cmap='magma')
    axes[1].set_title(f"Transformada Wavelet Morlet {titulo_extra}")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel("Frecuencia (Hz)")
    axes[1].set_ylim(0, 60)
    fig.colorbar(im, ax=axes[1], label="Magnitud")
    
    plt.tight_layout()
    plt.show()