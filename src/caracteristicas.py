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

from scipy import stats as _stats


def _entropia_normalizada(p):
    """Entropia de Shannon normalizada (0-1) de una distribucion de potencia."""
    p = np.asarray(p, dtype=float)
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p)))



def features_estadisticas(x):
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x, var_dx, var_ddx = np.var(x), np.var(dx), np.var(ddx)
    movilidad = np.sqrt(var_dx / (var_x + 1e-12))
    movilidad_dx = np.sqrt(var_ddx / (var_dx + 1e-12))
    return {
        "est_media": np.mean(x),
        "est_std": np.std(x),
        "est_asimetria": _stats.skew(x),
        "est_curtosis": _stats.kurtosis(x),
        "est_rms": np.sqrt(np.mean(x ** 2)),
        "est_rango": np.ptp(x),
        "est_long_linea": np.sum(np.abs(dx)),
        "est_hjorth_actividad": var_x,
        "est_hjorth_movilidad": movilidad,
        "est_hjorth_complejidad": movilidad_dx / (movilidad + 1e-12),
    }



def features_espectrales(x, fs=FS):
    frec, psd = calcular_psd(x, fs)
    abs_bandas = potencia_por_bandas(x, fs)          # potencia por bandas (absoluta)
    total = sum(abs_bandas.values()) + 1e-12
    out = {}
    for b, v in abs_bandas.items():
        out[f"psd_abs_{b}"] = v
        out[f"psd_rel_{b}"] = v / total              # potencia por bandas (relativa)
    m = (frec >= 0.5) & (frec <= 40)
    out["psd_entropia"] = _entropia_normalizada(psd[m])   # entropía espectral
    acum = np.cumsum(psd[m]) / (psd[m].sum() + 1e-12)
    out["psd_sef95"] = frec[m][np.searchsorted(acum, 0.95)]
    out["psd_frec_pico"] = frec[m][np.argmax(psd[m])]
    return out



def features_tiempo_frecuencia(x, fs=FS, wavelet="db4", nivel=5):
    out = {}
    # --- Wavelets: energía por subbanda + entropía wavelet
    nivel = min(nivel, pywt.dwt_max_level(len(x), wavelet))
    coefs = pywt.wavedec(x, wavelet, level=nivel)
    nombres = [f"A{nivel}"] + [f"D{nivel - i}" for i in range(nivel)]
    energia = np.array([np.sum(c ** 2) for c in coefs])
    rel = energia / (energia.sum() + 1e-12)
    for n, e, r in zip(nombres, energia, rel):
        out[f"wav_E_{n}"] = e                         # energía wavelet
        out[f"wav_rel_{n}"] = r
    out["wav_entropia"] = float(-np.sum(rel * np.log2(rel + 1e-12)))   # entropía wavelet
    # --- STFT: cómo cambia la potencia por banda a lo largo del tiempo
    frec, tiempos, P = calcular_stft(x, fs)
    m = (frec >= 0.5) & (frec <= 40)
    total_t = P[m].sum(axis=0) + 1e-12
    for b, (lo, hi) in BANDAS.items():
        mb = (frec >= lo) & (frec < hi)
        serie = P[mb].sum(axis=0) / total_t
        out[f"stft_{b}_media"] = serie.mean()
        out[f"stft_{b}_std"] = serie.std()
        out[f"stft_{b}_max"] = serie.max()
    ent_t = np.array([_entropia_normalizada(P[m][:, k]) for k in range(P.shape[1])])
    out["stft_entropia_media"] = ent_t.mean()
    out["stft_entropia_std"] = ent_t.std()
    return out


def extraer_features(x, fs=FS):
    return {**features_estadisticas(x), **features_espectrales(x, fs), **features_tiempo_frecuencia(x, fs)}


def matriz_features(senales, fs=FS):
    import pandas as pd
    return pd.DataFrame([extraer_features(x, fs) for x in senales])


def conjuntos_de_features(columnas):
    est = [c for c in columnas if c.startswith("est_")]
    esp = [c for c in columnas if c.startswith("psd_")]
    tf = [c for c in columnas if c.startswith("wav_") or c.startswith("stft_")]
    return {
        "A_estadisticas": est,
        "B_espectral_clasico": est + esp,         # FFT/PSD
        "C_tiempo_frecuencia": est + tf,          # STFT/wavelets
        "D_todo": est + esp + tf,
    }