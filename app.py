"""
Demostrador del Proyecto 7: detección de actividad epiléptica en EEG (Bonn).
Ejecutar desde la carpeta del proyecto:   streamlit run app.py
"""
import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import streamlit as st
from scipy.signal import stft, welch
from sklearn.feature_selection import f_classif

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.caracteristicas import BANDAS, FS, conjuntos_de_features, matriz_features
from src.modelos import crear_modelo
from src.preprocesamiento import filtrar_senal, segmentar_senal

RAW = os.path.join(ROOT, "data", "raw")
FEATURES = os.path.join(ROOT, "data", "processed", "features.csv")
FIGURAS = os.path.join(ROOT, "figuras")
VENTANA_S, PASO_S = 4.0, 2.0
NOMBRES_CLASE = {"Z": "Sano, ojos abiertos (Z)", "O": "Sano, ojos cerrados (O)",
                 "N": "Interictal, hemisferio opuesto (N)", "F": "Interictal, zona epileptógena (F)",
                 "S": "Crisis epiléptica (S)"}
MODELOS = {"C · Tiempo-frecuencia (STFT/wavelets)": "C_tiempo_frecuencia",
           "B · Espectral clásico (FFT/PSD)": "B_espectral_clasico"}

st.set_page_config(page_title="Detección de crisis en EEG", page_icon="🧠", layout="wide")


# ----------------------------------------------------------------------------- datos y modelo
@st.cache_data(show_spinner=False)
def cargar_features():
    if not os.path.exists(FEATURES):
        return None, None
    df = pd.read_csv(FEATURES)
    cols = [c for c in df.columns if c.startswith(("est_", "psd_", "wav_", "stft_"))]
    return df, cols


def listar_archivos_bonn():
    out = {}
    for c in ["Z", "O", "N", "F", "S"]:
        d = os.path.join(RAW, c)
        if os.path.isdir(d):
            out[c] = sorted(f for f in os.listdir(d) if f.lower().endswith(".txt"))
    return out


@st.cache_resource(show_spinner="Entrenando el modelo (unos segundos)...")
def entrenar(conjunto, excluir):
    df, cols = cargar_features()
    columnas = conjuntos_de_features(cols)[conjunto]
    datos = df[df["archivo"] != excluir] if excluir else df
    modelo = crear_modelo()
    modelo.fit(datos[columnas], datos["etiqueta"])
    return modelo, columnas


@st.cache_data(show_spinner=False)
def ranking_features(columnas):
    df, _ = cargar_features()
    F, _ = f_classif(df[list(columnas)], df["etiqueta"])
    return pd.Series(F, index=list(columnas)).sort_values(ascending=False)


@st.cache_data(show_spinner="Procesando el registro...")
def procesar(senal):
    filtrada = filtrar_senal(senal)
    ventanas = segmentar_senal(filtrada)
    return filtrada, ventanas, matriz_features(ventanas)


# ----------------------------------------------------------------------------- gráficas
def fig_senal(cruda, filtrada, win):
    t = np.arange(len(cruda)) / FS
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(t, cruda, lw=0.6, color="gray"); axes[0].set_ylabel("Cruda (µV)")
    axes[1].plot(t, filtrada, lw=0.6, color="tab:blue"); axes[1].set_ylabel("Filtrada (µV)")
    for ax in axes:
        ax.axvspan(win * PASO_S, win * PASO_S + VENTANA_S, color="orange", alpha=0.25)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("Tiempo (s)   ·   zona naranja = ventana seleccionada")
    fig.tight_layout()
    return fig


def fig_espectro(filtrada):
    fft = np.abs(np.fft.rfft(filtrada)); f = np.fft.rfftfreq(len(filtrada), 1 / FS)
    fp, p = welch(filtrada, fs=FS, nperseg=1024)
    tot = p[(fp >= 0.5) & (fp <= 40)].sum()
    rel = [p[(fp >= lo) & (fp < hi)].sum() / tot for lo, hi in BANDAS.values()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].plot(f, fft, color="darkorange", lw=0.8); axes[0].set_title("FFT"); axes[0].set_ylabel("Magnitud")
    m = fp <= 40
    axes[1].semilogy(fp[m], p[m], color="darkgreen"); axes[1].set_title("PSD (Welch)"); axes[1].set_ylabel("µV²/Hz")
    for ax in axes[:2]:
        ax.set_xlim(0, 40); ax.set_xlabel("Frecuencia (Hz)"); ax.grid(alpha=0.3)
    axes[2].bar(list(BANDAS.keys()), rel, color="tab:blue"); axes[2].set_title("Potencia relativa por banda")
    axes[2].set_ylabel("Fracción de la potencia")
    fig.tight_layout()
    return fig


def fig_tf(filtrada, win):
    t = np.arange(len(filtrada)) / FS
    f, ts, Z = stft(filtrada, fs=FS, nperseg=256, noverlap=128)
    S = 10 * np.log10(np.abs(Z[:, 1:-1]) ** 2 + 1e-12); ts = ts[1:-1]
    k = (f >= 0.5) & (f <= 40)
    frecs = np.linspace(1, 40, 120)
    c, fw = pywt.cwt(filtrada, pywt.central_frequency("morl") * FS / frecs, "morl", sampling_period=1 / FS)
    W = 20 * np.log10(np.abs(c) + 1e-12)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    im1 = axes[0].pcolormesh(ts, f, S, shading="auto", cmap="viridis",
                             vmin=np.percentile(S[k], 1), vmax=np.percentile(S[k], 99.5))
    axes[0].set_ylim(0.5, 40); axes[0].set_title("STFT"); axes[0].set_ylabel("Frecuencia (Hz)")
    fig.colorbar(im1, ax=axes[0], label="Potencia (dB)")
    im2 = axes[1].imshow(W, extent=[t[0], t[-1], fw[0], fw[-1]], aspect="auto", origin="lower", cmap="magma",
                         vmin=np.percentile(W, 1), vmax=np.percentile(W, 99.5), interpolation="antialiased")
    axes[1].set_ylim(1, 40); axes[1].set_title("Wavelet Morlet"); axes[1].set_ylabel("Frecuencia (Hz)")
    axes[1].set_xlabel("Tiempo (s)"); fig.colorbar(im2, ax=axes[1], label="Magnitud (dB)")
    for ax in axes:
        ax.axvline(win * PASO_S, color="white", ls="--", lw=1); ax.axvline(win * PASO_S + VENTANA_S, color="white", ls="--", lw=1)
    fig.tight_layout()
    return fig


def fig_perfil(feats_ventana, columnas):
    df, _ = cargar_features()
    top = list(ranking_features(tuple(columnas)).head(12).index)[::-1]
    mu, sd = df[top].mean(), df[top].std()
    z_v = (feats_ventana[top] - mu) / sd
    z_cri = (df[df["etiqueta"] == 1][top].mean() - mu) / sd
    z_no = (df[df["etiqueta"] == 0][top].mean() - mu) / sd
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(top))
    ax.scatter(z_no, y, color="tab:blue", s=60, label="Media no crisis", zorder=3)
    ax.scatter(z_cri, y, color="tab:red", s=60, label="Media crisis", zorder=3)
    ax.scatter(z_v, y, color="black", marker="D", s=70, label="Ventana seleccionada", zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(top); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Valor estandarizado (desviaciones estándar respecto a todo el dataset)")
    ax.set_title("Las 12 características más discriminantes"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_decision(puntajes, umbral_fraccion):
    colores = ["tab:red" if s >= 0 else "tab:blue" for s in puntajes]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.bar(np.arange(len(puntajes)) * PASO_S + VENTANA_S / 2, puntajes, width=PASO_S * 0.9, color=colores)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Tiempo (s) · centro de cada ventana"); ax.set_ylabel("Puntaje del SVM")
    ax.set_title("Puntaje por ventana (rojo = marcada como crisis, azul = no crisis)"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- interfaz
st.title("🧠 Detección de actividad epiléptica en EEG")
st.caption("Proyecto 7 · Procesamiento de Señales e Imágenes · Dataset de Bonn · "
           "¿El análisis tiempo-frecuencia mejora la detección frente al espectral clásico?")

df_feat, cols_feat = cargar_features()
if df_feat is None:
    st.error("No se encontró `data/processed/features.csv`. Ejecuta primero los notebooks 01 a 03 para generarlo.")
    st.stop()

archivos = listar_archivos_bonn()
with st.sidebar:
    st.header("Configuración")
    origen = st.radio("Fuente del EEG", ["Archivo del dataset de Bonn", "Subir un archivo .txt"])
    nombre_archivo, clase_real, senal = None, None, None
    if origen == "Archivo del dataset de Bonn":
        if not archivos:
            st.warning("No se encontró `data/raw/` con las carpetas Z, O, N, F, S.")
        else:
            clase = st.selectbox("Clase", list(archivos.keys()), format_func=lambda c: NOMBRES_CLASE[c])
            nombre_archivo = st.selectbox("Archivo", archivos[clase])
            clase_real = clase
            senal = np.loadtxt(os.path.join(RAW, clase, nombre_archivo))
    else:
        subido = st.file_uploader("EEG de un canal (una muestra por línea)", type=["txt", "csv"])
        if subido is not None:
            try:
                senal = np.loadtxt(io.BytesIO(subido.getvalue()), delimiter="," if subido.name.endswith(".csv") else None)
                nombre_archivo = subido.name
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")
    modelo_nombre = st.radio("Modelo", list(MODELOS.keys()))
    umbral = st.slider("Fracción de ventanas en crisis para decidir 'CRISIS'", 0.1, 0.9, 0.5, 0.05)
    excluir = False
    if clase_real is not None:
        excluir = st.checkbox("Entrenar sin este archivo (evaluación honesta)", value=True,
                              help="Si está activo, el modelo no vio este registro durante el entrenamiento.")
    st.divider()
    st.caption(f"Frecuencia de muestreo asumida: {FS} Hz · ventanas de 4 s con 50 % de solape.")

tab_res, tab_senal, tab_esp, tab_tf, tab_feat, tab_dec, tab_est = st.tabs(
    ["Resumen", "Señal y filtrado", "FFT / PSD", "Tiempo-frecuencia", "Características", "Decisión del modelo", "Resultados del estudio"])

if senal is None:
    with tab_res:
        st.info("Elige un archivo en la barra lateral para analizarlo.")
else:
    if len(senal) < 694:
        st.error("El registro es muy corto: se necesitan al menos 694 muestras (4 s).")
        st.stop()
    filtrada, ventanas, feats = procesar(senal)
    conjunto = MODELOS[modelo_nombre]
    modelo, columnas = entrenar(conjunto, nombre_archivo if excluir else None)
    puntajes = modelo.decision_function(feats[columnas])
    marcadas = puntajes >= 0
    fraccion = float(marcadas.mean())
    decision = "CRISIS" if fraccion >= umbral else "NO CRISIS"
    with st.sidebar:
        win = st.slider("Ventana a inspeccionar", 0, len(ventanas) - 1, int(np.argmax(puntajes)),
                        help="Por defecto, la ventana con mayor puntaje de crisis.")

    with tab_res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Decisión del modelo", decision)
        c2.metric("Ventanas marcadas", f"{marcadas.sum()} de {len(marcadas)}", f"{fraccion:.0%}")
        c3.metric("Duración", f"{len(senal) / FS:.1f} s", f"{len(senal)} muestras")
        c4.metric("Amplitud RMS", f"{np.sqrt(np.mean(filtrada ** 2)):.1f} µV")
        if clase_real is not None:
            real = "CRISIS" if clase_real == "S" else "NO CRISIS"
            (st.success if real == decision else st.error)(
                f"Clase real: **{NOMBRES_CLASE[clase_real]}** → {real}. "
                f"{'El modelo acertó.' if real == decision else 'El modelo se equivocó en este registro.'}")
            if excluir:
                st.caption("El modelo se entrenó sin este archivo, por lo que es una predicción honesta.")
        st.pyplot(fig_decision(puntajes, umbral), clear_figure=True)
        st.caption(f"Modelo: {modelo_nombre} · {len(columnas)} características · SVM RBF con clases balanceadas.")

    with tab_senal:
        st.pyplot(fig_senal(senal, filtrada, win), clear_figure=True)
        st.markdown("Filtro notch a 50 Hz y pasa-banda de 0.5–60 Hz (Butterworth de orden 4, `filtfilt`, sin z-score: "
                    "se conserva la amplitud real).")

    with tab_esp:
        st.pyplot(fig_espectro(filtrada), clear_figure=True)
        st.markdown("La FFT y la PSD resumen **todo el registro** en un espectro: dicen cuánta potencia hay en cada "
                    "frecuencia, pero no *cuándo* ocurre.")

    with tab_tf:
        st.pyplot(fig_tf(filtrada, win), clear_figure=True)
        st.markdown("La STFT y la wavelet conservan el **tiempo**: se ve cómo cambia la energía a lo largo del registro. "
                    "Las líneas blancas marcan la ventana seleccionada.")

    with tab_feat:
        fila = feats.iloc[win]
        st.subheader(f"Ventana {win} · {win * PASO_S:.0f}–{win * PASO_S + VENTANA_S:.0f} s · "
                     f"puntaje {puntajes[win]:+.2f} ({'crisis' if puntajes[win] >= 0 else 'no crisis'})")
        izq, der = st.columns([1.1, 1])
        with izq:
            st.pyplot(fig_perfil(fila, columnas), clear_figure=True)
        with der:
            tabla = pd.DataFrame({"característica": fila.index, "valor": fila.values})
            fam = {"est": "Estadística", "psd": "Espectral (PSD)", "wav": "Wavelet", "stft": "STFT"}
            tabla["familia"] = tabla["característica"].str.split("_").str[0].map(fam)
            tabla["usada por el modelo"] = tabla["característica"].isin(columnas)
            st.dataframe(tabla, height=420, width="stretch")
        st.download_button("Descargar características de todas las ventanas (CSV)", feats.to_csv(index=False),
                           file_name=f"caracteristicas_{nombre_archivo}.csv", mime="text/csv")

    with tab_dec:
        st.pyplot(fig_decision(puntajes, umbral), clear_figure=True)
        st.markdown(f"**Regla de decisión:** el registro se marca como crisis si al menos el **{umbral:.0%}** de sus "
                    f"ventanas tiene puntaje ≥ 0. En este registro: **{fraccion:.0%}** → **{decision}**.")
        comparacion = []
        for nombre, conj in MODELOS.items():
            m, cs = entrenar(conj, nombre_archivo if excluir else None)
            p = m.decision_function(feats[cs])
            comparacion.append({"Modelo": nombre, "Ventanas marcadas": f"{(p >= 0).sum()} de {len(p)}",
                                "Fracción": f"{(p >= 0).mean():.0%}",
                                "Decisión": "CRISIS" if (p >= 0).mean() >= umbral else "NO CRISIS"})
        st.markdown("**Los dos métodos sobre este mismo registro:**")
        st.dataframe(pd.DataFrame(comparacion), hide_index=True, width="stretch")

with tab_est:
    st.subheader("Resultados del estudio (validación cruzada agrupada por archivo)")
    for nombre_csv in ["resultados_fase6.csv", "resultados_fase5.csv"]:
        ruta = os.path.join(ROOT, "data", "processed", nombre_csv)
        if os.path.exists(ruta):
            r = pd.read_csv(ruta)
            m = ["sensibilidad", "especificidad", "f1", "auc"]
            g = r.groupby("conjunto")[m]
            st.markdown(f"**{nombre_csv}** (media ± desviación sobre {len(r) // r['conjunto'].nunique()} pares semilla × fold)")
            st.dataframe(g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str), width="stretch")
            break
    galeria = [("barras_B_vs_C.png", "Métricas por método"), ("roc_B_vs_C.png", "Curvas ROC"),
               ("metricas_con_IC.png", "Métricas con IC95 %"), ("bandas_y_wavelet.png", "Potencia por banda y energía wavelet"),
               ("ranking_familias.png", "Ranking de características"), ("espectros_sano_vs_crisis.png", "Espectros: sano vs crisis"),
               ("tf_sano_vs_crisis.png", "Tiempo-frecuencia: sano vs crisis"), ("ejemplos_FP.png", "Falsos positivos"),
               ("ejemplos_FN.png", "Falsos negativos")]
    disponibles = [(a, t) for a, t in galeria if os.path.exists(os.path.join(FIGURAS, a))]
    if disponibles:
        elegido = st.selectbox("Figura", [t for _, t in disponibles])
        st.image(os.path.join(FIGURAS, dict((t, a) for a, t in disponibles)[elegido]), width="stretch")
    else:
        st.info("Genera las figuras ejecutando los notebooks; aparecerán aquí desde la carpeta `figuras/`.")
