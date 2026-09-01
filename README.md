# Detección de Actividad Epiléptica en Señales EEG

Repositorio oficial para el proyecto de procesamiento de bioseñales enfocado en la detección automática de actividad epiléptica utilizando el conjunto de datos de Bonn (Bonn EEG Dataset).

---

## 👥 Equipo de Trabajo
* **Valentina Fonseca** 
* **Laura Olachica** 
* **Lorena Perez** 

---

## 🔬 Fase 1: Planteamiento del Problema e Hipótesis
* **Pregunta de Investigación:** ¿El análisis tiempo-frecuencia mejora la identificación de actividad epiléptica respecto al análisis espectral convencional?
* **Hipótesis:** Se espera que las técnicas de tiempo-frecuencia (STFT y/o Wavelet) permitan identificar con mayor claridad los cambios transitorios, temporales y frecuenciales asociados a la actividad epiléptica en comparación con el análisis espectral estático (FFT/PSD).

---

## 📂 Estructura del Repositorio

El proyecto se encuentra organizado de forma modular para garantizar la reproducibilidad y el trabajo colaborativo:

```text
eeg-epilepsy-project/
│
├── data/
│   ├── raw/                   <- Archivos .txt originales del dataset de Bonn (Carpetas Z, O, N, F, S).
│   └── processed/             <- Datos limpios y estructurados en formato .pkl (Fase 3).
│
├── notebooks/                 <- Cuadernos interactivos para validación y visualización.
│   ├── 01_Fases_2_y_3.ipynb   <- Adquisición, exploración y preprocesamiento de señales.
│   └── 03_Fase4_Extraccion... <- Extracción de características espectrales y tiempo-frecuencia.
│
├── src/                       <- Código fuente modular (funciones reutilizables).
│   ├── __init__.py
│   ├── dataset.py             <- Funciones de carga y verificación de calidad (Fase 2).
│   ├── preprocesamiento.py    <- Filtros Notch, Pasa-banda y normalización Z-score (Fase 3).
│   ├── caracteristicas.py     <- Extracción de PSD, STFT, Wavelet de Morlet y Bandas (Fase 4).
│   └── modelos.py             <- Lógica de clasificación y métricas comparativas (Fase 5 y 6).
│
├── docs/                      <- Presentaciones, infografías y marco teórico (Fase 1).
├── .gitignore                 <- Archivos excluidos del control de versiones (ej. /data).
├── requirements.txt           <- Librerías y dependencias del proyecto.
└── README.md                  <- Documentación principal del proyecto.