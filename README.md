# Detección de actividad epiléptica en EEG: ¿aporta el análisis tiempo-frecuencia?

Proyecto 7 de Procesamiento de Señales e Imágenes (ECCI). Comparación entre el análisis espectral clásico (FFT/PSD) y el análisis tiempo-frecuencia (STFT/wavelets) para detectar actividad epiléptica en el dataset de Bonn.

## Equipo
* Valentina Fonseca
* Laura Olachica
* Lorena Pérez

---

## Fase 1. Planteamiento del problema e hipótesis

**Pregunta de investigación:** ¿El análisis tiempo-frecuencia mejora la identificación de actividad epiléptica respecto al análisis espectral convencional?

**Hipótesis:** las técnicas de tiempo-frecuencia (STFT y wavelets) permiten identificar con mayor claridad los cambios transitorios, temporales y frecuenciales asociados a la actividad epiléptica que el análisis espectral estático (FFT/PSD).

**Hipótesis estadísticas** (comparación entre el conjunto de características B, espectral clásico, y C, tiempo-frecuencia):
* **H₀:** no hay diferencia en sensibilidad, especificidad, F1 ni AUC entre B y C.
* **H₁:** C obtiene mejor desempeño que B.

Se evalúan con validación cruzada agrupada por archivo, intervalos de confianza del 95 % por bootstrap y prueba de Wilcoxon pareada (α = 0.05).

---

## Fase 2. Datos

* **Fuente:** Andrzejak RG, Lehnertz K, Mormann F, Rieke C, David P, Elger CE. *Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: dependence on recording region and brain state.* Physical Review E 64, 061907 (2001). Dataset de la Universidad de Bonn (Departamento de Epileptología).
* **Contenido:** 5 conjuntos de 100 registros de un solo canal, 4097 muestras a 173.61 Hz (23.6 s por registro), sin valores faltantes. No incluye identificadores de paciente.

| Carpeta | Clase | Descripción |
|---|---|---|
| Z (A) | Sano, ojos abiertos | EEG de superficie |
| O (B) | Sano, ojos cerrados | EEG de superficie |
| N (C) | Interictal, hemisferio opuesto | Electrodos intracraneales |
| F (D) | Interictal, zona epileptógena | Electrodos intracraneales |
| S (E) | Ictal (crisis) | Electrodos intracraneales |

* **Tarea:** clasificación binaria, crisis (S) contra no crisis (Z, O, N, F). Desbalance 1:4.
* Los registros ya vienen filtrados con pasa-banda de 0.5 a 40 Hz.

---

## Fase 3. Preprocesamiento

1. **Filtro notch a 50 Hz** (la red eléctrica de Alemania, donde se registró el dataset).
2. **Pasa-banda Butterworth de orden 4, 0.5–60 Hz**, aplicado con `filtfilt` (filtrado de fase cero, para no desplazar eventos en el tiempo).
3. **Sin normalización z-score en el análisis.** Normalizar cada señal elimina la diferencia de amplitud, que es una de las características más discriminantes. El DataFrame conserva también una columna normalizada (`senal_limpia`) solo como referencia.
4. **Segmentación** en ventanas de 4 s con 50 % de solape (694 muestras): 5000 ventanas (1000 de crisis y 4000 de no crisis).
5. **Artefactos:** el dataset fue seleccionado por sus autores sin artefactos de movimiento ocular ni muscular, por lo que no se aplicó otra técnica de eliminación.

Como las ventanas se solapan, toda la validación se hace **agrupando por archivo** para evitar fuga de información entre entrenamiento y prueba.

---

## Fase 4. Características (53 por ventana)

| Familia | Nº | Contenido |
|---|---|---|
| Estadísticas (`est_`) | 10 | media, desviación estándar, asimetría, curtosis, RMS, rango, longitud de línea, parámetros de Hjorth |
| Espectrales, FFT/PSD (`psd_`) | 13 | potencia absoluta y relativa por banda (delta, theta, alfa, beta, gamma), entropía espectral, SEF95, frecuencia pico |
| Wavelet (`wav_`) | 13 | energía y energía relativa por subbanda con **DWT db4 de nivel 5**, entropía wavelet |
| STFT (`stft_`) | 17 | media, desviación y máximo de la potencia relativa por banda en el tiempo, entropía por ventana |

La transformada wavelet continua (Morlet) se usa **solo para las figuras** (mapas tiempo-frecuencia); las características de energía wavelet salen de la DWT.

Conjuntos comparados: **B** = estadísticas + PSD (23 características) y **C** = estadísticas + wavelet + STFT (40 características).

---

## Fase 5. Experimento comparativo

Mismo clasificador (SVM con kernel RBF y `class_weight="balanced"`), mismas ventanas y mismos folds para B y C. Validación cruzada estratificada de 5 folds, agrupada por archivo.

| Método | Sensibilidad | Especificidad | F1 | AUC |
|---|---|---|---|---|
| B: espectral clásico | 0.963 | 0.982 | 0.947 | 0.995 |
| C: tiempo-frecuencia | 0.977 | 0.988 | 0.964 | 0.998 |

*(media sobre 5 folds, una semilla; resultados completos en `data/processed/resultados_fase5.csv`)*

---

## Fase 6. Análisis estadístico y validación

Validación repetida con 10 semillas (50 pares semilla × fold), IC95 % por bootstrap con remuestreo de archivos, prueba de Wilcoxon pareada y un control con el mismo número de características (`C_top23`: las 23 mejores características de C, seleccionadas dentro de cada fold).

> **COMPLETAR con la salida del notebook 05:** tabla de la diferencia C − B con su IC95 % y p-valor por métrica, y el resultado del control `C_top23`.

---

## Fase 7. Interpretación biomédica y producto final

> **COMPLETAR con la salida del notebook 06:** en qué clases se concentran los falsos positivos y negativos, los archivos problemáticos, el perfil de características de los errores y las métricas por registro.

**Limitaciones**
* Un solo canal y sin identificadores de paciente: no se puede validar por sujeto.
* Registros ya filtrados y casi sin artefactos; el EEG clínico real es más ruidoso.
* Mezcla de electrodos de superficie (Z, O) e intracraneales (N, F, S): parte de la separación puede deberse al tipo de registro.
* C tiene más características que B; por eso se incluye el control `C_top23`.
* Los errores se concentran en pocos registros.

**Relevancia biomédica:** apoyo a la revisión de registros prolongados de video-EEG. Un falso negativo es una crisis no detectada y un falso positivo, una falsa alarma; por eso se reporta el punto de operación (especificidad alcanzable con una sensibilidad mínima). Es una prueba de concepto, no una herramienta diagnóstica.

**Demostrador:** el notebook 06 incluye `detectar_crisis`, que recibe un EEG crudo, lo filtra, segmenta, extrae las características y decide (crisis / no crisis) según la fracción de ventanas marcadas. Entrena y prueba con los mismos datos, así que ilustra el flujo y no mide rendimiento.

---

## Estructura del repositorio

```text
eeg-epilepsy-project/
├── data/
│   ├── raw/                  <- Bonn original: carpetas Z, O, N, F, S con los .txt (NO se sube a git)
│   └── processed/            <- .pkl (no se suben) y .csv de características y resultados
├── figuras/                  <- Figuras generadas por los notebooks
├── notebooks/
│   ├── 01_Fase2_Exploracion_Datos.ipynb
│   ├── 02_Fase3_Preprocesamiento.ipynb
│   ├── 03_Fase4_Caracteristicas.ipynb
│   ├── 04_Fase5_Comparacion.ipynb
│   ├── 05_Fase6_Validacion.ipynb
│   └── 06_Fase7_Interpretacion.ipynb
├── src/
│   ├── dataset.py            <- Carga y verificación de calidad (Fase 2)
│   ├── preprocesamiento.py   <- Filtrado y segmentación (Fase 3)
│   ├── caracteristicas.py    <- Características y figuras tiempo-frecuencia (Fase 4)
│   └── modelos.py            <- Comparación, validación estadística, errores y demostrador (Fases 5 a 7)
├── requirements.txt
└── README.md
```

## Cómo reproducir

1. **Clonar el repositorio** e instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. **Descargar el dataset de Bonn** y copiar sus carpetas en `data/raw/` de modo que existan `data/raw/Z`, `data/raw/O`, `data/raw/N`, `data/raw/F` y `data/raw/S`, cada una con 100 archivos `.txt` (en la carpeta N las extensiones pueden venir en mayúsculas: `.TXT`).
3. **Ejecutar los notebooks en este orden** (cada uno usa los archivos que genera el anterior):

| Orden | Notebook | Genera |
|---|---|---|
| 1 | `01_Fase2_Exploracion_Datos` | Descripción del dataset |
| 2 | `02_Fase3_Preprocesamiento` | `dataset_preprocesado.pkl`, `dataset_segmentado.pkl` |
| 3 | `03_Fase4_Caracteristicas` | `features.csv`, figuras de características |
| 4 | `04_Fase5_Comparacion` | `resultados_fase5.csv` |
| 5 | `05_Fase6_Validacion` | `resultados_fase6.csv`, figuras con intervalos de confianza |
| 6 | `06_Fase7_Interpretacion` | Análisis de errores y demostrador |

**Reproducibilidad:** todas las validaciones usan semillas fijas. Los pliegues de `StratifiedGroupKFold` pueden variar ligeramente entre versiones de scikit-learn, por lo que el entorno fija la versión (ver `requirements.txt`).