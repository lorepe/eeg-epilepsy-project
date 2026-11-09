
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def crear_modelo():
   
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))


def comparar(df_feat, y, grupos, conjuntos, n_splits=5, semilla=0):
    
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    filas = []
    for fold, (tr, te) in enumerate(cv.split(df_feat, y, grupos)):
        for nombre, cols in conjuntos.items():
            modelo = crear_modelo()
            modelo.fit(df_feat.iloc[tr][cols], y[tr])
            puntaje = modelo.decision_function(df_feat.iloc[te][cols])
            pred = (puntaje >= 0).astype(int)
            tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
            filas.append(dict(fold=fold, conjunto=nombre,
                              sensibilidad=tp / (tp + fn), especificidad=tn / (tn + fp),
                              f1=f1_score(y[te], pred), auc=roc_auc_score(y[te], puntaje)))
    return pd.DataFrame(filas)


def resumen(res):
    
    m = ["sensibilidad", "especificidad", "f1", "auc"]
    g = res.groupby("conjunto")[m]
    return g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)


def curvas_roc(df_feat, y, grupos, conjuntos, n_splits=5, semilla=0):
    
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    salida = {}
    for nombre, cols in conjuntos.items():
        puntajes = np.zeros(len(y))
        for tr, te in cv.split(df_feat, y, grupos):
            modelo = crear_modelo()
            modelo.fit(df_feat.iloc[tr][cols], y[tr])
            puntajes[te] = modelo.decision_function(df_feat.iloc[te][cols])
        fpr, tpr, _ = roc_curve(y, puntajes)
        salida[nombre] = (fpr, tpr, roc_auc_score(y, puntajes))
    return salida


# ============================================================
# FASE 6: validación estadística (varias semillas, IC95 %, pruebas)
# ============================================================
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif


def crear_modelo_k(k=None):
    """Igual que crear_modelo, pero con selección de las k mejores características DENTRO de cada fold."""
    pasos = [StandardScaler()]
    if k:
        pasos.append(SelectKBest(f_classif, k=k))
    pasos.append(SVC(kernel="rbf", class_weight="balanced"))
    return make_pipeline(*pasos)


def metricas_rapidas(y, puntaje, umbral=0.0):
    """Sensibilidad, especificidad, F1 y AUC a partir de un puntaje continuo (umbral 0 para el SVM)."""
    pred = puntaje >= umbral
    tp = np.sum(pred & (y == 1)); fn = np.sum(~pred & (y == 1))
    tn = np.sum(~pred & (y == 0)); fp = np.sum(pred & (y == 0))
    sens = tp / (tp + fn); esp = tn / (tn + fp)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return dict(sensibilidad=sens, especificidad=esp, f1=f1, auc=roc_auc_score(y, puntaje))


def comparar_semillas(df_feat, y, grupos, conjuntos, k_por_conjunto=None, n_splits=5, semillas=range(10)):
    """
    Validación cruzada agrupada por archivo, repetida con varias semillas.
    Devuelve (res, oof): res = métricas por semilla/fold/conjunto; oof = puntajes fuera de muestra
    de la primera semilla (para IC, errores y curvas).
    """
    k_por_conjunto = k_por_conjunto or {}
    semillas = list(semillas)
    filas, oof = [], {n: np.zeros(len(y)) for n in conjuntos}
    for semilla in semillas:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
        for fold, (tr, te) in enumerate(cv.split(df_feat, y, grupos)):
            for nombre, cols in conjuntos.items():
                modelo = crear_modelo_k(k_por_conjunto.get(nombre))
                modelo.fit(df_feat.iloc[tr][cols], y[tr])
                puntaje = modelo.decision_function(df_feat.iloc[te][cols])
                if semilla == semillas[0]:
                    oof[nombre][te] = puntaje
                filas.append(dict(semilla=semilla, fold=fold, conjunto=nombre, **metricas_rapidas(y[te], puntaje)))
    return pd.DataFrame(filas), oof


def resumen_semillas(res):
    """Media ± desviación sobre todos los pares semilla×fold."""
    m = ["sensibilidad", "especificidad", "f1", "auc"]
    g = res.groupby("conjunto")[m]
    return g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)


def wilcoxon_pareado(res, a="B_espectral_clasico", b="C_tiempo_frecuencia"):
    """Wilcoxon pareado (b vs a) sobre los mismos pares semilla×fold, para cada métrica."""
    filas = []
    for m in ["sensibilidad", "especificidad", "f1", "auc"]:
        t = res.pivot_table(index=["semilla", "fold"], columns="conjunto", values=m)
        d = t[b] - t[a]
        p = 1.0 if np.allclose(d, 0) else stats.wilcoxon(t[b], t[a]).pvalue
        filas.append(dict(metrica=m, media_a=t[a].mean(), media_b=t[b].mean(), diferencia=d.mean(),
                          pares_b_gana=int((d > 0).sum()), pares_empate=int((d == 0).sum()),
                          pares_a_gana=int((d < 0).sum()), n_pares=len(d), p_wilcoxon=float(f"{p:.3g}")))
    out = pd.DataFrame(filas)
    return out.round({c: 4 for c in out.columns if c != "p_wilcoxon"})


def bootstrap_por_archivo(y, grupos, puntajes, a="B_espectral_clasico", b="C_tiempo_frecuencia",
                          n_boot=2000, semilla=0):
    """
    IC 95 % por bootstrap con REMUESTREO DE ARCHIVOS (las ventanas de un mismo archivo van juntas).
    puntajes: dict {conjunto: puntajes fuera de muestra}. Devuelve (tabla IC por conjunto, tabla de la diferencia b - a).
    """
    rng = np.random.default_rng(semilla)
    archivos = np.unique(grupos)
    idx_por = {g: np.where(grupos == g)[0] for g in archivos}
    nombres = list(puntajes)
    nombres_m = ["sensibilidad", "especificidad", "f1", "auc"]
    boot = {n: {m: [] for m in nombres_m} for n in nombres}
    for _ in range(n_boot):
        idx = np.concatenate([idx_por[g] for g in rng.choice(archivos, size=len(archivos), replace=True)])
        for n in nombres:
            r = metricas_rapidas(y[idx], puntajes[n][idx])
            for m in nombres_m:
                boot[n][m].append(r[m])
    filas = []
    for n in nombres:
        punto = metricas_rapidas(y, puntajes[n])
        for m in nombres_m:
            lo, hi = np.percentile(boot[n][m], [2.5, 97.5])
            filas.append(dict(conjunto=n, metrica=m, estimado=punto[m], ic95_inf=lo, ic95_sup=hi))
    tabla_ic = pd.DataFrame(filas)
    pa, pb = metricas_rapidas(y, puntajes[a]), metricas_rapidas(y, puntajes[b])
    filas = []
    for m in nombres_m:
        d = np.array(boot[b][m]) - np.array(boot[a][m])
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        filas.append(dict(metrica=m, diferencia=pb[m] - pa[m], ic95_inf=lo, ic95_sup=hi,
                          p_bootstrap=p, ic_excluye_cero=bool(lo > 0 or hi < 0)))
    return tabla_ic.round(4), pd.DataFrame(filas).round(4)


# ============================================================
# FASE 7: análisis de falsos positivos / falsos negativos
# ============================================================
def tabla_oof(df_feat, y, puntajes):
    """Una fila por ventana con puntaje, predicción y tipo (TP, TN, FP, FN) para cada conjunto."""
    t = df_feat[["archivo", "carpeta_original", "ventana"]].copy()
    t["y"] = y
    for n, s in puntajes.items():
        pred = (s >= 0).astype(int)
        t[f"score_{n}"] = s
        t[f"tipo_{n}"] = np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 0), (y == 0) & (pred == 1)],
                                   ["TP", "TN", "FP"], default="FN")
    return t


def errores_por_clase(oof, nombre):
    """Cuántas ventanas de cada clase original (Z, O, N, F, S) caen en TP/TN/FP/FN y su tasa de error."""
    c = pd.crosstab(oof["carpeta_original"], oof[f"tipo_{nombre}"]).reindex(["Z", "O", "N", "F", "S"])
    for col in ["TP", "TN", "FP", "FN"]:
        if col not in c:
            c[col] = 0
    c = c[["TP", "TN", "FP", "FN"]]
    c["ventanas"] = c.sum(axis=1)
    c["tasa_error_%"] = ((c["FP"] + c["FN"]) / c["ventanas"] * 100).round(1)
    return c


def archivos_problematicos(oof, nombre, top=10):
    """Archivos con más ventanas mal clasificadas (de 10 posibles)."""
    e = oof[oof[f"tipo_{nombre}"].isin(["FP", "FN"])]
    return (e.groupby(["archivo", "carpeta_original", f"tipo_{nombre}"]).size()
              .rename("ventanas_erroneas").reset_index()
              .sort_values("ventanas_erroneas", ascending=False).head(top).reset_index(drop=True))


def perfil_de_errores(oof, df_feat, nombre, cols):
    """Media (en unidades de desviación estándar global) de características clave por tipo de resultado."""
    z = (df_feat[cols] - df_feat[cols].mean()) / df_feat[cols].std()
    z = z.assign(tipo=oof[f"tipo_{nombre}"].values)
    return z.groupby("tipo")[cols].mean().reindex(["TN", "FP", "FN", "TP"]).round(2)


def nivel_registro(oof, nombre, umbral_fraccion=0.5):
    """Decisión por registro (archivo): crisis si al menos la fracción indicada de ventanas se marca como crisis."""
    g = oof.groupby(["archivo", "carpeta_original"]).agg(
        y=("y", "max"), fraccion=(f"tipo_{nombre}", lambda s: s.isin(["TP", "FP"]).mean())).reset_index()
    g["pred"] = (g["fraccion"] >= umbral_fraccion).astype(int)
    tn, fp, fn, tp = confusion_matrix(g["y"], g["pred"], labels=[0, 1]).ravel()
    return g, dict(TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn), sensibilidad=tp / (tp + fn), especificidad=tn / (tn + fp))


def especificidad_a_sensibilidad(y, puntaje, objetivos=(0.90, 0.95, 0.99)):
    """Punto de operación: especificidad alcanzable para una sensibilidad mínima (relevante clínicamente)."""
    fpr, tpr, _ = roc_curve(y, puntaje)
    return {f"sens>={o:.0%}": round(float(1 - fpr[np.argmax(tpr >= o)]), 4) for o in objetivos}


def graficar_errores(df_seg, oof, nombre, tipo, n=3, fs=173.61):
    """Dibuja las ventanas con el error más 'seguro' (mayor |puntaje|) de un tipo (FP o FN)."""
    import matplotlib.pyplot as plt
    sel = oof[oof[f"tipo_{nombre}"] == tipo].copy()
    if sel.empty:
        return None
    sel["abs"] = sel[f"score_{nombre}"].abs()
    sel = sel.sort_values("abs", ascending=False).head(n)
    fig, axes = plt.subplots(len(sel), 1, figsize=(11, 2.6 * len(sel)), squeeze=False)
    for ax, (i, r) in zip(axes[:, 0], sel.iterrows()):
        x = df_seg.iloc[i]["senal"]
        ax.plot(np.arange(len(x)) / fs, x, lw=0.8, color="tab:red" if tipo == "FN" else "tab:orange")
        ax.set_title(f"{tipo} · {r['archivo']} (clase {r['carpeta_original']}) · ventana {r['ventana']} · puntaje {r[f'score_{nombre}']:.2f}", fontsize=10)
        ax.set_ylabel("µV"); ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("Tiempo (s)")
    plt.tight_layout()
    return fig


# ============================================================
# DEMOSTRADOR: entrenar el modelo final y detectar crisis en un registro nuevo
# ============================================================
def entrenar_modelo_final(df_feat, y, columnas, k=None):
    modelo = crear_modelo_k(k)
    modelo.fit(df_feat[columnas], y)
    return modelo


def detectar_crisis(senal_cruda, modelo, columnas, umbral_fraccion=0.5, fs=173.61):
    """Recibe un EEG crudo (1 canal), lo filtra, segmenta, extrae características y decide."""
    from src.preprocesamiento import filtrar_senal, segmentar_senal
    from src.caracteristicas import matriz_features
    ventanas = segmentar_senal(filtrar_senal(senal_cruda, fs), fs)
    puntajes = modelo.decision_function(matriz_features(ventanas, fs)[columnas])
    fraccion = float((puntajes >= 0).mean())
    return dict(decision="CRISIS" if fraccion >= umbral_fraccion else "NO CRISIS",
                fraccion_ventanas_crisis=round(fraccion, 2), n_ventanas=len(puntajes), puntajes=puntajes)