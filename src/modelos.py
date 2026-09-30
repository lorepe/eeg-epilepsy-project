
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