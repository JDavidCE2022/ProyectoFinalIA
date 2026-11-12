"""Modelos de la parte 1 implementados desde cero: regresión Ridge y Naive Bayes categórico."""
import json
import numpy as np
import pandas as pd
from datos import CLASES, PROC, construir_dataset

VARIABLES = ["clase_hoy", "tendencia", "lluvia", "viento", "temporada", "dia_manana"]


class Discretizador:
    """Convierte las variables continuas en categorías. Los cortes de viento se aprenden solo con entrenamiento."""

    def ajustar(self, df):
        self.cortes_viento = list(np.quantile(df["viento_media_kmh"], [1 / 3, 2 / 3]))
        return self

    def transformar(self, df):
        out = pd.DataFrame(index=df.index)
        out["clase_hoy"] = df["clase_hoy"]
        delta = df["pm25_media"] - df["pm25_ayer"]
        out["tendencia"] = np.where(delta > 3, "sube", np.where(delta < -3, "baja", "estable"))
        out["lluvia"] = pd.cut(df["lluvia_mm"], [-1, 1, 10, 1e9], labels=["seco", "moderada", "fuerte"]).astype(str)
        v1, v2 = self.cortes_viento
        out["viento"] = np.where(df["viento_media_kmh"] <= v1, "bajo", np.where(df["viento_media_kmh"] <= v2, "medio", "alto"))
        out["temporada"] = np.where(df["fecha_manana"].dt.month.isin([2, 3, 10, 11]), "critica", "normal")
        out["dia_manana"] = np.where(df["fecha_manana"].dt.dayofweek >= 5, "fin_semana", "laboral")
        return out


class NaiveBayesCategorico:
    """P(c | x) proporcional a P(c) * prod_j P(x_j | c), con suavizado de Laplace alpha."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def ajustar(self, X, y):
        self.clases = list(CLASES)
        self.valores = {v: sorted(X[v].unique()) for v in X.columns}
        n = len(y)
        self.log_prior = {c: np.log(((y == c).sum() + self.alpha) / (n + self.alpha * len(self.clases))) for c in self.clases}
        self.log_lik = {}
        for c in self.clases:
            Xc = X[y == c]
            for v in X.columns:
                k = len(self.valores[v])
                conteos = Xc[v].value_counts()
                for val in self.valores[v]:
                    self.log_lik[(c, v, val)] = np.log((conteos.get(val, 0) + self.alpha) / (len(Xc) + self.alpha * k))
        return self

    def _log_post(self, fila):
        s = {}
        for c in self.clases:
            total = self.log_prior[c]
            for v, val in fila.items():
                # Un valor nunca visto en entrenamiento no aporta evidencia
                total += self.log_lik.get((c, v, val), 0.0)
            s[c] = total
        return s

    def predecir_proba(self, X):
        filas = []
        for _, fila in X.iterrows():
            s = self._log_post(fila)
            m = max(s.values())
            e = {c: np.exp(v - m) for c, v in s.items()}
            z = sum(e.values())
            filas.append([e[c] / z for c in self.clases])
        return pd.DataFrame(filas, columns=self.clases, index=X.index)

    def predecir(self, X):
        return self.predecir_proba(X).idxmax(axis=1)

    def a_dict(self):
        return {"alpha": self.alpha, "clases": self.clases, "valores": self.valores,
                "log_prior": self.log_prior, "log_lik": {"|".join(k): v for k, v in self.log_lik.items()}}

    @classmethod
    def desde_dict(cls, d):
        m = cls(d["alpha"])
        m.clases, m.valores, m.log_prior = d["clases"], d["valores"], d["log_prior"]
        m.log_lik = {tuple(k.split("|")): v for k, v in d["log_lik"].items()}
        return m


class RegresionRidge:
    """Regresión lineal con penalización L2 resuelta por ecuaciones normales. Se ajusta sobre log(1 + PM2.5)."""

    def __init__(self, lam=1.0):
        self.lam = lam

    def ajustar(self, X, y):
        X = np.asarray(X, dtype=float)
        self.media, self.desv = X.mean(axis=0), X.std(axis=0)
        self.desv[self.desv == 0] = 1.0
        Z = (X - self.media) / self.desv
        A = np.c_[np.ones(len(Z)), Z]
        P = self.lam * np.eye(A.shape[1])
        P[0, 0] = 0.0  # el intercepto no se penaliza
        self.w = np.linalg.solve(A.T @ A + P, A.T @ np.log1p(np.asarray(y, dtype=float)))
        return self

    def predecir(self, X):
        Z = (np.asarray(X, dtype=float) - self.media) / self.desv
        return np.expm1(np.c_[np.ones(len(Z)), Z] @ self.w)


def a_clase(v, t1=12.0, t2=37.0):
    v = np.asarray(v)
    return np.where(v <= t1, "Buena", np.where(v <= t2, "Aceptable", "Dañina"))


def variables_regresion(ds, seriales):
    X = pd.DataFrame(index=ds.index)
    X["pm25_hoy"] = ds["pm25_media"]
    X["pm25_ayer"] = ds["pm25_ayer"]
    X["ciudad_hoy"] = ds["ciudad_hoy"]
    X["lluvia_mm"] = ds["lluvia_mm"]
    X["viento_kmh"] = ds["viento_media_kmh"]
    X["radiacion_mj"] = ds["radiacion_mj"]
    X["temporada_critica"] = ds["fecha_manana"].dt.month.isin([2, 3, 10, 11]).astype(int)
    X["manana_fin_semana"] = (ds["fecha_manana"].dt.dayofweek >= 5).astype(int)
    X["hoy_fin_semana"] = (ds["fecha"].dt.dayofweek >= 5).astype(int)
    X["manana_lunes"] = (ds["fecha_manana"].dt.dayofweek == 0).astype(int)
    for s in seriales:
        X[f"est_{s}"] = (ds["serial"] == s).astype(int)
    return X


def ajustar_umbrales(X, y_valor, y_clase, lam=1.0, k=5):
    """Busca los cortes de clase que maximizan el F1 macro con predicciones fuera de muestra (k bloques)."""
    n = len(X)
    pred = np.zeros(n)
    bloques = np.array_split(np.arange(n), k)
    for b in bloques:
        tr = np.setdiff1d(np.arange(n), b)
        pred[b] = RegresionRidge(lam).ajustar(X.iloc[tr], y_valor.iloc[tr]).predecir(X.iloc[b])
    mejor = (-1, 12.0, 37.0)
    for t1 in np.arange(10, 16.01, 0.5):
        for t2 in np.arange(30, 38.01, 0.5):
            f = metricas(y_clase, a_clase(pred, t1, t2))["f1_macro"]
            if f > mejor[0]:
                mejor = (f, float(t1), float(t2))
    return mejor[1], mejor[2]


def metricas(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    res = {"accuracy": float(np.mean(y_true == y_pred))}
    f1s = []
    for c in CLASES:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        res[f"recall_{c}"] = float(r)
        res[f"precision_{c}"] = float(p)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    res["f1_macro"] = float(np.mean(f1s))
    return res


def preparar(ds):
    ds = ds.copy()
    ds["ciudad_hoy"] = ds.groupby("fecha")["pm25_media"].transform("mean")
    return ds


def validacion_temporal(ds, meses_prueba, lam=1.0):
    """Entrena con todo lo anterior a cada mes y prueba en ese mes (ventana creciente)."""
    ds = preparar(ds)
    seriales = sorted(ds["serial"].unique())
    preds = []
    for mes in meses_prueba:
        ini = pd.Timestamp(mes)
        fin = ini + pd.offsets.MonthBegin(1)
        tr = ds[ds["fecha_manana"] < ini]
        te = ds[(ds["fecha_manana"] >= ini) & (ds["fecha_manana"] < fin)]
        Xtr, Xte = variables_regresion(tr, seriales), variables_regresion(te, seriales)
        t1, t2 = ajustar_umbrales(Xtr, tr["pm25_manana"], tr["clase_manana"], lam)
        reg = RegresionRidge(lam).ajustar(Xtr, tr["pm25_manana"])
        disc = Discretizador().ajustar(tr)
        nb = NaiveBayesCategorico().ajustar(disc.transformar(tr), tr["clase_manana"])
        out = te[["fecha", "fecha_manana", "serial", "pm25_media", "pm25_manana", "clase_hoy", "clase_manana"]].copy()
        out["mes"] = mes
        out["valor_ridge"] = reg.predecir(Xte)
        out["pred_ridge"] = a_clase(out["valor_ridge"], t1, t2)
        out["pred_ridge_cortes_ica"] = a_clase(out["valor_ridge"])
        out["pred_nb"] = nb.predecir(disc.transformar(te)).values
        out["pred_persistencia"] = te["clase_hoy"].values
        out["pred_mayoritaria"] = tr["clase_manana"].mode()[0]
        out["umbral_t1"], out["umbral_t2"] = t1, t2
        preds.append(out)
    return pd.concat(preds)


def tabla_metricas(p):
    filas = {}
    nombres = {"ridge": "Ridge + umbrales ajustados", "ridge_cortes_ica": "Ridge con cortes ICA",
               "nb": "Naive Bayes", "persistencia": "Persistencia", "mayoritaria": "Clase mayoritaria"}
    for k, nombre in nombres.items():
        m = metricas(p["clase_manana"], p[f"pred_{k}"])
        if k.startswith("ridge"):
            m["MAE"] = float(np.mean(np.abs(p["valor_ridge"] - p["pm25_manana"])))
        elif k == "persistencia":
            m["MAE"] = float(np.mean(np.abs(p["pm25_media"] - p["pm25_manana"])))
        else:
            m["MAE"] = np.nan
        filas[nombre] = m
    cols = ["MAE", "accuracy", "f1_macro", "recall_Dañina", "precision_Dañina", "recall_Buena"]
    return pd.DataFrame(filas).T[cols]


def entrenar_final(ds, lam=1.0):
    ds = preparar(ds)
    seriales = sorted(int(s) for s in ds["serial"].unique())
    X = variables_regresion(ds, seriales)
    t1, t2 = ajustar_umbrales(X, ds["pm25_manana"], ds["clase_manana"], lam)
    reg = RegresionRidge(lam).ajustar(X, ds["pm25_manana"])
    modelo = {"lam": lam, "seriales": seriales, "columnas": list(X.columns), "media": reg.media.tolist(),
              "desv": reg.desv.tolist(), "w": reg.w.tolist(), "t1": t1, "t2": t2}
    (PROC / "modelo_ridge.json").write_text(json.dumps(modelo, indent=1), encoding="utf-8")
    return modelo


def cargar_modelo():
    m = json.loads((PROC / "modelo_ridge.json").read_text(encoding="utf-8"))
    reg = RegresionRidge(m["lam"])
    reg.media, reg.desv, reg.w = np.array(m["media"]), np.array(m["desv"]), np.array(m["w"])
    return reg, m


MESES_PRUEBA = ["2023-11-01", "2023-12-01", "2024-01-01", "2024-02-01", "2024-03-01",
                "2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01"]

if __name__ == "__main__":
    ds = construir_dataset()
    p = validacion_temporal(ds, MESES_PRUEBA)
    tabla = tabla_metricas(p)
    print(tabla.round(3).to_string())
    p.to_csv(PROC / "predicciones_validacion.csv", index=False)
    tabla.round(4).to_csv(PROC / "metricas_modelo.csv")
    print(entrenar_final(ds)["t1"])
