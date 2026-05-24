"""
recommender.py
===============

Módulo encargado del sistema de recomendación musical.

Este archivo utiliza los modelos previamente entrenados:
- StandardScaler (preprocesado)
- MLPClassifier (clasificación supervisada)
- Dataset de canciones con clusters asignados

Su objetivo es:
- Recibir una canción (o sus características numéricas)
- Predecir el cluster al que pertenece
- Recomendar canciones similares pertenecientes al mismo cluster

Este módulo NO entrena modelos.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List

from sklearn.metrics.pairwise import euclidean_distances


# =========================
# Carga de artefactos
# =========================

def load_artifacts(
    scaler_path: str,
    classifier_path: str,
    dataset_path: str
):
    """
    Carga los modelos y el dataset necesarios para la recomendación.

    Parameters
    ----------
    scaler_path : str
        Ruta al archivo .pkl del StandardScaler.
    classifier_path : str
        Ruta al archivo .pkl del MLPClassifier.
    dataset_path : str
        Ruta al dataset con metadata y clusters.

    Returns
    -------
    Tuple
        - scaler
        - classifier
        - df_clustered
    """
    scaler = joblib.load(scaler_path)
    classifier = joblib.load(classifier_path)
    df_clustered = joblib.load(dataset_path)

    return scaler, classifier, df_clustered


# =========================
# Predicción de cluster
# =========================

def predict_cluster(
    song_features: Dict[str, float],
    scaler,
    classifier,
    feature_order: List[str]
) -> int:
    """
    Predice el cluster al que pertenece una canción nueva.

    Parameters
    ----------
    song_features : Dict[str, float]
        Diccionario con las features numéricas de la canción.
    scaler : StandardScaler
        Escalador entrenado en la fase de preprocesado.
    classifier : MLPClassifier
        Clasificador entrenado para predecir clusters.
    feature_order : List[str]
        Lista con el orden correcto de las features.

    Returns
    -------
    int
        Cluster predicho.
    """
    # Construir array en el orden correcto
    X = np.array([[song_features[f] for f in feature_order]])

    # Escalar
    X_scaled = scaler.transform(X)

    # Predicción del cluster
    cluster = classifier.predict(X_scaled)[0]

    return cluster


# =========================
# Recomendación
# =========================

def recommend_from_cluster(
    song_features: Dict[str, float],
    cluster: int,
    df_clustered: pd.DataFrame,
    feature_order: List[str],
    top_n: int = 1
) -> pd.DataFrame:
    """
    Recomienda canciones del mismo cluster basándose en
    cercanía euclídea dentro del espacio de features.

    Parameters
    ----------
    song_features : Dict[str, float]
        Features numéricas de la canción de entrada.
    cluster : int
        Cluster al que pertenece la canción.
    df_clustered : pd.DataFrame
        Dataset con metadata y clusters.
    feature_order : List[str]
        Lista con el orden de las features.
    top_n : int, optional
        Número de canciones a recomendar.

    Returns
    -------
    pd.DataFrame
        DataFrame con las canciones recomendadas.
    """
    # Filtrar canciones del mismo cluster
    df_cluster = df_clustered[df_clustered["cluster"] == cluster].copy()

    if df_cluster.empty:
        raise ValueError("No hay canciones disponibles en el cluster predicho.")

    # Vector de la canción de entrada
    song_vector = np.array([[song_features[f] for f in feature_order]])

    # Matriz de features del cluster
    X_cluster = df_cluster[feature_order].values

    # Cálculo de distancias
    distances = euclidean_distances(song_vector, X_cluster)[0]
    df_cluster["distance"] = distances

    # Ordenar por cercanía
    df_recommended = df_cluster.sort_values("distance").head(top_n)

    return df_recommended


# =========================
# Pipeline completo
# =========================

def recommend_song(
    song_features: Dict[str, float],
    scaler_path: str,
    classifier_path: str,
    dataset_path: str,
    feature_order: List[str],
    top_n: int = 1
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de recomendación:
    - Carga modelos y dataset
    - Predice el cluster
    - Recomienda canciones similares

    Parameters
    ----------
    song_features : Dict[str, float]
        Features numéricas de la canción de entrada.
    scaler_path : str
        Ruta al StandardScaler.
    classifier_path : str
        Ruta al MLPClassifier.
    dataset_path : str
        Ruta al dataset con clusters.
    feature_order : List[str]
        Lista con el orden de las features.
    top_n : int, optional
        Número de canciones a recomendar.

    Returns
    -------
    pd.DataFrame
        Canciones recomendadas.
    """
    scaler, classifier, df_clustered = load_artifacts(
        scaler_path,
        classifier_path,
        dataset_path
    )

    cluster = predict_cluster(
        song_features,
        scaler,
        classifier,
        feature_order
    )

    recommendations = recommend_from_cluster(
        song_features,
        cluster,
        df_clustered,
        feature_order,
        top_n
    )

    return recommendations


def recommend_from_existing_song(
    track_name: str,
    df_clustered: pd.DataFrame,
    feature_order: list,
    top_n: int = 3
) -> pd.DataFrame:
    """
    Recomienda canciones similares a una canción existente del dataset.

    Parameters
    ----------
    track_name : str
        Nombre de la canción de referencia.
    df_clustered : pd.DataFrame
        Dataset con metadata, features y clusters.
    feature_order : list
        Orden de las features numéricas.
    top_n : int
        Número de recomendaciones.

    Returns
    -------
    pd.DataFrame
        Canciones recomendadas.
    """
    song_row = df_clustered[df_clustered["track_name"] == track_name]

    if song_row.empty:
        raise ValueError(f"La canción '{track_name}' no existe en el dataset.")

    song_features = song_row.iloc[0][feature_order].to_dict()
    cluster = song_row.iloc[0]["cluster"]

    return recommend_from_cluster(
        song_features=song_features,
        cluster=cluster,
        df_clustered=df_clustered,
        feature_order=feature_order,
        top_n=top_n
    )

# =========================
# Ejecución directa (opcional)
# =========================

if __name__ == "__main__":
    """
    Ejemplo de uso del sistema de recomendación.
    """

    FEATURE_ORDER = [
        "duration_ms",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature"
    ]

    example_song = {
        "duration_ms": 210000,
        "danceability": 0.8,
        "energy": 0.75,
        "key": 5,
        "loudness": -6.0,
        "mode": 1,
        "speechiness": 0.05,
        "acousticness": 0.1,
        "instrumentalness": 0.0,
        "liveness": 0.15,
        "valence": 0.9,
        "tempo": 120.0,
        "time_signature": 4
    }

    recommendations = recommend_song(
        song_features=example_song,
        scaler_path="models/scaler.pkl",
        classifier_path="models/mlp.pkl",
        dataset_path="data/songs_with_clusters.pkl",
        feature_order=FEATURE_ORDER,
        top_n=3
    )

    print("Canciones recomendadas:")
    print(recommendations[["track_name", "artists", "track_genre"]])
