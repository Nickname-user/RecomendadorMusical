"""
clustering.py
==============

Módulo encargado del aprendizaje no supervisado mediante
Gaussian Mixture Models (GMM).

Responsabilidades principales:
- Entrenar un modelo de clustering basado en GMM.
- Asignar un cluster a cada canción.
- Guardar el modelo entrenado para su reutilización.
- Construir un dataset enriquecido con la información de clusters,
  que será utilizado posteriormente por el clasificador y el recomendador.

Este módulo NO realiza preprocesamiento ni clasificación supervisada.
"""

import pandas as pd
import joblib
from sklearn.mixture import GaussianMixture
from typing import Tuple


# =========================
# Funciones de clustering
# =========================

def train_gmm(
    X_scaled,
    n_components: int,
    random_state: int = 42
) -> GaussianMixture:
    """
    Entrena un modelo Gaussian Mixture Model (GMM) sobre las
    features numéricas previamente escaladas.

    Parameters
    ----------
    X_scaled : numpy.ndarray
        Matriz de features numéricas escaladas.
    n_components : int
        Número de componentes (clusters) del modelo.
    random_state : int, optional
        Semilla para garantizar reproducibilidad.

    Returns
    -------
    GaussianMixture
        Modelo GMM entrenado.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=300,
        random_state=random_state
    )
    gmm.fit(X_scaled)
    return gmm


def assign_clusters(
    gmm: GaussianMixture,
    X_scaled
):
    """
    Asigna a cada canción el cluster más probable según el modelo GMM.

    Parameters
    ----------
    gmm : GaussianMixture
        Modelo GMM entrenado.
    X_scaled : numpy.ndarray
        Matriz de features numéricas escaladas.

    Returns
    -------
    numpy.ndarray
        Array con el cluster asignado a cada canción.
    """
    return gmm.predict(X_scaled)


def compute_cluster_probabilities(
    gmm: GaussianMixture,
    X_scaled
):
    """
    Calcula las probabilidades de pertenencia de cada canción
    a cada cluster.

    Estas probabilidades pueden ser útiles para análisis posterior
    o para sistemas de recomendación más avanzados.

    Parameters
    ----------
    gmm : GaussianMixture
        Modelo GMM entrenado.
    X_scaled : numpy.ndarray
        Matriz de features numéricas escaladas.

    Returns
    -------
    numpy.ndarray
        Matriz de probabilidades (n_samples x n_clusters).
    """
    return gmm.predict_proba(X_scaled)


# =========================
# Construcción del dataset
# =========================

def build_clustered_dataset(
    df_metadata: pd.DataFrame,
    clusters
) -> pd.DataFrame:
    """
    Construye un DataFrame que combina la metadata de las canciones
    con el cluster asignado a cada una.

    La correspondencia se mantiene mediante el índice del DataFrame.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        DataFrame con la información descriptiva de las canciones.
    clusters : numpy.ndarray
        Cluster asignado a cada canción.

    Returns
    -------
    pd.DataFrame
        DataFrame enriquecido con la columna 'cluster'.
    """
    df_clustered = df_metadata.copy()
    df_clustered["cluster"] = clusters
    return df_clustered


# =========================
# Función de alto nivel
# =========================
def run_clustering(
    df_metadata: pd.DataFrame,
    X: pd.DataFrame,              # 👈 NUEVO
    X_scaled,
    n_components: int,
    model_path: str,
    dataset_path: str
):
    gmm = train_gmm(X_scaled, n_components)
    clusters = assign_clusters(gmm, X_scaled)

    df_clustered = df_metadata.copy()

    # Añadir features numéricas
    for col in X.columns:
        df_clustered[col] = X[col].values

    # Añadir cluster
    df_clustered["cluster"] = clusters

    joblib.dump(gmm, model_path)
    joblib.dump(df_clustered, dataset_path)

    return df_clustered, gmm
# =========================
# Ejecución directa (opcional)
# =========================

if __name__ == "__main__":
    """
    Ejecución de prueba del clustering.
    Este bloque permite ejecutar el archivo directamente
    si se dispone ya de los datos preprocesados.
    """

    # Ejemplo de uso (asumiendo que preprocess.py ya se ha ejecutado)
    METADATA_PATH = "data/metadata.pkl"
    FEATURES_PATH = "data/X_scaled.pkl"

    MODEL_PATH = "models/gmm.pkl"
    DATASET_PATH = "data/songs_with_clusters.pkl"

    N_COMPONENTS = 10

    # Carga de datos preprocesados
    df_metadata = joblib.load(METADATA_PATH)
    X_scaled = joblib.load(FEATURES_PATH)

    df_clustered, gmm = run_clustering(
        df_metadata=df_metadata,
        X_scaled=X_scaled,
        n_components=N_COMPONENTS,
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH
    )

    print("Clustering completado con éxito.")
    print(f"Número de canciones: {len(df_clustered)}")
    print(f"Número de clusters: {N_COMPONENTS}")
