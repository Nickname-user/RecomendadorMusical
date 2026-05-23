"""
preprocess.py
==============

Módulo encargado del preprocesamiento de los datos musicales obtenidos
desde Spotify. Este archivo se ocupa exclusivamente de preparar los datos
para su posterior uso en algoritmos de Machine Learning.

Responsabilidades principales:
- Cargar el dataset desde un archivo CSV.
- Limpiar los datos (duplicados y valores nulos).
- Separar las variables numéricas (features) de la información descriptiva
  (metadata).
- Escalar las variables numéricas.
- Guardar el escalador entrenado para garantizar consistencia futura.

Este módulo NO entrena modelos de clustering ni de clasificación.
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Tuple


# =========================
# Definición de columnas
# =========================

# Variables numéricas que se utilizarán para el entrenamiento de los modelos
NUMERIC_FEATURES = [
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

# Variables descriptivas que NO entran al modelo,
# pero se conservan para identificar y recuperar canciones
METADATA_COLUMNS = [
    "track_id",
    "track_name",
    "artists",
    "album_name",
    "track_genre",
    "explicit",
    "popularity"
]


# =========================
# Funciones principales
# =========================

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.

    Parameters
    ----------
    csv_path : str
        Ruta al archivo CSV con los datos de Spotify.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos cargados.
    """
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza una limpieza básica del dataset:
    - Elimina canciones duplicadas según el track_id.
    - Elimina filas con valores nulos en las variables numéricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio.
    """
    df = df.drop_duplicates(subset="track_id")
    df = df.dropna(subset=NUMERIC_FEATURES)
    return df


def split_metadata_and_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el DataFrame en:
    - Metadata: información descriptiva de las canciones.
    - Features numéricas: variables usadas por los modelos.

    La correspondencia entre ambos se mantiene mediante el índice.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame limpio.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - DataFrame de metadata.
        - DataFrame de features numéricas.
    """
    df_metadata = df[METADATA_COLUMNS].copy()
    X = df[NUMERIC_FEATURES].copy()
    return df_metadata, X


def scale_features(
    X: pd.DataFrame,
    scaler_path: str
):
    """
    Escala las variables numéricas utilizando StandardScaler y
    guarda el escalador entrenado en disco.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame con las variables numéricas.
    scaler_path : str
        Ruta donde se guardará el escalador (.pkl).

    Returns
    -------
    X_scaled : numpy.ndarray
        Matriz de features escaladas.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Guardar el escalador para reutilizarlo en inferencia
    joblib.dump(scaler, scaler_path)

    return X_scaled


# =========================
# Función de alto nivel
# =========================

def preprocess_dataset(
    csv_path: str,
    scaler_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta todo el pipeline de preprocesamiento:
    - Carga datos
    - Limpia el dataset
    - Separa metadata y features
    - Escala las features numéricas

    Esta función está pensada para ser llamada desde main.py.

    Parameters
    ----------
    csv_path : str
        Ruta al archivo CSV con los datos originales.
    scaler_path : str
        Ruta donde se guardará el escalador entrenado.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - DataFrame de metadata (sin escalar).
        - Matriz de features numéricas escaladas.
    """
    df = load_data(csv_path)
    df = clean_data(df)
    df_metadata, X = split_metadata_and_features(df)
    X_scaled = scale_features(X, scaler_path)

    return df_metadata, X_scaled


# =========================
# Ejecución directa (opcional)
# =========================

if __name__ == "__main__":
    """
    Ejecución de prueba del preprocesado.
    Este bloque permite ejecutar el archivo directamente
    para verificar que el preprocesamiento funciona correctamente.
    """

    CSV_PATH = "data/spotify.csv"
    SCALER_PATH = "models/scaler.pkl"

    metadata, X_scaled = preprocess_dataset(CSV_PATH, SCALER_PATH)

    print("Preprocesamiento completado con éxito.")
    print(f"Número de canciones procesadas: {len(metadata)}")
    print(f"Número de features numéricas: {X_scaled.shape[1]}")
