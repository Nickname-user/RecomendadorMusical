"""
main.py
========

Punto de entrada principal del proyecto de recomendación musical.

Este archivo orquesta todo el pipeline del sistema:
1. Preprocesado de los datos
2. Clustering no supervisado (GMM)
3. Clasificación supervisada (MLP)
4. Sistema de recomendación

El objetivo es separar claramente las fases de entrenamiento
y la fase de uso del recomendador.
"""

import joblib

from src.preprocess import preprocess_dataset
from src.clustering import run_clustering
from src.classifier import run_classification
from src.recommender import recommend_song, recommend_from_existing_song
from src.training import train_pipeline
from src.gui.gui_layout import RecommenderGUI


# =========================
# Configuración de rutas
# =========================

DATASET_PATH = "data/spotify.csv"

SCALER_PATH = "models/scaler.pkl"
GMM_MODEL_PATH = "models/gmm.pkl"
MLP_MODEL_PATH = "models/mlp.pkl"

CLUSTERED_DATASET_PATH = "data/songs_with_clusters.pkl"

# Número de clusters del GMM
N_COMPONENTS = 80

# Orden de las features numéricas (DEBE ser consistente en todo el proyecto)
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


# =========================
# Fase de entrenamiento
# =========================

def train_pipeline():
    """
    Ejecuta todo el pipeline de entrenamiento del sistema:
    - Preprocesado
    - Clustering (GMM)
    - Clasificación (MLP)

    Esta función debe ejecutarse UNA SOLA VEZ.
    """
    print("Iniciando preprocesado de datos...")
    df_metadata, X, X_scaled = preprocess_dataset(
        csv_path=DATASET_PATH,
        scaler_path=SCALER_PATH
    )

    print("Entrenando modelo de clustering (GMM)...")
    df_clustered, gmm = run_clustering(
        df_metadata=df_metadata,
        X=X,
        X_scaled=X_scaled,
        n_components=N_COMPONENTS,
        model_path=GMM_MODEL_PATH,
        dataset_path=CLUSTERED_DATASET_PATH
    )

    print("Entrenando clasificador supervisado (MLP)...")
    clusters = df_clustered["cluster"].values

    mlp, metrics = run_classification(
        X_scaled=X_scaled,
        clusters=clusters,
        model_path=MLP_MODEL_PATH
    )

    print("Entrenamiento completado.")
    print(f"Accuracy del clasificador: {metrics['accuracy']:.4f}")


# =========================
# Fase de recomendación
# =========================

def run_recommender():
    """
    Ejecuta un ejemplo de recomendación utilizando
    los modelos previamente entrenados.
    """
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

    print("Generando recomendaciones...")
    recommendations = recommend_song(
        song_features=example_song,
        scaler_path=SCALER_PATH,
        classifier_path=MLP_MODEL_PATH,
        dataset_path=CLUSTERED_DATASET_PATH,
        feature_order=FEATURE_ORDER,
        top_n=3
    )

    print("\nCanciones recomendadas:")
    print(recommendations[["track_name", "artists", "track_genre"]])


# =========================
# Ejecución principal
# =========================

if __name__ == "__main__":
    """
    Punto de entrada del programa.

    Cambiar TRAIN_MODEL a False si los modelos
    ya han sido entrenados previamente.
    """

    TRAIN_MODEL = False  # Cambiar a False para solo recomendar

    if TRAIN_MODEL:
        train_pipeline(DATASET_PATH, SCALER_PATH, GMM_MODEL_PATH, MLP_MODEL_PATH,CLUSTERED_DATASET_PATH,N_COMPONENTS)
    else:
        app = RecommenderGUI()
        app.mainloop()
