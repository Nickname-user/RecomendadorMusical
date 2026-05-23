"""
gui_commands.py
================

Lógica de la interfaz gráfica del recomendador musical.

Este módulo se encarga de:
- Resolver nombres de canciones introducidos por el usuario
- Usar TF-IDF para evitar errores semánticos
- Ejecutar recomendaciones en hilos separados (threading)
"""

import threading
import random
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender import recommend_from_existing_song


# =========================
# Carga del dataset
# =========================

DATASET_PATH = "data/songs_with_clusters.pkl"

df_clustered = joblib.load(DATASET_PATH)

# Nos aseguramos de que los títulos sean strings válidos
SONG_TITLES = (
    df_clustered["track_name"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# =========================
# TF-IDF: entrenamiento UNA SOLA VEZ
# =========================

"""
Aquí tratamos los títulos de canciones como documentos de texto.
TF-IDF asigna menos peso a palabras frecuentes ("quiero", "love")
y más peso a palabras discriminativas ("shuffle", "downtown").
"""


SPANISH_STOPWORDS = [
    "de", "la", "el", "los", "las", "un", "una",
    "y", "o", "en", "a", "con", "por", "para",
    "te", "me", "mi", "tu", "yo",
    "quiero", "estar", "ser", "es",
    "que", "donde", "cuando", "como"
]

ENGLISH_STOPWORDS = [
    "the", "of", "to", "and", "or", "in", "on",
    "a", "an", "with", "for",
    "i", "you", "me", "my", "your",
    "want", "be", "is", "are", "was",
    "where", "when", "how"
]

STOPWORDS = list(set(SPANISH_STOPWORDS + ENGLISH_STOPWORDS))


TFIDF_VECTORIZER = TfidfVectorizer(
    lowercase=True,
    stop_words=STOPWORDS  # stopwords en español
)

TFIDF_MATRIX = TFIDF_VECTORIZER.fit_transform(SONG_TITLES)


# =========================
# Utilidades
# =========================

def get_random_song() -> str:
    """
    Devuelve una canción aleatoria del dataset.
    """
    return random.choice(SONG_TITLES)


def find_best_song_match(user_input: str, threshold: float = 0.3) -> str:
    """
    Encuentra la canción más similar usando TF-IDF + cosine similarity.

    Parameters
    ----------
    user_input : str
        Título introducido por el usuario.
    threshold : float
        Umbral mínimo de similitud aceptable.

    Returns
    -------
    str
        Título de la canción más similar del dataset.

    Raises
    ------
    ValueError
        Si no se encuentra ninguna canción suficientemente similar.
    """
    # Vectorizamos la entrada del usuario
    query_vector = TFIDF_VECTORIZER.transform([user_input])

    # Calculamos similitud coseno
    similarities = cosine_similarity(query_vector, TFIDF_MATRIX)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < threshold:
        raise ValueError(
            "No se encontró ninguna canción suficientemente similar."
        )

    return SONG_TITLES[best_index]


# =========================
# Comando principal (con threading)
# =========================

def recommend_song_command(song_name: str, update_callback):
    """
    Ejecuta la recomendación en un hilo separado para evitar
    el bloqueo de la interfaz gráfica.

    Parameters
    ----------
    song_name : str
        Nombre de la canción introducida.
    update_callback : callable
        Función que actualiza la interfaz con los resultados.
    """

    def task():
        try:
            # Resolución del título con TF-IDF
            corrected_song = find_best_song_match(song_name)

            recommendations = recommend_from_existing_song(
                track_name=corrected_song,
                df_clustered=df_clustered,
                feature_order=[
                    "duration_ms", "danceability", "energy", "key",
                    "loudness", "mode", "speechiness", "acousticness",
                    "instrumentalness", "liveness", "valence",
                    "tempo", "time_signature"
                ],
                top_n=3
            )

            update_callback(
                corrected_song,
                recommendations[["track_name", "artists", "track_genre"]]
            )

        except Exception as e:
            update_callback(None, str(e))

    threading.Thread(target=task, daemon=True).start()