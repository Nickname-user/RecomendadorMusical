"""
gui_commands.py
================

Contiene la lógica asociada a los comandos de la interfaz gráfica.
Aquí NO se dibuja ninguna ventana: solo se ejecutan acciones.

Responsabilidades:
- Cargar modelos y dataset una sola vez
- Ejecutar recomendaciones sin bloquear la interfaz
- Gestionar errores y búsquedas aproximadas
"""

import threading
import random
import difflib
import joblib
import pandas as pd
import re
from src.recommender import recommend_from_existing_song


# =========================
# Carga de recursos (UNA VEZ)
# =========================

DATASET_PATH = "data/songs_with_clusters.pkl"

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

df_clustered = joblib.load(DATASET_PATH)
AVAILABLE_SONGS = (
    df_clustered["track_name"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# =========================
# Funciones auxiliares
# =========================


def normalize_title(title: str) -> list[str]:
    """
    Convierte un título en una lista de tokens normalizados.
    """
    title = title.lower()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    tokens = title.split()
    return tokens


def jaccard_similarity(tokens_a, tokens_b) -> float:
    '''
    Devuelve la cercanía de Jaccard
    '''
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def find_best_song_match(user_input: str, available_songs: list[str]) -> str:
    user_tokens = normalize_title(user_input)

    best_score = 0.0
    best_match = None

    for song in available_songs:
        song_tokens = normalize_title(song)
        score = jaccard_similarity(user_tokens, song_tokens)

        if score > best_score:
            best_score = score
            best_match = song

    if best_score < 0.5:
        raise ValueError("No se encontró ninguna canción suficientemente similar.")

    return best_match


def find_closest_song(song_name: str) -> str:
    """
    Busca el nombre de canción más parecido en el dataset
    utilizando coincidencia aproximada.

    Parameters
    ----------
    song_name : str
        Nombre introducido por el usuario.

    Returns
    -------
    str
        Nombre de la canción más similar encontrada.
    """
    matches = difflib.get_close_matches(
        song_name,
        AVAILABLE_SONGS,
        n=1,
        cutoff=0.6
    )

    if not matches:
        raise ValueError("No se encontró ninguna canción similar.")

    return matches[0]


def get_random_song() -> str:
    """
    Devuelve una canción aleatoria del dataset.
    """
    return random.choice(AVAILABLE_SONGS)


# =========================
# Comandos principales (threading)
# =========================

def recommend_song_command(
    song_name: str,
    update_callback
):
    """
    Ejecuta la recomendación en un hilo separado para evitar
    el bloqueo de la interfaz gráfica.

    Parameters
    ----------
    song_name : str
        Nombre de la canción introducida.
    update_callback : callable
        Función que actualiza la interfaz con el resultado.
    """

    def task():
        try:
            # Buscar coincidencia exacta o aproximada
            if song_name not in AVAILABLE_SONGS:
                try:
                    corrected_song = find_best_song_match(song_name,AVAILABLE_SONGS)
                except:
                    corrected_song = find_closest_song(song_name)
            else:
                corrected_song = song_name

            recommendations = recommend_from_existing_song(
                track_name=corrected_song,
                df_clustered=df_clustered,
                feature_order=FEATURE_ORDER,
                top_n=3
            )

            update_callback(
                corrected_song,
                recommendations[["track_name", "artists", "track_genre"]]
            )

        except Exception as e:
            update_callback(None, str(e))

    threading.Thread(target=task, daemon=True).start()
