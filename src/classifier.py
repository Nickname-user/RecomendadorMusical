"""
classifier.py
==============

Módulo encargado del aprendizaje supervisado mediante un
perceptrón multicapa (MLPClassifier).

El objetivo de este modelo es aprender a predecir el cluster
al que pertenece una canción a partir de sus características
numéricas, utilizando como etiquetas los clusters obtenidos
en la fase de clustering (GMM).

Esto permite clasificar nuevas canciones sin necesidad de
reentrenar el modelo de clustering.
"""

import joblib
import numpy as np
from typing import Tuple, Dict

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# Entrenamiento del modelo
# =========================

def train_mlp(
    X_scaled: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> MLPClassifier:
    """
    Entrena un clasificador MLP para predecir el cluster
    de una canción a partir de sus features numéricas.

    Parameters
    ----------
    X_scaled : np.ndarray
        Matriz de features numéricas escaladas.
    y : np.ndarray
        Vector de etiquetas (clusters asignados por el GMM).
    random_state : int, optional
        Semilla para reproducibilidad.

    Returns
    -------
    MLPClassifier
        Modelo MLP entrenado.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=5e-5,
        max_iter=800,
        early_stopping=False,
        random_state=random_state
    )

    mlp.fit(X_scaled, y)
    return mlp


# =========================
# Evaluación del modelo
# =========================

def evaluate_model(
    model: MLPClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, object]:
    """
    Evalúa el modelo MLP utilizando varias métricas
    de clasificación.

    Parameters
    ----------
    model : MLPClassifier
        Modelo entrenado.
    X_test : np.ndarray
        Datos de test.
    y_test : np.ndarray
        Etiquetas reales de test.

    Returns
    -------
    Dict[str, object]
        Diccionario con:
        - accuracy
        - matriz de confusión
        - classification report
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }

    return metrics


# =========================
# Pipeline completo
# =========================

def run_classification(
    X_scaled: np.ndarray,
    clusters: np.ndarray,
    model_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[MLPClassifier, Dict[str, object]]:
    """
    Ejecuta el pipeline completo de clasificación:
    - Divide los datos en entrenamiento y test.
    - Entrena el MLPClassifier.
    - Evalúa el modelo.
    - Guarda el modelo entrenado.

    Esta función está pensada para ser llamada desde main.py.

    Parameters
    ----------
    X_scaled : np.ndarray
        Matriz de features numéricas escaladas.
    clusters : np.ndarray
        Etiquetas de cluster obtenidas del GMM.
    model_path : str
        Ruta donde se guardará el modelo MLP entrenado (.pkl).
    test_size : float, optional
        Proporción del dataset destinada a test.
    random_state : int, optional
        Semilla para reproducibilidad.

    Returns
    -------
    Tuple[MLPClassifier, Dict[str, object]]
        - Modelo MLP entrenado.
        - Diccionario con métricas de evaluación.
    """
    # División train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        clusters,
        test_size=test_size,
        random_state=random_state,
        stratify=clusters
    )

    # Entrenamiento
    mlp = train_mlp(X_train, y_train, random_state)

    # Evaluación
    metrics = evaluate_model(mlp, X_test, y_test)

    # Guardado del modelo
    joblib.dump(mlp, model_path)

    return mlp, metrics


# =========================
# Ejecución directa (opcional)
# =========================

if __name__ == "__main__":
    """
    Ejecución de prueba del clasificador.
    Permite entrenar y evaluar el MLP si ya se dispone
    de las features escaladas y de los clusters.
    """

    FEATURES_PATH = "data/X_scaled.pkl"
    CLUSTERS_PATH = "data/clusters.pkl"
    MODEL_PATH = "models/mlp.pkl"

    # Carga de datos
    X_scaled = joblib.load(FEATURES_PATH)
    clusters = joblib.load(CLUSTERS_PATH)

    model, metrics = run_classification(
        X_scaled=X_scaled,
        clusters=clusters,
        model_path=MODEL_PATH
    )

    print("Clasificación completada con éxito.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification report:")
    print(metrics["classification_report"])
