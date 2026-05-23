# 🎵 Recomendador Musical basado en Machine Learning

Este proyecto implementa un **sistema de recomendación musical** a partir de datos acústicos de canciones de Spotify.  
El sistema agrupa canciones por similitud sonora y recomienda canciones similares a partir de una canción introducida por el usuario, combinando técnicas de **aprendizaje no supervisado, supervisado y procesamiento de texto**.

***

## 📌 Características principales

* ✅ Clustering no supervisado con **Gaussian Mixture Models (GMM)**
* ✅ Clasificación supervisada con **MLPClassifier**
* ✅ Recomendación basada en **similitud acústica**
* ✅ Búsqueda robusta de canciones mediante **TF‑IDF + similitud coseno**
* ✅ Soporte multilingüe (español e inglés)
* ✅ Interfaz gráfica con **Tkinter**
* ✅ Ejecución no bloqueante mediante **threading**
* ✅ Arquitectura modular y escalable

***

## 🧠 Enfoque del sistema

El sistema sigue el siguiente flujo:

1. **Preprocesado de datos**
   * Limpieza y selección de variables numéricas
   * Escalado con `StandardScaler`

2. **Clustering (aprendizaje no supervisado)**
   * Agrupación de canciones según sus características acústicas usando GMM

3. **Clasificación (aprendizaje supervisado)**
   * Entrenamiento de un MLP para predecir el cluster de nuevas canciones

4. **Recomendación**
   * Dada una canción, se buscan canciones del mismo cluster
   * Se recomiendan las más cercanas en el espacio de características

5. **Interfaz gráfica**
   * El usuario introduce una canción o selecciona una aleatoria
   * El sistema muestra recomendaciones similares

***

## 📁 Estructura del proyecto

```text
data/
  ├── spotify.csv                 # Dataset original
  └── songs_with_clusters.pkl     # Dataset procesado con clusters

models/
  ├── scaler.pkl                  # Escalador
  ├── gmm.pkl                     # Modelo de clustering
  └── mlp.pkl                     # Clasificador

src/
  ├── preprocess.py               # Preprocesado de datos
  ├── clustering.py               # Clustering con GMM
  ├── classifier.py               # Clasificación con MLP
  ├── recommender.py              # Sistema de recomendación
  ├── gui/
  │   ├── gui_layout.py           # Interfaz gráfica (Tkinter)
  │   └── gui_commands.py         # Lógica de la GUI + TF-IDF
  └── main.py                     # Punto de entrada del sistema
```

***

## ▶️ Cómo ejecutar el proyecto

### 1️⃣ Entrenamiento del sistema (solo la primera vez)

En `main.py`:

```python
TRAIN_MODEL = True
```

Ejecutar desde la raíz del proyecto:

```bash
python main.py
```

Esto:

* Preprocesa el dataset
* Entrena el clustering (GMM)
* Entrena el clasificador (MLP)
* Guarda los modelos y el dataset procesado

***

### 2️⃣ Uso normal (interfaz gráfica)

Una vez entrenado, cambiar en `main.py`:

```python
TRAIN_MODEL = False
```

Y ejecutar de nuevo:

```bash
python main.py
```

Se abrirá una **ventana gráfica** para usar el recomendador.

***

## 🖥️ Uso de la interfaz gráfica

La interfaz permite dos acciones:

### ✅ Introducir una canción manualmente

* Escribir el nombre de una canción (no es necesario que sea exacto).
* Pulsar **“Recomendar”**.
* El sistema busca la canción más cercana y devuelve recomendaciones.

### ✅ Canción aleatoria

* Pulsar **“Canción aleatoria”**.
* El sistema selecciona una canción del dataset y genera recomendaciones.

***

## 🔎 Búsqueda de canciones (TF‑IDF)

Para evitar errores semánticos al introducir títulos:

* Se utiliza **TF‑IDF + similitud coseno** sobre los nombres de las canciones.
* Se eliminan **stopwords en español e inglés** (por ejemplo: *de, quiero, the, want*).
* Esto evita emparejamientos incorrectos basados en palabras poco informativas.

Si no se encuentra ninguna canción suficientemente similar, el sistema muestra un mensaje de error controlado.

***

## 📊 Interpretación de las recomendaciones

* Las canciones recomendadas pertenecen al **mismo cluster acústico**
* La similitud se basa en **características musicales**, no en el género explícito
* Canciones de géneros distintos pueden aparecer juntas si su sonido es similar

Este comportamiento es esperado y forma parte del enfoque del sistema.

***

## 🚀 Posibles mejoras futuras

* Mostrar varias sugerencias de canciones similares al introducir un nombre
* Mostrar el score de similitud TF‑IDF al usuario
* Autocompletado de títulos en la interfaz
* Integración con la API de Spotify
* Uso de embeddings de texto para títulos

***

## 📄 Licencia

Uso académico y educativo.
