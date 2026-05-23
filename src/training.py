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
