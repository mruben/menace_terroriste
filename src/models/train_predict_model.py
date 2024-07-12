import numpy as np
import pandas as pd
import joblib

from src.features.build_features import (
    build_past_data,
    build_future_data,
    past_data_transformation,
    future_data_transformation
)
from src.features.utils import save_dataframe_to_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from ..config_loader import load_config
config = load_config()


def train_and_predict_model(past_data, future_data):
    """
    Entraîne un modèle de régression RandomForest avec une réduction
    de dimension LDA sur les données historiques (past_data) et fait
    des prédictions sur les données futures (future_data).

    Args:
        past_data (pd.DataFrame): Données historiques prétraitées.
        future_data (pd.DataFrame): Données futures prétraitées
                                    pour les prédictions.

    Returns:
        None
    """

    # On récupère nos jeux d'entrainement et de test
    (X_train_encoded,
        X_test_encoded,
        y_train,
        y_test,
        scaler) = past_data_transformation(past_data)

    # On enregistre le jeu de test de la variable cible dans un fichier csv
    save_dataframe_to_csv(
        pd.Series(y_test), config['paths']['y_test']
    )

    future_data_encoded = future_data_transformation(future_data, scaler)

    # On applique une réduction de dimension LDA sur les données
    # de test et d'entraînement
    lda = LDA()
    X_train_lda = lda.fit_transform(X_train_encoded, y_train)
    X_test_lda = lda.transform(X_test_encoded)
    future_data_encoded_lda = lda.transform(future_data_encoded)

    # On entraine un modèle RF avec la LDA
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_model.fit(X_train_lda, y_train)

    # Sauvegarde du modèle
    joblib.dump(rf_model, config['paths']['model'])

    # Prédictions du modèle sur les données historiques
    y_pred_past_lda = rf_model.predict(X_test_lda)

    # On enregistre les prédictions sur future_data dans un fichier csv
    save_dataframe_to_csv(
        pd.Series(y_pred_past_lda), config['paths']['predictions_past_data']
    )

    # Évaluation du modèle  sur les données historiques
    r2_lda = rf_model.score(X_test_lda, y_test)
    rmse_lda = np.sqrt(mean_squared_error(y_test, y_pred_past_lda))
    mae_lda = mean_absolute_error(y_test, y_pred_past_lda)
    pearson_corr_lda, _ = pearsonr(y_test, y_pred_past_lda)

    print("\nRandom Forest LDA R2:", r2_lda)
    print("Random Forest LDA RMSE:", rmse_lda)
    print("Random Forest LDA MAE:", mae_lda)
    print("Random Forest LDA Pearson Correlation:", pearson_corr_lda)

    # On utilise notre modèle pour faire des prédictions sur future_data
    y_pred_future_lda = rf_model.predict(future_data_encoded_lda)

    # On enregistre les prédictions sur future_data dans un fichier csv
    save_dataframe_to_csv(
        pd.Series(y_pred_future_lda),
        config['paths']['predictions_future_data']
    )


# Chargement des données historiques et création des features passées
past_data = build_past_data()

# Création des données futures basées sur les tendances passées
future_data = build_future_data(past_data)

# Entraînement d'un modèle Random Forest et prédictions sur les données futures
train_and_predict_model(past_data, future_data)
