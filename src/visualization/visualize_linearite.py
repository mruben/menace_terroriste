import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config_loader import load_config
config = load_config()


# Importer la fonction encode_and_transform depuis le fichier converti
def encode_and_transform(df):
    df_encoded = df.copy()
    # On standardise les variables numériques exceptées les variables 
        # catégorielles
    # ainsi que l'année et le mois qui subiront une autre transformation
    cat_columns = ['country', 'admin1']
    df_encoded_numerical_vars = [col for col in df_encoded.columns if col not 
                                 in cat_columns + ['year', 'month']]
    # Extraction des données numériques
    df_encoded_numerical = df_encoded[df_encoded_numerical_vars].values

    # Initialisation d'un scaler
    scaler = StandardScaler()
    df_encoded[df_encoded_numerical_vars] = scaler.fit_transform(
        df_encoded_numerical
        )

    # Hot One Encoding des variables catégorielles
    df_encoded = pd.get_dummies(df_encoded, columns=cat_columns, dtype=int)

    # Transformation trigonométrique de 'year' et 'month' pour la saisonnalité
    df_encoded['month_sin'] = np.sin(2 * np.pi * df_encoded['month'] / 12)
    df_encoded['month_cos'] = np.cos(2 * np.pi * df_encoded['month'] / 12)
    df_encoded['year_sin'] = np.sin(2 * np.pi * (
        df_encoded['year'] - df_encoded['year'].min()) /
                                    (df_encoded['year'].max() - df_encoded['year'].min() + 1))
    df_encoded['year_cos'] = np.cos(2 * np.pi * (
        df_encoded['year'] - df_encoded['year'].min()) /
                                    (df_encoded['year'].max() - df_encoded['year'].min() + 1))

    # Suppression des colonnes 'month' et 'year' après transformation
    df_encoded = df_encoded.drop(columns=['month', 'year'])

    return df_encoded


def linearite():
    df_past_data = pd.read_csv(config['paths']['past_data'])
    # Calculer la matrice de corrélation et afficher la heatmap
    corr_matrix = encode_and_transform(df_past_data).corr()
    fig, ax = plt.subplots(figsize=(40, 40))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
    ax.set_title('Heatmap des Corrélations')
    return fig
