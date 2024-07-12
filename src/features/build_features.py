import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.features.data_preprocessing import preprocessing_dataframe
from src.features.utils import save_dataframe_to_csv
from ..config_loader import load_config
config = load_config()


# CONSTRUCTION DU DATAFRAME HISTORIQUE "past_data"

def build_past_data():
    """
    Construit un ensemble de données historiques agrégé et prétraité à partir
    des données brutes ACLED, en comptant les occurrences des acteurs et des
    types d'événements, et en ajoutant des caractéristiques décalées.

    Returns:
        pandas.DataFrame: Le DataFrame agrégé et prétraité contenant
                            les données historiques.
    """

    df = preprocessing_dataframe(config['paths']['raw_data'])

    # On compte les occurences de chaque valeur des colonnes
    # 'actor1_type' et 'actor2_type'
    id_vars = ['event_date',
               'country',
               'admin1',
               'year',
               'month',
               'is_terrorist_group_related',
               'fatalities',
               'event_type']

    actor_counts = df.melt(id_vars=id_vars,
                           value_vars=['actor1_type', 'actor2_type'],
                           var_name='actor_role', value_name='actor')

    groupby_actor = ['country', 'admin1', 'year', 'month', 'actor']
    actor_counts = (
        actor_counts
        .groupby(groupby_actor)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # On compte les occurences de chaque type d'évènement
    groupby_vars = ['country', 'admin1', 'year', 'month']
    event_type_counts = df.pivot_table(
        index=groupby_vars,
        columns='event_type',
        aggfunc='size',
        fill_value=0).reset_index()

    # On agrège les données par mois, par pays, par sous-région
    past_data = df.groupby(groupby_vars).agg(
        total_events=('event_date', 'count'),
        terrorist_events=('is_terrorist_group_related', 'sum'),
        fatalities=('fatalities', 'sum')
    ).reset_index()

    # On fusionne les variables des acteurs et des types d'évènements
    # avec les données agrégées
    past_data = past_data.merge(actor_counts, on=groupby_vars, how='left')
    past_data = past_data.merge(event_type_counts, on=groupby_vars, how='left')

    # On supprime les colonnes 'None' et 'Protesters' qui concernent
    # les acteurs mais sont sans intérêt pour notre étude
    # ou bien la colonne 'Rebel Groups' qui est dérivée de la variable cible
    past_data = past_data.drop(
        columns=['None', 'Protesters', 'Rebel Groups'], axis=1
        )

    # On crée des features décalées (lags) pour capturer les tendances passées
    # des événements totaux et des événements terroristes
    for admin in past_data['admin1'].unique():
        # Pour chaque décalage (lag) de 1 à 6
        for lag in range(1, 7):
            # Crée une série décalée pour 'total_events' pour l'admin actuel
            lag_total_events = (
                past_data[past_data['admin1'] == admin]['total_events']
                .shift(lag)
            )
            # Crée une série décalée pour 'terrorist_events'
            # sans filtrer par admin
            lag_terrorist_events = (
                past_data[past_data['admin1'] == admin]['terrorist_events']
                .shift(lag)
            )
            # Ajoute la série décalée pour 'total_events' dans
            # une nouvelle colonne pour l'admin actuel
            past_data.loc[
                past_data['admin1'] == admin,
                f'total_events_lag_{lag}'
            ] = lag_total_events
            # Ajoute la série décalée pour 'terrorist_events' dans
            # une nouvelle colonne pour l'admin actuel
            past_data.loc[
                past_data['admin1'] == admin,
                f'terrorist_events_lag_{lag}'
            ] = lag_terrorist_events
            # Remplit les valeurs manquantes dans les colonnes décalées
            # avec des zéros
            columns_to_fill = [
                f'total_events_lag_{lag}',
                f'terrorist_events_lag_{lag}'
            ]
            past_data[columns_to_fill] = past_data[columns_to_fill].fillna(0)

    # On supprime la colonne "total_events" qui est trop corrélée
    # à notre variable cible
    past_data = past_data.drop(columns=['total_events'], axis=1)

    # On sauvegarde past_data
    save_dataframe_to_csv(past_data, config['paths']['past_data'])

    return past_data


# TRANSFORMATION DES DONNEES DU DATAFRAME "past_data"

def past_data_transformation(past_data):
    """
    Transforme les données historiques pour l'entraînement et le test de
    modèles de prédiction.

    Args:
        past_data (pandas.DataFrame): Le DataFrame contenant
        les données historiques.

    Returns:
        tuple: Contient les DataFrames et séries suivants après transformation:
            - X_train_encoded (pandas.DataFrame): Les données
            d'entraînement transformées.
            - X_test_encoded (pandas.DataFrame): Les données de
            test transformées.
            - y_train (pandas.Series): La variable cible pour l'entraînement.
            - y_test (pandas.Series): La variable cible pour le test.
            - scaler (StandardScaler): L'objet scaler utilisé pour
            standardiser les données.
    """

    # On trie notre dataframe par année puis mois dans l'ordre croissant
    # pour que les données de test concernent les événements les plus récents
    past_data = past_data.sort_values(by=['year', 'month'])

    # Séparation des variables explicatives de notre variable cible
    X = past_data.drop(columns='terrorist_events', axis=1)
    y = past_data['terrorist_events']

    # Division en ensembles d'entraînement et de test avec shuffle à False
    # pour que les données de test concernent les événements les plus récents
    # (6 derniers mois)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=False)

    # On sauvegarde une version non encodée de X_test pour traitement ultérieur
    save_dataframe_to_csv(X_test, config['paths']['X_test'])

    # On standardise les variables numériques exceptées
    # les variables catégorielles ainsi que l'année et le mois
    # qui subiront une autre transformation
    cat_columns = ['country', 'admin1']

    X_train_numerical_vars = [
        col for col in X_train.columns
        if col not in cat_columns + ['year', 'month']
    ]

    X_test_numerical_vars = [
        col for col in X_test.columns
        if col not in cat_columns + ['year', 'month']
    ]

    # Extraction des données numériques
    X_train_numerical = X_train[X_train_numerical_vars].values
    X_test_numerical = X_test[X_test_numerical_vars].values

    # Initialisation d'un scaler
    scaler = StandardScaler()
    X_train[X_train_numerical_vars] = scaler.fit_transform(X_train_numerical)
    X_test[X_test_numerical_vars] = scaler.transform(X_test_numerical)

    # Hot One Encoding des variables catégorielles
    X_train_encoded = pd.get_dummies(X_train, columns=cat_columns, dtype=int)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_columns, dtype=int)
    # On aligne les colonnes de X_test_encoded sur celles de X_train_encoded
    # pour s'assurer qu'ils ont les mêmes colonnes
    X_test_encoded = X_test_encoded.reindex(
        columns=X_train_encoded.columns,
        fill_value=0
    )

    # Transformation trigonométrique de 'year' et 'month' pour la saisonnalité
    X_train_encoded['month_sin'] = np.sin(
        2 * np.pi * X_train_encoded['month'] / 12
    )

    X_train_encoded['month_cos'] = np.cos(
        2 * np.pi * X_train_encoded['month'] / 12
    )

    X_train_encoded['year_sin'] = np.sin(
        2 * np.pi * (X_train_encoded['year'] - X_train_encoded['year'].min()) /
        (X_train_encoded['year'].max() - X_train_encoded['year'].min() + 1))

    X_train_encoded['year_cos'] = np.cos(
        2 * np.pi * (X_train_encoded['year'] - X_train_encoded['year'].min()) /
        (X_train_encoded['year'].max() - X_train_encoded['year'].min() + 1))

    X_test_encoded['month_sin'] = np.sin(
        2 * np.pi * X_test_encoded['month'] / 12
    )

    X_test_encoded['month_cos'] = np.cos(
        2 * np.pi * X_test_encoded['month'] / 12
    )

    X_test_encoded['year_sin'] = np.sin(
        2 * np.pi * (X_test_encoded['year'] - X_test_encoded['year'].min()) /
        (X_test_encoded['year'].max() - X_test_encoded['year'].min() + 1)
    )

    X_test_encoded['year_cos'] = np.cos(
        2 * np.pi * (X_test_encoded['year'] - X_test_encoded['year'].min()) /
        (X_test_encoded['year'].max() - X_test_encoded['year'].min() + 1)
    )

    # Suppression des colonnes 'month' et 'year' après transformation
    X_train_encoded = X_train_encoded.drop(columns=['month', 'year'])
    X_test_encoded = X_test_encoded.drop(columns=['month', 'year'])

    return (
        X_train_encoded,
        X_test_encoded,
        y_train,
        y_test,
        scaler
    )


# CONSTRUCTION DU DATAFRAME FUTUR "future_data"
# POUR LES PREDICTIONS DES 6 PROCHAINS MOIS

def build_future_data(past_data):
    """
    Construit un ensemble de données futures pour les six prochains mois
    basé sur les données historiques fournies, en utilisant des moyennes
    mensuelles et des valeurs décalées (lags) pour remplir les
    colonnes explicatives.

    Args:
        past_data (pandas.DataFrame): Le DataFrame contenant les données
                                        historiques prétraitées.

    Returns:
        pandas.DataFrame: Le DataFrame contenant les données futures pour les
                            six prochains mois.
    """

    unique_country_admin1 = past_data[['country', 'admin1']].drop_duplicates()

    # On souhaite d'abord obtenir le dernier mois de la dernière l'année
    # dans past_data
    last_year = past_data['year'].max()
    last_month = past_data[past_data['year'] == last_year]['month'].max()

    # On crée ensuite une liste de dictionnaires pour les 6 prochains mois
    # avec les colonnes explicatives vides
    future_data_list = []

    # On ajoute les 6 prochains mois à future_data_list avec les colonnes
    # connues "country" et "admin1" remplies
    for i in range(1, 7):
        next_month = last_month + i
        next_year = last_year
        if next_month > 12:
            next_month -= 12
            next_year += 1
        for _, row in unique_country_admin1.iterrows():
            future_data_list.append({
                'year': next_year,
                'month': next_month,
                'country': row['country'],
                'admin1': row['admin1']
            })

    # On crée notre dataframe "future_data" à partir de la liste
    # de dictionnaires
    future_data = pd.DataFrame(future_data_list, columns=past_data.columns)

    # On remplit les autres colonnes explicatives avec des valeurs manquantes
    for col in past_data.columns:
        if col not in ['year', 'month', 'country', 'admin1']:
            future_data[col] = np.nan

    columns_to_fill = [
        'fatalities',
        'Civilians',
        'External/Other Forces',
        'Identity Militias',
        'Political Militias',
        'State Forces',
        'Battles',
        'Explosions/Remote violence',
        'Strategic developments',
        'Violence against civilians',
        'total_events_lag_1',
        'terrorist_events_lag_1',
        'total_events_lag_2',
        'terrorist_events_lag_2',
        'total_events_lag_3',
        'terrorist_events_lag_3',
        'total_events_lag_4',
        'terrorist_events_lag_4',
        'total_events_lag_5',
        'terrorist_events_lag_5',
        'total_events_lag_6',
        'terrorist_events_lag_6'
    ]

    # On groupe les données par 'country', 'admin1', 'month'
    # pour calculer les moyennes dans past_data
    grouped_data = past_data.groupby(
        ['country', 'admin1', 'month']
        ).mean().reset_index()

    # On remplace les valeurs manquantes dans future_data
    # par les moyennes correspondantes de past_data en fonction du mois.
    # Ex : si on veut calculer les valeurs des variables du mois de juillet,
    # on va calculer la moyenne des variables des mois de juillet
    # des années précédentes pour avoir des valeurs de test réalistes.
    for index, row in future_data.iterrows():
        country = row['country']
        admin1 = row['admin1']
        month = row['month']
        for var in columns_to_fill:
            mean_value = grouped_data[
                (grouped_data['country'] == country) &
                (grouped_data['admin1'] == admin1) &
                (grouped_data['month'] == month)
            ][var].values
            if len(mean_value) > 0:
                future_data.at[index, var] = mean_value[0]

    # Certaines valeurs sont manquantes car les combinaisons
    # (country, admin1, month) n'existent pas, on va donc créer un dataframe
    # contenant les combinaisons uniques de (country, admin1, month)
    # dans future_data
    future_combinations = future_data[
        ['country', 'admin1', 'month']
    ].drop_duplicates()

    # Puis un dataframe contenant les combinaisons uniques de
    # (country, admin1, month) dans past_data
    monthly_combinations = past_data[
        ['country', 'admin1', 'month']
    ].drop_duplicates()

    # Ensuite on recherche les combinaisons présentes dans future_data
    # mais absentes dans past_data
    missing_combinations = future_combinations[
        ~future_combinations.isin(monthly_combinations)
    ].dropna()

    # En fonction de ces combinaisons, on vient remplir les valeurs manquantes
    # dans future_data à partir du calcul de la moyenne des autres mois pour
    # la combinaison (country, admin1)
    for index, row in missing_combinations.iterrows():
        # Sélection des lignes correspondant à la combinaison
        # de country et admin1 dans past_data
        matching_rows = past_data[
            (past_data['country'] == row['country']) &
            (past_data['admin1'] == row['admin1'])
        ]

        # Calcul de la moyenne des valeurs des variables pour tous
        # les mois existants
        mean_values = matching_rows[columns_to_fill].mean()
        # Remplissage des valeurs manquantes dans future_data avec
        # la moyenne calculée
        future_data.loc[
            (future_data['country'] == row['country']) &
            (future_data['admin1'] == row['admin1']) &
            (future_data['month'] == row['month']),
            columns_to_fill
        ] = mean_values.values

    # Enfin on remplace les NaN des lignes restantes par la valeur
    # la plus fréquente dans chaque colonne
    future_data = future_data.fillna(future_data.mode().iloc[0])

    # On supprime la colonne de variable cible et on vérifie qu'il n'y a plus
    # de NaN
    future_data = future_data.drop(columns=['terrorist_events'])
    future_data.info()

    # On arrondit les valeurs calculées pour les variables explicatives
    # afin d'obtenir des valeurs entières
    future_data = future_data.round(0)

    # On sauvegarde past_data
    save_dataframe_to_csv(future_data, config['paths']['future_data'])

    return future_data


# TRANSFORMATION DES DONNEES DU DATAFRAME "future_data"

def future_data_transformation(future_data, scaler):
    """
    Transforme les données futures en standardisant les variables numériques,
    en encodant les variables catégorielles et en appliquant des
    transformations trigonométriques aux variables de date pour
    capturer la saisonnalité.

    Args:
        - future_data (pandas.DataFrame): Le DataFrame contenant les données
        futures à transformer.
        - scaler (sklearn.preprocessing.StandardScaler): Le scaler ajusté sur
        les données d'entraînement.

    Returns:
        pandas.DataFrame: Le DataFrame contenant les données
                            futures transformées.
    """

    # On standardise les variables numériques exceptées
    # les variables catégorielles ainsi que l'année et le mois
    # qui subiront une autre transformation
    cat_columns = ['country', 'admin1']
    future_data_numerical_vars = [
        col for col in future_data.columns
        if col not in cat_columns + ['year', 'month']
    ]

    # Extraction des données numériques
    future_data_numerical = future_data[future_data_numerical_vars].values

    # On utilise le scaler ajusté sur les données d'entraînement
    # pour transformer les données du dataframe
    future_data[future_data_numerical_vars] = (
        scaler.transform(future_data_numerical)
    )

    # Hot One Encoding des variables catégorielles
    future_data = pd.get_dummies(
        future_data,
        columns=cat_columns,
        dtype=int
    )

    # Transformation trigonométrique de 'year' et 'month' pour la saisonnalité
    future_data['month_sin'] = np.sin(
        2 * np.pi * future_data['month'] / 12
    )

    future_data['month_cos'] = np.cos(
        2 * np.pi * future_data['month'] / 12
    )

    future_data['year_sin'] = np.sin(
        2 * np.pi * (future_data['year'] - future_data['year'].min()) /
        (future_data['year'].max() - future_data['year'].min() + 1)
    )

    future_data['year_cos'] = np.cos(
        2 * np.pi * (future_data['year'] - future_data['year'].min()) /
        (future_data['year'].max() - future_data['year'].min() + 1)
    )

    # Suppression des colonnes 'month' et 'year' après transformation
    future_data = future_data.drop(columns=['month', 'year'])

    return future_data
