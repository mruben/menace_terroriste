import pandas as pd

from src.features.utils import load_json_file, save_dataframe_to_csv
from ..config_loader import load_config
config = load_config()


# PREPROCESSING DU DATAFRAME

def preprocessing_dataframe(path):
    """
    La fonction preprocessing_dataframe prend en entrée le chemin d'accès
    à un fichier de données, effectue une série d'opérations de prétraitement
    sur ces données, les sauvegarde dans un fichier CSV, puis retourne
    le DataFrame prétraité.

    Paramètres:
        path (str): Chemin d'accès au fichier de données source.
                    Ce fichier doit être dans un format lisible par
                    la fonction load_dataset.

    Fonctionnalités:

        1. Chargement des données
        load_dataset(path): Charge les données à partir du chemin spécifié
        et retourne un DataFrame.

        2. Filtrage des colonnes
        columns_filtering(df): Applique un filtrage sur les colonnes
        du DataFrame. Cette fonction peut, par exemple, sélectionner
        un sous-ensemble de colonnes d'intérêt.

        3. Ajout de nouvelles colonnes
        add_new_columns(df): Ajoute de nouvelles colonnes calculées
        ou dérivées au DataFrame.

        4. Suppression de colonnes
        delete_columns(df): Supprime certaines colonnes du DataFrame,
        potentiellement celles qui ne sont plus nécessaires après
        les étapes précédentes.

        5. Imputation des valeurs manquantes
        fill_nan_values(df): Remplit les valeurs manquantes (NaN) dans
        le DataFrame, en utilisant une méthode appropriée telle que la moyenne,
        la médiane ou une valeur par défaut.

        6. Sauvegarde des données prétraitées
        save_dataframe_to_csv(df, config['paths']['processed_data']):
        Sauvegarde le DataFrame prétraité dans un fichier CSV à l'emplacement
        spécifié dans la configuration (config['paths']['processed_data']).

    Renvoie:
        df (pandas.DataFrame): Le DataFrame prétraité, après application de
                                toutes les étapes de transformation
                                et d'imputation.
    """

    df = load_dataset(path)
    df = columns_filtering(df)
    df = add_new_columns(df)
    df = delete_columns(df)
    df = fill_nan_values(df)

    # On sauvegarde le dataframe
    save_dataframe_to_csv(df, config['paths']['processed_data'])

    return df


# CHARGEMENT DU DATAFRAME

def load_dataset(path):
    """
    La fonction load_dataset lit un fichier CSV depuis le chemin spécifié et
    retourne les données sous forme de DataFrame Pandas.

    Paramètres:

        path (str): Chemin d'accès au fichier CSV à charger.
                    Ce chemin peut être absolu ou relatif.

    Fonctionnalités:

        Chargement du fichier CSV:
        Utilise pandas.read_csv pour lire le fichier CSV situé à l'emplacement
        spécifié par path.

        Retourne le DataFrame:
        Retourne un DataFrame Pandas contenant les données lues depuis
        le fichier CSV.

    Renvoie:
        df (pandas.DataFrame): Un DataFrame contenant les données lues depuis
                                le fichier CSV.
    """

    df = pd.read_csv(path)

    return df


# FILTRAGE DE COLONNES

def columns_filtering(df):
    """
    Filtre les lignes du DataFrame en fonction des types d'événements
    spécifiés et retourne le DataFrame filtré.

    Paramètres:
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données à filtrer. Il doit inclure
        une colonne 'event_type'.

    Retour:
    ------
    pandas.DataFrame
        Le DataFrame filtré contenant uniquement les lignes où 'event_type'
        est dans la liste des événements spécifiés.

    Exemple d'utilisation:
    ----------------------
    >>> df = pd.read_csv('path/to/data.csv')
    >>> filtered_df = columns_filtering(df)
    >>> print(filtered_df.head())

    Notes:
    ------
    - La colonne 'event_type' doit être présente dans le DataFrame fourni.
    - Les types d'événements filtrés sont: 'Battles',
    'Explosions/Remote violence', 'Violence against civilians',
    et 'Strategic developments'.
    """

    events_filter = ['Battles',
                     'Explosions/Remote violence',
                     'Violence against civilians',
                     'Strategic developments']

    df = df[df['event_type'].isin(events_filter)]

    return df


# CREATION DE NOUVELLES COLONNES

def add_new_columns(df):
    """
    Ajoute de nouvelles colonnes au DataFrame en fonction des dates,
    des acteurs, des interactions, des groupes terroristes, et des PMC russes.

    Paramètres:
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données d'origine.

    Retour:
    ------
    pandas.DataFrame
        Le DataFrame avec les nouvelles colonnes ajoutées.

    Fonctionnalités:
    ----------------
    - **add_new_colums_dates(df)**:
        - Convertit la colonne 'event_date' en type datetime.
        - Ajoute des colonnes 'month' et 'day' pour le mois et le jour
        de l'événement.

    - **add_new_colums_actors(df)**:
        - Charge les types d'acteurs à partir d'un fichier JSON.
        - Ajoute les colonnes 'actor1_type' et 'actor2_type' correspondant
        aux types d'acteurs.

    - **add_new_colums_interaction(df)**:
        - Charge les types d'interactions à partir d'un fichier JSON.
        - Ajoute une colonne 'interaction_type' correspondant aux types
        d'interactions.

    - **add_new_colums_terrorist_group(df)**:
        - Charge les affiliations des groupes terroristes à partir
        d'un fichier JSON.
        - Ajoute une colonne 'is_terrorist_group_related' indiquant si
        un acteur est une organisation terroriste.
        - Ajoute une colonne 'terrorist_group_filiation' indiquant
        l'organisation mère.

    - **add_new_colums_pmc_group(df)**:
        - Identifie les acteurs PMC russes spécifiques.
        - Ajoute une colonne 'is_pmc_related' indiquant si un acteur est
        un PMC russe.

    Exemple d'utilisation:
    ----------------------
    >>> df = pd.read_csv('path/to/data.csv')
    >>> df = add_new_columns(df)
    >>> print(df.head())

    Notes:
    ------
    - Les colonnes 'event_date', 'inter1', 'inter2', 'actor1', 'assoc_actor_1',
    'actor2', et 'assoc_actor_2' doivent être présentes dans le DataFrame.
    - Les fichiers JSON contenant les types d'acteurs, types d'interactions,
    affiliations des groupes terroristes et PMC russes doivent être spécifiés
    dans la configuration `config['paths']['json']`.
    - La fonction `load_json_file` doit être définie pour charger les données
    à partir des fichiers JSON.
    - La variable `config` doit contenir les chemins de fichiers appropriés.

    Dépendances:
    ------------
    - pandas doit être importé en tant que `pd`.
    - La fonction `load_json_file` doit être définie pour charger les types
    d'acteurs à partir d'un fichier JSON.
    - La variable `config` doit contenir les chemins de fichiers appropriés.
    """

    def add_new_colums_dates(df):
        # On transforme la colonne "event_date" en datetime pour la manipuler
        # plus facilement
        df['event_date'] = pd.to_datetime(df['event_date'])

        # On crée la colonne "month" en utilisant l'attribut month de datetime
        df['month'] = df['event_date'].dt.month
        # On ajoute la colonne juste après "year"
        df.insert(3, 'month', df.pop('month'))

        # On crée la colonne "day" en utilisant l'attribut day de datetime
        df['day'] = df['event_date'].dt.day
        # On ajoute la colonne juste après "month"
        df.insert(5, 'day', df.pop('day'))

        return df

    def add_new_colums_actors(df):
        # On charge les données du fichier json "actor_type" dans un
        # dictionnaire qui associe les valeurs des colonnes "inter1" et
        # "inter2" au nom de chaque catégorie d'acteur
        # source : codebook ACLED
        actor_type = load_json_file(
            config['paths']['json']['actor_type']
        )

        # On convertit les clés du dictionnaire en entiers
        actor_type = {int(k): v for k, v in actor_type.items()}

        # On ajoute les colonnes "actor1_type" et "actor2_type" au dataframe
        df['actor1_type'] = df['inter1'].map(actor_type)
        df['actor2_type'] = df['inter2'].map(actor_type)

        return df

    def add_new_colums_interaction(df):
        ##
        # On charge les données du fichier json "interaction_type" dans un
        # dictionnaire associant aux valeurs de la colonne "interaction"
        # les 2 acteurs impliqués dans une confrontation
        # source : codebook ACLED
        interaction_type = load_json_file(
            config['paths']['json']['interaction_type']
        )

        # On convertit les clés du dictionnaire en entiers
        interaction_type = {int(k): v for k, v in interaction_type.items()}

        # On ajoute une colonne "interaction_type" au dataframe
        df['interaction_type'] = df['interaction'].map(interaction_type)

        return df

    def add_new_colums_terrorist_group(df):
        # On crée une nouvelle colonne qui indique si pour un évènement l'un
        # des acteurs est une organisation terroriste
        # On charge les données du fichier json "terrorist_group_filiation"
        # dans un dictionnaire associant organisation terroriste et
        # organisation mère
        terrorist_group_filiation = load_json_file(
            config['paths']['json']['terrorist_group_filiation']
        )

        # On crée une liste contenant les organisations terroristes à partir
        # des indices de ce dictionnaire
        terrorist_groups = list(terrorist_group_filiation.keys())

        # On crée une fonction pour vérifier si un acteur est
        # une organisation terroriste
        def is_terrorist_actor(actor):
            return actor in terrorist_groups

        # On ajoute la colonne "is_terrorist_group_related" au dataframe
        df['is_terrorist_group_related'] = (
            df['actor1'].apply(is_terrorist_actor) |
            df['assoc_actor_1'].apply(is_terrorist_actor) |
            df['actor2'].apply(is_terrorist_actor) |
            df['assoc_actor_2'].apply(is_terrorist_actor)
        ).astype(int)

        # On crée une fonction de mapping pour associer les valeurs du
        # dictionnaire aux acteurs
        # S'il n'y a pas de valeur on retourne "None" car cela veut simplement
        # dire que l'évènement n'est pas lié à une organisation terroriste
        # et qu'il n'y a pas de lien de filiation avec une organisation mère
        def map_filiation(row):
            actors = ['actor1',
                      'assoc_actor_1',
                      'actor2',
                      'assoc_actor_2']

            for actor in actors:
                if row[actor] in terrorist_group_filiation:
                    return terrorist_group_filiation[row[actor]]
            return "None"

        # On ajoute la colonne "terrorist_group_filiation" au dataframe
        df['terrorist_group_filiation'] = df.apply(map_filiation, axis=1)

        return df

    def add_new_colums_pmc_group(df):
        # On crée une nouvelle colonne qui indique si pour un évènement l'un
        # des acteurs est une pmc russe
        pmc_groups = ['Wagner Group']

        # On crée une fonction pour vérifier si un acteur est une pmc russe
        def is_pmc_actor(actor):
            return actor in pmc_groups

        # On ajoute la colonne "is_pmc_related" au dataframe
        df['is_pmc_related'] = (
            df['actor1'].apply(is_pmc_actor) |
            df['assoc_actor_1'].apply(is_pmc_actor) |
            df['actor2'].apply(is_pmc_actor) |
            df['assoc_actor_2'].apply(is_pmc_actor)
        ).astype(int)

        return df

    # Ajout des différentes colonnes
    df = add_new_colums_dates(df)
    df = add_new_colums_actors(df)
    df = add_new_colums_interaction(df)
    df = add_new_colums_terrorist_group(df)
    df = add_new_colums_pmc_group(df)

    return df


# SUPPRESSION DE COLONNES

def delete_columns(df):
    """
    Supprime les colonnes inutiles du DataFrame et retourne le
    DataFrame modifié.

    Paramètres:
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données d'origine.

    Retour:
    ------
    pandas.DataFrame
        Le DataFrame avec les colonnes inutiles supprimées.

    Colonnes supprimées:
    -------------------
    - 'time_precision'
    - 'disorder_type'
    - 'sub_event_type'
    - 'actor1'
    - 'assoc_actor_1'
    - 'actor2'
    - 'assoc_actor_2'
    - 'inter1'
    - 'inter2'
    - 'interaction'
    - 'admin2'
    - 'admin3'
    - 'iso'
    - 'region'
    - 'location'
    - 'latitude'
    - 'longitude'
    - 'geo_precision'
    - 'source'
    - 'source_scale'
    - 'notes'
    - 'tags'
    - 'timestamp'
    - 'civilian_targeting'
    - 'event_id_cnty'

    Exemple d'utilisation:
    ----------------------
    >>> df = pd.read_csv('path/to/data.csv')
    >>> df = delete_columns(df)
    >>> print(df.head())

    Notes:
    ------
    - La liste des colonnes à supprimer est fixe et peut être modifiée selon
    les besoins spécifiques de l'analyse.
    - Assurez-vous que les colonnes listées existent dans le DataFrame pour
    éviter les erreurs lors de la suppression.
    """

    columns_to_drop = [
        'time_precision',
        'disorder_type',
        'sub_event_type',
        'actor1',
        'assoc_actor_1',
        'actor2',
        'assoc_actor_2',
        'inter1',
        'inter2',
        'interaction',
        'admin2',
        'admin3',
        'iso',
        'region',
        'location',
        'latitude',
        'longitude',
        'geo_precision',
        'source',
        'source_scale',
        'notes',
        'tags',
        'timestamp',
        'civilian_targeting',
        'event_id_cnty'
    ]

    df = df.drop(columns=columns_to_drop, axis=1)

    return df


# GESTION DES VALEURS MANQUANTES

def fill_nan_values(df):
    """
    Remplit les valeurs manquantes dans le DataFrame avec des
    valeurs appropriées.

    Paramètres:
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données d'origine avec des
        valeurs manquantes.

    Retour:
    ------
    pandas.DataFrame
        Le DataFrame avec les valeurs manquantes remplies.

    Valeurs manquantes remplies:
    ----------------------------
    - 'actor2_type': Les valeurs manquantes sont remplacées par "None"
    pour indiquer qu'il y a un seul acteur et non un manque de valeur.

    Exemple d'utilisation:
    ----------------------
    >>> df = pd.read_csv('path/to/data.csv')
    >>> df = fill_nan_values(df)
    >>> print(df.head())

    Notes:
    ------
    - La colonne 'actor2_type' doit être présente dans le DataFrame.
    - La fonction peut être modifiée pour inclure d'autres colonnes
    nécessitant des valeurs de remplacement spécifiques.
    """

    df['actor2_type'] = df['actor2_type'].fillna("None")

    return df
