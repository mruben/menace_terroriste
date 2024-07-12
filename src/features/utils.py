import json


def load_json_file(file_path):
    """
    Charge le contenu d'un fichier JSON et le renvoie sous forme
    de dictionnaire.

    Args:
        file_path (str): Le chemin vers le fichier JSON à charger.

    Returns:
        dict: Le contenu du fichier JSON sous forme de dictionnaire.
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def save_dataframe_to_csv(df, path):
    """
    Enregistre un DataFrame en tant que fichier CSV.

    Args:
        df (pandas.DataFrame): Le DataFrame à enregistrer.
        path (str): Le chemin où enregistrer le fichier CSV.

    Returns:
        None
    """

    df.to_csv(path, index=False, encoding='utf-8')
