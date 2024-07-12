import yaml
import os


def load_config():
    """
    Charge un fichier de configuration YAML à partir du chemin spécifié
    et retourne son contenu sous forme de dictionnaire.

    Returns:
        dict: Dictionnaire contenant les paramètres de configuration.
    """

    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    )

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
