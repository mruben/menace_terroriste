def convert_to_lowercase_and_replace_spaces(var):
    """
    Convertit une chaîne de caractères en minuscules et remplace
    les espaces par des tirets bas.

    Args:
        var (str): Chaîne de caractères à convertir.

    Returns:
        str: Chaîne convertie en minuscules avec les espaces remplacés
            par des tirets bas.
    """

    # Convertir la chaîne en minuscules
    lower_var = var.lower()
    # Remplacer les espaces par des tirets bas
    lower_var = lower_var.replace(' ', '_')

    return lower_var
