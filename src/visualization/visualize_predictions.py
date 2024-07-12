import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.visualization.utils import convert_to_lowercase_and_replace_spaces
from matplotlib.colors import to_hex
from src.config_loader import load_config
config = load_config()


def generate_plot_predictions_vs_observed(df_test, y_test, predictions_past):
    """
    Visualisation des prédictions par rapport au réel
    par mois de Novembre 2023 à Avril 2024
    au Mali, au Burkina Faso et au Niger
    des événements liés à des organisations terroristes.

    Args:
        df_test (DataFrame): DataFrame contenant les données de test.
        y_test (Series): Série contenant les valeurs réelles des événements.
        predictions_past (Series): Série contenant les prédictions passées.

    Returns:
        plt.Figure: Figure contenant le graphique comparatif.
    """

    # On ajoute les prédictions à l'ensemble de test
    df_test['terrorist_events'] = y_test
    df_test['terrorist_events_pred'] = predictions_past

    # On convertit 'year' et 'month' en une colonne datetime 'event_date'
    df_test['event_date'] = pd.to_datetime(
        df_test[['year', 'month']].assign(day=1)
    )

    # On regroupe les événements réels par mois
    real_events_by_month = (
        df_test.groupby('event_date')['terrorist_events'].sum()
    )

    # On regroupe les événements prédits par mois
    pred_events_by_month = df_test.groupby(
        'event_date')['terrorist_events_pred'].sum()

    # On crée notre graphique
    fig, ax = plt.subplots(figsize=(20, 8))

    ax.plot(
        real_events_by_month.index,
        real_events_by_month.values,
        label='réel',
        linestyle='-',
        marker='o',
        color='blue'
    )

    ax.plot(
        pred_events_by_month.index,
        pred_events_by_month.values,
        label='prédit',
        linestyle='--',
        marker='x',
        color='orange'
    )

    ax.legend()

    ax.set_title(
        "Evénements liés à des organisations terroristes, par mois"
        " de Novembre 2023 à Avril 2024, au Mali, au Burkina Faso et au Niger")

    ax.set_xlabel('Mois')
    ax.set_ylabel("Nombre d'événements")

    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%b %Y')
    )

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        config['paths']['figures'] +
        "predictions_vs_observed.png"
    )

    return fig


def generate_plot_predictions(country, df, display_by):
    """
    Génère un graphique des événements prédits par pays ou pour tous les pays,
    selon les données fournies dans le DataFrame.

    Args:
        country (str): Nom du pays à filtrer ou "all" pour tous les pays.
        df (DataFrame): DataFrame contenant les données à visualiser.
        display_by (str): Nom de la variable par laquelle grouper et afficher
                            les événements.

    Returns:
        plt.Figure: Figure contenant le graphique comparatif.
    """

    # Filtrer les données pour le pays renseigné en paramètre
    if (country in config['countries']['all']):
        df = df.loc[df['country'] == country]
        # Les variables qui nous intéressent pour le groupby
        group_by_var = ['year', 'month', 'country', 'admin1']
        # Le nom du fichier du graphique en fonction du pays
        filename = (
            "predictions_" +
            convert_to_lowercase_and_replace_spaces(country) +
            ".png"
        )

    # Sinon on exécute la suite du code sur tous les pays
    else:
        country = "Mali, Burkina Faso, Niger"
        # Les variables qui nous intéressent pour le groupby
        group_by_var = ['year', 'month', 'country']
        # Le nom du fichier pour le graphique de tous les pays
        filename = "predictions_all_countries.png"

    # Grouper les données filtrées par 'year', 'month', 'country', 'admin1'
    # et calculer la somme des 'terrorist_events'
    terrorist_events = df.groupby(
        group_by_var
    )['terrorist_events'].sum().reset_index()

    terrorist_events['date'] = pd.to_datetime(
        terrorist_events[['year', 'month']].assign(day=1)
    )

    terrorist_events.set_index('date', inplace=True)

    # Visualisation des événements prédits
    number_of_colors = 15

    # On utilise une palette de couleurs de matplotlib
    # en fixant un nombre maximum de couleurs possibles
    # afin d'en obtenir des différentes pour bien les distinguer visuellement
    cmap = plt.get_cmap('tab20')
    colors = [to_hex(cmap(i)) for i in range(number_of_colors)]

    dict_color = dict(zip(df[display_by].unique(), colors))

    legend_pops = []

    # On crée notre graphique
    fig, ax = plt.subplots(figsize=(20, 8))

    for display_by_unique in df[display_by].unique():
        filtered_terrorist_events = terrorist_events[
            terrorist_events[display_by] == display_by_unique
        ]

        predicted_terrorist_events = filtered_terrorist_events[
            (filtered_terrorist_events['month'] > 4) &
            (filtered_terrorist_events['year'] == 2024)
        ]

        observed_terrorist_events = filtered_terrorist_events[
            ~((filtered_terrorist_events['month'] > 4) &
                (filtered_terrorist_events['year'] == 2024))
        ]

        ax.plot(
            observed_terrorist_events.index,
            observed_terrorist_events['terrorist_events'],
            linestyle='-',
            marker='x',
            color=dict_color[display_by_unique]
        )

        ax.plot(
            predicted_terrorist_events.index,
            predicted_terrorist_events['terrorist_events'],
            linestyle='--',
            marker='x',
            color=dict_color[display_by_unique])

        legend_pops.append(
            mpatches.Patch(
                color=dict_color[display_by_unique],
                label=display_by_unique
            )
        )

    ax.legend(handles=legend_pops)

    ax.set_title(f"Evénements passés et prédits liés à des organisations"
                 f" terroristes, par mois de juin 2021 à Octobre 2024"
                 f" au {country}, par administration territoriale")

    ax.set_xlabel('Mois')
    ax.set_ylabel(f"Nombre d'événements par {display_by}")

    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%b %Y')
    )

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        config['paths']['figures'] +
        filename
    )

    return fig


# On récupère les jeux de données générés par le modèle de ML
past_data = pd.read_csv(config['paths']['past_data'])
predictions_past = pd.read_csv(config['paths']['predictions_past_data'])
future_data = pd.read_csv(config['paths']['future_data'])
predictions_future = pd.read_csv(config['paths']['predictions_future_data'])
future_data['terrorist_events'] = predictions_future.round(0)
df_test = pd.read_csv(config['paths']['X_test'])
y_test = pd.read_csv(config['paths']['y_test'])
# On créé un datframe contenant aussi bien les données passées que futures
all_data = pd.concat([past_data, future_data], axis=0, ignore_index=True)

# On génère les différents graphiques de visualisation des prédictions
generate_plot_predictions_vs_observed(df_test, y_test, predictions_past)
generate_plot_predictions('Niger', all_data, 'admin1')
generate_plot_predictions('Burkina Faso', all_data, 'admin1')
generate_plot_predictions('Mali', all_data, 'admin1')
generate_plot_predictions('All', all_data, 'country')
