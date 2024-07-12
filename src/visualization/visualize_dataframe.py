import pandas as pd
import matplotlib.pyplot as plt
from src.config_loader import load_config
config = load_config()


def generate_plot_events_vs_terrorist(df):
    """
    Génère un graphique comparant le nombre d'événements généraux et ceux liés
    à des organisations terroristes par mois de Juin 2021 à Avril 2024 au Mali,
    au Burkina Faso et au Niger.

    Args:
        df (DataFrame): DataFrame contenant les données des événements.

    Returns:
        matplotlib.figure.Figure: Figure contenant le graphique généré.
    """

    # On crée une série qui permet de grouper et de compter par mois
    # les évènements liés à des violences perpétrées
    # par des organisations terroristes
    terrorist_events_by_month = df[
        df['is_terrorist_group_related'] == 1
        ].groupby(pd.Grouper(key='event_date', freq='ME')).size()

    # On crée une autre série qui permet de grouper et de compter par mois
    # les évènements liés à des violences politiques de manière générale
    events_by_month = df.groupby(
        pd.Grouper(key='event_date', freq='ME')
    ).size()

    # On crée notre graphique
    fig, ax = plt.subplots(figsize=(20, 8))

    ax.set_xlim(pd.Timestamp(config['dates']['start']),
                pd.Timestamp(config['dates']['end']))

    ax.plot(
        events_by_month.index,
        events_by_month.values,
        label='Evènements de tout type')

    ax.plot(terrorist_events_by_month.index,
            terrorist_events_by_month.values,
            label='Liés au terrorisme')

    ax.set_xlabel('Mois')
    ax.set_ylabel('Nombre d\'événements')

    ax.set_title('Nombre d\'événements par mois de Juin 2021 à Avril 2024'
                 ' au Mali, au Burkina Faso et au Niger', fontsize=12)

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%b %Y')
    )

    ax.legend()

    # On enregistre le graphique
    fig.savefig(
        config['paths']['figures'] +
        "events_vs_terrorist.png"
    )

    return fig


def generate_plot_events_by_type(grouped, filename, terrorist_events):
    """
    Génère un graphique représentant le nombre d'événements par mois
    pour différents types d'événements.

    Args:
        grouped (DataFrame): DataFrame groupé par type d'événement et mois.
        filename (str): Nom du fichier dans lequel sauvegarder le graphique.
        terrorist_events (bool): Indique si l'on trace les événements
                                liés au terrorisme.

    Returns:
        None
    """

    event_types = grouped['event_type'].unique()

    # Ensuite on trace une courbe pour chaque type d'évènement
    fig, ax = plt.subplots(figsize=(20, 8))

    # On va utiliser un type de tracé différent pour chaque type d'évènement
    line_styles = ['-', '--', '-.', ':']
    for i, event in enumerate(event_types):
        ls = line_styles[i % len(line_styles)]
        data = grouped[grouped['event_type'] == event]
        ax.plot(data['event_date'], data['count'], label=event, linestyle=ls)

    ax.set_xlabel('Mois')
    ax.set_ylabel('Nombre d\'événements')
    if (terrorist_events is False):
        ax.set_title("Nombre d\'événements par mois de Juin 2021 à Avril 2024"
                     " au Mali, au Burkina Faso et au Niger"
                     " en fonction du type d'évènement",
                     fontsize=12)
    else:
        ax.set_title("Nombre d\'événements lié au terrorisme, par mois"
                     " de Juin 2021 à Avril 2024 au Mali, au Burkina Faso"
                     " et au Niger en fonction du type d'évènement",
                     fontsize=12)

    ax.legend()

    fig.savefig(
        config['paths']['figures'] +
        filename
    )

    return fig


def generate_plot_terrorist_events(df_country, country_name):
    """
    Génère un graphique comparant le nombre d'événements liés au terrorisme
    par mois de Juin 2021 à Avril 2024 au Mali, au Burkina Faso et au Niger

    Args:
        df_country (list): Liste de DataFrames contenant les données
                            des événements par pays.
        country_name (list): Liste des noms des pays correspondants.

    Returns:
        matplotlib.figure.Figure: Figure contenant le graphique généré.
    """

    # On crée notre graphique
    fig, ax = plt.subplots(figsize=(20, 8))

    ax.set_xlim(pd.Timestamp(config['dates']['start']),
                pd.Timestamp(config['dates']['end']))

    # On parcourt le dataframe de chaque pays
    for i, (df_country, country_name) in enumerate(
        zip(df_country_list, country_names)
    ):
        terrorist_events = df_country[
            df_country['is_terrorist_group_related'] == 1
        ].groupby(pd.Grouper(key='event_date', freq='ME')).size()

        # On trace une courbe pour afficher le nombre d'évènements
        # par mois liés à des violences perpétrées par des organisations
        # terroristes ainsi que les violences politiques quelque soit l'acteur
        ax.plot(
            terrorist_events.index,
            terrorist_events.values,
            label=country_name
        )

    ax.set_xlabel('Mois')
    ax.set_ylabel('Nombre d\'événements')

    ax.set_title('Nombre d\'événements liés au terrorisme,'
                 ' par mois de Juin 2021 à Avril 2024, par pays',
                 fontsize=12)

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%b %Y')
    )

    ax.legend()

    fig.savefig(
        config['paths']['figures'] +
        "terrorist_events.png"
    )

    return fig


def generate_plot_events_by_terrorist_filiation(df, country, filename):
    """
    Génère un diagramme représentant le nombre d'événements liés
    au terrorisme par mois pour chaque organisation terroriste
    spécifiée dans les pays Mali, Niger et Burkina Faso.

    Args:
        df (DataFrame): DataFrame contenant les données des événements.
        country (str): Nom du pays pour lequel générer le diagramme
                        ('All' pour tous soit Sahel, Mali et Burkina Faso).
        filename (str): Nom du fichier pour sauvegarder le diagramme.

    Returns:
        None
    """

    # On crée une série qui permet de grouper par mois les évènements
    # liés à des violences perpétrées par chaque organisation terroriste mère,
    # au Mali, au Niger et au Burkina Faso
    terrorism_ei = df[
        (df['is_terrorist_group_related'] == 1) &
        (df['terrorist_group_filiation'] == 'Islamic State')
    ].groupby(pd.Grouper(key='event_date', freq='ME')).size()

    terrorism_aq = df[
        (df['is_terrorist_group_related'] == 1) &
        (df['terrorist_group_filiation'] == 'Al-Qaeda')
    ].groupby(pd.Grouper(key='event_date', freq='ME')).size()

    # On trace un diagramme les courbes correspondants aux évènements
    # liés à des organisationsterroristes selon leur filiation
    fig, ax = plt.subplots(figsize=(20, 8))

    ax.set_xlim(pd.Timestamp(config['dates']['start']),
                pd.Timestamp(config['dates']['end']))

    ax.plot(terrorism_ei.index, terrorism_ei.values, label='Etat Islamique')
    ax.plot(terrorism_aq.index, terrorism_aq.values, label='Al Qaida')

    ax.set_xlabel('Mois')
    ax.set_ylabel('Nombre d\'événements')

    if (country == "All"):
        ax.set_title('Nombre d\'événements liés au terrorisme,'
                     ' par mois de Juin 2021 par organisation terroriste,'
                     ' au Mali, au Niger et au Burkina Faso',
                     fontsize=12)
    else:
        ax.set_title(f'Nombre d\'événements liés au terrorisme, par mois'
                     f' de Juin 2021 par organisation terroriste au {country}',
                     fontsize=12)

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%b %Y')
    )

    ax.legend()

    fig.savefig(
        config['paths']['figures'] +
        filename
    )

    return fig


def generate_plot_wagner_events(df_country, country_name):
    """
    Génère un graphique comparant le nombre d'événements impliquant Wagner
    par mois de Juin 2021 à Avril 2024 par pays.

    Args:
        df_country (list): Liste de DataFrames contenant les données
                            des événements par pays.
        country_name (list): Liste des noms des pays correspondants.

    Returns:
        matplotlib.figure.Figure: Figure contenant le graphique généré.
    """

    # On crée notre graphique
    fig, ax = plt.subplots(figsize=(20, 8))

    ax.set_xlim(pd.Timestamp(config['dates']['start']),
                pd.Timestamp(config['dates']['end']))

    # On parcourt le dataframe de chaque pays
    for i, (df_country, country_name) in enumerate(
        zip(df_country_list, country_names)
    ):
        # On crée une série qui permet de grouper par mois les évènements liés
        # à des pmc russes dans chaque pays étudié
        wagner_events = df_country[
            df_country['is_pmc_related'] == 1
        ].groupby(pd.Grouper(key='event_date', freq='ME')).size()

        ax.plot(
            wagner_events.index,
            wagner_events.values,
            label=country_name
        )

    ax.set_xlabel('Mois')
    ax.set_ylabel('Nombre d\'événements')

    ax.set_title('Nombre d\'événements impliquant Wagner,'
                 ' par mois de Juin 2021 à Avril 2024, par pays',
                 fontsize=12)

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%b %Y')
    )

    ax.legend()

    fig.savefig(
        config['paths']['figures'] +
        "wagner_events.png"
    )

    return fig


# Nombre d'événements impliquant Wagner et des organisations terroristes
# par mois de Juin 2021 à Avril 2024
# au Mali
def generate_plot_wagner_events_mali(df_mali):
    # On crée une série qui permet de grouper par mois les évènements
    # liés à des violences perpétrées
    # par chaque organisation terroriste mère, au Mali
    events_mali = df_mali.groupby(
        pd.Grouper(key='event_date', freq='ME')
        ).size()

    terrorism_mali = df_mali[
        df_mali['is_terrorist_group_related'] == 1
    ].groupby(pd.Grouper(key='event_date', freq='ME')).size()

    pmc_mali = df_mali[
        df_mali['is_pmc_related'] == 1
    ].groupby(pd.Grouper(key='event_date', freq='ME')).size()

    terrorism_pmc_mali = df_mali[(
        df_mali['is_terrorist_group_related'] == 1) &
        (df_mali['is_pmc_related'] == 1)
    ].groupby(pd.Grouper(key='event_date', freq='ME')).size()

    # On crée notre graphique
    fig, ax = plt.subplots(figsize=(20, 8))

    ax.set_xlim(pd.Timestamp(config['dates']['start']),
                pd.Timestamp(config['dates']['end']))

    ax.plot(
        events_mali.index,
        events_mali.values,
        label="Tout type d'évènement"
    )

    ax.plot(
        terrorism_mali.index,
        terrorism_mali.values,
        label='Impliquant une organisation terroriste'
    )

    ax.plot(
        pmc_mali.index,
        pmc_mali.values,
        label='Impliquant Wagner'
    )

    ax.plot(
        terrorism_pmc_mali.index,
        terrorism_pmc_mali.values,
        label='Impliquant Wagner et une organisation terroriste'
    )

    ax.set_xlabel('Mois')
    ax.set_ylabel('Nombre d\'événements')

    ax.set_title('Nombre d\'événements impliquant Wagner et des organisations'
                 ' terroristes, par mois de Juin 2021 à Avril 2024, au Mali',
                 fontsize=12)

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%b %Y')
    )

    ax.legend()

    # On enregistre le graphique
    fig.savefig(
        config['paths']['figures'] +
        "wagner_events_mali.png"
    )

    return fig


# On récupère le dataframe créé à partir du jeu de données brut
df = pd.read_csv(config['paths']['processed_data'])
df['event_date'] = pd.to_datetime(df['event_date'], format='%Y-%m-%d')

# On crée un dataframe par pays
df_burkina_faso = df[df['country'] == 'Burkina Faso']
df_mali = df[df['country'] == 'Mali']
df_niger = df[df['country'] == 'Niger']
df_country_list = [df_burkina_faso, df_mali, df_niger]
country_names = config['countries']['all']

# On regroupe les données par mois et par type d'évènement
grouped_events = df.groupby(
    ['event_type', pd.Grouper(key='event_date', freq='ME')]
).size().reset_index(name='count')

# On regroupe les données par mois et par type d'évènement
# pour les évènements liés à des organisations terroristes
grouped_terrorist_events = df[df['is_terrorist_group_related'] == 1].groupby(
    ['event_type', pd.Grouper(key='event_date', freq='ME')]
).size().reset_index(name='count')


# On génère les différents graphiques de visualisation
generate_plot_events_vs_terrorist(df)

generate_plot_events_by_type(
    grouped_events,
    'events_by_type.png',
    terrorist_events=False
)

generate_plot_events_by_type(
    grouped_terrorist_events,
    'terrorist_events_by_type.png',
    terrorist_events=True
)

generate_plot_terrorist_events(df_country_list, country_names)

generate_plot_events_by_terrorist_filiation(
    df,
    'All',
    'terrorist_events_by_filiation.png')

generate_plot_events_by_terrorist_filiation(
    df_mali,
    'Mali',
    'terrorist_events_by_filiation_mali.png')

generate_plot_wagner_events(df_country_list, country_names)

generate_plot_wagner_events_mali(df_mali)

