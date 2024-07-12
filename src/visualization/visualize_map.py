import pandas as pd
import geopandas as gpd
import plotly.express as px
from src.visualization.utils import convert_to_lowercase_and_replace_spaces
from src.features.utils import load_json_file
from src.config_loader import load_config
config = load_config()


def generate_geodataframe(df, mapping_dic, country):
    """
    Génère un GeoDataFrame en fusionnant les données géographiques
    avec les prédictions pour un pays spécifié.

    Args:
        df (DataFrame): DataFrame contenant les prédictions pour le pays.
        mapping_dic (dict): Dictionnaire de mapping pour associer les noms
                            des régions aux identifiants correspondants.
        country (str): Nom du pays pour lequel les données géographiques
                        doivent être chargées.

    Returns:
        GeoDataFrame: GeoDataFrame combinant les données géographiques
                    (shapefile) et les prédictions.
    """

    # On charge les fichiers .shp pour les sous-régions du pays rentré
    # en paramètre
    country_path = convert_to_lowercase_and_replace_spaces(country)
    gdf = gpd.read_file(config['paths']['shp'][country_path])

    # Pour chaque admin1 afin d'identifier on associe au nom
    # l'identifiant correspondant
    filtered_df = df[df['country'] == country]

    # On applique le mapping à notre dataframe qui contient les données pour
    # le pays donné en paramètre
    filtered_df['region_id'] = filtered_df['admin1'].map(mapping_dic)

    # On fusionne les données géographiques avec les prédictions
    gdf_combined = gdf.merge(
        filtered_df,
        how='left',
        left_index=True,
        right_on='region_id'
    )

    return gdf_combined


def generate_map():
    """
    Génère une carte choroplèthe interactive visualisant les prédictions
    d'événements impliquant des organisations terroristes pour les 6 prochains
    mois, par région pour les pays cibles (Mali, Niger, Burkina Faso).

    Returns:
        plotly.graph_objs._figure.Figure: Figure contenant la carte générée.
    """
    # Charger la configuration depuis config.yml

    # On récupère les jeux de données générés par le modèle de ML
    past_data = pd.read_csv(config['paths']['past_data'])
    future_data = pd.read_csv(config['paths']['future_data'])
    predictions_future = (
        pd.read_csv(config['paths']['predictions_future_data'])
    )

    # On ajoute les prédictions à notre dataframe future_data
    future_data['terrorist_events'] = predictions_future.round(0)

    # On souhaite ajouter une colonne 'terrorist_events_last_6_months'
    # à 'future_data' qui contient la moyenne de 'terrorist_events'
    # des 6 derniers mois pour chaque mois et sous-région
    # Convertir 'year' et 'month' en une seule colonne datetime
    # pour faciliter le tri
    past_data['date'] = pd.to_datetime(
        past_data[['year', 'month']].assign(day=1)
    )
    future_data['date'] = pd.to_datetime(
        future_data[['year', 'month']].assign(day=1)
    )

    # On trie les données par 'admin1' et 'date'
    past_data = past_data.sort_values(by=['admin1', 'date'])
    future_data = future_data.sort_values(by=['admin1', 'date'])

    # On calcule la moyenne mobile sur les 6 derniers mois dans past_data
    past_data['terrorist_events_last_6_months'] = (
        past_data.groupby('admin1')['terrorist_events']
        .transform(lambda x: x.rolling(window=6, min_periods=1).mean().round())
    )

    # On obtient les dernières valeurs de 'terrorist_events_last_6_months'
    # pour chaque 'admin1'
    last_6_months_avg = (
        past_data.groupby('admin1')
        .apply(lambda x: x.set_index('date').resample('M').last()
               .ffill().iloc[-1]['terrorist_events_last_6_months'])
    )

    # On ajoute cette information à future_data
    future_data['terrorist_events_last_6_months'] = (
        future_data['admin1'].map(last_6_months_avg)
    )

    # Ensuite on essaie d'identifier les sous-régions de chaque pays
    mapping_mali = load_json_file(
                config['paths']['json']['admin1_mali']
            )

    gdf_combined_mali = generate_geodataframe(
        future_data,
        mapping_mali,
        "Mali"
    )

    mapping_niger = load_json_file(
                config['paths']['json']['admin1_niger']
            )

    gdf_combined_niger = generate_geodataframe(
        future_data,
        mapping_niger,
        "Niger"
    )

    mapping_burkina = load_json_file(
                config['paths']['json']['admin1_burkina_faso']
            )

    gdf_combined_burkina = generate_geodataframe(
        future_data,
        mapping_burkina,
        "Burkina Faso"
    )

    # On fusionne les GeoDataFrames de tous les pays en un seul GeoDataFrame
    gdf_combined = pd.concat([
        gdf_combined_niger,
        gdf_combined_mali,
        gdf_combined_burkina
    ], axis=0, ignore_index=True)

    # On vérifie que nos données sont bien du type attendu pour
    # l'affichage sur notre map
    gdf_combined['region_id'] = gdf_combined['region_id'].astype(int)
    gdf_combined['month'] = gdf_combined['month'].astype(int)
    gdf_combined['terrorist_events'] = (
        gdf_combined['terrorist_events'].astype(int)
    )
    gdf_combined['terrorist_events_last_6_months'] = gdf_combined[
        'terrorist_events_last_6_months'
    ].astype(int)
    gdf_combined['admin1'] = gdf_combined['admin1'].astype(str)

    # On conserve uniquement les colonnes nécessaires
    columns_to_keep = [
        'geometry',
        'region_id',
        'month',
        'terrorist_events',
        'terrorist_events_last_6_months',
        'admin1'
    ]

    gdf_combined = gdf_combined[columns_to_keep]

    # On remplace les noms des mois
    mois_mapper = {
        1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril', 5: 'Mai',
        6: 'Juin', 7: 'Juillet', 8: 'Août', 9: 'Septembre', 10: 'Octobre',
        11: 'Novembre', 12: 'Décembre'
    }

    gdf_combined['month'] = gdf_combined['month'].replace(mois_mapper)

    # On crée une map qui permettra de visualiser pour chaque mois,
    # chaque sous-région des 3 pays cibles le nombre d'évènements
    # liés à des organisations terroristes
    fig = px.choropleth_mapbox(
        gdf_combined,
        geojson=gdf_combined.geometry.__geo_interface__,
        locations=gdf_combined.index,
        color='terrorist_events',
        animation_frame='month',
        hover_data={
            'region_id': False,
            'admin1': True,
            'month': True,
            'terrorist_events': True,
            'terrorist_events_last_6_months': True
        },
        title=(""),
        color_continuous_scale='Plasma',
        mapbox_style="carto-positron",
        center={"lat": 15, "lon": -1.5},
        zoom=5,
        opacity=0.5,
        labels={
            'month': 'mois',
            'admin1': 'région',
            'terrorist_events': 'Événements terroristes',
            'terrorist_events_last_6_months': '6 derniers mois'
            }
    )

    # On ajuste la taille de la carte
    fig.update_layout(
        height=800,
        width=1000
    )

    # On affiche la carte interactive
    # fig.show()

    return fig
