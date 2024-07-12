import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
)

from src.visualization.visualize_map import generate_map
from src.visualization.visualize_dataframe import (
    generate_plot_wagner_events,
    generate_plot_wagner_events_mali,
    generate_plot_events_vs_terrorist,
    generate_plot_events_by_terrorist_filiation,
    generate_plot_terrorist_events
)
from src.config_loader import load_config
config = load_config()


@st.cache_data
def load_data_display_plots():
    df = pd.read_csv(config['paths']['processed_data'])
    df['event_date'] = pd.to_datetime(df['event_date'], format='%Y-%m-%d')
    df_burkina_faso = df[df['country'] == 'Burkina Faso']
    df_mali = df[df['country'] == 'Mali']
    df_niger = df[df['country'] == 'Niger']
    df_country_list = [df_burkina_faso, df_mali, df_niger]
    country_names = config['countries']['all']

    return (
        df,
        df_burkina_faso,
        df_mali,
        df_niger,
        df_country_list,
        country_names
    )


@st.cache_data
def load_data_display_dataframe():
    past_data = pd.read_csv(config['paths']['past_data'])
    future_data = pd.read_csv(config['paths']['future_data'])
    predictions_future = pd.read_csv(
        config['paths']['predictions_future_data']
    )
    future_data['terrorist_events'] = predictions_future.round(0)

    return past_data, future_data


def display_dataframe(past_data, future_data):
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

    # On calcule la différence par rapport à la moyenne des 6 derniers mois
    future_data['diff'] = (
        future_data['terrorist_events'] -
        future_data['terrorist_events_last_6_months']
    )

    # On conserve uniquement les colonnes qui nous intéressent
    filtered_df = future_data[[
        'country',
        'admin1',
        'month',
        'terrorist_events',
        'terrorist_events_last_6_months',
        'diff'
    ]]

    # On convertit les chiffres en entiers
    filtered_df['diff'] = filtered_df['diff'].astype(int)

    filtered_df['terrorist_events'] = (
        filtered_df['terrorist_events'].astype(int)
    )

    filtered_df['terrorist_events_last_6_months'] = (
        filtered_df['terrorist_events_last_6_months'].astype(int)
    )

    # On remplace les noms des mois
    mois_mapper = {
        1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril', 5: 'Mai',
        6: 'Juin', 7: 'Juillet', 8: 'Août', 9: 'Septembre', 10: 'Octobre',
        11: 'Novembre', 12: 'Décembre'
    }

    filtered_df['month'] = filtered_df['month'].replace(mois_mapper)

    # On filtre par pays
    default_country = 'Tous les pays'
    selected_country = st.selectbox(
        'Filtrer par pays',
        [default_country] + list(filtered_df['country'].unique())
    )

    if selected_country != default_country:
        filtered_df = filtered_df[filtered_df['country'] == selected_country]

    # On filtre par région
    default_region = 'Toutes les régions'
    selected_region = st.selectbox(
        'Filtrer par région',
        [default_region] + list(filtered_df['admin1'].unique())
    )

    if selected_region != default_region:
        filtered_df = filtered_df[filtered_df['admin1'] == selected_region]

    # On filtre par mois
    default_month = 'Tous les mois'
    selected_month = st.selectbox(
        'Filtrer par mois',
        [default_month] + list(filtered_df['month'].unique())
    )

    if selected_month != default_month:
        filtered_df = filtered_df[filtered_df['month'] == selected_month]

    # On renomme les colonnes
    columns_name = {
        'country': 'Pays',
        'admin1': 'Région',
        'month': 'Mois',
        'terrorist_events': 'Événements terroristes',
        'terrorist_events_last_6_months': 'Moyenne des 6 derniers mois',
        'diff': 'diff'
    }

    filtered_df = filtered_df.rename(columns=columns_name)

    # On assigne le symbole en fonction de la différence
    filtered_df['diff'] = np.where(
        filtered_df['diff'] == 0,
        "=", np.where(filtered_df['diff'] > 0,
                      "+" + filtered_df['diff'].astype(str),
                      "-" + filtered_df['diff'].abs().astype(str)))

    # Fonction qui permet d'appliquer la couleur correspondante
    # en fonction du symbole de la valeur dans 'diff'
    def apply_styles(val):
        color = 'white'
        if val.startswith('+'):
            background_color = 'green'
        elif val.startswith('-'):
            background_color = 'red'
        else:
            background_color = ''
        return f'background-color: {background_color}; color: {color}'

    # On applique le style à la colonne 'diff'
    styled_df = filtered_df.style.applymap(apply_styles, subset=['diff'])

    # On affiche le dataframe sans l'index
    st.dataframe(
        styled_df,
        hide_index=True,
        width=None,
        use_container_width=True
    )


@st.cache_data
def display_map():
    fig_map = generate_map()
    st.plotly_chart(fig_map)


@st.cache_data
def display_metrics(past_data, future_data):
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

    # On calcule la différence par rapport à la moyenne des 6 derniers mois
    future_data['diff'] = (
        future_data['terrorist_events'] -
        future_data['terrorist_events_last_6_months']
    )

    # On convertit les chiffres en entiers
    future_data['diff'] = future_data['diff'].astype(int)

    future_data['terrorist_events'] = (
        future_data['terrorist_events'].astype(int)
    )

    future_data['terrorist_events_last_6_months'] = (
        future_data['terrorist_events_last_6_months'].astype(int)
    )

    # On groupe par pays et on calcule les agrégats
    grouped_df = future_data.groupby('country').agg(
        total=('terrorist_events', 'sum'),
        diff=('diff', 'sum'),
    ).reset_index()

    return grouped_df


def display_dashboard():
    st.write("""\n
        <div style="font-size: 16px;">
            Prédictions du nombre d'événements liés à des organisations
            terroristes d'Avril à Octobre 2024, au Mali, au Burkina Faso et
            au Niger
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2, col3 = st.columns((0.3, 0.3, 0.3), gap='medium')

    past_data, future_data = load_data_display_dataframe()

    df_metrics = display_metrics(past_data, future_data)

    with col1:
        st.metric(
            label="Mali",
            value=df_metrics[df_metrics.country == "Mali"]['total'],
            delta=int(df_metrics[df_metrics.country == "Mali"]['diff'].iloc[0])
        )

    with col2:
        st.metric(
            label="Burkina Faso",
            value=df_metrics[df_metrics.country == "Burkina Faso"]['total'],
            delta=int(
                df_metrics[df_metrics.country == "Burkina Faso"]['diff'].iloc[0]
            )
        )

    with col3:
        st.metric(
            label="Niger",
            value=df_metrics[df_metrics.country == "Niger"]['total'],
            delta=int(df_metrics[df_metrics.country == "Niger"]['diff'].iloc[0])
        )

    st.write("""\n
        <div style="font-size: 12px;">
            <i>(nombre total d'événements liés à des organisations terroristes,
            en comparatif de la moyenne des 6 derniers mois)</i>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns((2, 1), gap='medium')

    with col1:
        st.subheader('Visualisation des prédictions')

        # On crée les onglets
        tab1, tab2 = st.tabs(["Map", "DataFrame"])

        with tab1:
            display_map()

        with tab2:
            display_dataframe(past_data, future_data)

    with col2:
        st.subheader('Eléments de contexte')

        # On crée les onglets
        tab1, tab2 = st.tabs(["Terrorisme", "Wagner"])

        (
            df,
            df_burkina_faso,
            df_mali,
            df_niger,
            df_country_list,
            country_names
        ) = load_data_display_plots()

        with tab1:
            st.write("""
                Nombre d'événements par mois de Juin 2021 à Avril 2024 au Mali,
                    au Burkina Faso et au Niger
            """)

            fig1 = generate_plot_events_vs_terrorist(df)
            st.pyplot(fig1)

            st.divider()

            st.write("""
                    Nombre d'événements liés au terrorisme, par mois de Juin
                     2021 à Avril 2024 par organisation terroriste, au Mali,
                     au Niger et au Burkina Faso
            """)
            fig2 = generate_plot_events_by_terrorist_filiation(
                df,
                'All',
                'terrorist_events_by_filiation.png')
            st.pyplot(fig2)

            st.divider()

            st.write("""
                    Nombre d'événements liés au terrorisme,
                    par mois de Juin 2021 à Avril 2024, par pays
            """)

            fig3 = generate_plot_terrorist_events(
                df_country_list,
                country_names
            )
            st.pyplot(fig3)

        with tab2:
            st.write("""
                Nombre d'événements impliquant Wagner, par mois de Juin 2021
                    à Avril 2024, par pays
            """)
            fig1 = generate_plot_wagner_events(df_country_list, country_names)
            st.pyplot(fig1)

            st.divider()

            st.write("""
                Nombre d'événements impliquant Wagner et des organisations
                    terroristes, par mois de Juin 2021 à Avril 2024, au Mali
            """)

            fig2 = generate_plot_wagner_events_mali(df_mali)
            st.pyplot(fig2)
