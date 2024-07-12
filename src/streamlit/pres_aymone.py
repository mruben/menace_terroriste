import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt  # Corrected to pyplot
import seaborn as sns


import numpy as np

# Ajouter le chemin du fichier converti au système de chemin
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../notebooks'))
)

from src.visualization.visualize_linearite import linearite

from src.features.data_preprocessing import (
    load_dataset,
    columns_filtering,
    add_new_columns,
    delete_columns,
    fill_nan_values,
)

from src.visualization.visualize_dataframe import (
    generate_plot_events_vs_terrorist,
    generate_plot_events_by_type,
    generate_plot_terrorist_events,
    generate_plot_events_by_terrorist_filiation,
    generate_plot_wagner_events,
    generate_plot_wagner_events_mali,
)
from src.config_loader import load_config

# Charger la configuration depuis config.yml
config = load_config()

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


def pre_processing():
    st.header("Préparation des données")

    st.subheader("Introduction")
    
    st.write("""Afin de rendre notre dataset exploitable pour une analyse 
             exploratoire, nous avons
             effectué quelques modifications:
            """)
    st.write("")

    st.subheader("1. filtrage des lignes")

    st.write("")

    st.write("""       
             Afin de recueillir seulement les violences politiques, nous 
             avons filtré les lignes du Dataframe pour ne garder que les types
             d'évènements: 
             
             -"Battles": Ce sont des interactions violentes entre deux groupes 
                organisés, se déroulant à un moment et un lieu donnés. 
                Elles impliquent des combats directs.
             
             -"Explosions/Remote violence": événements de type 
                « explosions/violence à distance » comme des incidents au 
                cours desquels l’une des parties utilise des types d’armes qui,
                par nature, sont à distance et largement destructeurs (sans 
                confrontation directe).
             
             -"Violence against civilians": Ces événements impliquent des 
                violences infligées par un groupe armé organisé contre des 
                non-combattants non armés. 
             
             -"Strategic developments": Contrairement aux autres types 
                d'événements, ceux-ci incluent des actions significatives qui 
                ne se limitent pas à la violence physique directe. Ils 
                englobent des campagnes de recrutement, des pillages, 
                des incursions, l'annonce de pourparlers de paix, 
                l'arrestation de figures importantes, 
                l'établissement de quartiers généraux et des transferts 
                de territoire non violents. Ces événements capturent 
                des dynamiques stratégiques cruciales pour comprendre 
                les évolutions du conflit au-delà des confrontations armées.
                """)
    
    df = load_dataset(config['paths']['raw_data'])
    
    st.write("")
    
    df_filtered = columns_filtering(df)
    df_with_new_columns = add_new_columns(df_filtered)
    df_reduced = delete_columns(df_with_new_columns)
    df_filled_nan = fill_nan_values(df_reduced)

    if st.checkbox("Afficher le dataframe 1", key="checkbox_1"):
        st.dataframe(df_filtered)
        st.write("La dimension du dataframe 1 est de:", df_filtered.shape)
        
    st.write("""Ainsi nous sommes passés d'un dataset de 15075 lignes et 31 
             colonnes à un de 14054 lignes et 31 colonnes.Supprimant ainsi 7% des lignes""")
    
    st.write("")

    st.subheader("2. Ajout de colonnes")

    st.write("")

    st.write("""
             Ensuite, après avoir convertit la colonne "event_date"
             en type datetime, nous avons ajouté huit (8) colonnes:
             - les colonnes 'month' et 'day' pour le
             mois et le jour de l'évènement, 
             - les colonnes "actor1_type" et
             "actor2_type" correspondant aux types d'acteurs, 
             - une colonne 'interaction-type' correspondant aux types 
             d'interactions,
             - une colonne 'is_terrorist_group_related' indiquant si
             un acteur est une organisation terroriste.
             - une colonne 'terrorist_group_filiation' indiquant l'organisation
             mère de l'organisation terroriste
             - une colonne 'is_pmc_related' indiquant si un acteur est
             une PMC russe.
             """)

    st.write("")

    if st.checkbox("Afficher le dataframe 2", key="checkbox_2"):
        st.dataframe(df_with_new_columns)
        st.write("La dimension du dataframe est de 2:", df_with_new_columns.shape)
        
    st.write("""
             Ainsi, nous sommes passés à un dataframe de 31 colonnes
             à un de 39 colonnes.
             """)
    
    st.write("")
    st.subheader("3. Suppression de colonnes")
    st.write("")

    st.write("""
             Puis, nous avons procédé au nettoyage du dataframe 2
             en supprimant 25 colonnes dont nous n'avions pas besoin :
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
            """)

    st.write("")

    if st.checkbox("Afficher le dataframe 3", key="checkbox_3"):
        st.dataframe(df_reduced)
        st.write("La dimension du datframe 3 est de:", df_reduced.shape)
        
    st.write("""
             Ainsi, nous sommes passés de 39 à 14 colonnes.
             """)
    
    st.write("")
    st.subheader("4. Remplissage des valeurs manquantes.")
    st.write("")

    st.write("""
            Enfin, dans la colonne "actor2_type", nous avons décidé de 
            remplacer les valeurs manquantes par "None" pour indiquer 
            qu'il y a un seul acteur et non un manque de valeur.
             """)

    st.write("")

    if st.checkbox("Afficher le dataframe 4", key="checkbox_4"):
        st.dataframe(df_filled_nan)
        st.write("La dimension du dataframe 4 est toujours de:", df_filled_nan.shape)

    st.write("""
            Ainsi, après toutes ces modifications, on obtient un dataset 
            da 14054 lignes d'évènements et 14 colonnes (7% des lignes ont été 
            supprimées). 

             """)
    
    # st.divider()
    
    # st.subheader("OOOOOOO")
    
    # st.divider()


    # st.markdown('<u><b>Etape 2: pour écrire les titres des étapes en gras <b></u>',
    #             unsafe_allow_html=True)


@st.cache_data
def generate_plot_events_vs_terrorist_cached(df):
    return generate_plot_events_vs_terrorist(df)


@st.cache_data
def generate_plot_events_by_type_cached(grouped_events, filename, terrorist_events=False):
    return generate_plot_events_by_type(grouped_events, filename, terrorist_events)


@st.cache_data
def generate_plot_terrorist_events_cached(df_country_list, country_names):
    return generate_plot_terrorist_events(df_country_list, country_names)


@st.cache_data
def generate_plot_events_by_terrorist_filiation_cached(df, country, filename):
    return generate_plot_events_by_terrorist_filiation(df, country, filename)


@st.cache_data
def generate_plot_wagner_events_cached(df_country_list, country_names):
    return generate_plot_wagner_events(df_country_list, country_names)


@st.cache_data
def generate_plot_wagner_events_mali_cached(df_mali):
    return generate_plot_wagner_events_mali(df_mali)


def generate_plot():
    #st.header("Analyse exploratoire de la menace terroriste au Sahel")
    st.subheader("1. Etude de la menace terroriste ")

    st.write("""Ce graphique 1 représente les tendances et fréquences 
             mensuelles des événements de confits armés totaux et ceux liés 
             au terrorisme sur la période étudiée.
             """)
    fig1 = generate_plot_events_vs_terrorist_cached(df)
    st.pyplot(fig1)
    
    st.write(""" On remarque que les conflits liés au terrorisme contribuent 
             à 75% du nombre de conflits armés totaux dans cette zone.""")

    st.write("")
    st.write("")
    
    st.write("""Ce graphique 2 représente le nombre mensuel d'évènements liés 
             au terrorisme par type d'évènements dans ces 3 pays.
             """)
    fig2 = generate_plot_events_by_type_cached(
        grouped_terrorist_events,
        'terrorist_events_by_type.png',
        terrorist_events=True
    )
    st.pyplot(fig2)
    
    st.write(""" On remarque que les évènements de type "Battles", "strategic 
             development", "Violence againts civilians" sont les plus fréquents 
             lorsqu'un des acteurs est lié au terrorisme.
             """)
    
    st.write("")
    st.write("")

    st.write("""Ce graphique 3 représente le nombre mensuel d'évènements de 
             conflits armés liés au terrorisme dans chacun de ces 3 pays.""")
    fig3 = generate_plot_terrorist_events_cached(
        df_country_list, 
        country_names
    )
    st.pyplot(fig3)
    st.write("""Tout au long de la période étudiée, on remarque une légère 
             hausse de ces conflits au Mali et au Niger.
             Alors qu'au 
             Burkina Faso, depuis le début de l'année 2022,
             la fréquence de ces conflits diminue progressivement.
             Néanmoins, le Burkina Faso reste le pays le plus touché par 
             les conflits impliquant au moins une organisation liée au 
             terrorisme.""") 
  
    st.write("")
    st.write("")
    
    st.subheader("2. Etude de l'impact des PMC sur la situation sécuritaire.")
    
    st.write(" ")
    st.write("""
             Les sociétés militaires privées (PMC) russes, notamment le 
             groupe Wagner, sont présentes au Mali depuis la fin de 
             l'année 2021, au Burkina Faso depuis la fin de l'année 2022, et
             au Niger depuis avril 2024. Le Mali est le pays où la durée 
             d'installation des PMC (Wagner) est la plus significative. 
             Pour nous, il est donc plus pertinent de circonscrire l'étude de 
             l'impact des PMC russes à ce pays.

             Ce graphique 6 représente la progression des conflits impliquant 
             Wagner au Mali.""")
    
    fig6 = generate_plot_wagner_events_cached(
        df_country_list, 
        country_names
    )
    st.pyplot(fig6)
    
    st.write("""Au cours de ces 2 années de présence, le nombre d'évènements
             impliquant Wagner est en constante progression, atteingnant plus 
             60 événements pas mois en janvier 2024.""")    
    
    st.write("")
    st.write("")
    
    st.write("""Ce graphique 7 représente pour le Mali, le nombre de conflits 
                armés totaux (en bleu), ceux impliquant une organisation 
             liée au terrorisme (en orange), ceux impliquant Wagner (en vert) 
             et ceux impliquant Wagner et 
             une organisation liée au terrorisme (en rouge).""")
    fig7 = generate_plot_wagner_events_mali_cached(
        df_mali
    )
    st.pyplot(fig7)
    
    st.write("""Sur la période etudiée les conflits opposants
             Wagner à une organisation liée au terrorisme ne constituent pas 
             l'essentiel des conflits dans lequels Wagner s'implique.
             Depuis son arrivée, Wagner a mené seulement 25,7% de ses 
             combats contre ce type d'organisation.""")
    
    st.write(""" En conclusion, au Mali, sur toute la période étudiée, nous ne 
             remarquons
             pas un effet significatif de la présence de Wagner 
             sur la baisse de la fréquence des événements liés au 
             terrorisme.""")
