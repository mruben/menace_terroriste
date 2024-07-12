import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
)
from src.config_loader import load_config
from src.features.utils import load_json_file

# Charger la configuration depuis config.yml
config = load_config()

def partie_projet():

      
    st.write("""
             
        Ce projet a été réalisé dans le cadre de notre formation en data 
        science via l'organisme [DataScientest](https://datascientest.com).
        L'objectif est de prédire les événements impliquant des groupes 
        terroristes au Niger, au Mali, au Burkina Faso et en particulier 
        dans chaque sous région respective, pour les 6 prochains mois.

        """)
    
    st.image('references/misc/carte_sahel.png', width=600)

    st.divider()
    st.header("Genèse du projet")
    st.divider()

    st.write("""
        Ce sujet a été proposé à l’équipe pédagogique de 
        [DataScientest](https://datascientest.com)
        par Matthieu Ruben N’Dongo, qui au delà de son intérêt personnel 
        pour la géopolitique et l’Afrique depuis de nombreuses années 
        (lecture de publications de recherche, d’ouvrages spécialisés, 
        suivi de débats) a collaboré avec une journaliste spécialisée 
        sur le Sahel pendant plusieurs mois, notamment sur la recherche 
        d’informations (via des méthodes d’OSINT) concernant les réseaux 
        d’influence Russes au Sahel et en Afrique de l’Ouest.
        """)
    st.write("""
        Aymone Soh et Maïté Crayes l'ont rejoint pour former l'équipe projet.
        """)
    
    st.divider()    
    st.header("ACLED")
    st.divider()

    st.write("""
        [ACLED](https://acleddata.com) est une organisation internationale 
        se définissant comme indépendante et impartiale, et à but non lucratif 
        qui recueille des données sur les conflits violents et les 
        protestations dans tous 
        les pays et territoires du monde. Elle a pour but de recueillir 
        et enregistrer des informations sur la violence politique, les 
        manifestations (émeutes et protestations) et d'autres événements 
        non violents importants sur le plan politique. La finalité étant 
        de saisir les modes, la fréquence et l'intensité de la violence 
        politique et des manifestations.

        """)
    
    st.header("Objectifs")
    st.write("""
        L'objectif de ce projet est de développer un outil qui permette de 
        prédire les événements impliquant des groupes terroristes au Niger, 
        au Mali,au Burkina Faso et en particulier dans chaque sous région 
        respective,pour les 6 prochains mois.  
        Le but est également d'étudier l’impact des PMC Russes (Wagner, 
        Redut, Africa Corps, etc) dans la lutte contre le terrorisme depuis
        leur arrivée dans ces trois pays Sahéliens.
        """)
    
    st.write("""
        Exemple: 
        - si un événement implique un groupe terroriste et des civiles, 
        nous pouvons être surs qu'il s'agit d'un attentat terroriste;
        - par contre si les deux acteurs sont un groue terroriste et l'armée, 
        nous ne pouvons pas savoir à priori qui est à l'orginine de l'attaque 
        et qui l'a subit.
        
        """)
    
    

def partie_donnees():

    st.title("Le jeu de données")

    st.write("""
        [ACLED](https://acleddata.com) permet (après la création d’un compte 
        sur leur plateforme) d’accéder à leur base de données pour extraire 
        les informations qui nous intéressent.
             
        Les données sont composées de 15075 lignes et 31 colonnes.
        """)
    
    # afficher le dataframe initial
    df = pd.read_csv(config['paths']['raw_data'])
    st.dataframe(df.head(10))
    
    if st.checkbox("Afficher le Describe du dataframe", key="checkbox_1"):
        st.dataframe(df.describe())

    st.write("""
        Correspondances des identifiants pour les colonnes inter1, inter2 et interaction:
        """)
    
    actor_type = load_json_file(
                config['paths']['json']['actor_type']
            )
    df_actors = pd.DataFrame([actor_type])

    # Transposer le DataFrame
    df_actors_transposed = df_actors.transpose()
    df_actors_transposed.columns = ['Acteur']
    df_actors_transposed.index.name = 'Identifiant'
    st.dataframe(df_actors_transposed)    