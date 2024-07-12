import streamlit as st
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
)

from src.streamlit.pres_matthieu import (
    partie_modelisation,
    partie_ml,
    partie_conclusion
)

from src.streamlit.pres_aymone import (
    pre_processing,
    generate_plot
)

from src.streamlit.pres_maite import (
    partie_projet,
    partie_donnees
)

from src.streamlit.dashboard import display_dashboard


st.set_page_config(
    page_title="Menace terroriste au Sahel",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded")

# Création du menu latéral
menu = [
    "Le projet",
    "Le jeu de données",
    "Préparation des données",
    "Analyse exploratoire",
    "Modélisation",
    "Entraînement du modèle et prédictions",
    "Dashboard",
    "Conclusion"
]

st.sidebar.title('Analyse de la menace terroriste au Sahel')
choice = st.sidebar.radio("Sélectionnez une page", menu)


st.header('Analyse de la menace terroriste au Sahel', divider='rainbow')

# Affichage de la page correspondante en fonction du choix dans le menu
if choice == "Le projet":
    partie_projet()
elif choice == "Le jeu de données":
    partie_donnees()
elif choice == "Préparation des données":
    pre_processing()
elif choice == "Analyse exploratoire":
    generate_plot()
elif choice == "Modélisation":
    partie_modelisation()
elif choice == "Entraînement du modèle et prédictions":
    partie_ml()
elif choice == "Dashboard":
    display_dashboard()
elif choice == "Conclusion":
    partie_conclusion()

st.sidebar.header('Auteurs')

st.sidebar.write("""
    [Matthieu Ruben N’Dongo](https://www.linkedin.com/in/matthieuruben/?originalSubdomain=fr)
        """)

st.sidebar.write("""
    [Aymone Soh](https://www.linkedin.com/in/aymone-cynthia-soh-49144250/)
        """)

st.sidebar.write("""
    [Maïté Crayes](https://www.linkedin.com/in/maïté-crayes-422b9188/)
        """)
