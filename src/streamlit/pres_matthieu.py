import streamlit as st
import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
)

from src.features.build_features import (
    build_past_data,
    past_data_transformation
)

from src.visualization.visualize_predictions import (
    generate_plot_predictions_vs_observed,
    generate_plot_predictions
)

from src.config_loader import load_config
config = load_config()


@st.cache_data
def load_data_display_plots():
    past_data = pd.read_csv(config['paths']['past_data'])
    predictions_past = pd.read_csv(config['paths']['predictions_past_data'])
    future_data = pd.read_csv(config['paths']['future_data'])
    predictions_future = pd.read_csv(
        config['paths']['predictions_future_data']
    )
    future_data['terrorist_events'] = predictions_future.round(0)
    df_test = pd.read_csv(config['paths']['X_test'])
    y_test = pd.read_csv(config['paths']['y_test'])
    all_data = pd.concat([past_data, future_data], axis=0, ignore_index=True)

    return predictions_past, df_test, y_test, all_data


def partie_modelisation():
    st.title("Modélisation")

    # INTRODUCTION
    st.write("""
             Dans cette partie, nous allons d'abord vous présenter le processus
             utilisé pour créer les variables explicatives qui serviront à
             prédire notre variable cible, soit le nombre d'évènements liés à
             des organisations terroristes pour les 6 prochains mois, pour
             chaque région du Mali, du Burkina Faso et du Niger.
    """)

    st.write("""
        Nous avons identifié que nous étions dans un problème de Machine
             Learning correspondant à une régression non linéaire, nous
             entraînerons donc par la suite différents modèles de ML afin de
             calculer les prédictions.
    """)

    st.divider()

    # PAST_DATA
    st.subheader("Construction du dataframe 'past_data'")
    st.divider()

    st.write("""
        Nous avons d'abord créé un dataframe 'past_data' qui contient les
             données historiques, soient celles des 3 dernières années. Nous
             avons effectué les étapes suivantes pour le construire.
    """)

    st.write("\n")

    st.markdown('<u><b>Etape 1 : Regroupement des évènements<b></u>',
                unsafe_allow_html=True)

    if st.checkbox("Afficher le détail", key=1):
        st.markdown("""
            - Les événements sont regroupés par **pays**, **région**,
                    **année**, et **mois**.
            - Résultat : un dataframe d’un peu plus de **800 lignes**.
        """)

        st.write("\n")

    st.markdown('<u><b>Etape 2 : Calcul des variables catégorielles<b></u>',
                unsafe_allow_html=True)

    if st.checkbox("Afficher le détail", key=2):
        st.markdown("""
            1. Pour chaque région et mois :
            - **actors** : Comptabilisation des occurrences pour chaque
                    catégorie (ex : Forces gouvernementales, groupes rebelles,
                    etc.)
            - **event_type** : Comptabilisation des occurrences pour chaque
                    type d'événement (ex : violences contre des civils,
                    batailles, etc.)
            2. Simplification des variables :
            - La variable **interaction_type** (décrivant les deux acteurs
                    impliqués dans un événement) est supprimée car les
                    informations sont déjà présentes dans les différents
                    acteurs
            3. Ajout de nouvelles colonnes :
            - **terrorist_events** : Nombre d'événements liés à des
                    organisations terroristes, calculé à partir de la variable
                    `is_terrorist_related` du dataframe initial
            - Des lags (décalages) sur **6 mois** sont ajoutés pour les
                    variables **total_events** et **terrorist_events** afin de
                    capturer les tendances sur cette période
        """)

        df_past_data = pd.read_csv(config['paths']['past_data'])
        st.dataframe(df_past_data)

        st.write("\n")

    st.markdown(
        '<u><b>Etape 3 : Séparation et transformation des données<b></u>',
        unsafe_allow_html=True
    )

    if st.checkbox("Afficher le détail", key=3):
        st.markdown("""
            1. **Séparation en données d’entraînement et de test**
            - Le dataframe est trié par ordre croissant pour que les données de
                    test correspondent aux 6 derniers mois
            - Utilisation de la fonction `train_test_split` avec le paramètre
                    `shuffle=False`

            2. **Transformation des données**
            - **Variables numériques** : Standardisation à l’aide d’un
                    **StandardScaler**
            - **Variables catégorielles** (régions et pays) : Dichotomisation
                    pour une meilleure interprétation par les modèles de
                    Machine Learning

            3. **Capturer l'aspect temporel**
            - Transformation trigonométrique des variables **month** et
                    **year** pour capturer la périodicité et la saisonnalité
                    des relations temporelles
        """)

        code = """
        # Transformation trigonométrique de 'year' et 'month' pour
        # la saisonnalité
        X_train_encoded['month_sin'] = np.sin(
            2 * np.pi * X_train_encoded['month'] / 12
        )

        X_train_encoded['month_cos'] = np.cos(
            2 * np.pi * X_train_encoded['month'] / 12
        )

        X_train_encoded['year_sin'] = np.sin(
            2 * np.pi * (
                (X_train_encoded['year'] - X_train_encoded['year'].min()) /
                (X_train_encoded['year'].max() - X_train_encoded['year'].min() + 1)
            )

        X_train_encoded['year_cos'] = np.cos(
            2 * np.pi * (
                (X_train_encoded['year'] - X_train_encoded['year'].min()) /
                (X_train_encoded['year'].max() - X_train_encoded['year'].min() + 1)
            )
        """
        st.code(code)

        st.write("\n")

        st.write("""
            Les transformations et ajouts effectués permettent une meilleure
                analyse des tendances et des occurrences d'événements,
            notamment ceux liés au terrorisme. La variable `terrorist_events`
                 sera utilisée comme la variable cible pour nos modèles
                 prédictifs. Cette approche structurée permet d'exploiter au
                 maximum les données disponibles pour des analyses précises et
                 pertinentes.\nLa séparation soigneuse des données garantit
                 que les modèles seront testés sur des données récentes et
                 réalistes, tandis que les transformations trigonométriques
                 enrichissent l'analyse en intégrant des aspects saisonniers
                 et périodiques.
        """)

    st.divider()

    # FUTURE_DATA
    st.subheader("Construction du dataframe 'future_data'")
    st.divider()

    st.write("""
        Nous souhaitons construire un dataframe appelé **future_data** pour
             les 6 prochains mois et pour chaque région (seules variables que
             nous connaissons). **future_data** doit être identique au
             dataframe **past_data** en termes de variables pour utiliser
            le même modèle entraîné dessus.
    """)

    st.markdown('<u><b>Etape 1 : Remplissage des valeurs des variables'
                ' explicatives<b></u>',
                unsafe_allow_html=True)

    if st.checkbox("Afficher le détail", key=4):
        st.markdown("""
            - Défi : Comment remplir les valeurs des variables explicatives
                    (acteurs impliqués, type d’événement, etc.) sachant qu’il
                    est impossible de le savoir à l’avance ?
            - Solution : Calcul des moyennes à partir de **past_data** pour
                    remplir **future_data**
                1. Pour un mois et une région donnés dans **future_data**,
                    calcul de la moyenne pour chaque variable explicative à
                    partir des données historiques **past_data** pour la même
                    combinaison mois/région
                2. Si la correspondance mois/région n’existe pas (absence de
                    données passées), calcul de la moyenne à partir des données
                    des autres mois disponibles pour cette région (ceci
                    concerne environ une dizaine de lignes)
        """)

    st.markdown('<u><b>Etape 2 : Transformation du dataframe futur<b></u>',
                unsafe_allow_html=True)

    if st.checkbox("Afficher le détail", key=5):
        st.markdown("""
            - **future_data** subit les mêmes transformations que
                    **past_data** :
                - **Normalisation** des variables numériques.
                - **Dichotomisation** des variables catégorielles (régions et
                    pays)
                - **Transformations trigonométriques** des variables **month**
                    et **year**
        """)

        df_future_data = pd.read_csv(config['paths']['future_data'])
        st.dataframe(df_future_data)

        st.write("\n")

    st.write("""
        Le dataframe **future_data** ainsi construit et transformé nous permet
             de prédire les événements futurs en utilisant le modèle entraîné
             sur **past_data**. Cette méthodologie assure la cohérence entre
             les données historiques et les prévisions, tout en tenant compte
             des tendances et des caractéristiques
        spécifiques à chaque région et période.
    """)


def partie_ml():
    st.title("Entraînement du modèle et prédictions")

    st.write("""
        Une fois ces étapes de construction et transformation de nos
             dataframes terminées, nous avons entraîné différents modèles de
             régression avec différents paramètres dans l’optique de
             sélectionner le plus performant.
    """)

    st.write("""
        L'intégration systématique de la recherche de paramètres optimaux, de
             la réduction de dimension et des techniques d'ensemble (Bagging
             et Boosting) nous a permis d'explorer plusieurs approches pour
             améliorer les performances de nos modèles. Cette méthodologie
             rigoureuse assure que les modèles sont bien adaptés aux données
             et maximisent leur potentiel prédictif.
    """)

    st.divider()

    # ENTRAINEMENT MODELE
    st.subheader("Choix du meilleur modèle")
    st.divider()

    st.markdown("""
        1. **Modèles entraînés**
            - **Random Forest**
            - **XGBoost**
            - **SVM**

        2. **Techniques utilisées dans l'entraînement**
        - **Grid Search** pour la recherche des paramètres optimaux
        - **Réduction de dimension** avec :
            - PCA (Analyse en Composantes Principales)
            - LDA (Analyse Discriminante Linéaire)
            - UMAP (Uniform Manifold Approximation and Projection)
        - **Ensembles de techniques** pour améliorer les performances :
            - **Bagging**
            - **Boosting**
    """)

    st.write("\n")
    if st.checkbox("Afficher le détail des scores", key=6):
        st.image('references/misc/scores.png')

    st.divider()

    # CHOIX DU MEILLEUR MODELE
    st.subheader("Entraînement d'un Random Forest (avec ou sans LDA)")
    st.divider()

    # On récupère nos jeux d'entrainement et de test
    past_data = build_past_data()

    (
        X_train_encoded,
        X_test_encoded,
        y_train,
        y_test,
        scaler
    ) = past_data_transformation(past_data)

    use_lda = st.checkbox("Utiliser la réduction de dimension LDA")

    if use_lda:
        # On applique une réduction de dimension LDA sur les données
        # de test et d'entraînement
        lda = LDA()
        X_train_lda = lda.fit_transform(X_train_encoded, y_train)
        X_test_lda = lda.transform(X_test_encoded)
    else:
        X_train_lda = X_train_encoded
        X_test_lda = X_test_encoded

    # On entraine un modèle RF avec la LDA
    n_estimators = st.slider("Nombre d'estimators", 10, 200, 100)
    max_depth = st.slider("Profondeur maximale", 1, 20, 5)

    model = RandomForestRegressor(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth)

    model.fit(X_train_lda, y_train)

    # On prédit et on affiche les résultats
    accuracy = model.score(X_test_lda, y_test)
    st.markdown(f"""
    <div style="
        display: inline-block;
        padding: 10px 20px;
        border-radius: 25px;
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        text-align: center;">
        Accuracy: {accuracy:.4f}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # PREDICTIONS
    st.subheader("Prédictions")
    st.divider()

    predictions_past, df_test, y_test, all_data = load_data_display_plots()

    st.write("""
        Les prédictions que nous avons obtenues semblent cohérentes et
             réalistes. La courbe des prédictions de novembre 2023 à avril
             2024 par exemple suit celle des observations réelles, tant en
             terme d'ordre de grandeurs que de tendances.
    """)

    st.divider()

    st.markdown('<u><b>Visualisation des prédictions des 6 derniers mois<b></u>',
                unsafe_allow_html=True)

    st.write("Evénements liés à des organisations terroristes, par mois"
             " de Novembre 2023 à Avril 2024, au Mali, au Burkina Faso"
             " et au Niger")

    fig1 = generate_plot_predictions_vs_observed(
        df_test, y_test, predictions_past
    )
    st.pyplot(fig1)

    st.divider()

    st.markdown("<u><b>Evénements passés et prédits liés à des organisations"
                " terroristes, par mois de juin 2021 à Octobre 2024"
                " au Mali, au Burkina Faso et au Niger<b></u>",
                unsafe_allow_html=True)

    fig2 = generate_plot_predictions('All', all_data, 'country')
    st.pyplot(fig2)


def partie_conclusion():
    st.title("Conclusion")

    st.divider()

    st.subheader("Bilan")
    st.divider()

    st.write("""
        L'objectif de développer un outil qui permette de prédire les
        événements impliquant des groupes terroristes au Niger, au Mali,
        au Burkina Faso et en particulier dans chaque sous région,
        pour les 6 prochains mois, a été atteint.
        Compte tenu du sujet sensible de notre problématique, du fait que nous
             étions en même temps en train de nous former sur les technologies
             nécessaires à la réalisation du projet, et du temps restreint
             dans lequel nous l’avons réalisé, les graphiques qui ont été
             générés ainsi que la carte de visualisation des prédictions
             constitue un prototype à un projet plus ambitieux. En l’état, ils
             peuvent être utilisés à titre informatif par des chercheurs, des
             journalistes ou des personnes chargées de gestion de crises ou de
             questions sécuritaires au Sahel.
    """)

    st.divider()

    st.subheader("Perspectives")
    st.divider()

    st.write("""
        Comme piste d’amélioration, nous pouvons imaginer un modèle qui prend
        en compte plus de complexité, notamment concernant le contexte
        politique, qui capture de manière plus précise les tendances
        temporelles. 
       """)
    
    st.write("""
        Nous pouvons également adapter notre modèle pour
        qu’il prenne en compte des données provenant d’ACLED à partir de
        n’importe quelle date afin de générer des prédictions pour les 6
        prochains mois.
    """)

    st.write("""
        Etant donné que le Burkina Faso est le pays pour lequel nous avons le
        plus de données et en observant les résidus, nous pensons que le
        poids de ce pays est fort pour l'entrainement du modèle. Par
        conséquent il est moins performant pour le Niger et le Mali. Nous
        pourrions donc envisager d'entrainer le modèle pour chaque pays
        séparément, ou même faire un modèle par pays.
    """)

    st.divider()

    st.subheader("Problèmes rencontrés")
    st.divider()

    st.markdown("""
        - Construction du dataframe futur
        - Choix des variables explicatives
        - Intégration de la formation en parallèle du projet
    """)

    st.divider()

    st.subheader("Les points forts de l’équipe")
    st.divider()

    st.markdown("""
        - Un investissement à 100% de chacun des membres
        - Complémentarité des compétences
        - Une volonté de trouver du temps et de la disponibilité pour se réunir
        - De gros efforts de communications ont été faits ce qui a permis
                d’avancer de manière efficace, et toujours dans la
                bienveillance
        - L’ensemble des décisions ont été prises de manière collégiale, après
        discussions.
    """)
