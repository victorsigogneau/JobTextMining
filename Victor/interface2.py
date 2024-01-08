import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Fonction pour créer un nuage de mots avec une condition sur contratId
def generer_nuage_de_mots_pour_metier(contrat_id):
    # Ajoutez le code pour générer un nuage de mots avec des données fictives pour le contrat_id spécifié
    wordcloud_data = {'Python': 10, 'Java': 8, 'JavaScript': 6}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

# Fonction pour créer une carte fictive de la France avec une condition sur contratId
def generer_carte_france_pour_metier(contrat_id):
    # Ajoutez le code pour générer une carte fictive de la France avec des données fictives pour le contrat_id spécifié
    france_data = pd.DataFrame({
        'Région': ['Île-de-France', 'Auvergne-Rhône-Alpes', 'Provence-Alpes-Côte d\'Azur', 'Occitanie', 'Hauts-de-France'],
        'Valeur': [100, 80, 60, 40, 20]
    })
    fig_france = px.choropleth(france_data, 
                               locations='Région',
                               locationmode='country names',
                               color='Valeur',
                               color_continuous_scale='Viridis',
                               title=f'Carte fictive de la France pour le métier : {contrat_id}')
    return fig_france

# Fonction pour créer un Barplot avec une condition sur contratId
def generer_barplot_pour_metier(contrat_id):
    # Ajoutez le code pour générer un Barplot avec des données fictives pour le contrat_id spécifié
    data_barplot = pd.DataFrame({
        'Langage': ['Python', 'Java', 'JavaScript', 'C#', 'Ruby'],
        'Popularité': [30, 25, 20, 15, 10]
    })

    # Création du Barplot avec Plotly Express
    fig_barplot = px.bar(data_barplot, x='Langage', y='Popularité', title=f'Popularité des langages de programmation pour le métier : {contrat_id}', color_discrete_sequence=['#FF4B4B'])

    # Affichage du Barplot
    return fig_barplot

# Fonction pour calculer le nombre d'entreprises avec une condition sur contratId
def calculer_nombre_entreprises_pour_metier(contrat_id, date_filtre):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(DISTINCT entrepriseId) FROM OffresEmploi_Faits WHERE contratId = '{contrat_id}' AND strftime('%Y', dateCreation) = '{date_filtre}';")
    nombre_entreprises = cursor.fetchone()[0]
    conn.close()
    return nombre_entreprises

# Fonction pour calculer le nombre de villes avec une condition sur contratId
def calculer_nombre_villes_pour_metier(contrat_id, date_filtre):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(DISTINCT lieuTravailId) FROM OffresEmploi_Faits WHERE contratId = '{contrat_id}' AND strftime('%Y', dateCreation) = '{date_filtre}';")
    nombre_villes = cursor.fetchone()[0]
    conn.close()
    return nombre_villes

# Fonction pour calculer le nombre d'offres avec une condition sur contratId
def calculer_nombre_offres_pour_metier(contrat_id, date_filtre):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM OffresEmploi_Faits WHERE contratId = '{contrat_id}' AND strftime('%Y', dateCreation) = '{date_filtre}';")

    nombre_offres = cursor.fetchone()[0]
    conn.close()
    return nombre_offres

# Fonction pour calculer l'évolution d'offre par année en pourcentage avec une condition sur contratId
def calculer_evolution_offres_pour_metier(contrat_id, date_filtre):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    
    # Requête SQL pour obtenir le nombre d'offres créées en 2022 pour le contrat_id spécifié
    cursor.execute(f"SELECT COUNT(id) FROM OffresEmploi_Faits WHERE contratId = '{contrat_id}' AND dateCreation BETWEEN '2022-01-01' AND '2022-12-31' AND strftime('%Y', dateCreation) = '{date_filtre}';")
    offres_2022 = cursor.fetchone()[0]
    
    # Requête SQL pour obtenir le nombre d'offres créées en 2023 pour le contrat_id spécifié
    cursor.execute(f"SELECT COUNT(id) FROM OffresEmploi_Faits WHERE contratId = '{contrat_id}' AND dateCreation BETWEEN '2023-01-01' AND '2023-12-31' AND dateCreation <= '{date_filtre}';")
    offres_2023 = cursor.fetchone()[0]
    
    conn.close()
    
    # Calcul de l'évolution en pourcentage
    if offres_2022 == 0:
        return 0  # Éviter une division par zéro
    else:
        evolution = ((offres_2023 - offres_2022) / offres_2022) * 100
        return round(evolution, 2)

#############
# STREAMLIT #
#############

# CSS input
st.markdown(
"""
<style>
    input[type="text"] {
        color: #FF4B4B !important;
        font-weight: bold !important;
        font-size: larger !important;
        text-align: left !important;
    }
</style>
""",
unsafe_allow_html=True
)

# CSS subheader
st.markdown(
    """
    <style>
        div.stMarkdown.stSubheader {
            text-align: center !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Filtres
st.sidebar.title("Filtres")
# Année
selected_year = st.sidebar.slider("Année :", min_value=2018, max_value=2024, value=2024)
#Régions (requete SQL plus tard)
regions_france = ["Toute la France","Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Bretagne", "Centre-Val de Loire", "Corse", "Grand Est", "Hauts-de-France", "Île-de-France", "Normandie", "Nouvelle-Aquitaine", "Occitanie", "Pays de la Loire", "Provence-Alpes-Côte d'Azur"]
# Sélection de la région dans la barre latérale
selected_region = st.sidebar.selectbox("Région", regions_france)

# Page principale
selected_page="Accueil"

# Affichage du contenu 
if selected_page == "Accueil":
    st.title("JobAPP : Mieux comprendre")
    st.write("Cette application a pour objectif de fournir une compréhension approfondie des compétences demandées sur le marché de l'emploi en se basant sur les offres provenant de sites d'emploi renommés tels que Pôle Emploi et l'APEC. En explorant les données extraites de ces sources, les utilisateurs pourront analyser les tendances du marché, visualiser les différentes compétences recherchées, et obtenir des insights précieux pour orienter leurs choix professionnels. Que ce soit pour les demandeurs d'emploi cherchant à affiner leurs compétences ou les professionnels souhaitant rester informés des évolutions du marché du travail, cette application offre une plateforme interactive pour explorer et interpréter les données liées à l'emploi.")


# Champ de saisie de texte
    st.subheader(f"Quel métier voulez-vous connaitre?")
    recherche_metier = st.text_input("", value='', key='recherche_metier')
    st.markdown('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    st.markdown('<style>div.Widget.row-widget.stRadio > div > label{background-color: #FF0000;color:#FFFFFF;}</style>', unsafe_allow_html=True)

    # Vérification si un métier est saisi
    if recherche_metier:
        # Génération des graphiques et KPI
        st.title(f"Le métier de {recherche_metier}")

        # Affichage des KPI
        st.markdown(
            f"<style>"
            f"div {{ font-family: 'Segoe UI Emoji', sans-serif; }}"
            f"</style>"
            f"<div style='display:flex; justify-content: space-between;'>"
            f"<div style='border: 2px solid white; padding:10px; border-radius:10px; margin-right: 10px; text-align: center;'>"
            f"<strong style='color:#FF4B4B;'>Nombre d'offres :</strong><br>"
            f"{calculer_nombre_offres_pour_metier(recherche_metier, selected_year)}📑"
            f"</div>"
            f"<div style='border: 2px solid white; padding:10px; border-radius:10px; margin-right: 10px; text-align: center;'>"
            f"<strong style='color:#FF4B4B;'>Nombre d'entreprises :</strong><br>"
            f"{calculer_nombre_entreprises_pour_metier(recherche_metier, selected_year)}🏭"
            f"</div>"
            f"<div style='border: 2px solid white; padding:10px; border-radius:10px; text-align: center;'>"
            f"<strong style='color:#FF4B4B;'>Nombre de villes :</strong><br>"
            f"{calculer_nombre_villes_pour_metier(recherche_metier, selected_year)}🗺️"
            f"</div>"
            f"</div>"
            f"<br>",
            unsafe_allow_html=True
        )

        # Exemple de Nuage de mots pour le métier saisi
        st.subheader(f"Quelle compétences avoir ?")
        nuage_de_mots_metier = generer_nuage_de_mots_pour_metier(recherche_metier)
        st.pyplot(nuage_de_mots_metier)

        # Exemple de Carte fictive de la France pour le métier saisi
        st.subheader(f"Dans quel secteur ?")
        carte_france_metier = generer_carte_france_pour_metier(recherche_metier)
        st.plotly_chart(carte_france_metier)

        # Exemple de Barplot fictif pour le métier saisi
        st.subheader(f"Quel diplôme ?")
        barplot_fig_metier = generer_barplot_pour_metier(recherche_metier)
        st.plotly_chart(barplot_fig_metier)


