import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Fonction pour afficher les données d'une table
def afficher_table(table_name):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name};")
    data = cursor.fetchall()
    conn.close()
    return(data)

# Fonction pour créer un nuage de mots
def generer_nuage_de_mots():
    # Ajoutez le code pour générer un nuage de mots avec des données fictives
    wordcloud_data = {'Python': 10, 'Java': 8, 'JavaScript': 6}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

# Fonction pour créer une carte fictive de la France
def generer_carte_france():
    # Ajoutez le code pour générer une carte fictive de la France
    france_data = pd.DataFrame({
        'Région': ['Île-de-France', 'Auvergne-Rhône-Alpes', 'Provence-Alpes-Côte d\'Azur', 'Occitanie', 'Hauts-de-France'],
        'Valeur': [100, 80, 60, 40, 20]
    })
    fig_france = px.choropleth(france_data, 
                               locations='Région',
                               locationmode='country names',
                               color='Valeur',
                               color_continuous_scale='Viridis',
                               title='Carte fictive de la France')
    return fig_france

# Fonction pour créer un Barplot
def generer_barplot():
    # Ajoutez le code pour générer un Barplot avec des données fictives
    data_barplot = pd.DataFrame({
        'Langage': ['Python', 'Java', 'JavaScript', 'C#', 'Ruby'],
        'Popularité': [30, 25, 20, 15, 10]
    })

    # Création du Barplot avec Plotly Express
    fig_barplot = px.bar(data_barplot, x='Langage', y='Popularité', title='Popularité des langages de programmation')

    # Affichage du Barplot
    return fig_barplot

# Fonction pour calculer le nombre d'entreprises
def calculer_nombre_entreprises(date_filtre):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(DISTINCT entrepriseId) FROM OffresEmploi_Faits WHERE dateCreation <= '{date_filtre}';")
    nombre_entreprises = cursor.fetchone()[0]
    conn.close()
    return nombre_entreprises

# Fonction pour calculer le nombre de villes
def calculer_nombre_villes(date_filtre):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(DISTINCT lieuTravailId) FROM OffresEmploi_Faits WHERE dateCreation <= '{date_filtre}';")
    nombre_villes = cursor.fetchone()[0]
    conn.close()
    return nombre_villes

# Fonction pour calculer le nombre d'offres
def calculer_nombre_offres(date_filtre):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM OffresEmploi_Faits WHERE dateCreation <= '{date_filtre}';")
    nombre_offres = cursor.fetchone()[0]
    conn.close()
    return nombre_offres

# Fonction pour calculer l'évolution d'offre par année en pourcentage
def calculer_evolution_offres(date_filtre):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    
    # Requête SQL pour obtenir le nombre d'offres créées en 2022
    cursor.execute(f"SELECT COUNT(id) FROM OffresEmploi_Faits WHERE dateCreation BETWEEN '2022-01-01' AND '2022-12-31' AND dateCreation <= '{date_filtre}';")
    offres_2022 = cursor.fetchone()[0]
    
    # Requête SQL pour obtenir le nombre d'offres créées en 2023
    cursor.execute(f"SELECT COUNT(id) FROM OffresEmploi_Faits WHERE dateCreation BETWEEN '2023-01-01' AND '2023-12-31' AND dateCreation <= '{date_filtre}';")
    offres_2023 = cursor.fetchone()[0]
    
    conn.close()
    
    # Calcul de l'évolution en pourcentage
    if offres_2022 == 0:
        return 0  # Éviter une division par zéro
    else:
        evolution = ((offres_2023 - offres_2022) / offres_2022) * 100
        return round(evolution, 2)

# Connexion à la base de données SQLite
conn = sqlite3.connect('job_mining2.db')
cursor = conn.cursor()

# Exécution de la requête SQL pour obtenir la liste des tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [table[0] for table in cursor.fetchall()]

# Fermeture de la connexion à la base de données
conn.close()

# Sélection de la date avec le widget st.date_input
selected_date = st.sidebar.date_input("Sélectionner une date", pd.to_datetime('today'))

# Création de l'interface Streamlit avec plusieurs onglets
page = st.sidebar.selectbox("Sélectionnez un onglet :", ["Accueil", "Données", "Visualisation", "KPI"])

if page == "Accueil":
    st.title("Bienvenue sur l'onglet Accueil")
    st.write("Cette application permet d'explorer les données liées aux offres d'emploi, y compris des onglets pour afficher la liste des tables, une visualisation avec une carte de densité fictive de la France, un nuage de mots, et des KPI associés.")

elif page == "Données":
    st.title("Base de donées")

    # Liste des tables
    st.subheader("Liste des tables disponibles")
    for table in tables:
        st.write(table)

    # La table des faits
    st.subheader("Tables  ")
    st.table(afficher_table("OffresEmploi_Faits"))

elif page == "Visualisation":
    st.title("Visualisation avec Nuage de mots et Carte de la France")
    
    # Nuage de mots
    st.subheader("Nuage de mots")
    nuage_de_mots = generer_nuage_de_mots()
    st.pyplot(nuage_de_mots)

    # Carte fictive de la France
    st.subheader("Carte fictive de la France")
    carte_france = generer_carte_france()
    st.plotly_chart(carte_france)

    # Barplot fictif
    st.subheader("Barplot fictif")
    barplot_fig = generer_barplot()
    st.plotly_chart(barplot_fig)

elif page == "KPI":
    st.title("Indicateurs Clés de Performance (KPI)")
    
    # Affichage des KPI avec la date sélectionnée
    st.metric("Nombre d'offres", calculer_nombre_offres(selected_date))
    st.metric("Nombre d'entreprises", calculer_nombre_entreprises(selected_date))
    st.metric("Nombre de villes", calculer_nombre_villes(selected_date))   
    st.metric("Évolution d'offres par année (%)", calculer_evolution_offres(selected_date))
   
