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
    st.table(data)

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

# Connexion à la base de données SQLite
conn = sqlite3.connect('job_mining2.db')
cursor = conn.cursor()

# Exécution de la requête SQL pour obtenir la liste des tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [table[0] for table in cursor.fetchall()]

# Fermeture de la connexion à la base de données
conn.close()

# Création de l'interface Streamlit avec plusieurs onglets
page = st.sidebar.selectbox("Sélectionnez un onglet :", ["Accueil", "Données", "Visualisation", "KPI"])

if page == "Accueil":
    st.title("Bienvenue sur l'onglet Accueil")
    st.write("Cette application permet d'explorer les données liées aux offres d'emploi, y compris des onglets pour afficher la liste des tables, une visualisation avec une carte de densité fictive de la France, un nuage de mots, et des KPI associés.")

elif page == "Données":
    st.title("Liste des Tables disponibles")
    # Affichage de la liste des tables
    for table in tables:
        st.write(table)

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
    data_barplot = pd.DataFrame({
        'Langage': ['Python', 'Java', 'JavaScript', 'C#', 'Ruby'],
        'Popularité': [30, 25, 20, 15, 10]
    })
    fig_barplot = px.bar(data_barplot, x='Langage', y='Popularité', title='Popularité des langages de programmation')
    st.plotly_chart(fig_barplot)

elif page == "KPI":
    st.title("Indicateurs Clés de Performance (KPI)")
    
    # Fonction pour calculer l'évolution d'offre par année en pourcentage
    def calculer_evolution_offres():
        # Ajoutez le code pour calculer l'évolution des offres par année
        return 10.5  # Remplacez ceci par la vraie valeur calculée
    
    # Affichage des KPI
    st.metric("Nombre d'offres", 5000)
    st.metric("Nombre d'entreprises", 1500)
    st.metric("Nombre de villes", 200)
    st.metric("Évolution d'offres par année (%)", calculer_evolution_offres())

# Enregistrez ce script dans un fichier, puis exécutez-le avec Streamlit
# Exécutez avec la commande : streamlit run nom_du_fichier.py
