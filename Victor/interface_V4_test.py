import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import ssl 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.corpus.util import LazyCorpusLoader
import geopandas as gpd
import matplotlib.pyplot as plt

#ouvertuer carte
#Fonction pour traiter le texte recherche_metier
#récupérer la liste des ponctuations


ponctuations = list(string.punctuation)

#outil pour procéder à la lemmatisation - attention à charger le cas échéant
#nltk.download()
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()

#pour la tokénisation
from nltk.tokenize import word_tokenize

#liste des mots vides
from nltk.corpus import stopwords
nltk.download('french')
mots_vides = stopwords.words("french")

#********************************
#fonction pour nettoyage document (chaîne de caractères)
#le document revient sous la forme d'une liste de tokens
#********************************
def nettoyage_doc(doc_param):
    #passage en minuscule
    doc = doc_param.lower()
    #retrait des ponctuations
    doc = "".join([w for w in list(doc) if not w in ponctuations])
    #transformer le document en liste de termes par tokénisation
    doc = word_tokenize(doc)
    #lematisation de chaque terme
    doc = [lem.lemmatize(terme) for terme in doc]
    #retirer les stopwords
    doc = [w for w in doc if not w in mots_vides]
    #retirer les termes de moins de 3 caractères
    doc = [w for w in doc if len(w)>=1]
    #fin
    return doc


######################### NUAGE DE MOTS #########################

# Charger les ressources NLTK (à faire une seule fois)
nltk.download('punkt')
nltk.download('stopwords')

# Initialiser les outils de traitement de texte
ponctuations = set(string.punctuation)
stop_words = set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()

# def nettoyage_description(description):
       
#     # Tokenization
#     tokens = word_tokenize(description)
    
#     # Lemmatisation et retrait des mots vides
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    
#     return tokens

def extraire_competences(description):
    #Liste des compétences ciblées
    competences_cibles = [
    'python', 'r', 'sql', 'machine','learning', 'data','analyse', 'mining', 'warehouse',
    'big','data', 'hadoop', 'spark', 'visualization', 'tableau', 'power','bi', 'excel',
    'statistique', 'prediciton', 'natural language processing' 'nlp',
    'deep','learning', 'intelligence','artificielle', 'business','intelligence',
    'cleaning', 'quality', 'management', 'etl',
    'cloud','aws', 'azure', 'google'
]
        
    # Compter la fréquence des mots
    word_freq = Counter(description.split())
      
    # Filtrer les mots avec une fréquence minimale
    min_frequency = 2
    competences = [word for word, freq in word_freq.items() if freq >= min_frequency and word in competences_cibles]
    
    return competences

# Fonction pour créer un nuage de mots avec une condition sur contratId
def generer_nuage_de_mots_pour_metier(recherche_metier_nettoye):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()

    #Récupérer les descriptions des postes correspondants à la recherche
    query = f"SELECT DISTINCT description FROM OffresEmploi_Faits WHERE poste LIKE '%{recherche_metier_nettoye}%'"
    cursor.execute(query)
    resultat = cursor.fetchall()

    competences_data = []

    for row in resultat:
        description=row[0]  # Accéder à la première colonne du tuple (description)
        competences_poste = extraire_competences(description)
        competences_data.extend(competences_poste)

    # resultat = cursor.fetchone()

    # if resultat:
    #     description = resultat[0]  # Accédez à la première colonne du tuple (description)
    #     print(description)
    #     competences_data = extraire_competences(description)
        
   
    # Ajoutez le code pour générer un nuage de mots avec des données fictives pour le contrat_id spécifié
    wordcloud_data = Counter(competences_data)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

######################### CARTOGRAPHIE #########################

# Fonction pour créer une carte des régions
def generer_carte_region_pour_metier(recherche_metier_nettoye, date_filtre):
    
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()

    #### RECUPERE LES VALEURS ####
    # Utilisez une requête paramétrée pour éviter les injections SQL
    query = f"""
    SELECT LieuTravail_Dimension.nom_region AS Région, COUNT(DISTINCT OffresEmploi_Faits.id) AS Valeur 
    FROM OffresEmploi_Faits JOIN LieuTravail_Dimension ON OffresEmploi_Faits.lieuTravailId = LieuTravail_Dimension.id
    WHERE OffresEmploi_Faits.poste LIKE '%{recherche_metier_nettoye}%' AND strftime('%Y', OffresEmploi_Faits.dateCreation) = '{date_filtre}'
    GROUP BY LieuTravail_Dimension.nom_region
    """
    # Exécutez la requête avec un paramètre
    cursor.execute(query)
    # Récupérez toutes les lignes
    rows = cursor.fetchall()

    # Créez un DataFrame à partir des résultats de la requête
    region_data = pd.DataFrame(rows, columns=['Région', 'Valeur'])


    #### RECUPERE LES REGIONS ####
    # Utilisez une requête paramétrée pour éviter les injections SQL
    query = """
    SELECT DISTINCT LieuTravail_Dimension.nom_region
    FROM LieuTravail_Dimension 
    """

    # Exécutez la requête avec un paramètre
    cursor.execute(query)

    # Récupérez toutes les lignes
    rows = cursor.fetchall()

    # Créez un DataFrame à partir des résultats de la requête
    region_france = pd.DataFrame(rows, columns=['Région'])

    #Concatene les deux df
    region_carte_data=pd.merge(region_france, region_data, how='left', left_on='Région', right_on='Région')
    region_carte_data.fillna(0, inplace=True)
    conn.close()
    import json
    import plotly.express as px

    with open('regions.json', 'r') as file:
        regions_data = json.load(file)

    # On crée la carte chloropleth pour les régions
    fig_region = px.choropleth(
        region_carte_data,  # Remplacez df_region par vos données pour les régions
        geojson=regions_data,  # Utilisez les données chargées à partir du fichier JSON
        locations='Région',  # Remplacez nom_region par la colonne contenant le nom des régions dans votre dataframe
        color='Valeur',  # Remplacez Valeur fonciere par la colonne que vous voulez utiliser pour le remplissage
        color_continuous_scale='YlOrRd',  # La palette de couleur utilisée
        featureidkey="properties.libgeo",  # Indiquez le chemin aux IDs dans le GeoJSON
        range_color=[0, 1]  # Indiquez la plage de la légende
    )
    fig_region.update_geos(
        center={"lat": 45.8, "lon": 5},  # On centre la carte sur une coordonnée légèrement sur la droite de la france pour un meilleur affichage
        projection_scale=17,  # On ajuste l'échelle pour zoomer sur la phrance
        visible=False  # On enlève la carte du monde derrière
    )

    # On ajuste les paramètres graphiques de la carte
    # On ajuste les paramètres graphiques de la carte
    fig_region.update_layout(
        # On change ces dimensions
        autosize=True,
        width=500,
        height=600,
        # On retire la légende couleur
        coloraxis_showscale=False,
        margin={"r":0,"t":0,"l":0,"b":0}
    )


    # On change la couleur des tracés
    fig_region.update_traces(marker_line=dict(color="rgb(34,34,34)", width=1))


    return fig_region

# Fonction pour créer une carte des départements
def generer_carte_departement_pour_metier(recherche_metier_nettoye, date_filtre):
    
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()

    #### RECUPERE LES VALEURS ####
    # Utilisez une requête paramétrée pour éviter les injections SQL
    query = f"""
    SELECT LieuTravail_Dimension.nom_dep AS Département, COUNT(DISTINCT OffresEmploi_Faits.id) AS Valeur 
    FROM OffresEmploi_Faits JOIN LieuTravail_Dimension ON OffresEmploi_Faits.lieuTravailId = LieuTravail_Dimension.id
    WHERE OffresEmploi_Faits.poste LIKE '%{recherche_metier_nettoye}%' AND strftime('%Y', OffresEmploi_Faits.dateCreation) = '{date_filtre}'
    GROUP BY LieuTravail_Dimension.nom_dep
    """
    # Exécutez la requête avec un paramètre
    cursor.execute(query)
    # Récupérez toutes les lignes
    rows = cursor.fetchall()

    # Créez un DataFrame à partir des résultats de la requête
    dep_data = pd.DataFrame(rows, columns=['Département', 'Valeur'])


    #### RECUPERE LES DEPARTEMENTS ####
    # Charger le fichier Excel et sélectionner la colonne "nom_dep"
    dep_france = pd.read_excel('departements-region.xlsx', usecols=['nom_dep'])
    dep_france = dep_france.rename(columns={'nom_dep': 'Département'})

    #Concatene les deux df
    dep_carte_data=pd.merge(dep_france, dep_data, how='left', left_on='Département', right_on='Département')
    dep_carte_data.fillna(0, inplace=True)
    conn.close()
    import json
    import plotly.express as px

    with open('departement.json', 'r') as file:
        dep_data = json.load(file)

    # On crée la carte chloropleth pour les régions
    fig_region = px.choropleth(
        dep_carte_data,  # Remplacez df_region par vos données pour les régions
        geojson=dep_data,  # Utilisez les données chargées à partir du fichier JSON
        locations='Département',  # Remplacez nom_region par la colonne contenant le nom des régions dans votre dataframe
        color='Valeur',  # Remplacez Valeur fonciere par la colonne que vous voulez utiliser pour le remplissage
        color_continuous_scale='YlOrRd',  # La palette de couleur utilisée
        featureidkey="properties.libgeo",  # Indiquez le chemin aux IDs dans le GeoJSON
        range_color=[0, 5]  # Indiquez la plage de la légende
    )
    fig_region.update_geos(
        center={"lat": 45.8, "lon": 5},  # On centre la carte sur une coordonnée légèrement sur la droite de la france pour un meilleur affichage
        projection_scale=17,  # On ajuste l'échelle pour zoomer sur la phrance
        visible=False  # On enlève la carte du monde derrière
    )


    # On ajuste les paramètres graphiques de la carte
    # On ajuste les paramètres graphiques de la carte
    fig_region.update_layout(
        # On change ces dimensions
        autosize=True,
        width=500,
        height=600,
        # On retire la légende couleur
        coloraxis_showscale=False,
        margin={"r":0,"t":0,"l":0,"b":0}
    )


    # On change la couleur des tracés
    fig_region.update_traces(marker_line=dict(color="rgb(34,34,34)", width=1))


    return fig_region

# Fonction pour créer un Barplot avec une condition sur contratId
def generer_barplot_diplome_pour_metier(contrat_id):
    # Ajoutez le code pour générer un Barplot avec des données fictives pour le contrat_id spécifié
    data_barplot = pd.DataFrame({
        'Langage': ['Python', 'Java', 'JavaScript', 'C#', 'Ruby'],
        'Popularité': [30, 25, 20, 15, 10]
    })

    # Création du Barplot avec Plotly Express
    fig_barplot = px.bar(data_barplot, x='Langage', y='Popularité', title=f'Popularité des diplômes pour le métier de : {contrat_id}', color_discrete_sequence=['#FF4B4B'])

    # Affichage du Barplot
    return fig_barplot

######################### KPI #########################


# Fonction pour calculer le nombre d'entreprises avec une condition sur contratId
def calculer_nombre_entreprises_pour_metier(recherche_metier_nettoye, date_filtre):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(DISTINCT entrepriseId) FROM OffresEmploi_Faits WHERE poste LIKE '%{recherche_metier_nettoye}%' AND strftime('%Y', dateCreation) = '{date_filtre}';")
    nombre_entreprises = cursor.fetchone()[0]
    conn.close()
    return nombre_entreprises

# Fonction pour calculer le nombre de villes avec une condition sur contratId
def calculer_nombre_villes_pour_metier(recherche_metier_nettoye, date_filtre):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(DISTINCT lieuTravailId) FROM OffresEmploi_Faits WHERE poste LIKE '%{recherche_metier_nettoye}%' AND strftime('%Y', dateCreation) = '{date_filtre}';")
    nombre_villes = cursor.fetchone()[0]
    conn.close()
    return nombre_villes

# Fonction pour calculer le nombre d'offres avec une condition sur contratId
def calculer_nombre_offres_pour_metier(recherche_metier_nettoye, date_filtre):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(DISTINCT id) FROM OffresEmploi_Faits WHERE poste LIKE '%{recherche_metier_nettoye}%' AND strftime('%Y', dateCreation) = '{date_filtre}';")

    nombre_offres = cursor.fetchone()[0]
    conn.close()
    return nombre_offres

# Fonction pour calculer l'évolution d'offre par année en pourcentage avec une condition sur contratId
def calculer_evolution_offres_pour_metier(recherche_metier_nettoye, date_filtre):
    conn = sqlite3.connect('job_mining2.db')
    cursor = conn.cursor()
    
    # Requête SQL pour obtenir le nombre d'offres créées en 2022 pour le contrat_id spécifié
    cursor.execute(f"SELECT COUNT(DISTINCT id) FROM OffresEmploi_Faits WHERE poste LIKE '%{recherche_metier_nettoye}%' AND dateCreation BETWEEN '2022-01-01' AND '2022-12-31' AND strftime('%Y', dateCreation) = '{date_filtre}';")
    offres_2022 = cursor.fetchone()[0]
    
    # Requête SQL pour obtenir le nombre d'offres créées en 2023 pour le contrat_id spécifié
    cursor.execute(f"SELECT COUNT(DISTINCT id) FROM OffresEmploi_Faits WHERE poste LIKE '%{recherche_metier_nettoye}%' AND dateCreation BETWEEN '2023-01-01' AND '2023-12-31' AND dateCreation <= '{date_filtre}';")
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

        recherche_metier_nettoye = ' '.join(nettoyage_doc(recherche_metier))
        # Affichage des KPI
        st.markdown(
            f"<style>"
            f"div {{ font-family: 'Segoe UI Emoji', sans-serif; }}"
            f"</style>"
            f"<strong style='color:#FF4B4B;'>Le métier de {recherche_metier_nettoye} :</strong><br>"
            f"<div style='display:flex; justify-content: space-between;'>"
            f"<div style='border: 2px solid white; padding:10px; border-radius:10px; margin-right: 10px; text-align: center;'>"
            f"<strong style='color:#FF4B4B;'>Nombre d'offres :</strong><br>"
            f"{calculer_nombre_offres_pour_metier(recherche_metier_nettoye, selected_year)}📑"
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
        st.write(f"On peut voir que la région de <requete ou il y a le moins> est déserte pour le métier de {recherche_metier} mais la région <requete ou il y en a le plus> la moyenne générale en france est de <requete moyenne offre region>.")
        carte_france_metier = generer_carte_departement_pour_metier(recherche_metier,selected_year)
        st.plotly_chart(carte_france_metier)

        # Exemple de Barplot fictif pour le métier saisi
        st.subheader(f"Quel diplôme ?")
        barplot_fig_metier = generer_barplot_diplome_pour_metier(recherche_metier)
        st.plotly_chart(barplot_fig_metier)