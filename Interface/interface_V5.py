import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import ssl 
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.corpus.util import LazyCorpusLoader
import geopandas as gpd
import matplotlib.pyplot as plt
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, TfidfTransformer
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings

# Ignorer tous les avertissements
warnings.filterwarnings("ignore")

#********************************
#fonction pour nettoyage document (chaîne de caractères)
#le document revient sous la forme d'une liste de tokens
#********************************
def nettoyage_doc(doc_param):
    ponctuations = list(string.punctuation)

    ponctuations.insert(1,'°')

    #liste des chiffres
    chiffres = list("0123456789")
    lem = WordNetLemmatizer()
    mots_vides = stopwords.words("french")
    #passage en minuscule
    doc = doc_param.lower()
    #retrait des ponctuations
    doc = "".join([w for w in list(doc) if not w in ponctuations])
    #retirer les chiffres
    doc = "".join([w for w in list(doc) if not w in chiffres])
    #transformer le document en liste de termes par tokénisation
    doc = word_tokenize(doc)
    #lematisation de chaque terme
    doc = [lem.lemmatize(terme) for terme in doc]
    #retirer les stopwords
    doc = [w for w in doc if not w in mots_vides]
    #retirer les termes de moins de 3 caractères
    doc = [w for w in doc if len(w)>=3]
    #fin
    return doc

#************************************************************
#fonction pour nettoyage corpus
#attention, optionnellement les documents vides sont éliminés
#************************************************************
def nettoyage_corpus(corpus,vire_vide=True):
    #output
    output = [nettoyage_doc(doc) for doc in corpus if ((len(doc) > 0) or (vire_vide == False))]
    return output

######################### NUAGE DE MOTS #########################


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
def generer_nuage_de_mots_pour_metier(recherche_metier_nettoye,selected_region):
    conn = sqlite3.connect('job_mining.db')
    cursor = conn.cursor()


    if selected_region == "Toute la France":
        # Pas de filtre sur la région
        query = f"SELECT DISTINCT description FROM OffresEmploi_Faits WHERE poste LIKE '%{recherche_metier_nettoye}%'"
    else:
        # Filtre sur la région sélectionnée
        query = f"SELECT DISTINCT description FROM OffresEmploi_Faits O JOIN LieuTravail_Dimension L ON O.lieuTravailId = L.id WHERE poste LIKE '%{recherche_metier_nettoye}%' AND L.nom_region = '{selected_region}';"

    #Récupérer les descriptions des postes correspondants à la recherche
    cursor.execute(query)
    resultat = cursor.fetchall()

    competences_data = []

    for row in resultat:
        description=row[0]  # Accéder à la première colonne du tuple (description)
        competences_poste = extraire_competences(description)
        competences_data.extend(competences_poste)

    if not competences_data :
        return " Pas d'offres pour cette région "
    # Ajoutez le code pour générer un nuage de mots avec des données fictives pour le contrat_id spécifié
    else :
        wordcloud_data = Counter(competences_data)

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_data)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        return plt

######################### CARTOGRAPHIE #########################

# Fonction pour créer une carte des régions
def generer_carte_region_pour_metier(recherche_metier_nettoye):
    
    conn = sqlite3.connect('job_mining.db')
    cursor = conn.cursor()

    #### RECUPERE LES VALEURS ####
    # Utilisez une requête paramétrée pour éviter les injections SQL
    query = f"""
    SELECT LieuTravail_Dimension.nom_region AS Région, COUNT(DISTINCT OffresEmploi_Faits.id) AS Valeur 
    FROM OffresEmploi_Faits JOIN LieuTravail_Dimension ON OffresEmploi_Faits.lieuTravailId = LieuTravail_Dimension.id
    WHERE OffresEmploi_Faits.poste LIKE '%{recherche_metier_nettoye}%'
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
    #region_france = pd.DataFrame(rows, columns=['Région'])

    region_france = pd.read_excel('departements-region.xlsx', usecols=['nom_region'])
    region_france = region_france.rename(columns={'nom_region': 'Région'})
    #Concatene les deux df
    region_carte_data=pd.merge(region_france, region_data, how='left', left_on='Région', right_on='Région')
    region_carte_data.fillna(0, inplace=True)
    conn.close()
  
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
        range_color=[0, region_carte_data['Valeur'].max()],  # Utilisez la valeur maximale pour la plage de la légende
        labels={'Valeur': 'Nombre d\'offres'},  # Ajoutez une étiquette pour la légende
    )
    fig_region.update_geos(
        center={"lat": 46.6031, "lon": 1.8883},  # On centre la carte sur une coordonnée légèrement sur la droite de la france pour un meilleur affichage
        projection_scale=17,  # On ajuste l'échelle pour zoomer sur la phrance
        visible=False  # On enlève la carte du monde derrière
    )

    # On ajuste les paramètres graphiques de la carte
    fig_region.update_layout(
        # On change ces dimensions
        autosize=False,
        width=1000,
        height=1000,
        coloraxis_showscale=True,
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis_colorbar=dict(
        thickness=15,  # Ajustez l'épaisseur de la légende
        lenmode="fraction",
        len=0.2  # Ajustez la longueur de la légende
    )
    )


    # On change la couleur des tracés
    fig_region.update_traces(marker_line=dict(color="rgb(34,34,34)", width=1))


    return fig_region

# Fonction pour créer une carte des départements
def generer_carte_departement_pour_metier(recherche_metier_nettoye):
    
    conn = sqlite3.connect('job_mining.db')
    cursor = conn.cursor()

    #### RECUPERE LES VALEURS ####
    # Utilisez une requête paramétrée pour éviter les injections SQL
    query = f"""
    SELECT LieuTravail_Dimension.nom_dep AS Département, COUNT(DISTINCT OffresEmploi_Faits.id) AS Valeur 
    FROM OffresEmploi_Faits JOIN LieuTravail_Dimension ON OffresEmploi_Faits.lieuTravailId = LieuTravail_Dimension.id
    WHERE OffresEmploi_Faits.poste LIKE '%{recherche_metier_nettoye}%'
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
        range_color=[0,dep_carte_data['Valeur'].max()]  # Indiquez la plage de la légende
    )
    fig_region.update_geos(
        center={"lat": 46.6031, "lon": 1.8883},  # On centre la carte sur une coordonnée légèrement sur la droite de la france pour un meilleur affichage
        projection_scale=17,  # On ajuste l'échelle pour zoomer sur la phrance
        visible=False  # On enlève la carte du monde derrière
    )

    # On ajuste les paramètres graphiques de la carte
    # On ajuste les paramètres graphiques de la carte
    fig_region.update_layout(
        # On change ces dimensions
        autosize=False,
        width=1000,
        height=1000,
        coloraxis_showscale=True,
        margin={"r":0,"t":0,"l":0,"b":0}
    )


    # On change la couleur des tracés
    fig_region.update_traces(marker_line=dict(color="rgb(34,34,34)", width=1))


    return fig_region

# # Fonction pour créer un Barplot avec une condition sur contratId
# def generer_barplot_diplome_pour_metier(contrat_id):
#     # Ajoutez le code pour générer un Barplot avec des données fictives pour le contrat_id spécifié
#     data_barplot = pd.DataFrame({
#         'Langage': ['Python', 'Java', 'JavaScript', 'C#', 'Ruby'],
#         'Popularité': [30, 25, 20, 15, 10]
#     })

#     # Création du Barplot avec Plotly Express
#     fig_barplot = px.bar(data_barplot, x='Langage', y='Popularité', title=f'Popularité des diplômes pour le métier de : {contrat_id}', color_discrete_sequence=['#FF4B4B'])

#     # Affichage du Barplot
#     return fig_barplot

######################### KPI #########################


# Fonction pour calculer le nombre d'entreprises avec une condition sur contratId
def calculer_nombre_entreprises_pour_metier(recherche_metier_nettoye, selected_region):
    conn = sqlite3.connect('job_mining.db')
    cursor = conn.cursor()

    if selected_region == "Toute la France":
        # Pas de filtre sur la région
        cursor.execute(f"SELECT COUNT(DISTINCT entrepriseId) FROM OffresEmploi_Faits WHERE poste LIKE '%{recherche_metier_nettoye}%';")
    else:
        # Filtre sur la région sélectionnée
        cursor.execute(f"SELECT COUNT(DISTINCT entrepriseId) FROM OffresEmploi_Faits O JOIN LieuTravail_Dimension L ON O.lieuTravailId = L.id WHERE poste LIKE '%{recherche_metier_nettoye}%' AND L.nom_region = '{selected_region}';")

    
    nombre_entreprises = cursor.fetchone()[0]
    conn.close()
    return nombre_entreprises

# Fonction pour calculer le nombre de villes avec une condition sur contratId
def calculer_nombre_villes_pour_metier(recherche_metier_nettoye, selected_region):
    conn = sqlite3.connect('job_mining.db')
    cursor = conn.cursor()

    if selected_region == "Toute la France":
        cursor.execute(f"SELECT COUNT(DISTINCT lieuTravailId) FROM OffresEmploi_Faits WHERE poste LIKE '%{recherche_metier_nettoye}%';")
    else:
        # Filtre sur la région sélectionnée
        cursor.execute(f"SELECT COUNT(DISTINCT lieuTravailId) FROM OffresEmploi_Faits O JOIN LieuTravail_Dimension L ON O.lieuTravailId = L.id WHERE poste LIKE '%{recherche_metier_nettoye}%' AND L.nom_region = '{selected_region}';")

    
    nombre_villes = cursor.fetchone()[0]
    conn.close()
    return nombre_villes

# Fonction pour calculer le nombre d'offres avec une condition sur contratId
def calculer_nombre_offres_pour_metier(recherche_metier_nettoye, selected_region):
    conn = sqlite3.connect('job_mining.db')
    cursor = conn.cursor()

    if selected_region == "Toute la France":
        cursor.execute(f"SELECT COUNT(DISTINCT id) FROM OffresEmploi_Faits WHERE poste LIKE '%{recherche_metier_nettoye}%';")
    else:
        # Filtre sur la région sélectionnée
        cursor.execute(f"SELECT COUNT(DISTINCT O.id) FROM OffresEmploi_Faits O JOIN LieuTravail_Dimension L ON O.lieuTravailId = L.id WHERE poste LIKE '%{recherche_metier_nettoye}%' AND L.nom_region = '{selected_region}';")

    nombre_offres = cursor.fetchone()[0]
    conn.close()
    return nombre_offres

def max_min_region(recherche_metier_nettoye):

    conn = sqlite3.connect('job_mining.db')
    cursor = conn.cursor()
    # Requête pour trouver la région avec le moins d'offres
    query_min_offres = f"""
        SELECT LieuTravail_Dimension.nom_region
        FROM LieuTravail_Dimension
        JOIN OffresEmploi_Faits ON LieuTravail_Dimension.id = OffresEmploi_Faits.lieuTravailId
        WHERE OffresEmploi_Faits.poste LIKE '%{recherche_metier_nettoye}%'
        GROUP BY LieuTravail_Dimension.nom_region
        ORDER BY COUNT(DISTINCT OffresEmploi_Faits.id) ASC
        LIMIT 1
    """
    cursor.execute(query_min_offres)
    region_moins_offres = cursor.fetchone()[0]

    # Requête pour trouver la région avec le plus d'offres
    query_max_offres = f"""
        SELECT LieuTravail_Dimension.nom_region
        FROM LieuTravail_Dimension
        JOIN OffresEmploi_Faits ON LieuTravail_Dimension.id = OffresEmploi_Faits.lieuTravailId
        WHERE OffresEmploi_Faits.poste LIKE '%{recherche_metier_nettoye}%'
        GROUP BY LieuTravail_Dimension.nom_region
        ORDER BY COUNT(DISTINCT OffresEmploi_Faits.id) DESC
        LIMIT 1
    """
    cursor.execute(query_max_offres)
    region_plus_offres = cursor.fetchone()[0]

    # Requête pour trouver la moyenne du nombre d'offres par région
    query_moyenne_offres = f"""
        SELECT ROUND(AVG(nombre_offres),1)
        FROM (
            SELECT LieuTravail_Dimension.nom_region, COUNT(DISTINCT OffresEmploi_Faits.id) as nombre_offres
            FROM LieuTravail_Dimension
            JOIN OffresEmploi_Faits ON LieuTravail_Dimension.id = OffresEmploi_Faits.lieuTravailId
            WHERE OffresEmploi_Faits.poste LIKE '%{recherche_metier_nettoye}%'
            GROUP BY LieuTravail_Dimension.nom_region
        ) AS subquery
    """
    cursor.execute(query_moyenne_offres)
    moyenne_offres_region = cursor.fetchone()[0]

    # Remplacez le texte dans le message
    message = (
        f"On peut voir que la région de {region_moins_offres} est déserte pour le métier de {recherche_metier_nettoye}, "
        f"mais la région de {region_plus_offres} a le plus d'offres. En moyenne, la France a {moyenne_offres_region} offres par région."
    )

    conn.close()

    return message

    
#####################
# Analyses Globales #
#####################  

############# Traitement pour les analyses ##################
def traitement():

    conn = sqlite3.connect('job_mining.db')
    cursor = conn.cursor()

    #Récupérer les descriptions des postes correspondants à la recherche
    query = f"SELECT * FROM OffresEmploi_Faits"
    cursor.execute(query)
    resultat = cursor.fetchall()
    conn.close()  

    resultat = pd.DataFrame(resultat, columns=['id',
        'poste',
        'typeContrat',
        'dateCreation',
        'dateActualisation',
        'description',
        'nombrePostes',
        'salaireLibelle',
        'lieuTravailId',
        'entrepriseId',
        'qualificationId',
        'origineOffreId'])

    resultat['dateCreation'] = pd.to_datetime(resultat['dateCreation'], yearfirst=True)
    resultat['Mois'] = resultat['dateCreation'].dt.month
    resultat['description'] = nettoyage_corpus(resultat['description'])
    resultat['description'] = resultat['description'].apply(lambda liste: ' '.join(liste))
    resultat['poste'] = nettoyage_corpus(resultat['poste'])
    resultat['poste'] = resultat['poste'].apply(lambda liste: ' '.join(liste))

    resultat['Salchiffre'] = resultat['salaireLibelle'].astype(str)
    resultat['Salchiffre'] = resultat['Salchiffre'].apply(lambda x: ' '.join(re.findall(r'\d+', str(x))))
    len(resultat['Salchiffre'][0])
    for index, row in resultat.iterrows():
        # Accéder aux valeurs de chaque colonne pour la ligne actuelle
        valeur = row['Salchiffre']

        if(len(valeur) == 20):
            resultat.loc[index, 'Salchiffre'] = str(resultat.loc[index, 'Salchiffre'])[0:2] + str(resultat.loc[index, 'Salchiffre'])[8:11]
            
        if(len(valeur) == 18):
            resultat.loc[index, 'Salchiffre'] = str(resultat.loc[index, 'Salchiffre'])[0:2] + str(resultat.loc[index, 'Salchiffre'])[7:10]
            
        if(len(valeur) == 14):
            resultat.loc[index, 'Salchiffre'] = str(resultat.loc[index, 'Salchiffre'])[0:2] + str(resultat.loc[index, 'Salchiffre'])[5:8]
            
        if((len(valeur) == 11) | (len(valeur) == 10 )):
            resultat.loc[index, 'Salchiffre'] = str(resultat.loc[index, 'Salchiffre'])[0:2]
        
    for index, row in resultat.iterrows():
        valeur = row['Salchiffre']

        if(len(valeur) == 5):
            resultat.loc[index, ['Chiffre1', 'Chiffre2']] = resultat.loc[index, 'Salchiffre'].split()

            # Convertir les colonnes en entiers
            resultat.loc[index, ['Chiffre1', 'Chiffre2']] = resultat.loc[index, ['Chiffre1', 'Chiffre2']].astype(int)
            
            resultat.loc[index, 'Salchiffre'] = (resultat.loc[index, 'Chiffre1'] + resultat.loc[index, 'Chiffre2'])/ 2
            
            resultat.drop(columns=['Chiffre1', 'Chiffre2'], inplace=True)
    return resultat
    
    
############## Fonctions Graphiques ###############
       
def plot_metiers(resultat):
    # Votre code pour créer yr_5
    yr_5 = resultat.groupby('poste').count().sort_values('id', ascending=False).head(6)


    # Créer un graphique à barres interactif avec Plotly Express
    fig = px.bar(yr_5, x=yr_5.index, y='id', orientation='v',
                 labels={'id': 'Nombre de résultats', 'poste': 'Poste'} )

    # Personnaliser la mise en page
    fig.update_layout(
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        font=dict(size=12),
        showlegend=False  # Masquer la légende car la couleur est utilisée pour représenter la valeur
    )
    return fig

def plot_mois(resultat):
    monthly_counts = resultat['Mois'].value_counts().sort_index()

    # Créer un graphique barre avec Plotly Express
    fig = px.bar(x=monthly_counts.index, y=monthly_counts.values, 
                 labels={'x': 'Mois', 'y': 'Nombre d\'offres'})

    # Personnaliser la mise en page
    fig.update_layout(
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        font=dict(size=12),
        showlegend=False  # Masquer la légende car la couleur est utilisée pour représenter la valeur
    )

    return fig

def cam_contrat(resultat):
    count_by_category = resultat['typeContrat'].value_counts()

    # Créer un graphique de type camembert avec Plotly Express
    fig = px.pie(count_by_category, names=count_by_category.index, values=count_by_category.values,
                 labels={'names': 'Type de Contrat', 'values': 'Nombre'},
                 hole=0) 

    return fig

def salaire_m(resultat):
    salaire = resultat['Salchiffre']
    salaire.replace('', np.nan, inplace=True)
    salaire.dropna(inplace=True)
    salaire = salaire.apply(lambda x: float(x) if isinstance(x, (int, float)) else pd.to_numeric(x, errors='coerce'))
    salaire.astype(float)
    salaire = salaire.tolist()
    salaire_moyen = np.mean(salaire)
    salaire_moyen = round(salaire_moyen, 2)
    return salaire_moyen


def salaire_negociation(resultat):
    
    valeur = ''
    
    mask_null = (resultat['Salchiffre'] == valeur)
    mask_null = resultat['Salchiffre'].isnull()
    # Compter le nombre de lignes avec et sans valeurs nulles
    count_values = resultat['Salchiffre'].count()
    count_null = mask_null.sum()

    labels = ['Salaire fixé par l entreprise', 'A négocier']
    values = [count_values, count_null]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                                 insidetextorientation='radial',showlegend=False
                                )])

    fig.update_layout(showlegend=False)

    return fig

###########ETUDE CORPUS #########################



def corpus_mots(resultat):  
    nb_chars = [len(description) for description in resultat['description']]

    # Créer un histogramme avec Plotly Express
    fig = px.histogram(x=nb_chars, nbins=50, title='Nombre de caractères dans les descriptions',
                    labels={'x': 'Nombre de caractères', 'y': 'Fréquence'})
    return fig

def frequ_corspus(resultat):
    cleaned_corpus = [phrase.split() for phrase in resultat['description']]
    liste_resultante = [mot for liste in cleaned_corpus for mot in liste]
    word_counts = Counter(liste_resultante)

    # Sélection des 10 mots les plus fréquents
    top_words = dict(word_counts.most_common(10))

    # Créer un histogramme avec Plotly Express
    fig = px.bar(x=list(top_words.keys()), y=list(top_words.values()),
                labels={'x': 'Mots', 'y': 'Fréquence'},
                title='Mots les plus fréquents dans le corpus')
    return fig


###############TF IDF ########################################


def TF_IDF_dendogram(resultat):
    textes = resultat['description'].astype('str').tolist()


    # Utilisez TfidfVectorizer pour créer une matrice TF-IDF
    vectorizer = TfidfVectorizer()
    matrice_tfidf = vectorizer.fit_transform(textes)

    # Obtenez les noms des caractéristiques (mots)
    mots = vectorizer.get_feature_names_out()

    # Créez une DataFrame à partir de la matrice TF-IDF
    df_tfidf = pd.DataFrame(matrice_tfidf.toarray(), columns=mots)
    col = ['python', 'sql', 'machine','learning','data','analyse', 'mining', 'warehouse',
 'visualization', 'power','excel', 'savoirfaire', 'satisfaction',
'statistique', 'processing', 'nlp', 'social', 
'deep','learning','artificielle', 'spécialisée', 'spécialisé', 'soutien', 'quality', 'management', 'etl', 'software', 
'cloud','aws', 'azure', 'universitaire', 'télétravail','travail', 'tensorflow', 'support', 'suivre', 'structure', 'stratégie', 
'statistique', 'autonome', 'curiosité', 'flexible', 'adaptabilité']
    df_tfidf = df_tfidf[col]
    
    linkage_matrix = hierarchy.linkage(df_tfidf, method='ward')

# Créer un graphique dendrogramme
    fig, ax = plt.subplots(figsize=(10, 8))
    dendrogram = hierarchy.dendrogram(linkage_matrix, labels=df_tfidf.index, leaf_rotation=90)
    plt.title('Dendrogramme des Documents')
    return fig







def visu_tfidf(resultat):
    from scipy.cluster.hierarchy import fcluster
    textes = resultat['description'].astype('str').tolist()


    # Utilisez TfidfVectorizer pour créer une matrice TF-IDF
    vectorizer = TfidfVectorizer()
    matrice_tfidf = vectorizer.fit_transform(textes)

    # Obtenez les noms des caractéristiques (mots)
    mots = vectorizer.get_feature_names_out()

    # Créez une DataFrame à partir de la matrice TF-IDF
    df_tfidf = pd.DataFrame(matrice_tfidf.toarray(), columns=mots)
    col = ['python', 'sql', 'machine','learning','data','analyse', 'mining', 'warehouse',
 'visualization', 'power','excel', 'savoirfaire', 'satisfaction',
'statistique', 'processing', 'nlp', 'social', 
'deep','learning','artificielle', 'spécialisée', 'spécialisé', 'soutien', 'quality', 'management', 'etl', 'software', 
'cloud','aws', 'azure', 'universitaire', 'télétravail','travail', 'tensorflow', 'support', 'suivre', 'structure', 'stratégie', 
 'autonome', 'curiosité', 'flexible', 'adaptabilité']
    
    df_tfidf = df_tfidf[col]
    
    mots = df_tfidf.columns
    distances = hierarchy.linkage(df_tfidf, method='ward')
    # Spécifiez le nombre de clusters (ajustez en fonction de vos besoins)
    nombre_clusters = 2

    # Attribuez des clusters en fonction du dendrogramme
    clusters = fcluster(distances, t=nombre_clusters, criterion='maxclust')

    # Ajoutez les informations sur les clusters à votre DataFrame
    df_tfidf['cluster'] = clusters
    pca = PCA(n_components=2)
    resultats_acp = pca.fit_transform(df_tfidf)

    # Coordonnées des documents dans le plan factoriel
    coord_docs = pd.DataFrame(resultats_acp, columns=['Dimension 1', 'Dimension 2'])

    # Coordonnées des termes dans le plan factoriel (chargements)
    loadings = pd.DataFrame(pca.components_.T, index=col, columns=['Dimension 1', 'Dimension 2'])

    for cluster in df_tfidf['cluster'].unique():
        documents_cluster = df_tfidf[df_tfidf['cluster'] == cluster].index
        plt.scatter(coord_docs.loc[documents_cluster, 'Dimension 1'], coord_docs.loc[documents_cluster, 'Dimension 2'], label=f'Cluster {cluster}')

    # Créez un scatter plot pour les termes (utilisez les loadings)
    plt.scatter(loadings['Dimension 1'], loadings['Dimension 2'], color='red', marker='^', label='Termes')

    # Ajoutez des annotations pour chaque terme
    for i, txt in enumerate(mots):
        plt.annotate(txt, (loadings.iloc[i, 0], loadings.iloc[i, 1]), color='red', fontsize=8)

    # Ajoutez des labels aux axes
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Ajoutez la légende
    plt.legend()

    # Affichez le graphe
    plt.show()
    
############################## TABLES #####################################
def charger_table(selected_table):
    # Connexion à la base de données
    conn = sqlite3.connect('job_mining.db')

    # Charger les données de la table sélectionnée depuis la base de données
    query = f'SELECT * FROM {selected_table}'
    df = pd.read_sql_query(query, conn)

    # Fermer la connexion à la base de données
    conn.close()
    return df


#############
# STREAMLIT #
#############
st.set_page_config(page_title="Analyse des Offres d'Emploi", page_icon="📊")

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
# Changer le nom de l'onglet Streamlit


# Année
# selected_year = st.sidebar.slider("Année :", min_value=2018, max_value=2024, value=2024)
#Régions (requete SQL plus tard)

st.sidebar.title("JobAPP")
selected_page = st.sidebar.radio("Sélectionnez une page", ["Accueil", "Analyse",'Tables'])
# Page principale
#selected_page="Accueil"

# Affichage du contenu 
if selected_page == "Accueil":
    # Filtres
    st.sidebar.title("Filtres")
    regions_france = ['Toute la France','Auvergne-Rhone-Alpes','Bourgogne-Franche-Comte','Bretagne','Centre-Val de Loire','Corse','Grand-Est','Guadeloupe','Guyane','Hauts-de-France','Ile-de-France','La Reunion','Martinique','Mayotte','Normandie','Nouvelle Aquitaine','Occitanie','Pays de la Loire','Provence-Alpes-Cote d\'Azur']
    # Sélection de la région dans la barre latérale
    selected_region = st.sidebar.selectbox("Région", regions_france)
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
            f"{calculer_nombre_offres_pour_metier(recherche_metier_nettoye,selected_region)}📑"
            f"</div>"
            f"<div style='border: 2px solid white; padding:10px; border-radius:10px; margin-right: 10px; text-align: center;'>"
            f"<strong style='color:#FF4B4B;'>Nombre d'entreprises :</strong><br>"
            f"{calculer_nombre_entreprises_pour_metier(recherche_metier,selected_region)}🏭"
            f"</div>"
            f"<div style='border: 2px solid white; padding:10px; border-radius:10px; text-align: center;'>"
            f"<strong style='color:#FF4B4B;'>Nombre de villes :</strong><br>"
            f"{calculer_nombre_villes_pour_metier(recherche_metier,selected_region)}🗺️"
            f"</div>"
            f"</div>"
            f"<br>",
            unsafe_allow_html=True
        )

        # Exemple de Nuage de mots pour le métier saisi
        st.subheader(f"Quelle compétences avoir ?")
        nuage_de_mots_metier = generer_nuage_de_mots_pour_metier(recherche_metier,selected_region)
        if isinstance(nuage_de_mots_metier, str):
            st.write(nuage_de_mots_metier)
        else :
            st.pyplot(nuage_de_mots_metier)

        # Exemple de Carte fictive de la France pour le métier saisi
        st.subheader(f"Dans quel secteur ?")
        st.write(max_min_region(recherche_metier_nettoye))
        
        st.subheader(f"Cartographie des offres par régions")
        carte_region_metier = generer_carte_region_pour_metier(recherche_metier)
        st.plotly_chart(carte_region_metier)

        st.subheader(f"Cartographie des offres par départements")
        carte_dep_metier = generer_carte_departement_pour_metier(recherche_metier)
        st.plotly_chart(carte_dep_metier)

        


        # # Exemple de Barplot fictif pour le métier saisi
        # st.subheader(f"Quel diplôme ?")
        # barplot_fig_metier = generer_barplot_diplome_pour_metier(recherche_metier)
        # st.plotly_chart(barplot_fig_metier)

elif selected_page == "Analyse":

    st.title("Analyse des données d offres d emplois")
    
    recherche = traitement()
    
    # Metiers les plus recherchés par les entreprises
    st.subheader(f"Les metiers les plus recherchés en entreprise")
    metier = plot_metiers(recherche)
    st.plotly_chart(metier)
    st.write(f"Nous avons ici les métiers avec le plus de nombres d'offres d'emploi, les entreprises recherchent énorment de data analyst ")

    # Répartion des types de contrat
    st.subheader(f"Les types de contrat")
    barplot_fig_metier = cam_contrat(recherche)
    st.plotly_chart(barplot_fig_metier)
    st.write(f"Voici une représentation des contrats souhaités par les entreprises pour les métiers dans la Data. On aperçoit logiquement que les entreprises recherchent des CDI.")    

    # Nombre d'iffre d'emplo
    st.subheader(f"Répartition mensuelles des publications des offres")
    barplot_fig_metier = plot_mois(recherche)
    st.plotly_chart(barplot_fig_metier)
    st.write(f"Les recheches d'emplois ont été effectués en Janvier 2024, ainsi les annonces sont regroupés sur les mois de fin 2023 et debut 2024")

         
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader(f"Salaire moyen")
        st.subheader(f"{salaire_m(recherche) * 1000} €")
        st.write(f"C'est le salaire moyen que perçoivent les métiers de la data")
        st.markdown('<div class="vertical-center">', unsafe_allow_html=True)

    with col4:
        st.subheader(f"Salaire à négocier ?")
        cam_nego = salaire_negociation(recherche)
        st.plotly_chart(cam_nego, use_container_width=True)  # Utilisez use_container_width pour occuper toute la largeur
        st.write("Nous pouvons voir aussi que certains employeurs ne décrivent pas dans leurs annonces le salaire qu'ils souhaitent attribuer pour le poste")

        # Fermez la classe CSS pour centrer verticalement
        st.markdown("</div>", unsafe_allow_html=True)


    st.subheader(f"Analyses sur le corpus des descriptions")

    corpus_mot = corpus_mots(recherche)
    st.plotly_chart(corpus_mot)
    st.write(f"Voici une analyse du corpus utilisé, nous pouvons voirs le nombres de mots utilisés dans les descriptions principalement les descriptions contienent moins de 500 mots (après nettoyage) ") 
    

    mots = frequ_corspus(recherche)
    st.plotly_chart(mots)
    st.write(f"Nous pouvons voir ici les mots les plus utilisé dans les descriptions")
    
    st.subheader(f"TF_IDF")

    dendo = TF_IDF_dendogram(recherche)
    st.pyplot(dendo) 
    st.write(f"Voici le dendrogramme résultant d'un traitement TF-IDF appliqué au corpus de descriptions.")
    st.write(f"Par la suite, une analyse de clustering a été effectuée, mettant en évidence une séparation nette en deux classes distinctes au sein des descriptions.") 
    st.write(f"Cette observation suggère une structuration significative du contenu des descriptions, avec des similitudes marquées au sein de chaque classe et des différences notables entre les deux.")

elif selected_page == "Tables":
    # Liste des tables
    tables = ['LieuTravail_Dimension', 'Entreprise_Dimension', 'OrigineOffre_Dimension', 'Qualification_Dimension', 'OffresEmploi_Faits']
    # Sélection de la table à afficher
    selected_table = st.selectbox('Sélectionnez la table à afficher', tables)

    # Charger les données de la table sélectionnée
    df_selected = charger_table(selected_table)

    # Afficher les données de la table sélectionnée
    st.write(f'Données de la table {selected_table}')
    st.dataframe(df_selected)


    #visu_tfidf(recherche)
    #st.pyplot()