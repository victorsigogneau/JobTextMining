# Projet Informatique - Text Mining sur le Marché de l'Emploi

## Introduction
Ce projet vise à analyser le marché de l'emploi en France en utilisant des techniques de text mining. Les données sont extraites de sources telles que l'API de Pôle Emploi et le web scraping sur l'APEC.

## Contenu du Projet
- **Scripts Python**: Les scripts Python utilisés pour l'extraction, le traitement et l'analyse des données se trouvent dans le répertoire `scripts`.
- **Base de Données SQLite**: Les données sont stockées dans une base de données SQLite. Le schéma de la base de données est documenté dans le rapport.
- **Application Streamlit**: L'interface utilisateur interactive est construite avec Streamlit. Consultez le fichier `app.py` pour explorer l'application.

## Exécution du Projet

### Prérequis
- Docker installé

### Installation
1. Clonez le référentiel: `git clone https://github.com/votre-utilisateur/projet-text-mining.git`
2. Accédez au répertoire du projet: `cd projet-text-mining`
3. Installez les dépendances: `pip install -r requirements.txt`

### Exécution
1. Exécutez l'application Streamlit: `streamlit run app.py`
2. Ouvrez votre navigateur et accédez à l'URL indiquée par Streamlit.

## Structure du Répertoire
- `data/`: Ce répertoire contient les données extraites et peut également inclure les données prétraitées. 
    - `ExtractDATA_LoadDB.ipynb`: Script d'extraction et de prétraitement des données.
    - `analyses_globales.ipynb`: Analyses NLP globales utiles et mise en prodcution dans le Stramlit.
    - `departements-region.xlsx`: Tables des correspondances des régions, départements

- `interface/`: Dans ce répertoire, vous trouverez tous les composants nécessaires au lancement de l'application.
  - `interface.py`: Code source de l'application Streamlit.
  - `functions.py`: Functions de l'application Streamlit.
  - `job_mining.db`: Base de données SQLite.
  - `departement.json`: Coordonées des départements.
  - `regions.json`: Coordonnées des régions.
  - `departements-region.xlsx`: Tables des correspondances des régions, départements

- `README.md`: Vous êtes ici!

## Auteurs
- Cyrielle Barailler
- Célia Maurin
- Victor Sigogneau


