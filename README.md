# Text Mining sur le Marché de l'Emploi

## Introduction
Ce projet vise à explorer le marché de l’emploi en France. Pour cela, nous avons choisi de nous focaliser spécifiquement sur les opportunités dans le domaine de la data. En utilisant des techniques de web
scraping, nous avons collecté des données provenant d’offres d’emploi de l’APEC et de Pôle emploi. Ces
données ont alimenté une base de données modélisée comme un entrepôt, stockée dans SQLite.

Notre projet inclut une application web interactive développée avec Streamlit, offrant aux utilisateurs
une exploration facile et des analyses visuelles des offres d’emploi. L’application intègre également une di-
mension régionale pour des représentations cartographiques interactives. Enfin, pour une facilité d’accès,
l’application est encapsulée dans une image Docker, simplifiant ainsi le déploiement pour les utilisateurs.

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

### Exécution
1. `cd src/`
2. `docker compose -up`

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


