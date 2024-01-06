import sqlite3
import streamlit as st

# Connexion à la base de données SQLite
conn = sqlite3.connect('job_mining2.db')
cursor = conn.cursor()

# Exécution de la requête SQL pour obtenir la liste des tables
cursor.execute("SELECT * FROM OffresEmploi_Faits;")
data = cursor.fetchall()

# Fermeture de la connexion à la base de données
conn.close()

# Création de l'interface Streamlit
st.table(data)