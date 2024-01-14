import streamlit as st
from functions import *

# Application
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

#############
# STREAMLIT #
#############


# Sidebar
st.sidebar.title("JobAPP")
selected_page = st.sidebar.radio("Sélectionnez une page", ["Accueil", "Analyse",'Tables'])

# Affichage Accueil
if selected_page == "Accueil":
    # Filtres
    st.sidebar.title("Filtres")
    regions_france = ['Toute la France','Auvergne-Rhone-Alpes','Bourgogne-Franche-Comte','Bretagne','Centre-Val de Loire','Grand-Est','Hauts-de-France','Ile-de-France','Normandie','Nouvelle Aquitaine','Occitanie','Pays de la Loire','Provence-Alpes-Cote d Azur']
    # Sélection de la région dans la barre latérale
    selected_region = st.sidebar.selectbox("Région", regions_france)
    st.title("JobAPP : Mieux comprendre")
    st.write("Cette application a pour objectif de fournir une compréhension approfondie des compétences demandées sur le marché de l'emploi en se basant sur les offres provenant de sites d'emploi renommés tels que Pôle Emploi et l'APEC. En explorant les données extraites de ces sources, les utilisateurs pourront analyser les tendances du marché, visualiser les différentes compétences recherchées, et obtenir des insights précieux pour orienter leurs choix professionnels. Que ce soit pour les demandeurs d'emploi cherchant à affiner leurs compétences ou les professionnels souhaitant rester informés des évolutions du marché du travail, cette application offre une plateforme interactive pour explorer et interpréter les données liées à l'emploi.")


# Champ de saisie de texte - barre de recherche
    st.subheader(f"Quel métier voulez-vous connaître?")
    recherche_metier = st.text_input("", value='', key='recherche_metier')
    st.markdown('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    st.markdown('<style>div.Widget.row-widget.stRadio > div > label{background-color: #FF0000;color:#FFFFFF;}</style>', unsafe_allow_html=True)

    # Vérification si un métier est saisi
    if recherche_metier:

        st.title(f"Le métier de {recherche_metier}")
        recherche_metier_nettoye = ' '.join(nettoyage_doc(recherche_metier))
        # Affichage des KPI
        st.markdown(
            f"<style>"
            f"div {{ font-family: 'Segoe UI Emoji', sans-serif; }}"
            f"</style>"
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
            f"<div style='border: 2px solid white; padding:10px; border-radius:10px; text-align: center;'>"
            f"<strong style='color:#FF4B4B;'>Salaire moyen annuel :</strong><br>"
            f"{salaire_m_recherche(recherche_metier,selected_region)* 1000}€💵"
            f"</div>"
            f"</div>"
            f"<br>",
            unsafe_allow_html=True
        )

        # Exemple de Nuage de mots pour le métier saisi
        st.subheader(f"Quelles compétences avoir ?")
        nuage_de_mots_metier = generer_nuage_de_mots_pour_metier(recherche_metier,selected_region)
        if isinstance(nuage_de_mots_metier, str):
            st.write(nuage_de_mots_metier)
        else :
            st.pyplot(nuage_de_mots_metier)

        # Cartographie
        st.subheader(f"Dans quel secteur ?")
        st.write(max_min_region(recherche_metier_nettoye))
        
        # Regions
        st.subheader(f"Cartographie des offres par régions")
        carte_region_metier = generer_carte_region_pour_metier(recherche_metier)
        st.plotly_chart(carte_region_metier)

        # Départements
        st.subheader(f"Cartographie des offres par départements")
        carte_dep_metier = generer_carte_departement_pour_metier(recherche_metier)
        st.plotly_chart(carte_dep_metier)

# Affichage Analyse     
elif selected_page == "Analyse":

    st.title("Analyse des données d'offres d'emploi Data")
    recherche = traitement()
    
    # Metiers les plus recherchés par les entreprises
    st.subheader(f"Les métiers les plus recherchés en entreprise")
    metier = plot_metiers(recherche)
    st.plotly_chart(metier)
    st.write(f"Nous avons ici les métiers avec le plus d'offres d'emploi, les entreprises recherchent activement des Data Analysts ")

    # Répartion des types de contrat
    st.subheader(f"Les types de contrat")
    barplot_fig_metier = cam_contrat(recherche)
    st.plotly_chart(barplot_fig_metier)
    st.write(f"Voici une représentation des contrats souhaités par les entreprises pour les métiers dans la Data. On aperçoit que les entreprises recherchent majoritairement des CDI.")    

    # Nombre d'offre d'emploi
    st.subheader(f"Répartition mensuelle des publications des offres")
    barplot_fig_metier = plot_mois(recherche)
    st.plotly_chart(barplot_fig_metier)
    st.write(f"Les recherches d'emploi ont été effectués en Janvier 2024, ainsi les annonces sont regroupées sur les mois de fin 2023 et début 2024")

    # Separtion en deux colonnes
    col3, col4 = st.columns(2)
    with col3:
        # Salaire moyen 
        st.subheader(f"Salaire moyen annuel")
        st.subheader(f"{salaire_m(recherche) * 1000} €")
        st.write(f"C'est le salaire moyen annuel perçu par les métiers de la Data")
        st.markdown('<div class="vertical-center">', unsafe_allow_html=True)

    with col4:
        # Repartion des contrats
        st.subheader(f"Salaire à négocier ?")
        cam_nego = salaire_negociation(recherche)
        st.plotly_chart(cam_nego, use_container_width=True)  # Utilisez use_container_width pour occuper toute la largeur
        st.write("Nous pouvons voir que certains employeurs ne décrivent pas dans leurs annonces le salaire qu'ils souhaitent attribuer pour le poste")
        st.markdown("</div>", unsafe_allow_html=True)

    # Analyse des descriptions
    st.subheader(f"Analyses sur le corpus des descriptions")

    # Les mots les plus fréquents
    corpus_mot = corpus_mots(recherche)
    st.plotly_chart(corpus_mot)
    st.write(f"Voici une analyse du corpus utilisé, nous pouvons voir le nombre de mot dans les descriptions. Généralement les descriptions contiennent moins de 500 mots (après nettoyage) ") 
    mots = frequ_corspus(recherche)
    st.plotly_chart(mots)
    st.write(f"Nous pouvons voir ici les mots les plus utilisés dans les descriptions")
    
    # Matrice Tf_IDF
    st.subheader(f"TF_IDF")

    # Clustering des offres
    dendo = TF_IDF_dendogram(recherche)
    st.pyplot(dendo) 
    st.write(f"Voici le dendrogramme résultant d'un traitement TF-IDF appliqué au corpus des descriptions.")
    st.write(f"Par la suite, une analyse de clustering a été effectuée, mettant en évidence une séparation nette en deux classes distinctes au sein des descriptions.") 
    st.write(f"Cette observation suggère une structuration significative du contenu des descriptions, avec des similitudes marquées au sein de chaque classe et des différences notables entre les deux.")

# Affichage Tables
elif selected_page == "Tables":
    
    # Titre de la page
    st.title("Visualisation de la base de données")

    # Liste des tables
    tables = ['OffresEmploi_Faits','LieuTravail_Dimension', 'Entreprise_Dimension', 'OrigineOffre_Dimension', 'Qualification_Dimension']
    
    # Sélection de la table à afficher
    selected_table = st.selectbox('Sélectionnez la table à afficher', tables)

    # Charge et affiche
    df_selected = charger_table(selected_table)
    st.write(f'Données de la table {selected_table}')
    st.dataframe(df_selected)

