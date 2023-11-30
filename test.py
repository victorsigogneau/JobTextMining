from selenium import webdriver
from selenium.webdriver.common.keys import Keys
PATH = "C:/Program Files (x86)/chromedriver.exe"
driver = webdriver.Chrome()

driver.get("https://www.apec.fr/candidat/recherche-emploi.html/emploi?motsCles=data&typesConvention=143684&typesConvention=143685&typesConvention=143686&typesConvention=143687")
driver.implicitly_wait(5)

# Extraire le texte de la balise div avec la classe "number-candidat"
try:
    number_of_candidates_element = driver.find_element("class name", "number-candidat")
    text_content = number_of_candidates_element.find_element_by_tag_name("span").text
    print(text_content)
except Exception as e:
    print(f"Une erreur s'est produite : {e}")

# Fermez le navigateur
driver.quit()
