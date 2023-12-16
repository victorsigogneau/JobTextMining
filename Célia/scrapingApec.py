

import json

# pour exploiter les requêtes
from requests import post

# pour le contrôle des requêtes
from time import sleep, time
from random import randint
from abc import ABCMeta, abstractmethod
from warnings import warn



def convert_arr_2_string(arr, sep):
    """ Convert array to string with separator """
    return sep.join(arr)

def get_term(path):
    """ get term in a path. Otherwise, return 'Non renseigné' """
    if path is not None:
        return path.text.strip()
    return ''

def jprint(obj):
    """ convert array to json """
    # create a formatted string of the Python JSON object
    return json.dumps(obj, sort_keys=True) #, indent=4 

def post_data(root_path, payload, requests, start_time):
    """ post data and get the result  """
    response = post(root_path, json=payload)
    content = response.content
    
    ### pause de 8 à 15s
    sleep(randint(8, 15))
    
    ### afficher les informations sur les requêtes
    requests += 1 # incrémentation du nombre de requête
    elapsed_time = time() - start_time
    
    ### avertir si le code status est différent de 200
    if response.status_code != 200:
        warn('Request: {}; Status code:{}'.format(requests, requests/elapsed_time))
    
    ### stopper quand les requêtes atteignent le quota
    if requests > 200:
        warn('Nombre de requêtes trop important')
        return
    
    try:
        json_data = json.loads(content)
    except:
        json_data = ""
    
    return json_data


class scraping_jobs(metaclass=ABCMeta):
    
    def __init__(self, s_job, type_contract):
        self.s_job = s_job
        self.type_contract = type_contract
    
    @abstractmethod
    def scrap_job(self, dict_jobs, s_job, type_contract):
        pass


class scraping_jobs_apec(scraping_jobs):
    
    #
    def set_code_dpt(self, code_dpt):
        self.code_dpt = code_dpt
    
    #
    def scrap_job(self):
        ### paramètres pris
        # les termes doivent être séparés par ' ' ou '%20'
        param_search_words = self.s_job 
        # le numéro du département
        #param_search_location = self.code_dpt

        
        
        ### pages à parcourir
        pages = [str(i) for i in range(0, 20)]  #### le nombre de pages voulu 
        requests = 0
        start_time = time()
        
        dict_jobs = []
        
        ### parcours des pages
        for page in pages:
            #
            root_path = 'https://www.apec.fr/cms/webservices/rechercheOffre'
            payload = {
                #'lieux': [param_search_location],
                'typeClient': 'CADRE',
                'sorts' : [{
                    'type': 'SCORE',
                    'direction': 'DESCENDING'
                }],
                'pagination': {
                    'range': 20,
                    'startIndex': page
                },
                'activeFiltre': True,
                'pointGeolocDeReference': {
                    'distance': 0
                },
                'motsCles': param_search_words
            }
            
            json_data = post_data(root_path, payload, requests, start_time)
            
            ### vérifier l'existence de l'index 'resultats'
            if 'resultats' in json_data:
                result_containers = json_data['resultats']
            
                ### extraction des données du JSON renvoyé
                for result in result_containers:
                    #
                    dict_jobs.append({
                        'title' : result['intitule'],
                        'link' : 'https://www.apec.fr/candidat/recherche-emploi.html/emploi/detail-offre/'+result['numeroOffre'],
                        'location' : result['lieuTexte'],
                        'description' : result['texteOffre'],
                        'company' : result['nomCommercial'],
                        'note' : result['score'],
                        'salary' : result['salaireTexte'],
                        'publication_date' : result['datePublication'],
                    })
            
        ### retourne array
        return dict_jobs
    
    


s_job = "data"
city = ""
code_dpt = ""
type_contract = ''

arr_jobs = []

print('please wait, search in progress...')

## apec
sjapec = scraping_jobs_apec(s_job, type_contract)
sjapec.set_code_dpt(code_dpt)
dict_tmp = sjapec.scrap_job()
if len(dict_tmp) > 0:
    arr_jobs += dict_tmp


### impression des jobs en json
print(jprint(arr_jobs))


print(len( arr_jobs))
print(type(arr_jobs))

import pandas as pd

jobs = pd.DataFrame(arr_jobs)
print(jobs)

jobs.to_csv("jobsapec.csv", index=False, sep=";")



