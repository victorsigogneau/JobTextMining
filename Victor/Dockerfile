### j'attends un peu avant de tester ##

#nom image :> interface

FROM python:3.12.1

COPY requirement.txt /Victor/requirements.txt
WORKDIR /Victor

RUN pip install -r requirements.txt

COPY interface2.py interface2.py

#COPY job_mining2.db job_mining2.db

CMD ['streamlit', 'run', "interafce2.py"]

##le docker compose pas necessaire en fait ?? ###
