

#nom image :> streamlit_job:0.1

FROM python:3.9-slim

COPY . /Victor
WORKDIR /Victor

RUN pip install streamlit

RUN pip install -r requirement.txt


CMD ["streamlit", "run", "interface2.py"]

