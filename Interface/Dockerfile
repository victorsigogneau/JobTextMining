FROM python:3.9-slim

COPY . /Interface
WORKDIR /Interface

RUN pip install streamlit

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "interface.py"]

