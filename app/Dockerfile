FROM python:3.9-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y procps && pip install -U pip  \
    && rm /etc/localtime  \
    && ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r ./requirements.txt

COPY ./main.py /app/main.py

EXPOSE 5000

CMD ["uvicorn","main:app", "--host", "0.0.0.0", "--port", "5000"]
