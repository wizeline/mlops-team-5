FROM python:3.10.8-slim-bullseye
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]