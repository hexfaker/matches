FROM python:3.8

COPY . /app/matches
RUN pip install /app/matches

