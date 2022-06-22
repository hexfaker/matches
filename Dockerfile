FROM python:3.8-slim-buster AS poetry
ENV \
    DEBIAN_ENVIRONMENT=noninteractive \
    LC_ALL=C.UTF-8 LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=0

RUN pip install 'poetry'
RUN poetry config virtualenvs.create false

#
# Infer requirements
#
FROM poetry AS requirements
WORKDIR /app
COPY pyproject.toml /app/
RUN  poetry export -f requirements.txt -o requirements.txt


FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
COPY --from=requirements /app/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
