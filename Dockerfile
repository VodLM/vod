FROM python:3.11-slim-buster

RUN apt-get update
RUN apt-get install -y git build-essential

RUN useradd --create-home somebody
USER somebody

RUN pip install --user --upgrade pip

ENV POETRY_VERSION=1.3.0
ENV POETRY_HOME=/home/somebody/poetry
ENV POETRY_VENV=/home/somebody/venv
ENV POETRY_CACHE_DIR=/home/somebody/.cache

RUN python3 -m venv $POETRY_VENV
RUN $POETRY_VENV/bin/pip install -U pip setuptools
RUN $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /ds-repo-template
COPY . /ds-repo-template

COPY poetry.lock pyproject.toml ./
RUN poetry install

EXPOSE  8080

CMD [\
  "poetry", "run", \
  "uvicorn", "api.main:app", \
  "--port", "8080", \
  "--host", "0.0.0.0" \
  ]

