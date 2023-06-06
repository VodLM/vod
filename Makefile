DOCKERREGISTRY=raffle.azurecr.io
MODULE_NAME=ds-section-processor
GIT_HASH=$(shell git rev-parse HEAD)

.PHONY: docker clean lint test

all: clean lint test

container:
	docker build -t ${DOCKERREGISTRY}/${MODULE_NAME}:dev --build-arg GIT_HASH=${GIT_HASH} .

docker:
	docker build -t ${DOCKERREGISTRY}/${MODULE_NAME}:dev .

lint:
	poetry run pre-commit install
	poetry run pre-commit run --all-files

black:
	poetry run black .

test:
	poetry run pytest

ray_start:
	poetry run ray start --head --port=6379