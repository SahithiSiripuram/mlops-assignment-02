.PHONY: help setup data train test lint mlflow serve docker-build docker-run docker-stop \
        compose-up compose-down k8s-deploy k8s-deploy-local k8s-delete smoke monitor dvc-repro clean

PYTHON      ?= .venv/bin/python
IMAGE       ?= cats-vs-dogs-inference:local
PORT        ?= 8000
BASE_URL    ?= http://localhost:$(PORT)

help:  ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install dev dependencies
	python3.12 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt

data:  ## Download and preprocess the dataset (80/10/10 split, 224x224)
	$(PYTHON) -m src.data.download
	$(PYTHON) -m src.data.preprocess --force

train:  ## Train baseline + CNN + transfer model, tracked in MLflow
	$(PYTHON) -m src.models.train

test:  ## Run the unit tests with coverage
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

lint:  ## Lint the codebase
	$(PYTHON) -m ruff check src tests deployment scripts

mlflow:  ## Open the MLflow UI at http://localhost:5000
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:  ## Run the API locally with hot reload
	$(PYTHON) -m uvicorn src.api.main:app --reload --port $(PORT)

docker-build:  ## Build the inference image
	docker build -t $(IMAGE) .

docker-run:  ## Run the image locally on $(PORT)
	docker run -d --rm --name cats-vs-dogs-inference -p $(PORT):8000 $(IMAGE)

docker-stop:  ## Stop the local container
	-docker rm -f cats-vs-dogs-inference

compose-up:  ## Deploy with Docker Compose (inference + Prometheus)
	docker compose -f deployment/docker-compose.yml up -d --build

compose-down:  ## Tear down the Compose deployment
	docker compose -f deployment/docker-compose.yml down

k8s-deploy:  ## Deploy the published GHCR image to the current Kubernetes context
	kubectl apply -f deployment/k8s/deployment.yaml
	kubectl apply -f deployment/k8s/service.yaml
	kubectl rollout status deployment/cats-vs-dogs-inference --timeout=300s

k8s-deploy-local:  ## Deploy the locally built image (after `kind load docker-image`)
	sed 's|image: ghcr.io/.*|image: $(IMAGE)|' deployment/k8s/deployment.yaml | kubectl apply -f -
	kubectl apply -f deployment/k8s/service.yaml
	kubectl rollout status deployment/cats-vs-dogs-inference --timeout=300s

k8s-delete:  ## Remove the Kubernetes deployment
	-kubectl delete -f deployment/k8s/service.yaml
	-kubectl delete -f deployment/k8s/deployment.yaml

smoke:  ## Run the post-deploy smoke test against $(BASE_URL)
	$(PYTHON) deployment/smoke_test.py --base-url $(BASE_URL)

monitor:  ## Replay held-out images against the deployed service and report accuracy
	$(PYTHON) scripts/monitor_performance.py --base-url $(BASE_URL) --samples 60

dvc-repro:  ## Reproduce the full DVC pipeline (download -> preprocess -> train)
	dvc repro

clean:  ## Remove caches and generated artifacts (keeps data and models)
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
