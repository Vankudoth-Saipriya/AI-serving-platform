import yaml
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"

def test_dockerfile_exists_and_valid():
    assert DOCKERFILE_PATH.exists()
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in content
    assert "WORKDIR /app" in content
    assert "EXPOSE 8000" in content
    assert "HEALTHCHECK" in content
    assert "uvicorn" in content

def test_dockerignore_rules():
    assert DOCKERIGNORE_PATH.exists()
    content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
    assert ".venv" in content
    assert ".git" in content
    assert ".pytest_cache" in content

def test_docker_compose_schema_and_services():
    assert COMPOSE_PATH.exists()
    with open(COMPOSE_PATH, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    assert "services" in compose_data
    services = compose_data["services"]

    # 1. API Service
    assert "api" in services
    assert "8000:8000" in services["api"]["ports"]
    assert "healthcheck" in services["api"]

    # 2. Prometheus Service
    assert "prometheus" in services
    assert "9090:9090" in services["prometheus"]["ports"]
    assert any("prometheus.yml" in v for v in services["prometheus"]["volumes"])

    # 3. Grafana Service
    assert "grafana" in services
    assert "3000:3000" in services["grafana"]["ports"]
    assert any("provisioning" in v for v in services["grafana"]["volumes"])
    assert any("dashboards" in v for v in services["grafana"]["volumes"])
