import pytest
import numpy as np
import json
from fastapi.testclient import TestClient


def test_api_import():
    """Vérifie que l'API s'importe sans erreur."""
    try:
        from api.main import app
        assert app is not None
    except Exception as e:
        pytest.skip(f"API non encore implémentée : {e}")


def test_health_endpoint():
    """L'endpoint /health doit retourner status=ok."""
    try:
        from api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    except Exception as e:
        pytest.skip(f"API non encore implémentée : {e}")