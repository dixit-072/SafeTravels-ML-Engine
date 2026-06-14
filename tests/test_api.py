import os
import pytest
import requests
from dotenv import load_dotenv

# Ingest configuration mappings from your hidden local register file
load_dotenv()

# 🟢 PRODUCTION VECTOR TARGETING: Point the validation test right at your live Render cluster
BASE_URL = "https://safetravels-ml-engine.onrender.com"

def test_backend_health_endpoint():
    """Verifies the core production FastAPI ASGI service layer on Render is live."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        assert response.status_code == 200
        # Adapt asserting check variables to match production API layout schemas
        assert "status" in response.json() or "model_loaded" in response.json()
        print("\n🟢 Production Render Health Connection Check: PASSED!")
    except requests.exceptions.ConnectionError:
        pytest.fail("Cannot bridge network link to live Render server clusters. Check domain name routing.")

def test_prediction_schema_rejection():
    """Asserts that invalid payload envelopes are strictly blocked by production validation gates."""
    invalid_payload = {"location_query": "Shimla", "target_date": "not-a-date"}
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_payload, timeout=10)
        assert response.status_code == 422  # Unprocessable Entity
        print("🟢 Production Pydantic Core Gateway Gate Rejection Check: PASSED!")
    except requests.exceptions.ConnectionError:
        pytest.fail("Cannot connect to production server gates layout channels.")