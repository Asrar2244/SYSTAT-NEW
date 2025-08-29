import pytest
import json
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_paired_t_test_valid(client):
    payload = {
        "vehicle": [55, 45, 65, 54, 43, 45, 54, 63, 73, 36, 65],
        "drugs": [74, 85, 76, 58, 67, 47, 56, 92, 71, 93, 86]
    }
    response = client.post('/hyphothesis/api/paired-t-test-api', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "T-Test Results" in data
    assert "t-Statistic" in data["T-Test Results"]

def test_paired_t_test_missing_key(client):
    payload = {
        "vehicle": [55, 45, 65]
    }
    response = client.post('/hyphothesis/api/paired-t-test-api', json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_paired_t_test_non_list_input(client):
    payload = {
        "vehicle": "not a list",
        "drugs": [74, 85, 76]
    }
    response = client.post('/hyphothesis/api/paired-t-test-api', json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_paired_t_test_unequal_lengths(client):
    payload = {
        "vehicle": [55, 45, 65],
        "drugs": [74, 85]
    }
    response = client.post('/hyphothesis/api/paired-t-test-api', json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_paired_t_test_invalid_json(client):
    response = client.post('/hyphothesis/api/paired-t-test-api', data="invalid json", content_type='application/json')
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data
