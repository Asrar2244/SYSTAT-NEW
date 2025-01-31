import pytest
from ...app import create_app

@pytest.fixture
def client():
    '''flask application setup'''
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_valid_z_test(client):
    '''Test Z-test with valid input'''
    response = client.post('/hyphothesis/api/z-test', json={
        "Alpha_value": 0.05,
        "Yates_correction": 1,
        "Confidence_interval": 95,
        "Data": [[40, 0.3], [160, 0.7]]
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "message" in data
    assert "results" in data
    assert data["results"]["p_value"] < 0.05

def test_invalid_alpha(client):
    """Test invalid Alpha_value (out of range)"""
    response = client.post('/hyphothesis/api/z-test', json={
        "Alpha_value": 1.5,
        "Yates_correction": 0,
        "Confidence_interval": 95,
        "Data": [[40, 0.3], [160, 0.7]]
    })
    assert response.status_code == 400
    assert response.get_json()["error"] == "Alpha_value must be between 0 and 1."

def test_invalid_yates_correction(client):
    """Test invalid Yates_correction (not 0 or 1)"""
    response = client.post('/hyphothesis/api/z-test', json={
        "Alpha_value": 0.05,
        "Yates_correction": 2,
        "Confidence_interval": 95,
        "Data": [[40, 0.3], [160, 0.7]]
    })
    assert response.status_code == 400
    assert response.get_json()["error"] == "Yates_correction must be either 0 or 1."

def test_invalid_confidence_interval(client):
    """Test invalid Confidence_interval (out of range)"""
    response = client.post('/hyphothesis/api/z-test', json={
        "Alpha_value": 0.05,
        "Yates_correction": 0,
        "Confidence_interval": 150,
        "Data": [[40, 0.3], [160, 0.7]]
    })
    assert response.status_code == 400
    assert response.get_json()["error"] == "Confidence_interval must be between 1 and 99."

def test_invalid_data_structure(client):
    """Test invalid Data structure"""
    response = client.post('/hyphothesis/api/z-test', json={
        "Alpha_value": 0.05,
        "Yates_correction": 0,
        "Confidence_interval": 95,
        "Data": [[40], [160, 0.7]]
    })
    assert response.status_code == 400
    assert response.get_json()["error"] == "Data must contain two rows and two columns."

def test_missing_fields(client):
    """Test request missing required fields"""
    response = client.post('/hyphothesis/api/z-test', json={
        "Yates_correction": 0,
        "Confidence_interval": 95
    })
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_division_by_zero(client):
    """Test case where division by zero might occur"""
    response = client.post('/hyphothesis/api/z-test', json={
        "Alpha_value": 0.05,
        "Yates_correction": 0,
        "Confidence_interval": 95,
        "Data": [[0, 0.3], [160, 0.7]]  # Size of 0 will cause ZeroDivisionError
    })
    assert response.status_code == 400
    assert response.get_json()["error"] == "Division by zero encountered during calculation."

def test_unexpected_error(client, monkeypatch):
    """Test unexpected server error handling"""

    def mock_get_json(self):
        raise Exception("Unexpected error!")

    monkeypatch.setattr("flask.Request.get_json", mock_get_json)

    response = client.post('/hyphothesis/api/z-test', json={})
    assert response.status_code == 500
    assert response.get_json()["error"] == "An unexpected error occurred. Please try again later."
