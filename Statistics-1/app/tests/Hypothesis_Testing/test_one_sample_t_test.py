import pytest
import json
from app import create_app

class TestOneSampleTTestAPI:
    
    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_valid_t_test(self, client):
        """Test One-Sample T-Test with valid input"""
        response = client.post('/hyphothesis/api/one-sample-t-test', json={
            "sample": [55, 45, 65, 54, 43, 45, 54, 63, 73, 36, 65],
            "population_mean": 50,
            "alternative": "two-sided",
            "confidence_level": 0.95
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "Test Type" in data
        assert "Sample Statistics" in data
        assert "Conclusion" in data

    def test_missing_required_keys(self, client):
        """Test request with missing required keys."""
        response = client.post('/hyphothesis/api/one-sample-t-test', json={
            "population_mean": 50
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
    
    def test_invalid_confidence_level(self, client):
        """Test request with an invalid confidence level."""
        response = client.post('/hyphothesis/api/one-sample-t-test', json={
            "sample": [55, 45, 65, 54],
            "population_mean": 50,
            "confidence_level": 1.5
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
    
    def test_invalid_sample_size(self, client):
        """Test request with a sample size too small."""
        response = client.post('/hyphothesis/api/one-sample-t-test', json={
            "sample": [55],
            "population_mean": 50,
            "confidence_level": 0.95
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
    
    def test_invalid_alternative_hypothesis(self, client):
        """Test request with an invalid alternative hypothesis."""
        response = client.post('/hyphothesis/api/one-sample-t-test', json={
            "sample": [55, 45, 65, 54],
            "population_mean": 50,
            "alternative": "invalid_option"
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
    
    def test_zero_variance_sample(self, client):
        """Test request where all sample values are identical, causing zero variance."""
        response = client.post('/hyphothesis/api/one-sample-t-test', json={
            "sample": [50, 50, 50, 50, 50],
            "population_mean": 50
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
