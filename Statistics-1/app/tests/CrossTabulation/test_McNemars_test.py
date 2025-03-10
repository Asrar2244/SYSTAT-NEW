import json
import pytest
from app import create_app

class TestMcNemarsTestAPI:
    
    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client
    
    def test_valid_mcnemars_test(self, client):
        """Test McNemar's Test with valid 2x2 contingency table."""
        response = client.post('/mcnemars-test', json={
            "data": [[10, 5], [3, 20]]
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "McNemar's Test Results" in data
        assert "P-Value" in data["McNemar's Test Results"]
    
    def test_invalid_input_not_2x2(self, client):
        """Test request with invalid input that is not a 2x2 contingency table."""
        response = client.post('/mcnemars-test', json={
            "data": [[10, 5, 2], [3, 20, 4]]
        })
        assert response.status_code == 400
        assert response.get_json()["error"] == "Invalid input: Data must be a 2x2 contingency table."
    
    def test_non_numeric_data(self, client):
        """Test request where contingency table contains non-numeric data."""
        response = client.post('/mcnemars-test', json={
            "data": [["ten", 5], [3, 20]]
        })
        assert response.status_code == 400
        assert "error" in response.get_json()
    
    def test_empty_data(self, client):
        """Test request with an empty data list."""
        response = client.post('/mcnemars-test', json={
            "data": []
        })
        assert response.status_code == 400
        assert response.get_json()["error"] == "Invalid input: Data cannot be empty."
    
    def test_low_discordant_pairs(self, client):
        """Test McNemar's Test with low discordant pairs, should return a warning."""
        response = client.post('/mcnemars-test', json={
            "data": [[50, 1], [1, 50]]
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "McNemar's Test Results" in data
        assert "warning" in data["McNemar's Test Results"]