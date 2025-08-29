import json
import pytest
from app import create_app

class TestFishersExactTestAPI:
    
    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client
    
    def test_valid_fishers_exact_test_wide_format(self, client):
        """Test Fisher's Exact Test with valid 2x2 contingency table (Wide Format)."""
        response = client.post('/fisher-exact-test', json={
            "DB": True,
            "columns": ["Smoker", "Non-Smoker"],
            "rows": ["Cancer", "Non-Cancer"],
            "data": [[5, 2], [4, 8]],
            "switch_to_chi_square": "no"
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "Fisher's Exact Test Results" in data
        assert "P-Value" in data["Fisher's Exact Test Results"]
        assert "Odds Ratio" in data["Fisher's Exact Test Results"]
    
    def test_valid_fishers_exact_test_long_format(self, client):
        """Test Fisher's Exact Test with valid categorical data (Long Format)."""
        response = client.post('/fisher-exact-test', json={
            "DB": False,
            "data": [
                {"Group": "Smoker", "Category": "Cancer"},
                {"Group": "Smoker", "Category": "Cancer"},
                {"Group": "Smoker", "Category": "Non-Cancer"},
                {"Group": "Non-Smoker", "Category": "Cancer"},
                {"Group": "Non-Smoker", "Category": "Non-Cancer"},
                {"Group": "Non-Smoker", "Category": "Non-Cancer"}
            ],
            "switch_to_chi_square": "no"
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "Fisher's Exact Test Results" in data
    
    def test_invalid_input_not_2x2(self, client):
        """Test request with invalid input that is not a 2x2 contingency table."""
        response = client.post('/fisher-exact-test', json={
            "DB": True,
            "columns": ["Smoker", "Non-Smoker", "Other"],
            "rows": ["Cancer", "Non-Cancer"],
            "data": [[5, 2, 1], [4, 8, 3]],
            "switch_to_chi_square": "no"
        })
        assert response.status_code == 400
        assert response.get_json()["error"] == "Invalid input: Data must be a 2x2 contingency table."
    
    def test_switch_to_chi_square(self, client):
        """Test switch to Chi-Square when cell count exceeds 100."""
        response = client.post('/fisher-exact-test', json={
            "DB": True,
            "columns": ["Smoker", "Non-Smoker"],
            "rows": ["Cancer", "Non-Cancer"],
            "data": [[150, 200], [300, 500]],
            "switch_to_chi_square": "yes"
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "Chi-Square Test Results" in data
    
    def test_missing_required_keys(self, client):
        """Test request with missing required keys."""
        response = client.post('/fisher-exact-test', json={
            "DB": True,
            "data": [[5, 2], [4, 8]]
        })
        assert response.status_code == 400
        assert response.get_json()["error"] == "Invalid input: Missing required columns or rows."
    
    def test_non_numeric_data(self, client):
        """Test request where contingency table contains non-numeric data."""
        response = client.post('/fisher-exact-test', json={
            "DB": True,
            "columns": ["Smoker", "Non-Smoker"],
            "rows": ["Cancer", "Non-Cancer"],
            "data": [["five", 2], [4, 8]],
            "switch_to_chi_square": "no"
        })
        assert response.status_code == 400
        assert "error" in response.get_json()
    
    def test_empty_data(self, client):
        """Test request with an empty data list."""
        response = client.post('/fisher-exact-test', json={
            "DB": True,
            "columns": ["Smoker", "Non-Smoker"],
            "rows": ["Cancer", "Non-Cancer"],
            "data": [],
            "switch_to_chi_square": "no"
        })
        assert response.status_code == 400
        assert response.get_json()["error"] == "Invalid input: Data cannot be empty."
