import json
import pytest
from app import create_app

class TestRelativeRiskAPI:
    
    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client
    
    def test_valid_relative_risk_wide_format(self, client):
        """Test Relative Risk with valid 2x2 contingency table (Wide Format)."""
        response = client.post('/relative-risk', json={
            "DB": True,
            "data": [[30, 20], [50, 40]],
            "columns": ["Outcome_Yes", "Outcome_No"],
            "rows": ["Treatment", "Control"],
            "alpha": 0.05,
            "yates_correction": True,
            "confidence_level": 95,
            "first_row_treatment": False
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "Relative Risk" in data
        assert "P-Value" in data
        assert "Chi-square Statistic" in data
    
    def test_valid_relative_risk_long_format(self, client):
        """Test Relative Risk with valid categorical data (Long Format)."""
        response = client.post('/relative-risk', json={
            "DB": False,
            "data": [
                {"Group": "Treatment", "Outcome": "Yes"},
                {"Group": "Treatment", "Outcome": "Yes"},
                {"Group": "Treatment", "Outcome": "No"},
                {"Group": "Treatment", "Outcome": "No"},
                {"Group": "Treatment", "Outcome": "Yes"},
                {"Group": "Control", "Outcome": "No"},
                {"Group": "Control", "Outcome": "Yes"},
                {"Group": "Control", "Outcome": "No"},
                {"Group": "Control", "Outcome": "No"},
                {"Group": "Control", "Outcome": "No"}
            ],
            "alpha": 0.05,
            "yates_correction": False,
            "confidence_level": 95,
            "first_row_treatment": False
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "Relative Risk" in data
    
    def test_invalid_wide_format_input(self, client):
        """Test request with invalid input that is not a 2x2 contingency table."""
        response = client.post('/relative-risk', json={
            "DB": True,
            "data": [[30, 20, 10], [50, 40, 30]],
            "columns": ["Outcome_Yes", "Outcome_No", "Other"],
            "rows": ["Treatment", "Control"],
            "alpha": 0.05
        })
        assert response.status_code == 400
        assert "error" in response.get_json()
    
    def test_invalid_long_format_missing_columns(self, client):
        """Test request with missing required keys in long format."""
        response = client.post('/relative-risk', json={
            "DB": False,
            "data": [
                {"Group": "Treatment", "Outcome": "Yes"},
                {"Group": "Treatment"},
                {"Outcome": "No"}
            ],
            "alpha": 0.05
        })
        assert response.status_code == 400
        assert "error" in response.get_json()
    
    def test_non_numeric_wide_format_data(self, client):
        """Test request where contingency table contains non-numeric data."""
        response = client.post('/relative-risk', json={
            "DB": True,
            "data": [["thirty", 20], [50, 40]],
            "columns": ["Outcome_Yes", "Outcome_No"],
            "rows": ["Treatment", "Control"],
            "alpha": 0.05
        })
        assert response.status_code == 400
        assert "error" in response.get_json()
    
    def test_empty_data(self, client):
        """Test request with an empty data list."""
        response = client.post('/relative-risk', json={
            "DB": True,
            "data": [],
            "columns": ["Outcome_Yes", "Outcome_No"],
            "rows": ["Treatment", "Control"],
            "alpha": 0.05
        })
        assert response.status_code == 400
        assert "error" in response.get_json()
    
    def test_confidence_interval_out_of_range(self, client):
        """Test request with confidence level outside valid range."""
        response = client.post('/relative-risk', json={
            "DB": True,
            "data": [[30, 20], [50, 40]],
            "columns": ["Outcome_Yes", "Outcome_No"],
            "rows": ["Treatment", "Control"],
            "alpha": 0.05,
            "confidence_level": 150  # Invalid confidence level
        })
        assert response.status_code == 400
        assert "error" in response.get_json()
