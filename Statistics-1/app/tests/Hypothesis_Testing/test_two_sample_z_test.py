import json
import pytest
from app import create_app

class TestTwoSampleZTestAPI:
    
    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client
    
    def test_valid_z_test(self, client):
        """Test Z-test with valid input"""
        response = client.post('/hyphothesis/api/two-sample-ztest', json={
             "column": "test_scores",
            "group_column": "group",
            "std1": 10.5,
            "std2": 9.8,
            "confidence": 0.95,
            "alternative": "two-sided",
            "data": [
                {"group": "A", "test_scores": 85},
                {"group": "A", "test_scores": 90},
                {"group": "B", "test_scores": 78},
                {"group": "B", "test_scores": 82}
            ]
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "z_stat" in data
        assert "p_value" in data
        assert "confidence_interval" in data

    def test_missing_required_keys(self, client):
        """Test request with missing required keys."""
        response = client.post('/hyphothesis/api/two-sample-ztest', json={
            "group_column": "group",
            "data": [
                {"group": "A", "test_scores": 85},
                {"group": "A", "test_scores": 90}
            ]
        })
        assert response.status_code == 400
        data = response.get_json()
        assert response.get_json()["error"] == "Invalid input value: Both 'column' and 'group_column' are required."

    def test_invalid_confidence_level(self, client):
        """Test request with an invalid confidence level."""
        response = client.post('/hyphothesis/api/two-sample-ztest', json={
            "column": "test_scores",
            "group_column": "group",
            "confidence": 1.5,
            "data": [
                {"group": "A", "test_scores": 85},
                {"group": "B", "test_scores": 78}
            ]
        })
        assert response.status_code == 400
        data = response.get_json()
        assert response.get_json()["error"] == "Invalid input value: Confidence level must be between 0 and 1."

    def test_invalid_grouping(self, client):
        """Test request with more than two groups."""
        response = client.post('/hyphothesis/api/two-sample-ztest', json={
            "column": "test_scores",
            "group_column": "group",
            "confidence": 0.95,
            "data": [
                {"group": "A", "test_scores": 85},
                {"group": "B", "test_scores": 78},
                {"group": "C", "test_scores": 90}
            ]
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert response.get_json()["error"] == "Ensure exactly two groups."

    def test_zero_division_case(self, client):
        """Test request where one group has zero samples."""
        response = client.post('/hyphothesis/api/two-sample-ztest', json={
            "column": "test_scores",
            "group_column": "group",
            "confidence": 0.95,
            "data": [
                {"group": "A", "test_scores": 85}
            ]
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert response.get_json()["error"] == "Ensure exactly two groups."

    def test_non_numeric_data(self, client):
        """Test request where test_scores contain non-numeric data."""
        response = client.post('/hyphothesis/api/two-sample-ztest', json={
            "column": "test_scores",
            "group_column": "group",
            "confidence": 0.95,
            "data": [
                {"group": "A", "test_scores": "eighty-five"},
                {"group": "B", "test_scores": 78}
            ]
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_empty_data(self, client):
        """Test request with an empty data list."""
        response = client.post('/hyphothesis/api/two-sample-ztest', json={
            "column": "test_scores",
            "group_column": "group",
            "confidence": 0.95,
            "data": []
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert response.get_json()["error"] ==  "Missing required field: 'group'"
