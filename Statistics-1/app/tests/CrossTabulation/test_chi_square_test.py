import pytest
import json
from app import create_app

class TestChiSquareTestAPI:

    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_valid_wide_format_input(self, client):
        """Test Chi-Square test with valid wide-format input."""
        response = client.post('/hypothesis/api/chi-square-test', json={
            "columns": ["High School", "Middle School", "Bachelors", "Masters", "Ph.D"],
            "rows": ["Never Married", "Married", "Divorced", "Widowed"],
            "data": [
                [18, 36, 21, 9, 6],
                [12, 36, 45, 36, 21],
                [6, 9, 9, 3, 3],
                [3, 9, 9, 60, 3]
            ],
            "alpha": 0.05,
            "yates_correction": False,
            "use_fishers_test": False,
            "tables": {
                "counts": True,
                "percentages": True,
                "residuals": True
            },
            "test_statistics": {
                "pearson": True,
                "log_likelihood": True
            },
            "other_statistics": {
                "phi": True,
                "cramers_v": True
            }
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "Test Type" in data
        assert "Chi-Square Statistic" in data
        assert "P-Value" in data
        assert "Conclusion" in data

    def test_valid_long_format_input(self, client):
        """Test Chi-Square test with valid long-format input."""
        response = client.post('/hypothesis/api/chi-square-test', json={
            "data": [
                {"Group": "Never Married", "Category": "High School"},
                {"Group": "Never Married", "Category": "Middle School"},
                {"Group": "Never Married", "Category": "Bachelors"},
                {"Group": "Never Married", "Category": "Masters"},
                {"Group": "Never Married", "Category": "Ph.D"},
                {"Group": "Married", "Category": "High School"},
                {"Group": "Married", "Category": "Middle School"},
                {"Group": "Married", "Category": "Bachelors"},
                {"Group": "Married", "Category": "Masters"},
                {"Group": "Married", "Category": "Ph.D"},
                {"Group": "Divorced", "Category": "High School"},
                {"Group": "Divorced", "Category": "Middle School"},
                {"Group": "Divorced", "Category": "Bachelors"},
                {"Group": "Divorced", "Category": "Masters"},
                {"Group": "Divorced", "Category": "Ph.D"},
                {"Group": "Widowed", "Category": "High School"},
                {"Group": "Widowed", "Category": "Middle School"},
                {"Group": "Widowed", "Category": "Bachelors"},
                {"Group": "Widowed", "Category": "Masters"},
                {"Group": "Widowed", "Category": "Ph.D"}
            ],
            "alpha": 0.05,
            "yates_correction": False,
            "use_fishers_test": False,
            "tables": {
                "counts": True,
                "percentages": False,
                "residuals": True
            },
            "test_statistics": {
                "pearson": True,
                "log_likelihood": True
            },
            "other_statistics": {
                "phi": True,
                "cramers_v": True
            }
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "Test Type" in data
        assert "Chi-Square Statistic" in data
        assert "P-Value" in data
        assert "Conclusion" in data

    def test_missing_required_keys(self, client):
        """Test request with missing required keys."""
        response = client.post('/hypothesis/api/chi-square-test', json={
            "columns": ["Smoker", "Non-Smoker"]
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_invalid_alpha_value(self, client):
        """Test request with an invalid alpha value."""
        response = client.post('/hypothesis/api/chi-square-test', json={
            "columns": ["Smoker", "Non-Smoker"],
            "rows": ["Cancer", "Non-Cancer"],
            "data": [[5, 2], [4, 8]],
            "alpha": 1.5,
            "yates_correction": True,
            "use_fishers_test": False
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_mismatched_list_lengths(self, client):
        """Test request with mismatched columns and data row lengths."""
        response = client.post('/hypothesis/api/chi-square-test', json={
            "columns": ["Smoker", "Non-Smoker"],
            "rows": ["Cancer", "Non-Cancer"],
            "data": [[5, 2, 3], [4, 8]],
            "alpha": 0.05,
            "yates_correction": True,
            "use_fishers_test": False
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_zero_counts(self, client):
        """Test request where all counts are zero."""
        response = client.post('/hypothesis/api/chi-square-test', json={
            "columns": ["Smoker", "Non-Smoker"],
            "rows": ["Cancer", "Non-Cancer"],
            "data": [[0, 0], [0, 0]],
            "alpha": 0.05,
            "yates_correction": True,
            "use_fishers_test": False
        })
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_fishers_exact_test_case(self, client):
        """Test Fisher's Exact Test when required."""
        response = client.post('/hypothesis/api/chi-square-test', json={
            "columns": ["Smoker", "Non-Smoker"],
            "rows": ["Cancer", "Non-Cancer"],
            "data": [[5, 2], [4, 8]],
            "alpha": 0.05,
            "yates_correction": True,
            "use_fishers_test": True
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "Test Type" in data
        assert "P-Value" in data
        assert "Conclusion" in data

    def test_yates_correction_case(self, client):
        """Test Chi-Square test with Yates' correction applied."""
        response = client.post('/hypothesis/api/chi-square-test', json={
            "columns": ["Smoker", "Non-Smoker"],
            "rows": ["Cancer", "Non-Cancer"],
            "data": [[5, 2], [4, 8]],
            "alpha": 0.05,
            "yates_correction": True,
            "use_fishers_test": False
        })
        assert response.status_code == 200
        data = response.get_json()
        assert "Test Type" in data
        assert "Chi-Square Statistic" in data
        assert "P-Value" in data
        assert "Conclusion" in data
