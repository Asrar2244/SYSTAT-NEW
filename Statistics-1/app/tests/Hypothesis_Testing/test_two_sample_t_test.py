import pytest
import json
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestTwoSampleTTestAPI:
    def test_valid_input(self, client):
        payload = {
            "vehicle": [55, 45, 65, 54, 43, 45, 54, 63, 73, 36, 65],
            "drugs": [74, 85, 76, 58, 67, 47, 56, 92, 71, 93, 86]
        }
        response = client.post('/hyphothesis/api/two-sample-t-test', json=payload)
        assert response.status_code == 200
        data = response.get_json()
        assert "Difference of Means" in data
        assert "Equal Variances Assumed (Student's t-test)" in data
        assert "Equal Variances Not Assumed (Welch's t-test)" in data

    def test_missing_keys(self, client):
        payload = {"vehicle": [55, 45, 65]}
        response = client.post('/hyphothesis/api/two-sample-t-test', json=payload)
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_non_numeric_values(self, client):
        payload = {"vehicle": [55, "A", 65], "drugs": [74, 85, 76]}
        response = client.post('/hyphothesis/api/two-sample-t-test', json=payload)
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_empty_lists(self, client):
        payload = {"vehicle": [], "drugs": []}
        response = client.post('/hyphothesis/api/two-sample-t-test', json=payload)
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_invalid_json(self, client):
        response = client.post('/hyphothesis/api/two-sample-t-test', data="invalid json", content_type='application/json')
        assert response.status_code == 500
        assert "error" in response.get_json()

    def test_large_data_input(self, client):
        payload = {
            "group1": list(range(1000)),
            "group2": list(range(500, 1500))
        }
        response = client.post('/hyphothesis/api/two-sample-t-test', json=payload)
        assert response.status_code == 200
        data = response.get_json()
        assert "Difference of Means" in data
