"""
Integration tests for FastAPI endpoints.

Tests API routes, authentication, and response formats.
"""

import pytest
from datetime import datetime


class TestPredictionEndpoints:
    """Test prediction API endpoints."""
    
    def test_health_check(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'status' in data
        assert 'timestamp' in data
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint."""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'name' in data
        assert 'version' in data
        assert 'endpoints' in data
    
    def test_metrics_endpoint(self, api_client):
        """Test Prometheus metrics endpoint."""
        response = api_client.get("/metrics")
        
        assert response.status_code == 200
        # Should return Prometheus text format
        assert 'text/plain' in response.headers.get('content-type', '')
    
    def test_predict_without_auth(self, api_client, sample_prediction_request):
        """Test prediction endpoint without API key."""
        response = api_client.post(
            "/api/v1/predict",
            json=sample_prediction_request
        )
        
        # Should require authentication
        assert response.status_code == 422  # Validation error - missing header
    
    def test_predict_with_invalid_auth(self, api_client, sample_prediction_request):
        """Test prediction with invalid API key."""
        response = api_client.post(
            "/api/v1/predict",
            json=sample_prediction_request,
            headers={"X-API-Key": "invalid-key"}
        )
        
        assert response.status_code == 401
    
    @pytest.mark.skip(reason="Requires model to be loaded - may fail in test environment")
    def test_predict_with_auth(self, api_client, sample_prediction_request, mock_api_key):
        """Test successful prediction."""
        # Set environment variable for test
        import os
        os.environ['API_KEY'] = mock_api_key
        
        response = api_client.post(
            "/api/v1/predict",
            json=sample_prediction_request,
            headers={"X-API-Key": mock_api_key}
        )
        
        # May fail if model not loaded, but should not be auth error
        assert response.status_code != 401
    
    def test_predict_invalid_data(self, api_client, mock_api_key):
        """Test prediction with invalid data."""
        import os
        os.environ['API_KEY'] = mock_api_key
        
        invalid_request = {
            "timestamp": "invalid-date",
            "features": {}
        }
        
        response = api_client.post(
            "/api/v1/predict",
            json=invalid_request,
            headers={"X-API-Key": mock_api_key}
        )
        
        assert response.status_code in [422, 500]  # Validation or processing error


class TestMonitoringEndpoints:
    """Test monitoring API endpoints."""
    
    def test_drift_endpoint_without_auth(self, api_client):
        """Test drift endpoint without authentication."""
        response = api_client.get("/api/v1/monitoring/drift")
        
        assert response.status_code == 422  # Missing header
    
    def test_drift_endpoint_with_auth(self, api_client, mock_api_key):
        """Test drift endpoint with authentication."""
        import os
        os.environ['API_KEY'] = mock_api_key
        
        response = api_client.get(
            "/api/v1/monitoring/drift",
            headers={"X-API-Key": mock_api_key}
        )
        
        # Should not be auth error
        assert response.status_code != 401
    
    def test_metrics_endpoint_with_auth(self, api_client, mock_api_key):
        """Test metrics endpoint."""
        import os
        os.environ['API_KEY'] = mock_api_key
        
        response = api_client.get(
            "/api/v1/monitoring/metrics",
            headers={"X-API-Key": mock_api_key}
        )
        
        assert response.status_code != 401
    
    def test_health_endpoint(self, api_client):
        """Test monitoring health endpoint."""
        response = api_client.get("/api/v1/monitoring/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'status' in data
        assert 'components' in data


class TestModelManagementEndpoints:
    """Test model management API endpoints."""
    
    def test_retrain_without_admin_auth(self, api_client):
        """Test retrain endpoint without admin key."""
        response = api_client.post("/api/v1/models/retrain")
        
        assert response.status_code == 422  # Missing header
    
    def test_retrain_with_invalid_admin_auth(self, api_client):
        """Test retrain with invalid admin key."""
        response = api_client.post(
            "/api/v1/models/retrain",
            headers={"X-Admin-Key": "invalid-admin-key"}
        )
        
        assert response.status_code == 401
    
    @pytest.mark.skip(reason="Requires database - may fail in test environment")
    def test_retrain_with_admin_auth(self, api_client, mock_admin_key):
        """Test successful retrain trigger."""
        import os
        os.environ['ADMIN_API_KEY'] = mock_admin_key
        
        response = api_client.post(
            "/api/v1/models/retrain",
            json={"trigger_reason": "test"},
            headers={"X-Admin-Key": mock_admin_key}
        )
        
        # Should not be auth error
        assert response.status_code != 401
    
    def test_versions_endpoint(self, api_client, mock_admin_key):
        """Test model versions endpoint."""
        import os
        os.environ['ADMIN_API_KEY'] = mock_admin_key
        
        response = api_client.get(
            "/api/v1/models/versions",
            headers={"X-Admin-Key": mock_admin_key}
        )
        
        # Should not be auth error
        assert response.status_code != 401


class TestDatabaseOperations:
    """Test database CRUD operations."""
    
    def test_create_prediction(self, test_db):
        """Test creating a prediction record."""
        pred_data = {
            'timestamp': datetime.now(),
            'features': {'rsi_14': 45.3},
            'prediction': 'UP',
            'probability': 0.73,
            'model_version': 'v1.0.0',
            'latency_ms': 25.5
        }
        
        result = test_db.create_prediction(pred_data)
        
        assert 'id' in result
        assert result['prediction'] == 'UP'
        assert result['probability'] == 0.73
    
    def test_get_predictions(self, test_db):
        """Test retrieving predictions."""
        # Create some predictions
        for i in range(5):
            test_db.create_prediction({
                'timestamp': datetime.now(),
                'features': {'test': i},
                'prediction': 'UP' if i % 2 == 0 else 'DOWN',
                'probability': 0.5 + i * 0.1,
                'model_version': 'v1.0.0',
                'latency_ms': 25.0
            })
        
        predictions = test_db.get_predictions(limit=10)
        
        assert len(predictions) == 5
    
    def test_create_model_version(self, test_db):
        """Test creating a model version."""
        model_data = {
            'version': 'v1.0.0',
            'metrics': {'accuracy': 0.78},
            'is_production': True
        }
        
        result = test_db.create_model_version(model_data)
        
        assert result['version'] == 'v1.0.0'
        assert result['is_production'] is True
    
    def test_get_production_model(self, test_db):
        """Test getting production model."""
        # Create a production model
        test_db.create_model_version({
            'version': 'v1.0.0',
            'metrics': {},
            'is_production': True
        })
        
        prod_model = test_db.get_production_model()
        
        assert prod_model is not None
        assert prod_model['version'] == 'v1.0.0'
    
    def test_set_production_model(self, test_db):
        """Test setting production model."""
        # Create two models
        test_db.create_model_version({
            'version': 'v1.0.0',
            'metrics': {},
            'is_production': True
        })
        
        test_db.create_model_version({
            'version': 'v2.0.0',
            'metrics': {},
            'is_production': False
        })
        
        # Switch production to v2.0.0
        success = test_db.set_production_model('v2.0.0')
        assert success is True
        
        # Verify
        prod_model = test_db.get_production_model()
        assert prod_model['version'] == 'v2.0.0'
    
    def test_create_drift_report(self, test_db):
        """Test creating drift report."""
        report_data = {
            'drift_detected': True,
            'drift_score': 0.28,
            'drifted_features': ['rsi_14', 'macd'],
            'recommendation': 'Retrain model'
        }
        
        result = test_db.create_drift_report(report_data)
        
        assert result['drift_detected'] is True
        assert result['drift_score'] == 0.28
        assert len(result['drifted_features']) == 2
    
    def test_create_retraining_job(self, test_db):
        """Test creating retraining job."""
        job_data = {
            'job_id': 'test-job-123',
            'trigger_reason': 'manual',
            'status': 'pending'
        }
        
        result = test_db.create_retraining_job(job_data)
        
        assert result['job_id'] == 'test-job-123'
        assert result['status'] == 'pending'
    
    def test_update_job_status(self, test_db):
        """Test updating job status."""
        # Create job
        test_db.create_retraining_job({
            'job_id': 'test-job-456',
            'trigger_reason': 'drift',
            'status': 'pending'
        })
        
        # Update status
        success = test_db.update_job_status(
            'test-job-456',
            'completed',
            {'new_model_version': 'v2.0.0', 'improvement': 0.05}
        )
        
        assert success is True
        
        # Verify
        job = test_db.get_job_status('test-job-456')
        assert job['status'] == 'completed'
        assert job['new_model_version'] == 'v2.0.0'
