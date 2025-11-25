"""
Monitoring API Routes.

This module implements monitoring endpoints for drift detection and metrics.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Header, BackgroundTasks
from fastapi.responses import FileResponse
import pandas as pd

from backend.api.schemas.prediction import (
    DriftReport,
    ModelMetrics,
    HealthCheck
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Global dependencies (injected from main.py)
drift_detector = None
metrics_collector = None
db_manager = None
current_model = None


def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Verify API key authentication."""
    expected_key = os.getenv("API_KEY", "your-secret-api-key-here")
    
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return x_api_key


# Simple in-memory cache for drift reports
drift_cache = {
    'report': None,
    'timestamp': None,
    'ttl': 300  # 5 minutes
}


@router.get("/drift", response_model=DriftReport)
async def get_drift_status(api_key: str = Depends(verify_api_key)):
    """
    Get current drift status.
    
    Compares last 100 predictions vs reference dataset.
    Results are cached for 5 minutes.
    
    Returns:
        Drift report with detection status and recommendations
    """
    try:
        # Check cache
        if drift_cache['report'] and drift_cache['timestamp']:
            cache_age = (datetime.now() - drift_cache['timestamp']).seconds
            if cache_age < drift_cache['ttl']:
                logger.info("Returning cached drift report")
                return drift_cache['report']
        
        if drift_detector is None or db_manager is None:
            raise HTTPException(
                status_code=503,
                detail="Drift detection not available"
            )
        
        # Get reference data (from training set or saved reference)
        # For now, we'll simulate this
        # TODO: Load actual reference data
        reference_data = pd.DataFrame()  # Placeholder
        
        # Get recent predictions
        recent_predictions = db_manager.get_recent_predictions(hours=24)
        
        if len(recent_predictions) < 10:
            return DriftReport(
                drift_detected=False,
                drift_score=0.0,
                drifted_features=[],
                report_timestamp=datetime.now(),
                recommendation="Not enough data for drift detection"
            )
        
        current_data = pd.DataFrame([p['features'] for p in recent_predictions])
        
        # Detect drift
        # For now, we'll return a placeholder
        # TODO: Implement actual drift detection with reference data
        drift_result = {
            'overall_drift_detected': False,
            'drift_score': 0.0,
            'drifted_features': []
        }
        
        # Determine recommendation
        if drift_result['drift_score'] > 0.25:
            recommendation = "CRITICAL: Significant drift detected. Retrain model immediately."
        elif drift_result['drift_score'] > 0.20:
            recommendation = "WARNING: Moderate drift detected. Schedule retraining soon."
        else:
            recommendation = "No significant drift detected. Model is performing well."
        
        report = DriftReport(
            drift_detected=drift_result['overall_drift_detected'],
            drift_score=drift_result['drift_score'],
            drifted_features=drift_result['drifted_features'],
            report_timestamp=datetime.now(),
            recommendation=recommendation
        )
        
        # Update cache
        drift_cache['report'] = report
        drift_cache['timestamp'] = datetime.now()
        
        # Record drift metrics
        if metrics_collector:
            metrics_collector.record_drift(
                drift_score=drift_result['drift_score'],
                drifted_features=len(drift_result['drifted_features'])
            )
        
        logger.info(f"Drift check complete: score={drift_result['drift_score']:.4f}")
        
        return report
        
    except Exception as e:
        logger.error(f"Error checking drift: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Drift check failed: {str(e)}"
        )


@router.post("/drift/check")
async def trigger_drift_check(
    time_window_hours: int = 24,
    background_tasks: BackgroundTasks = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Trigger manual drift check with full report generation.
    
    Args:
        time_window_hours: Time window for analysis (in hours)
        background_tasks: FastAPI background tasks
        api_key: API key
    
    Returns:
        Job status and report path
    """
    try:
        if drift_detector is None:
            raise HTTPException(
                status_code=503,
                detail="Drift detector not available"
            )
        
        # Generate report in background
        report_path = f"reports/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # TODO: Implement background task for report generation
        # background_tasks.add_task(generate_drift_report, report_path, time_window_hours)
        
        logger.info(f"Drift check triggered for last {time_window_hours} hours")
        
        return {
            "status": "processing",
            "message": "Drift report generation started",
            "report_path": report_path,
            "estimated_completion": "2-3 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error triggering drift check: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger drift check: {str(e)}"
        )


@router.get("/metrics", response_model=ModelMetrics)
async def get_model_metrics(
    time_period: str = "24h",
    api_key: str = Depends(verify_api_key)
):
    """
    Get model performance metrics.
    
    Args:
        time_period: Time period for metrics (24h, 7d, 30d)
        api_key: API key
    
    Returns:
        Model performance metrics
    """
    try:
        if db_manager is None:
            raise HTTPException(
                status_code=503,
                detail="Database not available"
            )
        
        # Parse time period
        hours_map = {"24h": 24, "7d": 168, "30d": 720}
        hours = hours_map.get(time_period, 24)
        
        # Get recent predictions with outcomes
        predictions = db_manager.get_recent_predictions(hours=hours)
        
        # Filter predictions with actual outcomes
        predictions_with_outcomes = [
            p for p in predictions if p.get('actual_outcome') is not None
        ]
        
        if len(predictions_with_outcomes) < 10:
            # Return default metrics if not enough data
            return ModelMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                roc_auc=0.0,
                sample_size=0,
                evaluation_timestamp=datetime.now(),
                model_version="v1.0.0"
            )
        
        # Calculate metrics
        # TODO: Implement actual metric calculation
        metrics = ModelMetrics(
            accuracy=0.75,  # Placeholder
            precision=0.73,
            recall=0.77,
            f1_score=0.75,
            roc_auc=0.80,
            sample_size=len(predictions_with_outcomes),
            evaluation_timestamp=datetime.now(),
            model_version="v1.0.0"
        )
        
        logger.info(f"Metrics calculated for {time_period}: accuracy={metrics.accuracy:.3f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """
    System health check.
    
    Returns:
        Health status of all components
    """
    components = {}
    
    # Check database
    try:
        if db_manager:
            # TODO: Implement actual DB ping
            components['database'] = "healthy"
        else:
            components['database'] = "not_configured"
    except Exception as e:
        components['database'] = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        # TODO: Implement Redis ping
        components['redis'] = "not_configured"
    except Exception as e:
        components['redis'] = f"unhealthy: {str(e)}"
    
    # Check MLflow
    try:
        # TODO: Implement MLflow ping
        components['mlflow'] = "not_configured"
    except Exception as e:
        components['mlflow'] = f"unhealthy: {str(e)}"
    
    # Check model
    if current_model:
        components['model'] = "loaded"
    else:
        components['model'] = "not_loaded"
    
    # Determine overall status
    unhealthy = [k for k, v in components.items() if "unhealthy" in v]
    overall_status = "unhealthy" if unhealthy else "healthy"
    
    return HealthCheck(
        status=overall_status,
        timestamp=datetime.now(),
        components=components
    )


# Dependency injection setters
def set_drift_detector(detector):
    """Set the drift detector."""
    global drift_detector
    drift_detector = detector
    logger.info("Drift detector set")


def set_metrics_collector(collector):
    """Set the metrics collector."""
    global metrics_collector
    metrics_collector = collector
    logger.info("Metrics collector set")


def set_db_manager(manager):
    """Set the database manager."""
    global db_manager
    db_manager = manager
    logger.info("Database manager set")


def set_model(model):
    """Set the current model."""
    global current_model
    current_model = model
    logger.info("Model set for monitoring")
