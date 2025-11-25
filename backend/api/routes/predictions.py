"""
Prediction API Routes.

This module implements prediction endpoints for single and batch predictions.
"""

import logging
import time
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

from backend.api.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionHistoryResponse,
    PredictionHistory,
    ErrorResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predictions"])

# These will be injected via dependency injection in main.py
current_model = None
metrics_collector = None
db_manager = None


def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Verify API key authentication."""
    import os
    expected_key = os.getenv("API_KEY", "your-secret-api-key-here")
    
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return x_api_key


@router.post("/", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Make a single price prediction.
    
    Args:
        request: Prediction request with features
        api_key: API key for authentication
    
    Returns:
        Prediction response with direction and probability
    """
    try:
        start_time = time.time()
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([request.features])
        
        # Make prediction
        if current_model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        prediction_class = current_model.predict(feature_df)[0]
        probabilities = current_model.predict_proba(feature_df)[0]
        
        # Get prediction label and probability
        prediction_label = "UP" if prediction_class == 1 else "DOWN"
        probability = float(probabilities[prediction_class])
        confidence = float(max(probabilities))
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Record metrics
        if metrics_collector:
            metrics_collector.record_prediction(
                latency=latency,
                model_version="v1.0.0",  # TODO: Get from model registry
                prediction_class=prediction_label
            )
        
        # Store prediction in database
        if db_manager:
            try:
                db_manager.create_prediction({
                    'timestamp': request.timestamp,
                    'features': request.features,
                    'prediction': prediction_label,
                    'probability': probability,
                    'model_version': "v1.0.0",
                    'latency_ms': latency * 1000
                })
            except Exception as e:
                logger.error(f"Failed to store prediction: {str(e)}")
        
        response = PredictionResponse(
            prediction=prediction_label,
            probability=probability,
            confidence=confidence,
            model_version="v1.0.0",
            timestamp=datetime.now()
        )
        
        logger.info(
            f"Prediction made: {prediction_label} "
            f"(prob={probability:.3f}, latency={latency:.3f}s)"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        
        if metrics_collector:
            metrics_collector.record_prediction_error(str(type(e).__name__))
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Make batch predictions.
    
    Args:
        request: Batch prediction request
        api_key: API key for authentication
    
    Returns:
        Batch prediction response
    """
    try:
        start_time = time.time()
        
        predictions = []
        
        for pred_request in request.predictions:
            # Convert features to DataFrame
            feature_df = pd.DataFrame([pred_request.features])
            
            # Make prediction
            prediction_class = current_model.predict(feature_df)[0]
            probabilities = current_model.predict_proba(feature_df)[0]
            
            prediction_label = "UP" if prediction_class == 1 else "DOWN"
            probability = float(probabilities[prediction_class])
            confidence = float(max(probabilities))
            
            predictions.append(PredictionResponse(
                prediction=prediction_label,
                probability=probability,
                confidence=confidence,
                model_version="v1.0.0",
                timestamp=datetime.now()
            ))
        
        processing_time = time.time() - start_time
        
        # Record batch metrics
        if metrics_collector:
            for pred in predictions:
                metrics_collector.record_prediction(
                    latency=processing_time / len(predictions),
                    model_version="v1.0.0",
                    prediction_class=pred.prediction
                )
        
        logger.info(
            f"Batch prediction completed: {len(predictions)} predictions "
            f"in {processing_time:.3f}s"
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    start_date: str = None,
    end_date: str = None,
    limit: int = 100,
    model_version: str = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get prediction history.
    
    Args:
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        limit: Maximum number of results
        model_version: Filter by model version
        api_key: API key for authentication
    
    Returns:
        Historical predictions
    """
    try:
        if db_manager is None:
            raise HTTPException(
                status_code=503,
                detail="Database not available"
            )
        
        # Get predictions from database
        predictions = db_manager.get_predictions(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            model_version=model_version
        )
        
        # Convert to response format
        history = [
            PredictionHistory(
                id=pred['id'],
                timestamp=pred['timestamp'],
                features=pred['features'],
                prediction=pred['prediction'],
                probability=pred['probability'],
                actual_outcome=pred.get('actual_outcome'),
                model_version=pred['model_version'],
                latency_ms=pred['latency_ms']
            )
            for pred in predictions
        ]
        
        return PredictionHistoryResponse(
            predictions=history,
            total=len(history),
            returned=len(history)
        )
        
    except Exception as e:
        logger.error(f"Error fetching prediction history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch history: {str(e)}"
        )


# Dependency injection setters (called from main.py)
def set_model(model):
    """Set the current production model."""
    global current_model
    current_model = model
    logger.info("Prediction model set")


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
