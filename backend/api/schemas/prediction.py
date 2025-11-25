"""
Pydantic schemas for prediction API.

This module defines request/response models for the prediction endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request model for price prediction."""
    
    timestamp: datetime = Field(
        ...,
        description="Timestamp for the prediction"
    )
    features: Dict[str, float] = Field(
        ...,
        description="Feature dictionary for prediction",
        example={
            "rsi_14": 45.3,
            "macd": 0.52,
            "ma_7": 42150.5,
            "return_1h": 0.002,
            "volatility_24h": 0.015
        }
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00",
                "features": {
                    "rsi_14": 45.3,
                    "macd": 0.52,
                    "ma_7": 42150.5
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response model for price prediction."""
    
    prediction: str = Field(
        ...,
        description="Predicted direction: UP or DOWN"
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of the predicted class"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (max probability)"
    )
    model_version: str = Field(
        ...,
        description="Version of the model used"
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp when prediction was made"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "UP",
                "probability": 0.73,
                "confidence": 0.73,
                "model_version": "v1.2.0",
                "timestamp": "2024-01-15T10:30:05"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    predictions: List[PredictionRequest] = Field(
        ...,
        description="List of prediction requests"
    )
    
    @validator('predictions')
    def check_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 predictions")
        if len(v) == 0:
            raise ValueError("Batch cannot be empty")
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of prediction responses"
    )
    total: int = Field(
        ...,
        description="Total number of predictions"
    )
    processing_time: float = Field(
        ...,
        description="Total processing time in seconds"
    )


class PredictionHistoryQuery(BaseModel):
    """Query parameters for prediction history."""
    
    start_date: Optional[datetime] = Field(
        None,
        description="Start date for history query"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="End date for history query"
    )
    limit: int = Field(
        100,
        ge=1,
        le=1000,
        description="Maximum number of results"
    )
    model_version: Optional[str] = Field(
        None,
        description="Filter by model version"
    )


class PredictionHistory(BaseModel):
    """Historical prediction record."""
    
    id: str = Field(..., description="Prediction ID")
    timestamp: datetime
    features: Dict[str, float]
    prediction: str
    probability: float
    actual_outcome: Optional[str] = None
    model_version: str
    latency_ms: float


class PredictionHistoryResponse(BaseModel):
    """Response model for prediction history."""
    
    predictions: List[PredictionHistory]
    total: int = Field(..., description="Total records matching query")
    returned: int = Field(..., description="Number of records returned")


class DriftReport(BaseModel):
    """Drift detection report."""
    
    drift_detected: bool = Field(
        ...,
        description="Whether significant drift was detected"
    )
    drift_score: float = Field(
        ...,
        ge=0.0,
        description="Overall drift score (PSI)"
    )
    drifted_features: List[str] = Field(
        ...,
        description="List of features that have drifted"
    )
    report_timestamp: datetime = Field(
        ...,
        description="When the report was generated"
    )
    recommendation: str = Field(
        ...,
        description="Recommended action based on drift analysis"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "drift_detected": True,
                "drift_score": 0.28,
                "drifted_features": ["rsi_14", "volatility_24h", "ma_7"],
                "report_timestamp": "2024-01-15T10:30:00",
                "recommendation": "Consider retraining the model due to significant drift"
            }
        }


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    roc_auc: float = Field(..., ge=0.0, le=1.0)
    sample_size: int = Field(..., ge=0)
    evaluation_timestamp: datetime
    model_version: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "accuracy": 0.78,
                "precision": 0.76,
                "recall": 0.80,
                "f1_score": 0.78,
                "roc_auc": 0.82,
                "sample_size": 5000,
                "evaluation_timestamp": "2024-01-15T10:00:00",
                "model_version": "v1.2.0"
            }
        }


class HealthCheck(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Overall system status")
    timestamp: datetime
    components: Dict[str, str] = Field(
        ...,
        description="Status of individual components"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "components": {
                    "database": "healthy",
                    "redis": "healthy",
                    "mlflow": "healthy",
                    "model": "loaded"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Prediction failed",
                "detail": "Invalid feature format",
                "timestamp": "2024-01-15T10:30:00"
            }
        }
