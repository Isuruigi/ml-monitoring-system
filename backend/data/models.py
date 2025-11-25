"""
SQLAlchemy Database Models.

This module defines database models for the ML Monitoring System using SQLAlchemy ORM.
"""

import uuid
from datetime import datetime
from typing import Dict, Any
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    Text, JSON, Enum as SQLEnum, Index
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import enum

Base = declarative_base()


class JobStatus(str, enum.Enum):
    """Retraining job status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Prediction(Base):
    """
    Prediction table storing all model predictions.
    """
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    features = Column(JSON, nullable=False)
    prediction = Column(String(10), nullable=False)  # UP or DOWN
    probability = Column(Float, nullable=False)
    actual_outcome = Column(String(10), nullable=True)  # Filled later
    model_version = Column(String(50), nullable=False, index=True)
    latency_ms = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Create composite index for common queries
    __table_args__ = (
        Index('idx_timestamp_version', 'timestamp', 'model_version'),
        Index('idx_created_model', 'created_at', 'model_version'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': str(self.id),
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'features': self.features,
            'prediction': self.prediction,
            'probability': self.probability,
            'actual_outcome': self.actual_outcome,
            'model_version': self.model_version,
            'latency_ms': self.latency_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ModelVersion(Base):
    """
    Model version table for tracking deployed models.
    """
    __tablename__ = "model_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(String(50), unique=True, nullable=False, index=True)
    mlflow_run_id = Column(String(100), nullable=True)
    metrics = Column(JSON, nullable=False, default={})
    hyperparameters = Column(JSON, nullable=True)
    is_production = Column(Boolean, default=False, index=True)
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': str(self.id),
            'version': self.version,
            'mlflow_run_id': self.mlflow_run_id,
            'metrics': self.metrics,
            'hyperparameters': self.hyperparameters,
            'is_production': self.is_production,
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'notes': self.notes
        }


class DriftReport(Base):
    """
    Drift report table for storing drift detection results.
    """
    __tablename__ = "drift_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    drift_detected = Column(Boolean, nullable=False)
    drift_score = Column(Float, nullable=False)
    drifted_features = Column(ARRAY(String), nullable=False, default=[])
    feature_drift_scores = Column(JSON, nullable=True)
    report_path = Column(String(500), nullable=True)
    recommendation = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': str(self.id),
            'report_timestamp': self.report_timestamp.isoformat() if self.report_timestamp else None,
            'drift_detected': self.drift_detected,
            'drift_score': self.drift_score,
            'drifted_features': self.drifted_features or [],
            'feature_drift_scores': self.feature_drift_scores,
            'report_path': self.report_path,
            'recommendation': self.recommendation,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class RetrainingJob(Base):
    """
    Retraining job table for tracking model retraining operations.
    """
    __tablename__ = "retraining_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(100), unique=True, nullable=False, index=True)
    status = Column(SQLEnum(JobStatus), nullable=False, default=JobStatus.PENDING)
    trigger_reason = Column(String(200), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    old_model_version = Column(String(50), nullable=True)
    new_model_version = Column(String(50), nullable=True)
    improvement = Column(Float, nullable=True)
    final_metrics = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Index for status queries
    __table_args__ = (
        Index('idx_status_created', 'status', 'created_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': str(self.id),
            'job_id': self.job_id,
            'status': self.status.value if isinstance(self.status, JobStatus) else self.status,
            'trigger_reason': self.trigger_reason,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'old_model_version': self.old_model_version,
            'new_model_version': self.new_model_version,
            'improvement': self.improvement,
            'final_metrics': self.final_metrics,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class SystemMetric(Base):
    """
    System metrics table for tracking performance over time.
    """
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Composite index for time-series queries
    __table_args__ = (
        Index('idx_metric_time', 'metric_name', 'timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': str(self.id),
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'model_version': self.model_version,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# Example usage
if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create in-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:", echo=True)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Test creating a prediction
    pred = Prediction(
        timestamp=datetime.now(),
        features={"rsi_14": 45.3, "macd": 0.52},
        prediction="UP",
        probability=0.73,
        model_version="v1.0.0",
        latency_ms=25.5
    )
    
    session.add(pred)
    session.commit()
    
    # Query
    predictions = session.query(Prediction).all()
    print(f"Created {len(predictions)} prediction(s)")
    print(predictions[0].to_dict())
    
    session.close()
