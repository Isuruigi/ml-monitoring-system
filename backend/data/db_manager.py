"""
Database Manager Module.

This module provides database connection management and CRUD operations
for the ML Monitoring System.
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, and_, desc, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from backend.data.models import (
    Base, Prediction, ModelVersion, DriftReport,
    RetrainingJob, SystemMetric, JobStatus
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for handling all database operations.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL (defaults to env var)
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://mluser:mlpassword@localhost:5432/ml_monitoring"
        )
        
        # Create engine
        self.engine = create_engine(
            self.database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
            echo=False  # Set to True for SQL logging
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"DatabaseManager initialized with {self.database_url.split('@')[1] if '@' in self.database_url else 'database'}")
    
    def init_db(self) -> None:
        """Initialize database by creating all tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Get database session context manager.
        
        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def close(self) -> None:
        """Close database connection."""
        self.engine.dispose()
        logger.info("Database connections closed")
    
    # ============ Prediction Operations ============
    
    def create_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new prediction record.
        
        Args:
            data: Prediction data dictionary
        
        Returns:
            Created prediction as dictionary
        """
        try:
            with self.get_session() as session:
                prediction = Prediction(
                    timestamp=data['timestamp'],
                    features=data['features'],
                    prediction=data['prediction'],
                    probability=data['probability'],
                    model_version=data['model_version'],
                    latency_ms=data['latency_ms'],
                    actual_outcome=data.get('actual_outcome')
                )
                session.add(prediction)
                session.flush()
                
                result = prediction.to_dict()
                logger.debug(f"Created prediction: {result['id']}")
                return result
                
        except Exception as e:
            logger.error(f"Error creating prediction: {str(e)}")
            raise
    
    def get_predictions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        model_version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get predictions with optional filters.
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            limit: Maximum number of results
            model_version: Filter by model version
        
        Returns:
            List of prediction dictionaries
        """
        try:
            with self.get_session() as session:
                query = session.query(Prediction)
                
                # Apply filters
                if start_date:
                    query = query.filter(Prediction.timestamp >= start_date)
                if end_date:
                    query = query.filter(Prediction.timestamp <= end_date)
                if model_version:
                    query = query.filter(Prediction.model_version == model_version)
                
                # Order by timestamp descending and limit
                query = query.order_by(desc(Prediction.timestamp)).limit(limit)
                
                predictions = query.all()
                return [p.to_dict() for p in predictions]
                
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            return []
    
    def update_actual_outcome(
        self,
        prediction_id: str,
        outcome: str
    ) -> bool:
        """
        Update actual outcome for a prediction.
        
        Args:
            prediction_id: Prediction UUID
            outcome: Actual outcome (UP/DOWN)
        
        Returns:
            True if successful
        """
        try:
            with self.get_session() as session:
                prediction = session.query(Prediction).filter(
                    Prediction.id == prediction_id
                ).first()
                
                if prediction:
                    prediction.actual_outcome = outcome
                    logger.debug(f"Updated outcome for prediction {prediction_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error updating outcome: {str(e)}")
            return False
    
    def get_recent_predictions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get predictions from the last N hours.
        
        Args:
            hours: Number of hours to look back
        
        Returns:
            List of prediction dictionaries
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with self.get_session() as session:
                predictions = session.query(Prediction).filter(
                    Prediction.timestamp >= cutoff_time
                ).order_by(desc(Prediction.timestamp)).all()
                
                return [p.to_dict() for p in predictions]
                
        except Exception as e:
            logger.error(f"Error getting recent predictions: {str(e)}")
            return []
    
    # ============ Model Version Operations ============
    
    def create_model_version(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new model version record.
        
        Args:
            data: Model version data
        
        Returns:
            Created model version as dictionary
        """
        try:
            with self.get_session() as session:
                model_version = ModelVersion(
                    version=data['version'],
                    mlflow_run_id=data.get('mlflow_run_id'),
                    metrics=data.get('metrics', {}),
                    hyperparameters=data.get('hyperparameters'),
                    is_production=data.get('is_production', False),
                    deployed_at=data.get('deployed_at'),
                    notes=data.get('notes')
                )
                session.add(model_version)
                session.flush()
                
                result = model_version.to_dict()
                logger.info(f"Created model version: {result['version']}")
                return result
                
        except Exception as e:
            logger.error(f"Error creating model version: {str(e)}")
            raise
    
    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """
        Get current production model.
        
        Returns:
            Production model dictionary or None
        """
        try:
            with self.get_session() as session:
                model = session.query(ModelVersion).filter(
                    ModelVersion.is_production == True
                ).first()
                
                return model.to_dict() if model else None
                
        except Exception as e:
            logger.error(f"Error getting production model: {str(e)}")
            return None
    
    def set_production_model(self, version: str) -> bool:
        """
        Set a model version as production.
        
        Args:
            version: Model version string
        
        Returns:
            True if successful
        """
        try:
            with self.get_session() as session:
                # Unset current production model
                session.query(ModelVersion).filter(
                    ModelVersion.is_production == True
                ).update({'is_production': False})
                
                # Set new production model
                model = session.query(ModelVersion).filter(
                    ModelVersion.version == version
                ).first()
                
                if model:
                    model.is_production = True
                    model.deployed_at = datetime.now()
                    logger.info(f"Set production model to {version}")
                    return True
                
                logger.warning(f"Model version {version} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error setting production model: {str(e)}")
            return False
    
    def get_model_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get model version history.
        
        Args:
            limit: Maximum number of versions to return
        
        Returns:
            List of model version dictionaries
        """
        try:
            with self.get_session() as session:
                models = session.query(ModelVersion).order_by(
                    desc(ModelVersion.created_at)
                ).limit(limit).all()
                
                return [m.to_dict() for m in models]
                
        except Exception as e:
            logger.error(f"Error getting model history: {str(e)}")
            return []
    
    # ============ Drift Report Operations ============
    
    def create_drift_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new drift report.
        
        Args:
            data: Drift report data
        
        Returns:
            Created drift report as dictionary
        """
        try:
            with self.get_session() as session:
                report = DriftReport(
                    report_timestamp=data.get('report_timestamp', datetime.now()),
                    drift_detected=data['drift_detected'],
                    drift_score=data['drift_score'],
                    drifted_features=data.get('drifted_features', []),
                    feature_drift_scores=data.get('feature_drift_scores'),
                    report_path=data.get('report_path'),
                    recommendation=data.get('recommendation')
                )
                session.add(report)
                session.flush()
                
                result = report.to_dict()
                logger.info(f"Created drift report: {result['id']}")
                return result
                
        except Exception as e:
            logger.error(f"Error creating drift report: {str(e)}")
            raise
    
    def get_latest_drift_report(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent drift report.
        
        Returns:
            Latest drift report or None
        """
        try:
            with self.get_session() as session:
                report = session.query(DriftReport).order_by(
                    desc(DriftReport.report_timestamp)
                ).first()
                
                return report.to_dict() if report else None
                
        except Exception as e:
            logger.error(f"Error getting latest drift report: {str(e)}")
            return None
    
    def get_drift_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get drift history for the last N days.
        
        Args:
            days: Number of days to look back
        
        Returns:
            List of drift reports
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            with self.get_session() as session:
                reports = session.query(DriftReport).filter(
                    DriftReport.report_timestamp >= cutoff_date
                ).order_by(desc(DriftReport.report_timestamp)).all()
                
                return [r.to_dict() for r in reports]
                
        except Exception as e:
            logger.error(f"Error getting drift history: {str(e)}")
            return []
    
    # ============ Retraining Job Operations ============
    
    def create_retraining_job(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new retraining job.
        
        Args:
            data: Job data
        
        Returns:
            Created job as dictionary
        """
        try:
            with self.get_session() as session:
                job = RetrainingJob(
                    job_id=data['job_id'],
                    status=JobStatus(data.get('status', 'pending')),
                    trigger_reason=data['trigger_reason'],
                    start_time=data.get('start_time', datetime.now()),
                    old_model_version=data.get('old_model_version')
                )
                session.add(job)
                session.flush()
                
                result = job.to_dict()
                logger.info(f"Created retraining job: {result['job_id']}")
                return result
                
        except Exception as e:
            logger.error(f"Error creating retraining job: {str(e)}")
            raise
    
    def update_job_status(
        self,
        job_id: str,
        status: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update retraining job status.
        
        Args:
            job_id: Job identifier
            status: New status
            data: Additional data to update
        
        Returns:
            True if successful
        """
        try:
            with self.get_session() as session:
                job = session.query(RetrainingJob).filter(
                    RetrainingJob.job_id == job_id
                ).first()
                
                if job:
                    job.status = JobStatus(status)
                    
                    if data:
                        if 'end_time' in data:
                            job.end_time = data['end_time']
                        if 'new_model_version' in data:
                            job.new_model_version = data['new_model_version']
                        if 'improvement' in data:
                            job.improvement = data['improvement']
                        if 'final_metrics' in data:
                            job.final_metrics = data['final_metrics']
                        if 'error_message' in data:
                            job.error_message = data['error_message']
                    
                    logger.info(f"Updated job {job_id} to status {status}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error updating job status: {str(e)}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get retraining job status.
        
        Args:
            job_id: Job identifier
        
        Returns:
            Job data or None
        """
        try:
            with self.get_session() as session:
                job = session.query(RetrainingJob).filter(
                    RetrainingJob.job_id == job_id
                ).first()
                
                return job.to_dict() if job else None
                
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            return None
    
    # ============ Analytics Queries ============
    
    def get_model_performance_trend(
        self,
        days: int = 30,
        model_version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get model performance trend over time.
        
        Args:
            days: Number of days to analyze
            model_version: Filter by model version
        
        Returns:
            List of performance data points
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            with self.get_session() as session:
                query = session.query(Prediction).filter(
                    Prediction.timestamp >= cutoff_date,
                    Prediction.actual_outcome != None
                )
                
                if model_version:
                    query = query.filter(Prediction.model_version == model_version)
                
                predictions = query.all()
                
                # Calculate daily accuracy
                daily_performance = {}
                for pred in predictions:
                    date_key = pred.timestamp.date()
                    if date_key not in daily_performance:
                        daily_performance[date_key] = {'correct': 0, 'total': 0}
                    
                    daily_performance[date_key]['total'] += 1
                    if pred.prediction == pred.actual_outcome:
                        daily_performance[date_key]['correct'] += 1
                
                # Convert to list
                trend = []
                for date, stats in sorted(daily_performance.items()):
                    trend.append({
                        'date': date.isoformat(),
                        'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                        'sample_size': stats['total']
                    })
                
                return trend
                
        except Exception as e:
            logger.error(f"Error getting performance trend: {str(e)}")
            return []
    
    def get_prediction_volume_stats(self) -> Dict[str, Any]:
        """
        Get prediction volume statistics.
        
        Returns:
            Volume statistics
        """
        try:
            with self.get_session() as session:
                # Total predictions
                total = session.query(func.count(Prediction.id)).scalar()
                
                # Last 24 hours
                cutoff_24h = datetime.now() - timedelta(hours=24)
                last_24h = session.query(func.count(Prediction.id)).filter(
                    Prediction.timestamp >= cutoff_24h
                ).scalar()
                
                # Last 7 days
                cutoff_7d = datetime.now() - timedelta(days=7)
                last_7d = session.query(func.count(Prediction.id)).filter(
                    Prediction.timestamp >= cutoff_7d
                ).scalar()
                
                return {
                    'total': total,
                    'last_24h': last_24h,
                    'last_7d': last_7d,
                    'avg_per_hour': last_24h / 24 if last_24h else 0,
                    'avg_per_day': last_7d / 7 if last_7d else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting volume stats: {str(e)}")
            return {}


# Example usage
if __name__ == "__main__":
    # Initialize database manager
    db = DatabaseManager("sqlite:///test.db")
    
    # Create tables
    db.init_db()
    
    # Create a prediction
    pred_data = {
        'timestamp': datetime.now(),
        'features': {'rsi_14': 45.3, 'macd': 0.52},
        'prediction': 'UP',
        'probability': 0.73,
        'model_version': 'v1.0.0',
        'latency_ms': 25.5
    }
    
    pred = db.create_prediction(pred_data)
    print(f"Created prediction: {pred['id']}")
    
    # Get recent predictions
    recent = db.get_recent_predictions(hours=24)
    print(f"Recent predictions: {len(recent)}")
    
    # Create model version
    model_data = {
        'version': 'v1.0.0',
        'metrics': {'accuracy': 0.78, 'precision': 0.76},
        'is_production': True
    }
    
    model = db.create_model_version(model_data)
    print(f"Created model version: {model['version']}")
    
    # Get volume stats
    stats = db.get_prediction_volume_stats()
    print(f"Volume stats: {stats}")
    
    # Clean up
    db.close()
