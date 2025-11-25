"""
Metrics Collection Module using Prometheus.

This module provides comprehensive metrics collection for ML model monitoring,
including performance metrics, drift metrics, and system metrics.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from prometheus_client import (
    Counter, Gauge, Histogram, Info,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Metrics Collector using Prometheus for ML model monitoring.
    
    Tracks model performance, drift, and system metrics.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize the MetricsCollector.
        
        Args:
            registry: Prometheus registry (creates new if None)
        """
        self.registry = registry or CollectorRegistry()
        
        # Model Performance Metrics
        self.prediction_accuracy = Gauge(
            'model_prediction_accuracy',
            'Current model prediction accuracy',
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Model prediction latency in seconds',
            buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5),
            registry=self.registry
        )
        
        self.predictions_total = Counter(
            'model_predictions_total',
            'Total number of predictions made',
            ['model_version', 'prediction_class'],
            registry=self.registry
        )
        
        self.prediction_errors = Counter(
            'model_prediction_errors_total',
            'Total number of prediction errors',
            ['error_type'],
            registry=self.registry
        )
        
        # Drift Metrics
        self.data_drift_score = Gauge(
            'model_data_drift_score',
            'Current data drift score (PSI)',
            registry=self.registry
        )
        
        self.prediction_drift_score = Gauge(
            'model_prediction_drift_score',
            'Current prediction drift score',
            registry=self.registry
        )
        
        self.features_drifted_count = Gauge(
            'model_features_drifted_count',
            'Number of features currently drifted',
            registry=self.registry
        )
        
        # System Metrics
        self.model_version_info = Info(
            'model_version',
            'Current production model version',
            registry=self.registry
        )
        
        self.retraining_triggered = Counter(
            'model_retraining_triggered_total',
            'Number of times retraining was triggered',
            ['trigger_reason'],
            registry=self.registry
        )
        
        self.last_retraining_timestamp = Gauge(
            'model_last_retraining_timestamp',
            'Timestamp of last retraining (Unix time)',
            registry=self.registry
        )
        
        # Additional Performance Metrics
        self.precision_score = Gauge(
            'model_precision_score',
            'Current model precision',
            registry=self.registry
        )
        
        self.recall_score = Gauge(
            'model_recall_score',
            'Current model recall',
            registry=self.registry
        )
        
        self.f1_score = Gauge(
            'model_f1_score',
            'Current model F1 score',
            registry=self.registry
        )
        
        self.roc_auc_score = Gauge(
            'model_roc_auc_score',
            'Current model ROC-AUC score',
            registry=self.registry
        )
        
        logger.info("MetricsCollector initialized with Prometheus metrics")
    
    def record_prediction(
        self,
        latency: float,
        model_version: str,
        prediction_class: str = "unknown",
        accuracy: Optional[float] = None
    ) -> None:
        """
        Record a prediction event.
        
        Args:
            latency: Prediction latency in seconds
            model_version: Version of model used
            prediction_class: Predicted class (UP/DOWN)
            accuracy: Current accuracy if available
        """
        # Record latency
        self.prediction_latency.observe(latency)
        
        # Increment prediction counter
        self.predictions_total.labels(
            model_version=model_version,
            prediction_class=prediction_class
        ).inc()
        
        # Update accuracy if provided
        if accuracy is not None:
            self.prediction_accuracy.set(accuracy)
        
        logger.debug(
            f"Recorded prediction: version={model_version}, "
            f"class={prediction_class}, latency={latency:.3f}s"
        )
    
    def record_prediction_error(self, error_type: str) -> None:
        """
        Record a prediction error.
        
        Args:
            error_type: Type of error encountered
        """
        self.prediction_errors.labels(error_type=error_type).inc()
        logger.warning(f"Recorded prediction error: {error_type}")
    
    def record_drift(
        self,
        drift_score: float,
        drifted_features: int,
        prediction_drift: Optional[float] = None
    ) -> None:
        """
        Record drift metrics.
        
        Args:
            drift_score: Overall data drift score
            drifted_features: Number of drifted features
            prediction_drift: Prediction drift score if available
        """
        self.data_drift_score.set(drift_score)
        self.features_drifted_count.set(drifted_features)
        
        if prediction_drift is not None:
            self.prediction_drift_score.set(prediction_drift)
        
        logger.info(
            f"Recorded drift: score={drift_score:.4f}, "
            f"drifted_features={drifted_features}"
        )
    
    def record_retraining(
        self,
        trigger_reason: str,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Record a retraining event.
        
        Args:
            trigger_reason: Reason for triggering retraining
            timestamp: Unix timestamp (uses current time if None)
        """
        self.retraining_triggered.labels(
            trigger_reason=trigger_reason
        ).inc()
        
        if timestamp is None:
            timestamp = time.time()
        
        self.last_retraining_timestamp.set(timestamp)
        
        logger.info(
            f"Recorded retraining: reason={trigger_reason}, "
            f"timestamp={datetime.fromtimestamp(timestamp)}"
        )
    
    def update_model_version(self, version: str, metadata: Dict[str, str] = None) -> None:
        """
        Update current model version information.
        
        Args:
            version: Model version string
            metadata: Additional metadata about the model
        """
        info_dict = {'version': version}
        
        if metadata:
            info_dict.update(metadata)
        
        self.model_version_info.info(info_dict)
        logger.info(f"Updated model version to {version}")
    
    def update_performance_metrics(
        self,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1: Optional[float] = None,
        roc_auc: Optional[float] = None
    ) -> None:
        """
        Update model performance metrics.
        
        Args:
            accuracy: Model accuracy
            precision: Model precision
            recall: Model recall
            f1: Model F1 score
            roc_auc: Model ROC-AUC score
        """
        if accuracy is not None:
            self.prediction_accuracy.set(accuracy)
        
        if precision is not None:
            self.precision_score.set(precision)
        
        if recall is not None:
            self.recall_score.set(recall)
        
        if f1 is not None:
            self.f1_score.set(f1)
        
        if roc_auc is not None:
            self.roc_auc_score.set(roc_auc)
        
        logger.info(
            f"Updated performance metrics: accuracy={accuracy}, "
            f"precision={precision}, recall={recall}, f1={f1}, roc_auc={roc_auc}"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metric values.
        
        Returns:
            Dictionary of current metrics
        """
        # Note: This is a simplified version. In production, you'd
        # query the registry for actual values
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'note': 'Use Prometheus /metrics endpoint for full metrics'
        }
        
        return metrics
    
    def expose_metrics(self) -> bytes:
        """
        Expose metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """
        Get content type for metrics endpoint.
        
        Returns:
            Prometheus content type
        """
        return CONTENT_TYPE_LATEST


# Example usage
if __name__ == "__main__":
    import time
    
    # Initialize metrics collector
    collector = MetricsCollector()
    
    # Update model version
    collector.update_model_version(
        "v1.0.0",
        metadata={"algorithm": "XGBoost", "features": "42"}
    )
    
    # Simulate some predictions
    for i in range(10):
        start_time = time.time()
        
        # Simulate prediction work
        time.sleep(0.01 + (i % 3) * 0.01)
        
        latency = time.time() - start_time
        prediction_class = "UP" if i % 2 == 0 else "DOWN"
        
        collector.record_prediction(
            latency=latency,
            model_version="v1.0.0",
            prediction_class=prediction_class,
            accuracy=0.75 + (i % 5) * 0.01
        )
    
    # Record some drift
    collector.record_drift(
        drift_score=0.15,
        drifted_features=3,
        prediction_drift=0.12
    )
    
    # Record retraining
    collector.record_retraining(trigger_reason="scheduled")
    
    # Update performance metrics
    collector.update_performance_metrics(
        accuracy=0.78,
        precision=0.76,
        recall=0.80,
        f1=0.78,
        roc_auc=0.82
    )
    
    # Get metrics in Prometheus format
    print("\nPrometheus Metrics:")
    print(collector.expose_metrics().decode('utf-8'))
