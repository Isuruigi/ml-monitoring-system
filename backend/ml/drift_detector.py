"""
Drift Detection Module using Evidently AI.

This module provides comprehensive drift detection capabilities for monitoring
data drift, prediction drift, and target drift in ML models.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from scipy import stats
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Drift Detector using Evidently AI for monitoring ML model drift.
    
    Monitors three types of drift:
    - Data drift: Changes in input feature distributions
    - Prediction drift: Changes in model output distributions
    - Target drift: Changes in actual target distributions
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.25,
        warning_threshold: float = 0.20
    ):
        """
        Initialize the DriftDetector.
        
        Args:
            drift_threshold: PSI threshold for drift alert
            warning_threshold: PSI threshold for drift warning
        """
        self.drift_threshold = drift_threshold
        self.warning_threshold = warning_threshold
        self.reference_stats: Dict[str, Any] = {}
        
        logger.info(
            f"DriftDetector initialized with thresholds: "
            f"warning={warning_threshold}, alert={drift_threshold}"
        )
    
    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect data drift in input features.
        
        Args:
            reference_data: Reference (baseline) dataset
            current_data: Current dataset to compare
            feature_columns: List of feature columns to check (None = all numeric)
        
        Returns:
            Dictionary with drift detection results
        """
        logger.info("Detecting data drift...")
        
        if feature_columns is None:
            feature_columns = reference_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
        
        drift_results = {
            'overall_drift_detected': False,
            'drift_score': 0.0,
            'drifted_features': [],
            'feature_drift_scores': {},
            'timestamp': datetime.now().isoformat()
        }
        
        total_psi = 0.0
        
        for feature in feature_columns:
            try:
                # Calculate PSI for this feature
                psi = self.calculate_psi(
                    reference_data[feature].dropna().values,
                    current_data[feature].dropna().values
                )
                
                drift_results['feature_drift_scores'][feature] = float(psi)
                total_psi += psi
                
                # Check if feature has drifted
                if psi > self.drift_threshold:
                    drift_results['drifted_features'].append(feature)
                    logger.warning(
                        f"Feature '{feature}' has drifted: PSI={psi:.4f}"
                    )
                elif psi > self.warning_threshold:
                    logger.info(
                        f"Feature '{feature}' warning: PSI={psi:.4f}"
                    )
                
            except Exception as e:
                logger.error(f"Error calculating drift for {feature}: {str(e)}")
        
        # Calculate overall drift score (average PSI)
        drift_results['drift_score'] = total_psi / len(feature_columns)
        drift_results['overall_drift_detected'] = (
            drift_results['drift_score'] > self.drift_threshold
        )
        
        logger.info(
            f"Data drift detection complete. Overall drift score: "
            f"{drift_results['drift_score']:.4f}, "
            f"Drifted features: {len(drift_results['drifted_features'])}"
        )
        
        return drift_results
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect prediction drift using chi-square test.
        
        Args:
            reference_predictions: Reference predictions
            current_predictions: Current predictions
        
        Returns:
            Dictionary with drift metrics
        """
        logger.info("Detecting prediction drift...")
        
        try:
            # Calculate PSI for predictions
            psi = self.calculate_psi(reference_predictions, current_predictions)
            
            # Chi-square test for distribution similarity
            ref_counts = np.bincount(reference_predictions.astype(int))
            curr_counts = np.bincount(current_predictions.astype(int))
            
            # Ensure same length
            max_len = max(len(ref_counts), len(curr_counts))
            ref_counts = np.pad(ref_counts, (0, max_len - len(ref_counts)))
            curr_counts = np.pad(curr_counts, (0, max_len - len(curr_counts)))
            
            chi2_stat, p_value = stats.chisquare(curr_counts, ref_counts)
            
            result = {
                'psi': float(psi),
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'drift_detected': p_value < 0.05,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(
                f"Prediction drift: PSI={psi:.4f}, "
                f"chi2={chi2_stat:.4f}, p-value={p_value:.4f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting prediction drift: {str(e)}")
            return {
                'error': str(e),
                'drift_detected': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI Interpretation:
        - PSI < 0.10: No significant population change
        - 0.10 <= PSI < 0.20: Moderate population change
        - PSI >= 0.20: Significant population change
        
        Args:
            expected: Expected (reference) distribution
            actual: Actual (current) distribution
            buckets: Number of quantile buckets
        
        Returns:
            PSI value
        """
        # Remove NaN values
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        
        if len(expected) == 0 or len(actual) == 0:
            logger.warning("Empty array for PSI calculation")
            return 0.0
        
        # Create quantile bins from expected distribution
        breakpoints = np.linspace(0, 100, buckets + 1)
        breakpoints = np.unique(np.percentile(expected, breakpoints))
        
        # If we have fewer unique values than buckets, adjust
        if len(breakpoints) <= 2:
            logger.warning(
                f"Not enough unique values for PSI calculation: "
                f"{len(breakpoints)} breakpoints"
            )
            return 0.0
        
        # Bucket the data
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * 
                     np.log(actual_percents / expected_percents))
        
        return float(psi)
    
    def generate_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: str = "reports/drift_report.html"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive Evidently drift report.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            output_path: Path to save HTML report
        
        Returns:
            Summary statistics from the report
        """
        logger.info("Generating drift report...")
        
        try:
            # Create output directory
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create Evidently report
            report = Report(metrics=[
                DataDriftPreset(),
                DatasetDriftMetric(),
                DatasetMissingValuesMetric()
            ])
            
            # Run the report
            report.run(
                reference_data=reference_data,
                current_data=current_data
            )
            
            # Save HTML report
            report.save_html(output_path)
            
            # Extract summary
            report_dict = report.as_dict()
            
            # Get dataset drift metric
            dataset_drift = None
            for metric in report_dict.get('metrics', []):
                if metric.get('metric') == 'DatasetDriftMetric':
                    dataset_drift = metric.get('result', {})
                    break
            
            summary = {
                'report_path': output_path,
                'drift_detected': dataset_drift.get('dataset_drift', False) if dataset_drift else False,
                'drift_share': dataset_drift.get('drift_share', 0.0) if dataset_drift else 0.0,
                'number_of_drifted_columns': dataset_drift.get('number_of_drifted_columns', 0) if dataset_drift else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Drift report saved to {output_path}")
            logger.info(f"Drift summary: {summary}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating drift report: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_reference_distribution(
        self,
        reference_data: pd.DataFrame,
        filepath: str = "models/reference_distribution.json"
    ) -> bool:
        """
        Save reference distribution statistics.
        
        Args:
            reference_data: Reference dataset
            filepath: Path to save statistics
        
        Returns:
            True if successful
        """
        try:
            numeric_cols = reference_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            
            stats_dict = {}
            for col in numeric_cols:
                data = reference_data[col].dropna().values
                stats_dict[col] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'percentiles': {
                        '25': float(np.percentile(data, 25)),
                        '50': float(np.percentile(data, 50)),
                        '75': float(np.percentile(data, 75))
                    }
                }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            
            self.reference_stats = stats_dict
            logger.info(f"Reference distribution saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving reference distribution: {str(e)}")
            return False
    
    def load_reference_distribution(self, filepath: str) -> bool:
        """
        Load reference distribution statistics.
        
        Args:
            filepath: Path to load statistics from
        
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                self.reference_stats = json.load(f)
            
            logger.info(f"Reference distribution loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading reference distribution: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Create synthetic reference and current data
    np.random.seed(42)
    n_samples = 1000
    
    reference_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(5, 2, n_samples),
        'feature_3': np.random. exponential(1, n_samples),
        'prediction': np.random.randint(0, 2, n_samples)
    })
    
    # Current data with drift
    current_data = pd.DataFrame({
        'feature_1': np.random.normal(0.5, 1.2, n_samples),  # Shifted mean
        'feature_2': np.random.normal(5, 2, n_samples),      # No drift
        'feature_3': np.random.exponential(1.5, n_samples),  # Different scale
        'prediction': np.random.randint(0, 2, n_samples)
    })
    
    # Initialize detector
    detector = DriftDetector(drift_threshold=0.25, warning_threshold=0.20)
    
    # Detect data drift
    drift_result = detector.detect_data_drift(
        reference_data,
        current_data,
        feature_columns=['feature_1', 'feature_2', 'feature_3']
    )
    
    print("\nData Drift Results:")
    print(json.dumps(drift_result, indent=2))
    
    # Detect prediction drift
    pred_drift = detector.detect_prediction_drift(
        reference_data['prediction'].values,
        current_data['prediction'].values
    )
    
    print("\nPrediction Drift Results:")
    print(json.dumps(pred_drift, indent=2))
    
    # Generate drift report
    report_summary = detector.generate_drift_report(
        reference_data.drop('prediction', axis=1),
        current_data.drop('prediction', axis=1)
    )
    
    print("\nDrift Report Summary:")
    print(json.dumps(report_summary, indent=2))
    
    # Save reference distribution
    detector.save_reference_distribution(reference_data)
