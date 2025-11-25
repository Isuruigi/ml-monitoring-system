"""
Automated Model Retraining Module.

This module implements automated retraining pipeline with trigger conditions,
validation suite, and safety features.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from backend.data.data_loader import DataLoader
from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.model import PricePredictor
from backend.ml.drift_detector import DriftDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRetrainer:
    """
    Automated model retraining system with safety features.
    
    Handles retraining triggers, validation, and deployment.
    """
    
    def __init__(
        self,
        ticker: str = "BTC-USD",
        drift_threshold: float = 0.25,
        accuracy_threshold: float = 0.55,
        cooldown_hours: int = 12
    ):
        """
        Initialize the ModelRetrainer.
        
        Args:
            ticker: Asset ticker symbol
            drift_threshold: Drift score threshold for retraining
            accuracy_threshold: Minimum accuracy threshold
            cooldown_hours: Minimum hours between retraining runs
        """
        self.ticker = ticker
        self.drift_threshold = drift_threshold
        self.accuracy_threshold = accuracy_threshold
        self.cooldown_hours = cooldown_hours
        
        self.last_retraining_time: Optional[datetime] = None
        self.model_history: list = []
        
        logger.info(
            f"ModelRetrainer initialized for {ticker} "
            f"(drift_threshold={drift_threshold}, "
            f"accuracy_threshold={accuracy_threshold})"
        )
    
    def should_retrain(
        self,
        drift_score: float,
        current_accuracy: Optional[float] = None,
        days_since_last: Optional[int] = None
    ) -> tuple[bool, str]:
        """
        Determine if retraining should be triggered.
        
        Args:
            drift_score: Current drift score
            current_accuracy: Current model accuracy
            days_since_last: Days since last retraining
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        # Check cooldown period
        if self.last_retraining_time:
            time_since = datetime.now() - self.last_retraining_time
            if time_since < timedelta(hours=self.cooldown_hours):
                hours_remaining = self.cooldown_hours - (time_since.seconds / 3600)
                return False, f"Cooldown active ({hours_remaining:.1f}h remaining)"
        
        # Check drift
        if drift_score > self.drift_threshold:
            return True, f"Data drift exceeded threshold ({drift_score:.3f} > {self.drift_threshold})"
        
        # Check accuracy
        if current_accuracy and current_accuracy < self.accuracy_threshold:
            return True, f"Accuracy below threshold ({current_accuracy:.3f} < {self.accuracy_threshold})"
        
        # Check scheduled retraining
        if days_since_last and days_since_last >= 7:
            return True, f"Scheduled retraining (last: {days_since_last} days ago)"
        
        return False, "No retraining needed"
    
    def retrain_pipeline(
        self,
        job_id: str,
        trigger_reason: str,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Execute full retraining pipeline.
        
        Args:
            job_id: Unique job identifier
            trigger_reason: Reason for retraining
            lookback_days: Days of historical data to use
        
        Returns:
            Retraining report dictionary
        """
        logger.info(f"Starting retraining pipeline (job_id={job_id}, reason={trigger_reason})")
        
        start_time = datetime.now()
        report = {
            'job_id': job_id,
            'trigger_reason': trigger_reason,
            'start_time': start_time,
            'status': 'running',
            'steps': []
        }
        
        try:
            # Step 1: Fetch latest data
            logger.info("Step 1: Fetching latest data...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            data_loader = DataLoader(ticker=self.ticker, interval="1h")
            df = data_loader.fetch_historical_data(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            if df is None or len(df) < 100:
                raise ValueError("Insufficient data for retraining")
            
            report['steps'].append({
                'step': 'data_fetch',
                'status': 'success',
                'rows': len(df)
            })
            
            # Step 2: Feature engineering
            logger.info("Step 2: Engineering features...")
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.fit_transform(df)
            
            # Remove NaN values
            df_features = df_features.dropna()
            
            report['steps'].append({
                'step': 'feature_engineering',
                'status': 'success',
                'features': len(feature_engineer.get_feature_names())
            })
            
            # Step 3: Prepare target variable
            logger.info("Step 3: Preparing target...")
            df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
            df_features = df_features.dropna()
            
            # Step 4: Split data
            logger.info("Step 4: Splitting data...")
            train_size = int(0.6 * len(df_features))
            val_size = int(0.2 * len(df_features))
            
            feature_cols = feature_engineer.get_feature_names()
            
            X = df_features[feature_cols]
            y = df_features['target']
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size+val_size]
            y_val = y[train_size:train_size+val_size]
            X_test = X[train_size+val_size:]
            y_test = y[train_size+val_size:]
            
            report['steps'].append({
                'step': 'data_split',
                'status': 'success',
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            })
            
            # Step 5: Train new model
            logger.info("Step 5: Training new model...")
            new_model = PricePredictor(
                max_depth=6,
                learning_rate=0.05,
                n_estimators=150
            )
            
            train_metrics = new_model.train(
                X_train, y_train,
                X_val, y_val,
                experiment_name="automated_retraining",
                run_name=f"retrain_{job_id}"
            )
            
            report['steps'].append({
                'step': 'model_training',
                'status': 'success',
                'metrics': train_metrics
            })
            
            # Step 6: Evaluate on test set
            logger.info("Step 6: Evaluating new model...")
            test_metrics = new_model.evaluate(X_test, y_test)
            
            report['steps'].append({
                'step': 'model_evaluation',
                'status': 'success',
                'metrics': test_metrics
            })
            
            # Step 7: Compare with current production model
            logger.info("Step 7: Comparing with production model...")
            # TODO: Load current production model and compare
            # For now, we'll assume new model is better if accuracy > threshold
            
            is_better = test_metrics['accuracy'] >= self.accuracy_threshold
            improvement = test_metrics['accuracy'] - 0.50  # Placeholder baseline
            
            report['steps'].append({
                'step': 'model_comparison',
                'status': 'success',
                'new_model_better': is_better,
                'improvement': improvement
            })
            
            # Step 8: Validation suite
            logger.info("Step 8: Running validation suite...")
            validation_results = self.validate_model(
                new_model,
                X_test,
                y_test,
                test_metrics
            )
            
            report['steps'].append({
                'step': 'validation',
                'status': 'success',
                'results': validation_results
            })
            
            # Step 9: Save or deploy model
            if is_better and validation_results['passed']:
                logger.info("Step 9: Saving new model...")
                model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model_path = f"models/{model_version}.pkl"
                
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                new_model.save_model(model_path)
                
                # Save feature engineer config
                feature_engineer.save_config(f"models/{model_version}_features.json")
                
                report['new_model_version'] = model_version
                report['model_path'] = model_path
                report['deployed'] = True
                
                self.last_retraining_time = datetime.now()
                
                report['steps'].append({
                    'step': 'model_deployment',
                    'status': 'success',
                    'version': model_version
                })
            else:
                logger.warning("New model did not pass validation or is not better")
                report['deployed'] = False
                report['reason'] = "Model did not meet deployment criteria"
                
                report['steps'].append({
                    'step': 'model_deployment',
                    'status': 'skipped',
                    'reason': 'Did not pass validation'
                })
            
            # Finalize report
            report['status'] = 'completed'
            report['end_time'] = datetime.now()
            report['duration_seconds'] = (report['end_time'] - start_time).total_seconds()
            report['final_metrics'] = test_metrics
            
            logger.info(
                f"Retraining pipeline completed successfully "
                f"(duration={report['duration_seconds']:.1f}s)"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Retraining pipeline failed: {str(e)}", exc_info=True)
            
            report['status'] = 'failed'
            report['error'] = str(e)
            report['end_time'] = datetime.now()
            report['duration_seconds'] = (report['end_time'] - start_time).total_seconds()
            
            return report
    
    def validate_model(
        self,
        model: PricePredictor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Run validation suite on new model.
        
        Args:
            model: Model to validate
            X_test: Test features
            y_test: Test target
            metrics: Evaluation metrics
        
        Returns:
            Validation results
        """
        logger.info("Running model validation suite...")
        
        validation = {
            'checks': [],
            'passed': True
        }
        
        # Check 1: Accuracy threshold
        accuracy_ok = metrics['accuracy'] >= self.accuracy_threshold - 0.02
        validation['checks'].append({
            'name': 'accuracy_threshold',
            'passed': accuracy_ok,
            'value': metrics['accuracy'],
            'threshold': self.accuracy_threshold - 0.02
        })
        
        # Check 2: Prediction latency
        import time
        start = time.time()
        _ = model.predict(X_test.head(100))
        latency = (time.time() - start) / 100  # Per-prediction latency
        
        latency_ok = latency < 0.1  # 100ms threshold
        validation['checks'].append({
            'name': 'prediction_latency',
            'passed': latency_ok,
            'value': latency,
            'threshold': 0.1
        })
        
        # Check 3: No severe class imbalance in predictions
        predictions = model.predict(X_test)
        class_balance = predictions.mean()  # Proportion of class 1
        
        balance_ok = 0.3 < class_balance < 0.7
        validation['checks'].append({
            'name': 'prediction_balance',
            'passed': balance_ok,
            'value': class_balance,
            'range': [0.3, 0.7]
        })
        
        # Overall validation
        validation['passed'] = all(check['passed'] for check in validation['checks'])
        
        logger.info(f"Validation suite: {'PASSED' if validation['passed'] else 'FAILED'}")
        
        return validation
    
    def generate_model_card(
        self,
        model_version: str,
        metrics: Dict[str, float],
        training_data_stats: Dict[str, Any]
    ) -> str:
        """
        Generate model card documentation.
        
        Args:
            model_version: Model version string
            metrics: Performance metrics
            training_data_stats: Statistics about training data
        
        Returns:
            Model card markdown content
        """
        card = f"""# Model Card: {model_version}

## Model Details
- **Version**: {model_version}
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Algorithm**: XGBoost Classifier
- **Task**: Binary classification (price direction prediction)

## Performance Metrics
- **Accuracy**: {metrics.get('accuracy', 0):.4f}
- **Precision**: {metrics.get('precision', 0):.4f}
- **Recall**: {metrics.get('recall', 0):.4f}
- **F1 Score**: {metrics.get('f1_score', 0):.4f}
- **ROC-AUC**: {metrics.get('roc_auc', 0):.4f}

## Training Data
- **Ticker**: {self.ticker}
- **Samples**: {training_data_stats.get('samples', 'N/A')}
- **Features**: {training_data_stats.get('features', 'N/A')}
- **Date Range**: {training_data_stats.get('date_range', 'N/A')}

## Intended Use
This model predicts whether the price will go UP or DOWN in the next hour based on technical indicators and price features.

## Limitations
- Performance may degrade during high volatility periods
- Not suitable for very short-term trading (< 5 minutes)
- Requires continuous monitoring for data drift

## Ethical Considerations
- This model is for educational/research purposes
- Trading decisions should not rely solely on model predictions
- Past performance does not guarantee future results
"""
        
        return card


# Example usage
if __name__ == "__main__":
    retrainer = ModelRetrainer(
        ticker="BTC-USD",
        drift_threshold=0.25,
        accuracy_threshold=0.55
    )
    
    # Check if should retrain
    should, reason = retrainer.should_retrain(
        drift_score=0.28,
        current_accuracy=0.52,
        days_since_last=5
    )
    
    print(f"Should retrain: {should}")
    print(f"Reason: {reason}")
    
    if should:
        # Run retraining
        report = retrainer.retrain_pipeline(
            job_id="test_job_001",
            trigger_reason=reason
        )
        
        print(f"\nRetraining Report:")
        print(f"Status: {report['status']}")
        print(f"Duration: {report.get('duration_seconds', 0):.1f}s")
        if report.get('final_metrics'):
            print(f"Final Accuracy: {report['final_metrics']['accuracy']:.4f}")
