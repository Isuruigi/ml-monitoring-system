"""
Price Prediction Model Module.

This module implements an XGBoost-based price predictor with MLflow integration
for experiment tracking and model registry.
"""

import logging
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PricePredictor:
    """
    Price Predictor using XGBoost for binary classification (price up/down).
    
    Predicts whether the price will go up or down in the next hour based on
    technical indicators and features.
    """
    
    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        """
        Initialize the PricePredictor.
        
        Args:
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            n_estimators: Number of boosting rounds
            min_child_weight: Minimum sum of instance weight in a child
            gamma: Minimum loss reduction for split
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            random_state: Random seed
        """
        self.params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: Optional[list] = None
        self.is_trained = False
        
        logger.info(f"PricePredictor initialized with params: {self.params}")
    
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary target variable (1 if price goes up, 0 if down).
        
        Args:
            df: DataFrame with 'close' column
        
        Returns:
            Binary target series
        """
        # Target: 1 if next hour's close > current close, 0 otherwise
        target = (df['close'].shift(-1) > df['close']).astype(int)
        return target
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        early_stopping_rounds: int = 50,
        experiment_name: str = "price_prediction",
        run_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train the XGBoost model with MLflow tracking.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            early_stopping_rounds: Early stopping patience
            experiment_name: MLflow experiment name
            run_name: MLflow run name
        
        Returns:
            Dictionary of training metrics
        """
        logger.info("Starting model training...")
        
        # Calculate scale_pos_weight for class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        logger.info(
            f"Class distribution - Negative: {neg_count}, "
            f"Positive: {pos_count}, Scale: {scale_pos_weight:.2f}"
        )
        
        # Update params with scale_pos_weight
        self.params['scale_pos_weight'] = scale_pos_weight
        
        # Set up MLflow
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {str(e)}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(self.params)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Initialize model
            self.model = xgb.XGBClassifier(**self.params)
            
            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
            
            self.feature_names = list(X_train.columns)
            self.is_trained = True
            
            # Evaluate on validation set
            metrics = self.evaluate(X_val, y_val)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log feature importance
            self._log_feature_importance()
            
            # Log model
            mlflow.xgboost.log_model(self.model, "model")
            
            logger.info(f"Training complete. Validation metrics: {metrics}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save_model(self, path: str) -> bool:
        """
        Save model to disk using joblib.
        
        Args:
            path: File path to save model
        
        Returns:
            True if successful
        """
        if not self.is_trained or self.model is None:
            logger.error("Cannot save untrained model")
            return False
        
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'params': self.params,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'saved_at': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, path)
            logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load model from disk.
        
        Args:
            path: File path to load model from
        
        Returns:
            True if successful
        """
        try:
            model_data = joblib.load(path)
            
            self.model = model_data['model']
            self.params = model_data['params']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained first")
        
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n)
    
    def _log_feature_importance(self) -> None:
        """Log feature importance plot to MLflow."""
        try:
            importance_df = self.get_feature_importance(top_n=20)
            
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plot_path = "feature_importance.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
            
            # Clean up
            Path(plot_path).unlink(missing_ok=True)
            
        except Exception as e:
            logger.warning(f"Could not log feature importance: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Initialize predictor
    predictor = PricePredictor(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100
    )
    
    # Train model
    metrics = predictor.train(
        X_train, y_train,
        X_val, y_val,
        experiment_name="test_experiment"
    )
    
    print(f"\nTraining metrics: {metrics}")
    
    # Evaluate on test set
    test_metrics = predictor.evaluate(X_test, y_test)
    print(f"\nTest metrics: {test_metrics}")
    
    # Get feature importance
    importance = predictor.get_feature_importance()
    print(f"\nTop features:\n{importance}")
    
    # Save model
    predictor.save_model("test_model.pkl")
    
    # Load model
    new_predictor = PricePredictor()
    new_predictor.load_model("test_model.pkl")
    
    # Make predictions
    predictions = new_predictor.predict(X_test[:5])
    print(f"\nSample predictions: {predictions}")
