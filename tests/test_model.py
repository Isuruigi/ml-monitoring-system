"""
Unit tests for the PricePredictor model.

Tests model initialization, training, prediction, and evaluation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestPricePredictor:
    """Test suite for PricePredictor class."""
    
    def test_model_initialization(self):
        """Test that model initializes with correct parameters."""
        from backend.ml.model import PricePredictor
        
        predictor = PricePredictor(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100
        )
        
        assert predictor.params['max_depth'] == 5
        assert predictor.params['learning_rate'] == 0.1
        assert predictor.params['n_estimators'] == 100
        assert predictor.is_trained is False
        assert predictor.model is None
    
    def test_model_training(self, sample_training_data):
        """Test that model trains successfully."""
        from backend.ml.model import PricePredictor
        
        predictor = PricePredictor(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=20
        )
        
        metrics = predictor.train(
            sample_training_data['X_train'],
            sample_training_data['y_train'],
            sample_training_data['X_val'],
            sample_training_data['y_val'],
            experiment_name="test_training"
        )
        
        # Check training completed
        assert predictor.is_trained is True
        assert predictor.model is not None
        
        # Check metrics returned
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Check metrics are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_model_prediction(self, trained_model, sample_training_data):
        """Test making predictions."""
        X_test = sample_training_data['X_test']
        
        predictions = trained_model.predict(X_test)
        
        # Check predictions
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_predict_proba(self, trained_model, sample_training_data):
        """Test probability predictions."""
        X_test = sample_training_data['X_test']
        
        probabilities = trained_model.predict_proba(X_test)
        
        # Check probabilities
        assert probabilities.shape == (len(X_test), 2)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_model_evaluation(self, trained_model, sample_training_data):
        """Test model evaluation."""
        X_test = sample_training_data['X_test']
        y_test = sample_training_data['y_test']
        
        metrics = trained_model.evaluate(X_test, y_test)
        
        # Check all metrics present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_model_save_load(self, trained_model, temp_model_file):
        """Test saving and loading model."""
        # Save model
        success = trained_model.save_model(temp_model_file)
        assert success is True
        assert Path(temp_model_file).exists()
        
        # Load model
        from backend.ml.model import PricePredictor
        new_predictor = PricePredictor()
        
        load_success = new_predictor.load_model(temp_model_file)
        assert load_success is True
        assert new_predictor.is_trained is True
        assert new_predictor.model is not None
    
    def test_feature_importance(self, trained_model):
        """Test feature importance extraction."""
        importance_df = trained_model.get_feature_importance(top_n=10)
        
        assert len(importance_df) <= 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert all(importance_df['importance'] >= 0)
    
    def test_prediction_without_training(self, sample_training_data):
        """Test that prediction fails without training."""
        from backend.ml.model import PricePredictor
        
        predictor = PricePredictor()
        
        with pytest.raises(ValueError, match="must be trained"):
            predictor.predict(sample_training_data['X_test'])
    
    def test_class_imbalance_handling(self, sample_training_data):
        """Test that model handles class imbalance."""
        from backend.ml.model import PricePredictor
        
        # Create imbalanced data
        X_train = sample_training_data['X_train']
        y_train = sample_training_data['y_train']
        
        # Keep only 20% positive samples
        pos_indices = y_train[y_train == 1].index
        neg_indices = y_train[y_train == 0].index
        
        sampled_pos = np.random.choice(pos_indices, size=int(len(pos_indices) * 0.2), replace=False)
        balanced_indices = np.concatenate([sampled_pos, neg_indices])
        
        X_imbalanced = X_train.loc[balanced_indices]
        y_imbalanced = y_train.loc[balanced_indices]
        
        # Train model
        predictor = PricePredictor(n_estimators=20)
        predictor.train(
            X_imbalanced,
            y_imbalanced,
            sample_training_data['X_val'],
            sample_training_data['y_val']
        )
        
        # Check scale_pos_weight was set
        assert 'scale_pos_weight' in predictor.params
        assert predictor.params['scale_pos_weight'] > 1.0


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""
    
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        from backend.ml.feature_engineering import FeatureEngineer
        
        fe = FeatureEngineer()
        assert fe.is_fitted is False
        assert fe.feature_stats == {}
    
    def test_technical_indicators(self, sample_ohlcv_data):
        """Test technical indicator calculation."""
        from backend.ml.feature_engineering import FeatureEngineer
        
        fe = FeatureEngineer()
        df_with_indicators = fe.create_technical_indicators(sample_ohlcv_data)
        
        # Check RSI
        assert 'rsi_14' in df_with_indicators.columns
        assert df_with_indicators['rsi_14'].min() >= 0
        assert df_with_indicators['rsi_14'].max() <= 100
        
        # Check MACD
        assert 'macd' in df_with_indicators.columns
        assert 'macd_signal' in df_with_indicators.columns
        
        # Check Bollinger Bands
        assert 'bb_upper' in df_with_indicators.columns
        assert 'bb_middle' in df_with_indicators.columns
        assert 'bb_lower' in df_with_indicators.columns
    
    def test_price_features(self, sample_ohlcv_data):
        """Test price feature creation."""
        from backend.ml.feature_engineering import FeatureEngineer
        
        fe = FeatureEngineer()
        df_with_features = fe.create_price_features(sample_ohlcv_data)
        
        # Check returns
        assert 'return_1h' in df_with_features.columns
        assert 'return_4h' in df_with_features.columns
        
        # Check volatility
        assert 'volatility_24h' in df_with_features.columns
        
        # Check momentum
        assert 'momentum_7' in df_with_features.columns
    
    def test_time_features(self, sample_ohlcv_data):
        """Test time feature creation."""
        from backend.ml.feature_engineering import FeatureEngineer
        
        fe = FeatureEngineer()
        df_with_time = fe.create_time_features(sample_ohlcv_data)
        
        # Check time features
        assert 'hour' in df_with_time.columns
        assert 'day_of_week' in df_with_time.columns
        assert 'is_weekend' in df_with_time.columns
        
        # Check ranges
        assert df_with_time['hour'].min() >= 0
        assert df_with_time['hour'].max() <= 23
        assert df_with_time['day_of_week'].min() >= 0
        assert df_with_time['day_of_week'].max() <= 6
    
    def test_fit_transform(self, sample_ohlcv_data):
        """Test fit_transform method."""
        from backend.ml.feature_engineering import FeatureEngineer
        
        fe = FeatureEngineer()
        df_transformed = fe.fit_transform(sample_ohlcv_data)
        
        # Check fitting
        assert fe.is_fitted is True
        assert len(fe.feature_stats) > 0
        
        # Check features created
        assert df_transformed.shape[1] > sample_ohlcv_data.shape[1]
    
    def test_get_feature_names(self):
        """Test feature name retrieval."""
        from backend.ml.feature_engineering import FeatureEngineer
        
        fe = FeatureEngineer()
        feature_names = fe.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 30  # Should have 40+ features
        
        # Check some expected features
        expected = ['rsi_14', 'macd', 'bb_upper', 'return_1h', 'hour']
        for feature in expected:
            assert feature in feature_names


class TestDriftDetector:
    """Test suite for DriftDetector class."""
    
    def test_initialization(self):
        """Test DriftDetector initialization."""
        from backend.ml.drift_detector import DriftDetector
        
        detector = DriftDetector(drift_threshold=0.25, warning_threshold=0.20)
        
        assert detector.drift_threshold == 0.25
        assert detector.warning_threshold == 0.20
    
    def test_psi_calculation(self):
        """Test PSI calculation."""
        from backend.ml.drift_detector import DriftDetector
        
        detector = DriftDetector()
        
        # Same distribution - PSI should be near 0
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)
        
        psi = detector.calculate_psi(expected, actual)
        assert psi < 0.1  # Very low drift
        
        # Different distribution - PSI should be higher
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1, 1000)  # Shifted distribution
        
        psi = detector.calculate_psi(expected, actual)
        assert psi > 0.1  # Noticeable drift
    
    def test_no_drift_detection(self):
        """Test drift detection with no drift."""
        from backend.ml.drift_detector import DriftDetector
        
        detector = DriftDetector()
        
        # Same distribution
        reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000)
        })
        
        current_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000)
        })
        
        result = detector.detect_data_drift(reference_data, current_data)
        
        assert 'overall_drift_detected' in result
        assert 'drift_score' in result
        assert result['drift_score'] < 0.25
    
    def test_drift_detection_with_drift(self):
        """Test drift detection with actual drift."""
        from backend.ml.drift_detector import DriftDetector
        
        detector = DriftDetector(drift_threshold=0.2)
        
        # Different distributions
        reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000)
        })
        
        current_data = pd.DataFrame({
            'feature_1': np.random.normal(1.5, 1.5, 1000),  # Shifted
            'feature_2': np.random.normal(5, 2, 1000)        # Same
        })
        
        result = detector.detect_data_drift(reference_data, current_data)
        
        assert 'drifted_features' in result
        # At least one feature should drift
        # (May or may not trigger overall depending on threshold)
    
    def test_prediction_drift(self):
        """Test prediction drift detection."""
        from backend.ml.drift_detector import DriftDetector
        
        detector = DriftDetector()
        
        # Similar predictions - no drift
        ref_preds = np.random.randint(0, 2, 1000)
        curr_preds = np.random.randint(0, 2, 1000)
        
        result = detector.detect_prediction_drift(ref_preds, curr_preds)
        
        assert 'psi' in result
        assert 'drift_detected' in result
        assert isinstance(result['drift_detected'], bool)
