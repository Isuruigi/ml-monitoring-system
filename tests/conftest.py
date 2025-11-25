"""
Pytest configuration and fixtures.

This module provides shared fixtures for all tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

# Add backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
    
    # Generate realistic price data
    base_price = 40000
    price_changes = np.random.randn(n_samples).cumsum() * 100
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': base_price + price_changes,
        'high': base_price + price_changes + np.abs(np.random.randn(n_samples) * 50),
        'low': base_price + price_changes - np.abs(np.random.randn(n_samples) * 50),
        'close': base_price + price_changes + np.random.randn(n_samples) * 20,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    return df


@pytest.fixture
def sample_features_data(sample_ohlcv_data):
    """Generate sample data with features."""
    from backend.ml.feature_engineering import FeatureEngineer
    
    fe = FeatureEngineer()
    df_features = fe.fit_transform(sample_ohlcv_data)
    df_features = df_features.dropna()
    
    return df_features


@pytest.fixture
def sample_training_data(sample_features_data):
    """Generate sample training data with train/val/test splits."""
    from backend.ml.feature_engineering import FeatureEngineer
    
    df = sample_features_data.copy()
    
    # Create target
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    # Get feature columns
    fe = FeatureEngineer()
    feature_cols = fe.get_feature_names()
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Split
    train_size = int(0.6 * len(df))
    val_size = int(0.2 * len(df))
    
    X = df[available_features]
    y = df['target']
    
    return {
        'X_train': X[:train_size],
        'y_train': y[:train_size],
        'X_val': X[train_size:train_size+val_size],
        'y_val': y[train_size:train_size+val_size],
        'X_test': X[train_size+val_size:],
        'y_test': y[train_size+val_size:],
        'feature_cols': available_features
    }


@pytest.fixture
def trained_model(sample_training_data):
    """Provide a trained model for testing."""
    from backend.ml.model import PricePredictor
    
    predictor = PricePredictor(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=20  # Small for faster testing
    )
    
    predictor.train(
        sample_training_data['X_train'],
        sample_training_data['y_train'],
        sample_training_data['X_val'],
        sample_training_data['y_val'],
        experiment_name="test_experiment"
    )
    
    return predictor


@pytest.fixture
def temp_model_file():
    """Provide a temporary file for model saving."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        filepath = f.name
    
    yield filepath
    
    # Cleanup
    if os.path.exists(filepath):
        os.unlink(filepath)


@pytest.fixture
def test_db():
    """Provide a test database."""
    from backend.data.db_manager import DatabaseManager
    
    # Use in-memory SQLite for testing
    db = DatabaseManager("sqlite:///:memory:")
    db.init_db()
    
    yield db
    
    db.close()


@pytest.fixture
def api_client():
    """Provide FastAPI test client."""
    from fastapi.testclient import TestClient
    from backend.api.main import app
    
    client = TestClient(app)
    return client


@pytest.fixture
def mock_api_key():
    """Provide test API key."""
    return "test-api-key-12345"


@pytest.fixture
def mock_admin_key():
    """Provide test admin API key."""
    return "test-admin-key-12345"


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request data."""
    return {
        "timestamp": datetime.now().isoformat(),
        "features": {
            "rsi_14": 45.3,
            "macd": 0.52,
            "ma_7": 42150.5,
            "return_1h": 0.002,
            "volatility_24h": 0.015,
            "bb_upper": 43000,
            "bb_lower": 41000,
            "obv": 1000000
        }
    }
