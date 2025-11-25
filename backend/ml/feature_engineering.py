"""
Feature Engineering Module for ML Monitoring System.

This module provides comprehensive feature engineering for financial time series data,
including technical indicators, price features, and time-based features.
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature Engineering class for creating technical indicators and derived features.
    
    Implements various technical indicators and price-based features commonly used
    in financial time series prediction.
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.feature_stats: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        logger.info("FeatureEngineer initialized")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            period: RSI period (default: 14)
        
        Returns:
            RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(
        self, 
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        
        Returns:
            Dictionary with macd, signal, and histogram
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: int = 2
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Number of standard deviations
        
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        middle_band = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return {
            'bb_upper': upper_band,
            'bb_middle': middle_band,
            'bb_lower': lower_band,
            'bb_width': (upper_band - lower_band) / middle_band
        }
    
    def _calculate_obv(
        self,
        prices: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            prices: Price series
            volume: Volume series
        
        Returns:
            OBV values
        """
        obv = (np.sign(prices.diff()) * volume).fillna(0).cumsum()
        return obv
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        logger.info("Calculating technical indicators...")
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], period=14)
        
        # MACD
        macd_dict = self._calculate_macd(df['close'])
        df['macd'] = macd_dict['macd']
        df['macd_signal'] = macd_dict['macd_signal']
        df['macd_histogram'] = macd_dict['macd_histogram']
        
        # Bollinger Bands
        bb_dict = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_dict['bb_upper']
        df['bb_middle'] = bb_dict['bb_middle']
        df['bb_lower'] = bb_dict['bb_lower']
        df['bb_width'] = bb_dict['bb_width']
        
        # Moving Averages
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_14'] = df['close'].rolling(window=14).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()
        
        # Exponential Moving Averages
        df['ema_7'] = df['close'].ewm(span=7, adjust=False).mean()
        df['ema_14'] = df['close'].ewm(span=14, adjust=False).mean()
        
        # Volume indicators
        df['obv'] = self._calculate_obv(df['close'], df['volume'])
        df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
        df['volume_roc'] = df['volume'].pct_change(periods=1)
        
        logger.info(f"Created {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} technical indicators")
        
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added price features
        """
        df = df.copy()
        
        logger.info("Calculating price features...")
        
        # Returns
        df['return_1h'] = df['close'].pct_change(periods=1)
        df['return_4h'] = df['close'].pct_change(periods=4)
        df['return_24h'] = df['close'].pct_change(periods=24)
        
        # Volatility
        df['volatility_24h'] = df['return_1h'].rolling(window=24).std()
        df['volatility_7d'] = df['return_1h'].rolling(window=168).std()  # 7 days
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['hl_spread_ma_7'] = df['hl_spread'].rolling(window=7).mean()
        
        # Price momentum
        df['momentum_7'] = df['close'] - df['close'].shift(7)
        df['momentum_14'] = df['close'] - df['close'].shift(14)
        df['momentum_30'] = df['close'] - df['close'].shift(30)
        
        # Price position relative to high/low
        df['price_to_high_24h'] = df['close'] / df['high'].rolling(window=24).max()
        df['price_to_low_24h'] = df['close'] / df['low'].rolling(window=24).min()
        
        logger.info(f"Created {12} price features")
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: DataFrame with timestamp column
        
        Returns:
            DataFrame with added time features
        """
        df = df.copy()
        
        logger.info("Calculating time features...")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Hour of day (0-23)
        df['hour'] = df['timestamp'].dt.hour
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Is weekend (binary)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Day of month
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Month of year
        df['month'] = df['timestamp'].dt.month
        
        logger.info(f"Created 5 time features")
        
        return df
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit the feature engineer on training data.
        
        Args:
            df: Training DataFrame
        
        Returns:
            Self
        """
        logger.info("Fitting FeatureEngineer...")
        
        # Create all features
        df_features = self.transform(df)
        
        # Get feature columns (exclude timestamp and target if present)
        feature_cols = [col for col in df_features.columns 
                       if col not in ['timestamp', 'target', 'prediction']]
        
        # Calculate statistics
        self.feature_stats = {
            'mean': df_features[feature_cols].mean().to_dict(),
            'std': df_features[feature_cols].std().to_dict(),
            'min': df_features[feature_cols].min().to_dict(),
            'max': df_features[feature_cols].max().to_dict()
        }
        
        # Fit scaler (on non-NaN values)
        df_clean = df_features[feature_cols].dropna()
        if len(df_clean) > 0:
            self.scaler.fit(df_clean)
        
        self.is_fitted = True
        logger.info(f"FeatureEngineer fitted on {len(feature_cols)} features")
        
        return self
    
    def transform(self, df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
        """
        Transform data by creating all features.
        
        Args:
            df: Input DataFrame with OHLCV data
            normalize: Whether to normalize features
        
        Returns:
            DataFrame with all features
        """
        logger.info("Transforming data...")
        
        df_transformed = df.copy()
        
        # Create all feature types
        df_transformed = self.create_technical_indicators(df_transformed)
        df_transformed = self.create_price_features(df_transformed)
        df_transformed = self.create_time_features(df_transformed)
        
        # Normalize if requested and fitted
        if normalize and self.is_fitted:
            feature_cols = [col for col in df_transformed.columns 
                           if col not in ['timestamp', 'target', 'prediction']]
            
            # Only normalize non-NaN values
            mask = df_transformed[feature_cols].notna().all(axis=1)
            if mask.sum() > 0:
                df_transformed.loc[mask, feature_cols] = self.scaler.transform(
                    df_transformed.loc[mask, feature_cols]
                )
        
        logger.info(f"Transformation complete. Shape: {df_transformed.shape}")
        
        return df_transformed
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        normalize: bool = False
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input DataFrame
            normalize: Whether to normalize features
        
        Returns:
            Transformed DataFrame
        """
        self.fit(df)
        return self.transform(df, normalize=normalize)
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names.
        
        Returns:
            List of feature names
        """
        # Create a sample dataframe to get feature names
        sample_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        df_features = self.transform(sample_df)
        feature_cols = [col for col in df_features.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return feature_cols
    
    def save_config(self, filepath: str) -> bool:
        """
        Save feature configuration to JSON.
        
        Args:
            filepath: Path to save configuration
        
        Returns:
            True if successful
        """
        try:
            config = {
                'is_fitted': self.is_fitted,
                'feature_stats': self.feature_stats,
                'feature_names': self.get_feature_names()
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def load_config(self, filepath: str) -> bool:
        """
        Load feature configuration from JSON.
        
        Args:
            filepath: Path to load configuration from
        
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            self.is_fitted = config.get('is_fitted', False)
            self.feature_stats = config.get('feature_stats', {})
            
            logger.info(f"Configuration loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Get feature names
    features = fe.get_feature_names()
    print(f"\nTotal features: {len(features)}")
    print(f"\nFeature names:\n{features}")
    
    # Fit and transform
    df_transformed = fe.fit_transform(sample_data)
    
    print(f"\nOriginal shape: {sample_data.shape}")
    print(f"Transformed shape: {df_transformed.shape}")
    print(f"\nSample features:\n{df_transformed.head()}")
    
    # Save configuration
    fe.save_config("feature_config.json")
