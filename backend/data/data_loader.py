"""
Data Loader Module for ML Monitoring System.

This module provides functionality to fetch cryptocurrency and stock data
using yfinance with proper error handling, rate limiting, and logging.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd
import yfinance as yf
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    DataLoader class for fetching cryptocurrency and stock market data.
    
    Attributes:
        ticker (str): The ticker symbol (e.g., 'BTC-USD', 'AAPL')
        interval (str): Data interval ('1m', '5m', '15m', '1h', '1d', etc.)
        rate_limit_delay (float): Delay between API calls to respect rate limits
    """
    
    def __init__(
        self, 
        ticker: str, 
        interval: str = "1h",
        rate_limit_delay: float = 1.0
    ):
        """
        Initialize the DataLoader.
        
        Args:
            ticker: Ticker symbol to fetch data for
            interval: Time interval for data points
            rate_limit_delay: Seconds to wait between API calls
        """
        self.ticker = ticker
        self.interval = interval
        self.rate_limit_delay = rate_limit_delay
        self.yf_ticker = yf.Ticker(ticker)
        self._last_request_time = 0
        
        logger.info(
            f"DataLoader initialized for {ticker} with interval {interval}"
        )
    
    def _respect_rate_limit(self) -> None:
        """Ensure rate limiting by waiting if necessary."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def fetch_historical_data(
        self,
        start: str,
        end: str,
        retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for the specified date range.
        
        Args:
            start: Start date in 'YYYY-MM-DD' format
            end: End date in 'YYYY-MM-DD' format
            retries: Number of retry attempts on failure
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        self._respect_rate_limit()
        
        for attempt in range(retries):
            try:
                logger.info(
                    f"Fetching historical data for {self.ticker} "
                    f"from {start} to {end} (attempt {attempt + 1}/{retries})"
                )
                
                df = self.yf_ticker.history(
                    start=start,
                    end=end,
                    interval=self.interval
                )
                
                if df.empty:
                    logger.warning(
                        f"No data returned for {self.ticker} "
                        f"from {start} to {end}"
                    )
                    return None
                
                # Clean column names
                df.columns = df.columns.str.lower()
                
                # Reset index to make datetime a column
                df.reset_index(inplace=True)
                df.rename(columns={'date': 'timestamp'}, inplace=True)
                
                logger.info(
                    f"Successfully fetched {len(df)} rows "
                    f"for {self.ticker}"
                )
                
                return df
                
            except Exception as e:
                logger.error(
                    f"Error fetching historical data (attempt {attempt + 1}/"
                    f"{retries}): {str(e)}"
                )
                
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to fetch data after {retries} attempts"
                    )
                    return None
        
        return None
    
    def fetch_live_data(
        self,
        lookback_hours: int = 24
    ) -> Optional[pd.DataFrame]:
        """
        Fetch recent data to simulate real-time data stream.
        
        Args:
            lookback_hours: Number of hours to look back
        
        Returns:
            DataFrame with recent OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=lookback_hours)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(
            f"Fetching live data for {self.ticker} "
            f"(last {lookback_hours} hours)"
        )
        
        return self.fetch_historical_data(start_str, end_str)
    
    def get_latest_price(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest price information.
        
        Returns:
            Dictionary with latest price data or None if failed
        """
        self._respect_rate_limit()
        
        try:
            info = self.yf_ticker.info
            
            latest_data = {
                'ticker': self.ticker,
                'price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'volume': info.get('volume') or info.get('regularMarketVolume'),
                'market_cap': info.get('marketCap'),
                'timestamp': datetime.now()
            }
            
            logger.info(
                f"Latest price for {self.ticker}: "
                f"${latest_data.get('price', 'N/A')}"
            )
            
            return latest_data
            
        except Exception as e:
            logger.error(f"Error fetching latest price: {str(e)}")
            return None
    
    def save_to_csv(
        self,
        df: pd.DataFrame,
        output_dir: str = "data",
        filename: Optional[str] = None
    ) -> bool:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            output_dir: Directory to save the file
            filename: Custom filename (auto-generated if None)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{self.ticker}_{self.interval}_{timestamp}.csv"
            
            filepath = Path(output_dir) / filename
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            
            logger.info(f"Data saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to CSV: {str(e)}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get ticker information.
        
        Returns:
            Dictionary with ticker metadata
        """
        self._respect_rate_limit()
        
        try:
            info = self.yf_ticker.info
            return {
                'ticker': self.ticker,
                'name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            logger.error(f"Error fetching ticker info: {str(e)}")
            return {'ticker': self.ticker, 'error': str(e)}


# Example usage
if __name__ == "__main__":
    # Initialize loader for Bitcoin
    loader = DataLoader(ticker="BTC-USD", interval="1h")
    
    # Fetch historical data
    df = loader.fetch_historical_data(
        start="2023-01-01",
        end="2024-01-01"
    )
    
    if df is not None:
        print(f"\nFetched {len(df)} rows")
        print(f"\nFirst few rows:\n{df.head()}")
        print(f"\nData info:\n{df.info()}")
        
        # Save to CSV
        loader.save_to_csv(df, filename="btc_historical.csv")
    
    # Fetch live data
    live_df = loader.fetch_live_data(lookback_hours=24)
    if live_df is not None:
        print(f"\nLive data: {len(live_df)} rows")
    
    # Get latest price
    latest = loader.get_latest_price()
    if latest:
        print(f"\nLatest price info: {latest}")
    
    # Get ticker info
    info = loader.get_info()
    print(f"\nTicker info: {info}")
