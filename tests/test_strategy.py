"""
Unit tests for the Strategy class in strategy.py
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategy import Strategy


class TestStrategyInit:
    """Tests for Strategy initialization."""
    
    def test_strategy_initializes_successfully(self):
        """Test that Strategy can be instantiated."""
        strategy = Strategy()
        assert strategy is not None


class TestCalculateIndicators:
    """Tests for the calculate_indicators method."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
        
        data = pd.DataFrame({
            'Open': close_prices - np.random.rand(100),
            'High': close_prices + np.random.rand(100) * 2,
            'Low': close_prices - np.random.rand(100) * 2,
            'Close': close_prices,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        return data
    
    def test_calculate_indicators_adds_macd_columns(self, sample_ohlcv_data):
        """Test that MACD columns are added to the dataframe."""
        strategy = Strategy()
        result = strategy.calculate_indicators(sample_ohlcv_data)
        
        assert 'MACD_12_26_9' in result.columns
        assert 'MACDs_12_26_9' in result.columns
        assert 'MACDh_12_26_9' in result.columns
    
    def test_calculate_indicators_adds_rsi_column(self, sample_ohlcv_data):
        """Test that RSI column is added to the dataframe."""
        strategy = Strategy()
        result = strategy.calculate_indicators(sample_ohlcv_data)
        
        assert 'RSI_14' in result.columns
    
    def test_calculate_indicators_preserves_original_columns(self, sample_ohlcv_data):
        """Test that original OHLCV columns are preserved."""
        strategy = Strategy()
        result = strategy.calculate_indicators(sample_ohlcv_data)
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            assert col in result.columns
    
    def test_calculate_indicators_returns_dataframe(self, sample_ohlcv_data):
        """Test that the method returns a DataFrame."""
        strategy = Strategy()
        result = strategy.calculate_indicators(sample_ohlcv_data)
        
        assert isinstance(result, pd.DataFrame)


class TestGenerateSignals:
    """Tests for the generate_signals method."""
    
    @pytest.fixture
    def strategy(self):
        """Create a Strategy instance for testing."""
        return Strategy()
    
    def test_buy_signal_when_conditions_met(self, strategy):
        """Test BUY signal when MACD > Signal, MACDh > 0, and RSI < 70."""
        row = pd.Series({
            'MACD_12_26_9': 1.5,
            'MACDs_12_26_9': 1.0,
            'MACDh_12_26_9': 0.5,
            'RSI_14': 50.0
        })
        assert strategy.generate_signals(row) == 'BUY'
    
    def test_sell_signal_when_conditions_met(self, strategy):
        """Test SELL signal when MACD < Signal, MACDh < 0, and RSI > 30."""
        row = pd.Series({
            'MACD_12_26_9': 0.5,
            'MACDs_12_26_9': 1.0,
            'MACDh_12_26_9': -0.5,
            'RSI_14': 50.0
        })
        assert strategy.generate_signals(row) == 'SELL'
    
    def test_hold_signal_when_rsi_overbought(self, strategy):
        """Test HOLD signal when RSI >= 70 (overbought)."""
        row = pd.Series({
            'MACD_12_26_9': 1.5,
            'MACDs_12_26_9': 1.0,
            'MACDh_12_26_9': 0.5,
            'RSI_14': 75.0  # Overbought
        })
        assert strategy.generate_signals(row) == 'HOLD'
    
    def test_hold_signal_when_rsi_oversold(self, strategy):
        """Test HOLD signal when RSI <= 30 (oversold)."""
        row = pd.Series({
            'MACD_12_26_9': 0.5,
            'MACDs_12_26_9': 1.0,
            'MACDh_12_26_9': -0.5,
            'RSI_14': 25.0  # Oversold
        })
        assert strategy.generate_signals(row) == 'HOLD'
    
    def test_hold_signal_when_macd_missing(self, strategy):
        """Test HOLD signal when MACD values are missing."""
        row = pd.Series({
            'MACDs_12_26_9': 1.0,
            'MACDh_12_26_9': 0.5,
            'RSI_14': 50.0
        })
        assert strategy.generate_signals(row) == 'HOLD'
    
    def test_hold_signal_when_rsi_missing(self, strategy):
        """Test HOLD signal when RSI is missing."""
        row = pd.Series({
            'MACD_12_26_9': 1.5,
            'MACDs_12_26_9': 1.0,
            'MACDh_12_26_9': 0.5
        })
        assert strategy.generate_signals(row) == 'HOLD'
    
    def test_hold_signal_when_values_are_nan(self, strategy):
        """Test HOLD signal when indicator values are NaN."""
        row = pd.Series({
            'MACD_12_26_9': np.nan,
            'MACDs_12_26_9': 1.0,
            'MACDh_12_26_9': 0.5,
            'RSI_14': 50.0
        })
        assert strategy.generate_signals(row) == 'HOLD'
    
    def test_hold_signal_when_macdh_zero(self, strategy):
        """Test HOLD signal when MACDh is zero."""
        row = pd.Series({
            'MACD_12_26_9': 1.5,
            'MACDs_12_26_9': 1.0,
            'MACDh_12_26_9': 0.0,
            'RSI_14': 50.0
        })
        assert strategy.generate_signals(row) == 'HOLD'
    
    def test_hold_signal_when_macdh_missing_defaults_to_zero(self, strategy):
        """Test that missing MACDh defaults to 0 and results in HOLD."""
        row = pd.Series({
            'MACD_12_26_9': 1.5,
            'MACDs_12_26_9': 1.0,
            'RSI_14': 50.0
        })
        # MACDh defaults to 0, so condition for BUY (MACDh > 0) fails
        assert strategy.generate_signals(row) == 'HOLD'
    
    def test_boundary_rsi_at_70(self, strategy):
        """Test signal at RSI boundary of 70."""
        row = pd.Series({
            'MACD_12_26_9': 1.5,
            'MACDs_12_26_9': 1.0,
            'MACDh_12_26_9': 0.5,
            'RSI_14': 70.0  # Exactly at boundary
        })
        # RSI < 70 is required for BUY, so 70 should result in HOLD
        assert strategy.generate_signals(row) == 'HOLD'
    
    def test_boundary_rsi_at_30(self, strategy):
        """Test signal at RSI boundary of 30."""
        row = pd.Series({
            'MACD_12_26_9': 0.5,
            'MACDs_12_26_9': 1.0,
            'MACDh_12_26_9': -0.5,
            'RSI_14': 30.0  # Exactly at boundary
        })
        # RSI > 30 is required for SELL, so 30 should result in HOLD
        assert strategy.generate_signals(row) == 'HOLD'
