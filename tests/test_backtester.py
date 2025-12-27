"""
Unit tests for the Backtester class in backtester.py
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtester import Backtester, fetch_ohlcv_data


class TestBacktesterInit:
    """Tests for Backtester initialization."""
    
    @pytest.fixture
    def valid_ohlcv_data(self):
        """Create valid OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        return pd.DataFrame({
            'Open': [100.0] * 50,
            'High': [105.0] * 50,
            'Low': [95.0] * 50,
            'Close': [102.0] * 50,
            'Volume': [1000] * 50
        }, index=dates)
    
    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy for testing."""
        strategy = Mock()
        strategy.calculate_indicators = Mock(return_value=pd.DataFrame())
        strategy.generate_signals = Mock(return_value='HOLD')
        return strategy
    
    def test_init_with_valid_data(self, valid_ohlcv_data, mock_strategy):
        """Test successful initialization with valid data."""
        backtester = Backtester(valid_ohlcv_data, mock_strategy)
        
        assert backtester.initial_capital == 10000.0
        assert backtester.commission_rate == 0.001
        assert backtester.position_size == 0.95
        assert backtester.capital == 10000.0
        assert backtester.holdings == 0.0
        assert backtester.trade_log == []
        assert backtester.portfolio_value == []
    
    def test_init_with_custom_parameters(self, valid_ohlcv_data, mock_strategy):
        """Test initialization with custom parameters."""
        backtester = Backtester(
            valid_ohlcv_data, 
            mock_strategy,
            initial_capital=50000.0,
            commission_rate=0.002,
            position_size=0.8
        )
        
        assert backtester.initial_capital == 50000.0
        assert backtester.commission_rate == 0.002
        assert backtester.position_size == 0.8
    
    def test_init_raises_error_without_datetime_index(self, mock_strategy):
        """Test that ValueError is raised when data doesn't have DatetimeIndex."""
        data = pd.DataFrame({
            'Open': [100.0],
            'High': [105.0],
            'Low': [95.0],
            'Close': [102.0],
            'Volume': [1000]
        })
        
        with pytest.raises(ValueError, match="Data must have a DatetimeIndex"):
            Backtester(data, mock_strategy)
    
    def test_init_raises_error_missing_columns(self, mock_strategy):
        """Test that ValueError is raised when required columns are missing."""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Open': [100.0] * 5,
            'Close': [102.0] * 5
        }, index=dates)
        
        with pytest.raises(ValueError, match="Data must contain"):
            Backtester(data, mock_strategy)
    
    def test_init_copies_data(self, valid_ohlcv_data, mock_strategy):
        """Test that data is copied, not referenced."""
        backtester = Backtester(valid_ohlcv_data, mock_strategy)
        
        # Modify original data
        valid_ohlcv_data.iloc[0, 0] = 999.0
        
        # Backtester's data should be unchanged
        assert backtester.data.iloc[0, 0] != 999.0


class TestExecuteTrade:
    """Tests for the _execute_trade method."""
    
    @pytest.fixture
    def backtester_with_capital(self):
        """Create a backtester with capital for testing trades."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'Open': [100.0] * 50,
            'High': [105.0] * 50,
            'Low': [95.0] * 50,
            'Close': [102.0] * 50,
            'Volume': [1000] * 50
        }, index=dates)
        
        mock_strategy = Mock()
        return Backtester(data, mock_strategy, initial_capital=10000.0, commission_rate=0.001, position_size=0.95)
    
    def test_buy_trade_reduces_capital(self, backtester_with_capital):
        """Test that BUY trade reduces capital correctly."""
        initial_capital = backtester_with_capital.capital
        date = pd.Timestamp('2023-01-01')
        
        backtester_with_capital._execute_trade(date, 'BUY', 100.0)
        
        assert backtester_with_capital.capital < initial_capital
        assert backtester_with_capital.holdings > 0
    
    def test_buy_trade_logs_transaction(self, backtester_with_capital):
        """Test that BUY trade is logged correctly."""
        date = pd.Timestamp('2023-01-01')
        
        backtester_with_capital._execute_trade(date, 'BUY', 100.0)
        
        assert len(backtester_with_capital.trade_log) == 1
        assert backtester_with_capital.trade_log[0]['type'] == 'BUY'
        assert backtester_with_capital.trade_log[0]['price'] == 100.0
    
    def test_sell_trade_increases_capital(self, backtester_with_capital):
        """Test that SELL trade increases capital correctly."""
        # First buy some holdings
        date = pd.Timestamp('2023-01-01')
        backtester_with_capital._execute_trade(date, 'BUY', 100.0)
        
        capital_after_buy = backtester_with_capital.capital
        
        # Then sell at higher price
        backtester_with_capital._execute_trade(date, 'SELL', 110.0)
        
        assert backtester_with_capital.capital > capital_after_buy
        assert backtester_with_capital.holdings == 0.0
    
    def test_sell_trade_logs_transaction(self, backtester_with_capital):
        """Test that SELL trade is logged correctly."""
        date = pd.Timestamp('2023-01-01')
        
        # First buy
        backtester_with_capital._execute_trade(date, 'BUY', 100.0)
        # Then sell
        backtester_with_capital._execute_trade(date, 'SELL', 110.0)
        
        assert len(backtester_with_capital.trade_log) == 2
        assert backtester_with_capital.trade_log[1]['type'] == 'SELL'
    
    def test_buy_with_no_capital_does_nothing(self, backtester_with_capital):
        """Test that BUY with no capital doesn't execute."""
        backtester_with_capital.capital = 0.0
        date = pd.Timestamp('2023-01-01')
        
        backtester_with_capital._execute_trade(date, 'BUY', 100.0)
        
        assert len(backtester_with_capital.trade_log) == 0
        assert backtester_with_capital.holdings == 0.0
    
    def test_sell_with_no_holdings_does_nothing(self, backtester_with_capital):
        """Test that SELL with no holdings doesn't execute."""
        date = pd.Timestamp('2023-01-01')
        
        backtester_with_capital._execute_trade(date, 'SELL', 100.0)
        
        assert len(backtester_with_capital.trade_log) == 0
    
    def test_commission_is_applied_on_buy(self, backtester_with_capital):
        """Test that commission is correctly applied on BUY."""
        date = pd.Timestamp('2023-01-01')
        
        backtester_with_capital._execute_trade(date, 'BUY', 100.0)
        
        trade = backtester_with_capital.trade_log[0]
        expected_commission = trade['quantity'] * trade['price'] * 0.001
        assert abs(trade['commission'] - expected_commission) < 0.01
    
    def test_commission_is_applied_on_sell(self, backtester_with_capital):
        """Test that commission is correctly applied on SELL."""
        date = pd.Timestamp('2023-01-01')
        
        backtester_with_capital._execute_trade(date, 'BUY', 100.0)
        backtester_with_capital._execute_trade(date, 'SELL', 100.0)
        
        sell_trade = backtester_with_capital.trade_log[1]
        expected_commission = sell_trade['quantity'] * sell_trade['price'] * 0.001
        assert abs(sell_trade['commission'] - expected_commission) < 0.01


class TestRunBacktest:
    """Tests for the run_backtest method."""
    
    @pytest.fixture
    def sample_data_with_indicators(self):
        """Create sample data with pre-calculated indicators."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 109.0, 108.0, 107.0, 106.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 99.0, 98.0, 97.0, 96.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 106.0, 105.0, 104.0, 103.0],
            'Volume': [1000] * 10,
            'MACD_12_26_9': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -0.1, -0.2, -0.3, -0.4],
            'MACDs_12_26_9': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, -0.1, -0.2, -0.3],
            'MACDh_12_26_9': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1],
            'RSI_14': [50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 55.0, 50.0, 45.0, 40.0]
        }, index=dates)
    
    def test_run_backtest_tracks_portfolio_value(self, sample_data_with_indicators):
        """Test that portfolio value is tracked during backtest."""
        mock_strategy = Mock()
        mock_strategy.calculate_indicators = Mock(return_value=sample_data_with_indicators)
        mock_strategy.generate_signals = Mock(return_value='HOLD')
        
        backtester = Backtester(sample_data_with_indicators, mock_strategy)
        backtester.run_backtest()
        
        assert len(backtester.portfolio_value) == len(sample_data_with_indicators)
    
    def test_run_backtest_sells_remaining_holdings(self, sample_data_with_indicators):
        """Test that remaining holdings are sold at end of backtest."""
        mock_strategy = Mock()
        mock_strategy.calculate_indicators = Mock(return_value=sample_data_with_indicators)
        # Return BUY on first call, then HOLD for the rest
        mock_strategy.generate_signals = Mock(side_effect=['BUY'] + ['HOLD'] * 9)
        
        backtester = Backtester(sample_data_with_indicators, mock_strategy)
        backtester.run_backtest()
        
        # Should have sold at end
        assert backtester.holdings == 0.0
    
    def test_run_backtest_handles_empty_data_after_dropna(self):
        """Test that backtest handles empty data gracefully."""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Open': [np.nan] * 5,
            'High': [np.nan] * 5,
            'Low': [np.nan] * 5,
            'Close': [np.nan] * 5,
            'Volume': [np.nan] * 5
        }, index=dates)
        
        mock_strategy = Mock()
        mock_strategy.calculate_indicators = Mock(return_value=data)
        
        backtester = Backtester(
            pd.DataFrame({
                'Open': [100.0] * 5,
                'High': [105.0] * 5,
                'Low': [95.0] * 5,
                'Close': [102.0] * 5,
                'Volume': [1000] * 5
            }, index=dates),
            mock_strategy
        )
        
        # Should not raise an error
        backtester.run_backtest()


class TestGetResults:
    """Tests for the get_results method."""
    
    @pytest.fixture
    def backtester_after_trades(self):
        """Create a backtester that has executed some trades."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Open': [100.0] * 10,
            'High': [105.0] * 10,
            'Low': [95.0] * 10,
            'Close': [100.0, 105.0, 110.0, 108.0, 106.0, 104.0, 102.0, 100.0, 98.0, 96.0],
            'Volume': [1000] * 10
        }, index=dates)
        
        mock_strategy = Mock()
        mock_strategy.calculate_indicators = Mock(return_value=data)
        # BUY at 100, SELL at 110, BUY at 104, SELL at 96
        mock_strategy.generate_signals = Mock(side_effect=[
            'BUY', 'HOLD', 'SELL', 'HOLD', 'HOLD', 'BUY', 'HOLD', 'HOLD', 'HOLD', 'SELL'
        ])
        
        backtester = Backtester(data, mock_strategy, initial_capital=10000.0)
        backtester.run_backtest()
        return backtester
    
    def test_get_results_returns_dict(self, backtester_after_trades):
        """Test that get_results returns a dictionary."""
        results = backtester_after_trades.get_results()
        assert isinstance(results, dict)
    
    def test_get_results_contains_required_keys(self, backtester_after_trades):
        """Test that results contain all required keys."""
        results = backtester_after_trades.get_results()
        
        required_keys = [
            'initial_capital', 'final_capital', 'final_holdings',
            'final_total_value', 'pnl_absolute', 'pnl_percentage',
            'total_trades', 'winning_trades', 'losing_trades',
            'win_rate', 'trade_log', 'portfolio_history'
        ]
        
        for key in required_keys:
            assert key in results
    
    def test_get_results_trade_log_is_dataframe(self, backtester_after_trades):
        """Test that trade_log is a DataFrame."""
        results = backtester_after_trades.get_results()
        assert isinstance(results['trade_log'], pd.DataFrame)
    
    def test_get_results_portfolio_history_is_dataframe(self, backtester_after_trades):
        """Test that portfolio_history is a DataFrame."""
        results = backtester_after_trades.get_results()
        assert isinstance(results['portfolio_history'], pd.DataFrame)
    
    def test_get_results_pnl_calculation(self, backtester_after_trades):
        """Test that PnL is calculated correctly."""
        results = backtester_after_trades.get_results()
        
        expected_pnl = results['final_total_value'] - results['initial_capital']
        assert abs(results['pnl_absolute'] - expected_pnl) < 0.01
    
    def test_get_results_pnl_percentage_calculation(self, backtester_after_trades):
        """Test that PnL percentage is calculated correctly."""
        results = backtester_after_trades.get_results()
        
        expected_pnl_pct = (results['pnl_absolute'] / results['initial_capital']) * 100
        assert abs(results['pnl_percentage'] - expected_pnl_pct) < 0.01
    
    def test_get_results_win_rate_calculation(self, backtester_after_trades):
        """Test that win rate is between 0 and 1."""
        results = backtester_after_trades.get_results()
        
        assert 0 <= results['win_rate'] <= 1
    
    def test_get_results_with_no_trades(self):
        """Test results when no trades were executed."""
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Open': [100.0] * 5,
            'High': [105.0] * 5,
            'Low': [95.0] * 5,
            'Close': [102.0] * 5,
            'Volume': [1000] * 5
        }, index=dates)
        
        mock_strategy = Mock()
        mock_strategy.calculate_indicators = Mock(return_value=data)
        mock_strategy.generate_signals = Mock(return_value='HOLD')
        
        backtester = Backtester(data, mock_strategy)
        backtester.run_backtest()
        results = backtester.get_results()
        
        assert results['total_trades'] == 0
        assert results['winning_trades'] == 0
        assert results['losing_trades'] == 0
        assert results['win_rate'] == 0


class TestFetchOhlcvData:
    """Tests for the fetch_ohlcv_data function."""
    
    @patch('backtester.ccxt')
    def test_fetch_returns_dataframe_on_success(self, mock_ccxt):
        """Test that fetch returns a DataFrame on successful fetch."""
        # Setup mock
        mock_exchange = MagicMock()
        mock_exchange.parse8601.return_value = 1640995200000
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 100.0, 105.0, 95.0, 102.0, 1000],
            [1641081600000, 102.0, 107.0, 97.0, 104.0, 1100],
        ]
        mock_ccxt.kraken.return_value = mock_exchange
        
        result = fetch_ohlcv_data('kraken', 'BTC/USDT', '1d', '2022-01-01', limit=2)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
    
    @patch('backtester.ccxt')
    def test_fetch_returns_empty_dataframe_on_no_data(self, mock_ccxt):
        """Test that fetch returns empty DataFrame when no data is returned."""
        mock_exchange = MagicMock()
        mock_exchange.parse8601.return_value = 1640995200000
        mock_exchange.fetch_ohlcv.return_value = []
        mock_ccxt.kraken.return_value = mock_exchange
        
        result = fetch_ohlcv_data('kraken', 'BTC/USDT', '1d', '2022-01-01')
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    @patch('backtester.ccxt')
    def test_fetch_handles_network_error(self, mock_ccxt):
        """Test that fetch handles network errors gracefully."""
        import ccxt as real_ccxt
        
        mock_exchange = MagicMock()
        mock_exchange.parse8601.return_value = 1640995200000
        mock_exchange.fetch_ohlcv.side_effect = real_ccxt.NetworkError("Network error")
        mock_ccxt.kraken.return_value = mock_exchange
        mock_ccxt.NetworkError = real_ccxt.NetworkError
        
        result = fetch_ohlcv_data('kraken', 'BTC/USDT', '1d', '2022-01-01')
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    @patch('backtester.ccxt')
    def test_fetch_handles_exchange_error(self, mock_ccxt):
        """Test that fetch handles exchange errors gracefully."""
        import ccxt as real_ccxt
        
        mock_exchange = MagicMock()
        mock_exchange.parse8601.return_value = 1640995200000
        mock_exchange.fetch_ohlcv.side_effect = real_ccxt.ExchangeError("Exchange error")
        mock_ccxt.kraken.return_value = mock_exchange
        mock_ccxt.NetworkError = real_ccxt.NetworkError
        mock_ccxt.ExchangeError = real_ccxt.ExchangeError
        
        result = fetch_ohlcv_data('kraken', 'BTC/USDT', '1d', '2022-01-01')
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    @patch('backtester.ccxt')
    def test_fetch_creates_correct_columns(self, mock_ccxt):
        """Test that fetched data has correct column names."""
        mock_exchange = MagicMock()
        mock_exchange.parse8601.return_value = 1640995200000
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 100.0, 105.0, 95.0, 102.0, 1000],
        ]
        mock_ccxt.kraken.return_value = mock_exchange
        
        result = fetch_ohlcv_data('kraken', 'BTC/USDT', '1d', '2022-01-01', limit=1)
        
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            assert col in result.columns
    
    @patch('backtester.ccxt')
    def test_fetch_sets_datetime_index(self, mock_ccxt):
        """Test that fetched data has DatetimeIndex."""
        mock_exchange = MagicMock()
        mock_exchange.parse8601.return_value = 1640995200000
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 100.0, 105.0, 95.0, 102.0, 1000],
        ]
        mock_ccxt.kraken.return_value = mock_exchange
        
        result = fetch_ohlcv_data('kraken', 'BTC/USDT', '1d', '2022-01-01', limit=1)
        
        assert isinstance(result.index, pd.DatetimeIndex)
