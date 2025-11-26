import pandas as pd
import numpy as np
import logging
import ccxt
import asyncio 
from strategy import Strategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Backtester:
    """
    A simple backtesting engine for trading strategies.
    """
    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_capital: float = 10000.0, commission_rate: float = 0.001):
        """
        Initializes the backtester with historical data and trading parameters.

        Args:
            data (pd.DataFrame): Historical OHLCV data with a DateTime index.
            strategy (Strategy): The trading strategy to use.
            initial_capital (float): Starting capital for the backtest.
            commission_rate (float): Commission rate per trade (e.g., 0.001 for 0.1%).
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex.")
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("Data must contain 'Open', 'High', 'Low', 'Close', 'Volume' columns.")
        if data.isnull().values.any():
            logger.warning("Input data contains NaN values. These will be dropped during indicator calculation.")

        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.capital = initial_capital
        self.holdings = 0.0
        self.trade_log = []
        self.portfolio_value = []

        logger.info(f"Backtester initialized with initial capital: {self.initial_capital}")

    def _execute_trade(self, date, signal, price):
        """
        Simulates executing a trade (buy or sell).

        Args:
            date (datetime): The timestamp of the trade.
            signal (str): 'BUY' or 'SELL'.
            price (float): The price at which the trade is executed.
        """
        if signal == 'BUY':
            if self.capital > 0:
                amount_to_invest = self.capital * 0.95
                if amount_to_invest == 0: return

                quantity_to_buy = amount_to_invest / price
                commission = quantity_to_buy * price * self.commission_rate
                
                if self.capital >= (amount_to_invest + commission):
                    self.capital -= (amount_to_invest + commission)
                    self.holdings += quantity_to_buy
                    self.trade_log.append({
                        'date': date,
                        'type': 'BUY',
                        'price': price,
                        'quantity': quantity_to_buy,
                        'commission': commission,
                        'capital_after_trade': self.capital,
                        'holdings_after_trade': self.holdings
                    })
                    logger.debug(f"{date}: BUY {quantity_to_buy:.6f} at {price:.2f}. Capital: {self.capital:.2f}, Holdings: {self.holdings:.6f}")
                else:
                    logger.debug(f"{date}: Insufficient capital to BUY (wanted {amount_to_invest + commission:.2f}, have {self.capital:.2f})")

        elif signal == 'SELL':
            if self.holdings > 0:
                amount_to_sell = self.holdings * price
                commission = amount_to_sell * self.commission_rate
                
                if amount_to_sell >= commission:
                    self.capital += (amount_to_sell - commission)
                    quantity_sold = self.holdings
                    self.holdings = 0.0
                    self.trade_log.append({
                        'date': date,
                        'type': 'SELL',
                        'price': price,
                        'quantity': quantity_sold,
                        'commission': commission,
                        'capital_after_trade': self.capital,
                        'holdings_after_trade': self.holdings
                    })
                    logger.debug(f"{date}: SELL {quantity_sold:.6f} at {price:.2f}. Capital: {self.capital:.2f}, Holdings: {self.holdings:.6f}")
                else:
                    logger.debug(f"{date}: Holdings value ({amount_to_sell:.2f}) less than commission ({commission:.2f}). No SELL.")

    def run_backtest(self):
        """
        Runs the backtest over the historical data.
        """
        logger.info("Starting backtest...")
        self.data = self.strategy.calculate_indicators(self.data)

        initial_rows = len(self.data)
        self.data.dropna(inplace=True)
        dropped_rows = initial_rows - len(self.data)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows due to NaN values after indicator calculation.")
        if self.data.empty:
            logger.error("No data remaining after dropping NaNs. Backtest cannot proceed.")
            return

        for index, row in self.data.iterrows():
            current_price = row['Close']
            signal = self.strategy.generate_signals(row)

            current_portfolio_value = self.capital + (self.holdings * current_price)
            self.portfolio_value.append({
                'date': index,
                'total_value': current_portfolio_value,
                'capital': self.capital,
                'holdings_value': self.holdings * current_price
            })

            if signal == 'BUY' and self.capital > 0 and self.holdings == 0:
                self._execute_trade(index, 'BUY', current_price)
            elif signal == 'SELL' and self.holdings > 0:
                self._execute_trade(index, 'SELL', current_price)

        if self.holdings > 0:
            last_price = self.data.iloc[-1]['Close']
            self._execute_trade(self.data.index[-1], 'SELL', last_price)
            logger.info("Sold remaining holdings at the end of the backtest.")

        logger.info("Backtest completed.")

    def get_results(self):
        """
        Returns the backtest results, including final capital, PnL, and trade log.

        Returns:
            dict: A dictionary containing backtest summary and details.
        """
        if self.data.empty:
            logger.error("No data available to calculate final portfolio value.")
            final_market_price = 0
        else:
            final_market_price = self.data.iloc[-1]['Close']

        final_value = self.capital + (self.holdings * final_market_price)
        pnl = final_value - self.initial_capital
        total_trades = len(self.trade_log)

        winning_trades = 0
        losing_trades = 0
        win_rate = 0
        
        # A "completed trade" is a buy-sell pair. We calculate PnL for each pair.
        if total_trades > 0:
            completed_trades_pnl = []
            # The strategy logic ensures a simple sequence of BUY, SELL, BUY, SELL...
            # This loop pairs each SELL with its preceding BUY to calculate PnL.
            for i in range(len(self.trade_log)):
                if self.trade_log[i]['type'] == 'SELL':
                    if i > 0 and self.trade_log[i-1]['type'] == 'BUY':
                        buy_trade = self.trade_log[i-1]
                        sell_trade = self.trade_log[i]
                        
                        # PnL = (sell revenue - sell commission) - (buy cost + buy commission)
                        buy_cost = (buy_trade['price'] * buy_trade['quantity'])
                        sell_revenue = (sell_trade['price'] * sell_trade['quantity'])
                        
                        single_trade_pnl = sell_revenue - buy_cost - buy_trade['commission'] - sell_trade['commission']
                        completed_trades_pnl.append(single_trade_pnl)

            if completed_trades_pnl:
                winning_trades = sum(1 for pnl in completed_trades_pnl if pnl > 0)
                losing_trades = sum(1 for pnl in completed_trades_pnl if pnl < 0)
                
                num_completed_trades = len(completed_trades_pnl)
                if num_completed_trades > 0:
                    win_rate = winning_trades / num_completed_trades
        else:
            winning_trades, losing_trades, win_rate = 0, 0, 0


        portfolio_df = pd.DataFrame(self.portfolio_value).set_index('date')

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'final_holdings': self.holdings,
            'final_total_value': final_value,
            'pnl_absolute': pnl,
            'pnl_percentage': (pnl / self.initial_capital) * 100 if self.initial_capital != 0 else 0,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'trade_log': pd.DataFrame(self.trade_log),
            'portfolio_history': portfolio_df
        }

# Function to fetch OHLCV data using CCXT
def fetch_ohlcv_data(exchange_id: str, symbol: str, timeframe: str, since_date: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetches historical OHLCV data from an exchange using CCXT.

    Args:
    exchange_id (str): The ID of the exchange (e.g., 'kraken', 'binance').
    symbol (str): The trading pair (e.g., 'BTC/USDT').
    timeframe (str): The OHLCV timeframe (e.g., '1d', '4h').
    since_date (str): The start date in 'YYYY-MM-DD' format.
    limit (int): The maximum number of candles to fetch.

    Returns:
    pd.DataFrame: DataFrame with OHLCV data, indexed by datetime.
    """
    try:
        # Use the standard synchronous ccxt module
        exchange_class = getattr(ccxt, exchange_id)

        exchange = exchange_class({
            'enableRateLimit': True,
        })

        since_timestamp = exchange.parse8601(f'{since_date}T00:00:00Z')

        logger.info(f"Fetching {limit} {symbol} {timeframe} data from {exchange_id} starting from {since_date}...")

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit)

        if not ohlcv:
            logger.warning(f"No OHLCV data fetched for {symbol} from {exchange_id}.")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv)

        if df.shape[1] != 6:
            logger.error(f"Unexpected number of columns returned by ccxt: {df.shape[1]}. Expected 6.")
            return pd.DataFrame()

        df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.index.name = 'Date'

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"Successfully fetched {len(df)} data points.")
        return df

    except ccxt.NetworkError as e:
        logger.error(f"Network error while fetching data: {e}")
        return pd.DataFrame()
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error while fetching data: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

# Example Usage:
async def main():
    """Main function to run the backtest."""
    # --- Fetch Data using CCXT ---
    # Fetch daily BTC/USDT data from Kraken for the last 1000 days
    data_for_backtest = fetch_ohlcv_data(
        exchange_id='kraken',
        symbol='BTC/USDT',
        timeframe='1d',
        since_date='2022-01-01', # Start date for historical data
        limit=1000 # Max number of candles to fetch
    )

    if data_for_backtest.empty:
        logger.error("Failed to fetch data for backtest. Exiting.")
        return

    # --- Initialize and Run Backtester ---
    # Ensure all data columns are numeric before passing to Backtester
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data_for_backtest[col] = pd.to_numeric(data_for_backtest[col], errors='coerce')
    data_for_backtest.dropna(inplace=True) # Drop any rows with NaN introduced by coercion

    # Initialize the strategy
    strategy = Strategy()

    # Initialize the backtester with the strategy
    backtester = Backtester(data=data_for_backtest, strategy=strategy, initial_capital=100000, commission_rate=0.00075)
    backtester.run_backtest()
    results = backtester.get_results()

    # --- Display Results ---
    print("\n--- Backtest Summary ---")
    print(f"Initial Capital: ${results['initial_capital']:.2f}")
    print(f"Final Total Value: ${results['final_total_value']:.2f}")
    print(f"Absolute PnL: ${results['pnl_absolute']:.2f}")
    print(f"Percentage PnL: {results['pnl_percentage']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Losing Trades: {results['losing_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")

    print("\n--- Trade Log (first 5 entries) ---")
    print(results['trade_log'].head())

    print("\n--- Portfolio Value History (first 5 entries) ---")
    print(results['portfolio_history'].head())

    # --- Visualisation (Optional, requires matplotlib) ---
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 7))
        plt.plot(results['portfolio_history'].index, results['portfolio_history']['total_value'], label='Portfolio Value')
        plt.plot(results['portfolio_history'].index, results['portfolio_history']['capital'], label='Capital')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        plt.show()

    except ImportError:
        logger.warning("Matplotlib not installed. Skipping visualization.")

if __name__ == "__main__":
    asyncio.run(main()) # Run the async main function