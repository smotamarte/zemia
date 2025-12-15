import pandas as pd
import pandas_ta_classic as ta
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Strategy:
    """
    Encapsulates the trading strategy logic based on MACD and RSI indicators.
    """
    def __init__(self):
        """
        Initializes the strategy.
        """
        logger.info("Strategy initialized.")

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates MACD and RSI indicators and adds them to the DataFrame.

        Args:
            data (pd.DataFrame): The input OHLCV data.

        Returns:
            pd.DataFrame: The data with indicators added.
        """
        data.ta.macd(append=True)  # Adds MACD, MACDH, MACDS
        data.ta.rsi(append=True)
        logger.info("Calculated MACD and RSI indicators.")
        return data

    def generate_signals(self, row: pd.Series) -> str:
        """
        Generates buy/sell/hold signals based on MACD and RSI.

        Buy Signal:
            - MACD Line crosses above Signal Line (MACD_12_26_9 > MACDS_12_26_9)
            - RSI is not overbought (RSI_14 < 70)
        Sell Signal:
            - MACD Line crosses below Signal Line (MACD_12_26_9 < MACDS_12_26_9)
            - RSI is not oversold (RSI_14 > 30)

        Args:
            row (pd.Series): A row of the data DataFrame at a specific timestamp.

        Returns:
            str: 'BUY', 'SELL', or 'HOLD'.
        """
        macd_line = row.get('MACD_12_26_9')
        signal_line = row.get('MACDs_12_26_9')
        rsi = row.get('RSI_14')

        if macd_line is None or signal_line is None or rsi is None or np.isnan(macd_line) or np.isnan(signal_line) or np.isnan(rsi):
            return 'HOLD'

        if macd_line > signal_line and row.get('MACDh_12_26_9', 0) > 0 and rsi < 70:
            return 'BUY'
        elif macd_line < signal_line and row.get('MACDh_12_26_9', 0) < 0 and rsi > 30:
            return 'SELL'
        else:
            return 'HOLD'
