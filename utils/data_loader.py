"""
Tool to download stock return data, to avoid calling yfinance multiple times.
Reads tickers from files and saves as CSV, or if already exists, returns df

Top 120 S&P500 companies by market cap (2015-11-06 to 2025-11-04),
excluding columns (tickers) and rows (days) with missing or no data
(e.g. data from no-trading days, null)
"""

import yfinance as yf
import pandas as pd
from pathlib import Path


CSV_PATH = Path("data/stock_returns.csv")
TICKERS_PATH = Path("data/tickers.txt")
with TICKERS_PATH.open() as f:
    TICKERS = [line.strip() for line in f if line.strip()]

# Not inclusive
START_DATE = "2015-11-05"
END_DATE = "2025-11-05"


def _check_requested_data_present(existing_df, tickers, start_date, end_date):
    """Check if existing csv (df) matches the requested one."""
    if not set(tickers).issubset(existing_df.columns):
        return False

    # Check if year and month match, as days may be cleaned due to no data
    first_month = str(existing_df.index[0])[:7]
    last_month = str(existing_df.index[-1])[:7]

    start_month = start_date[:7]
    end_month = end_date[:7]

    return (first_month == start_month) and (last_month == end_month)


def get_stock_returns(path=CSV_PATH, tickers=TICKERS, start_date=START_DATE, end_date=END_DATE):
    """Fetches daily stock return % for given tickers and date rabge (not inclusive)
    either from an existing file or yfinance.

    Uses either given or stored parameters.

    Args:
        path (Path / str): Data store. Defaults to CSV_PATH.
        tickers (list[str]): List of stock ticker symbols. Defaults to TICKERS.
        start_date (str): year-month-day format start date. Defaults to START_DATE.
        end_date (str): year-month-day format end date. Defaults to END_DATE.

    Returns:
        pd.DataFrame: Daily percentage returns for each ticker.
    """
    if path.exists():
        csv_df = pd.read_csv(path, index_col=0, parse_dates=True)
        if _check_requested_data_present(csv_df, tickers, start_date, end_date):
            print("Loading stock return data...")

            return csv_df

    print(f"Attempting to download stock return data for {tickers}")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]

    # Drop empty columns and rows, convert prices to percentage returns
    returns = data.dropna(axis=1).pct_change().dropna()

    fetched_tickers = returns.columns.tolist()
    missing_tickers = [t for t in tickers if t not in fetched_tickers]

    print(f"Downloaded {len(fetched_tickers)} / {len(tickers)} tickers")

    if missing_tickers:
        print(f"Missing tickers: {missing_tickers}")

    path.parent.mkdir(exist_ok=True)
    returns.to_csv(path)

    return returns


if __name__ == "__main__":
    returns = get_stock_returns()
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    print(returns.head())
