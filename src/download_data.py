import yfinance as yf


def download_stock_data(ticker="PYPL", start="2014-01-01", end="2024-01-01"):
    """
    Download historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol.
        start (str): Start date.
        end (str): End date.

    Returns:
        pandas.DataFrame: Historical stock price data.
    """
    data = yf.download(ticker, start=start, end=end)
    return data