import pandas as pd
from ta import momentum, add_all_ta_features
from ta.utils import dropna


def oversold(series: pd.Series, threshold: float = 29.5) -> bool:
    return series.values[-1] <= threshold


def overbought(series: pd.Series, threshold: float = 70.5) -> bool:
    return series.values[-1] >= threshold


def get_rsi(prices: list) -> pd.Series:
    df = clean_data(prices=prices)
    return momentum.rsi(df['close'], window=14)


def get_all_features(prices: list) -> pd.DataFrame:
    df = clean_data(prices=prices)
    df = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)

    return df


def clean_data(prices: list) -> pd.DataFrame:
    # Load data
    df = pd.DataFrame(prices)

    # Clean NaN values
    df = dropna(df)

    return df
