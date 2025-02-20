from typing import Callable, Any
import numpy as np
import pandas as pd

def group_weighted_mean_factory(df: pd.DataFrame, weight_col_name: str) -> Callable:
    def group_weighted_mean(x: pd.Series) -> float:
        try:
            return np.average(x, weights=df.loc[x.index, weight_col_name])
        except ZeroDivisionError:
            return np.average(x)
    return group_weighted_mean

def calculate_roi(df: pd.DataFrame) -> pd.DataFrame:
    df["daily_overall_roi"] = df["total_amount"] / df["amount_last"]
    df["daily_amount_roi"] = df["amount"] / df["amount_last"]
    df["daily_fee_roi"] = df["fee"] / df["amount_last"]
    return df

def cumsum_mpz(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mpz_arr = df["liquidity_mpz"].to_numpy().cumsum()
    df["net_liquidity"] = mpz_arr
    return df
