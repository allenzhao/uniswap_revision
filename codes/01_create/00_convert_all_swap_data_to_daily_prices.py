# Note: here the OHLC we are directly using converted base to avoid problems
# In the preprocess step we need to use the 0 base ones

import pandas as pd
import numpy as np
import os
from codes.shared_library.utils import POOL_ADDR, get_parent, POOL_INFO


def agg_ohlcv(x):
    price_arr = x['price'].values
    tick_arr = x['tick'].values
    names = {
        'low': min(price_arr) if len(price_arr) > 0 else np.nan,
        'high': max(price_arr) if len(price_arr) > 0 else np.nan,
        'open': price_arr[0] if len(price_arr) > 0 else np.nan,
        'close': price_arr[-1] if len(price_arr) > 0 else np.nan,
        'low_tick': min(tick_arr) if len(tick_arr) > 0 else np.nan,
        'high_tick': max(tick_arr) if len(tick_arr) > 0 else np.nan,
        'open_tick': tick_arr[0] if len(tick_arr) > 0 else np.nan,
        'close_tick': tick_arr[-1] if len(tick_arr) > 0 else np.nan,
        'volume_crypto_abs': sum(x['amount_crypto_abs'].values) if len(x['amount_crypto_abs'].values) > 0 else 0,
        'volume_stable_abs': sum(x['amount_stable_abs'].values) if len(x['amount_stable_abs'].values) > 0 else 0,
        'volume_crypto_net': sum(x['amount_crypto'].values) if len(x['amount_crypto'].values) > 0 else 0,
        'volume_stable_net': sum(x['amount_stable'].values) if len(x['amount_stable'].values) > 0 else 0,
        'volume_usd': sum(x['amount'].values) if len(x['amount'].values) > 0 else 0,
        'volume_crypto_net_usd': sum(x['amount_crypto_usd'].values) if len(x['amount_crypto_usd'].values) > 0 else 0,
        'volume_stable_net_usd': sum(x['amount_stable_usd'].values) if len(x['amount_stable_usd'].values) > 0 else 0,
        'buying_crypto_trade_cnt': sum(x['buying_crypto_trade'].values) if len(
            x['buying_crypto_trade'].values) > 0 else 0,
        'buying_stable_trade_cnt': sum(x['buying_stable_trade'].values) if len(
            x['buying_stable_trade'].values) > 0 else 0,
    }
    return pd.Series(names)


if __name__ == "__main__":
    raw_folder_path = os.path.join(get_parent(), "data", "raw")
    file_path = os.path.join(raw_folder_path, "all_swaps.csv")
    all_swaps_df = pd.read_csv(file_path, low_memory=False, parse_dates=["block_timestamp"])
    # Fix issue
    all_swaps_df = all_swaps_df[all_swaps_df["tx_hash"]!='0xcdf9d46f009c8fe02b04889b5e927f3a49004ac246cd76140ff8890563b9374b'].copy()
    all_daily = pd.DataFrame()
    all_weekly = pd.DataFrame()
    for pool_addr in POOL_ADDR:
        print(f"Working on {pool_addr}")
        pool_info = POOL_INFO[pool_addr]
        current_pool_df = all_swaps_df[all_swaps_df["pool_address"] == pool_addr].copy().set_index(
            "block_timestamp").sort_values(by='block_timestamp')
        if pool_info.base_token0:
            current_pool_df["price"] = current_pool_df["price_0_1"]
            current_pool_df["amount_stable_abs"] = current_pool_df["amount0_adjusted"].abs()
            current_pool_df["amount_crypto_abs"] = current_pool_df["amount1_adjusted"].abs()
            current_pool_df["amount_stable"] = current_pool_df["amount0_adjusted"]
            current_pool_df["amount_crypto"] = current_pool_df["amount1_adjusted"]  # > 0 means more crypto goes in
            current_pool_df["amount_stable_usd"] = current_pool_df["amount0_usd"]
            current_pool_df["amount_crypto_usd"] = current_pool_df["amount1_usd"]
            current_pool_df["buying_crypto_trade"] = current_pool_df["amount1_adjusted"] < 0
            current_pool_df["buying_stable_trade"] = current_pool_df["amount0_adjusted"] < 0
        else:
            current_pool_df["price"] = current_pool_df["price_1_0"]
            current_pool_df["amount_stable_abs"] = current_pool_df["amount1_adjusted"].abs()
            current_pool_df["amount_crypto_abs"] = current_pool_df["amount0_adjusted"].abs()
            current_pool_df["amount_stable"] = current_pool_df["amount1_adjusted"]
            current_pool_df["amount_crypto"] = current_pool_df["amount0_adjusted"]
            current_pool_df["amount_stable_usd"] = current_pool_df["amount1_usd"]
            current_pool_df["amount_crypto_usd"] = current_pool_df["amount0_usd"]
            current_pool_df["buying_crypto_trade"] = current_pool_df["amount0_adjusted"] < 0
            current_pool_df["buying_stable_trade"] = current_pool_df["amount1_adjusted"] < 0
        current_pool_df["amount"] = (current_pool_df["amount0_usd"].abs() + current_pool_df["amount1_usd"].abs()) / 2
        print("Resampling by day")
        current_pool_df_ohlcv_by_day = current_pool_df.resample('D').apply(agg_ohlcv)
        current_pool_df_ohlcv_by_day = current_pool_df_ohlcv_by_day.ffill().reset_index()
        print("Resampling by week")
        current_pool_df_ohlcv_by_week = current_pool_df.resample('W', label='left').apply(agg_ohlcv)
        current_pool_df_ohlcv_by_week = current_pool_df_ohlcv_by_week.ffill().reset_index()
        current_pool_df_ohlcv_by_day["pool_address"] = pool_addr
        current_pool_df_ohlcv_by_day.rename(columns={"block_timestamp": "date"}, inplace=True)
        current_pool_df_ohlcv_by_week["pool_address"] = pool_addr
        current_pool_df_ohlcv_by_week.rename(columns={"block_timestamp": "week"}, inplace=True)
        all_daily = pd.concat([all_daily, current_pool_df_ohlcv_by_day], ignore_index=True)
        all_weekly = pd.concat([all_weekly, current_pool_df_ohlcv_by_week], ignore_index=True)
        print("Done with this pool")
    all_daily.to_csv(os.path.join(raw_folder_path, "daily_pool_agg_results.csv"), index=False)
    all_weekly.to_csv(os.path.join(raw_folder_path, "weekly_pool_agg_results.csv"), index=False)
