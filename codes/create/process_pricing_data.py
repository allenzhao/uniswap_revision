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
        'buy_trades': sum(x['buying_crypto_trade'].values) if len(x['buying_crypto_trade'].values) > 0 else 0,
        'sell_trades': sum(x['buying_crypto_trade'].values) if len(x['buying_stable_trade'].values) > 0 else 0,
        'price_avg': np.mean(price_arr) if len(price_arr) > 0 else np.nan,
        'price_std': np.std(price_arr) if len(price_arr) > 0 else np.nan,
        'tx_users': len(np.unique(x.index)) if len(x.index) > 0 else 0,
        'log_buy': np.log(sum(x['buying_crypto_trade'].values) + 1) if len(x['buying_crypto_trade'].values) > 0 else np.nan,
        'log_sell': np.log(sum(~x['buying_crypto_trade'].values) + 1) if len(x['buying_crypto_trade'].values) > 0 else np.nan,
        'log_price': np.log(np.mean(price_arr) + 1) if len(price_arr) > 0 else np.nan,
        'log_vol': np.log(sum(x['amount'].values) + 1) if len(x['amount'].values) > 0 else np.nan,
        'log_price_std': np.log(np.std(price_arr)) if len(price_arr) > 0 and np.std(price_arr) > 0 else np.nan,
        'log_tx_users': np.log(len(np.unique(x.index)) + 1) if len(x.index) > 0 else np.nan,
    }
    return pd.Series(names)


def process_pool_pricing_data(frequencies=None):
    """
    Process pool pricing data and generate aggregated results at specified frequencies.
    
    Args:
        frequencies (list, optional): List of frequencies to aggregate data by. 
            Valid options are 'D' (daily), 'W' (weekly), 'M' (monthly).
            If None, defaults to ['D', 'W', 'M'].
    
    Returns:
        dict: Dictionary containing DataFrames for each frequency level
    """
    if frequencies is None:
        frequencies = ['D', 'W', 'M']
    
    frequency_labels = {'D': 'daily', 'W': 'weekly', 'M': 'monthly'}
    
    # Validate input frequencies
    valid_frequencies = set(['D', 'W', 'M'])
    if not all(freq in valid_frequencies for freq in frequencies):
        raise ValueError(f"Invalid frequency. Must be one of {valid_frequencies}")

    raw_folder_path = os.path.join(get_parent(), "data", "raw")
    
    # Read and concatenate all split CSV files
    all_swaps_df = pd.DataFrame()
    for i in range(1, 3):  # We have 2 parts
        file_path = os.path.join(raw_folder_path, f"all_swaps_part{i}.csv")
        if os.path.exists(file_path):
            print(f"Reading part {i}...")
            df_part = pd.read_csv(file_path, low_memory=False, parse_dates=["block_timestamp"])
            all_swaps_df = pd.concat([all_swaps_df, df_part], ignore_index=True)
    
    # Fix issue
    all_swaps_df = all_swaps_df[
        all_swaps_df["tx_hash"] != '0xcdf9d46f009c8fe02b04889b5e927f3a49004ac246cd76140ff8890563b9374b'].copy()

    all_results = {freq: pd.DataFrame() for freq in frequencies}

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
            current_pool_df["amount_crypto"] = current_pool_df["amount1_adjusted"]
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

        for freq in frequencies:
            print(f"Resampling by {frequency_labels[freq]}")
            label_param = {}
            if freq == 'W':
                label_param['label'] = 'left'
            resampled_df = current_pool_df.resample(freq, **label_param).apply(agg_ohlcv)
            resampled_df = resampled_df.ffill().reset_index()
            resampled_df["pool_address"] = pool_addr
            resampled_df.rename(columns={"block_timestamp": "date"}, inplace=True)
            all_results[freq] = pd.concat([all_results[freq], resampled_df], ignore_index=True)
        print("Done with this pool")

    # Save results
    for freq in frequencies:
        freq_label = frequency_labels[freq]
        output_filename = f"{freq_label}_pool_agg_results.csv"
        all_results[freq].to_csv(os.path.join(raw_folder_path, output_filename), index=False)
    
    return all_results


if __name__ == "__main__":
    # Run with default settings
    process_pool_pricing_data()
