import os
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from gmpy2 import mpz
from scipy.stats.mstats import winsorize

from codes.shared_library.utils import POOL_INFO, UNISWAP_NFT_MANAGER, get_parent, UNISWAP_MIGRATOR
from codes.shared_library.position_utils import (
    get_relevant_position_ids_for_increase_only,
    get_relevant_position_ids_for_decrease
)
from codes.shared_library.data_utils import cumsum_mpz

class UserLevelAnalyzer:
    def __init__(self, pool_addr: str, cumulative_mode: bool = False, debug: bool = False):
        self.pool_addr = pool_addr
        self.cumulative_mode = cumulative_mode
        self.debug = debug
        self.data_folder_path = os.path.join(get_parent(), "data")
        self.pickle_path = os.path.join(self.data_folder_path, 'raw', 'pkl')
        self.pool_info = POOL_INFO[pool_addr]
        
    def load_data(self) -> None:
        """Load required data files."""
        self.data_df = pd.read_pickle(os.path.join(self.pickle_path, f"input_info_{self.pool_addr}.pkl"))
        self.data_df["position_id"] = self.data_df["position_id"].astype(str)
        self.data_df["sc"] = self.data_df["nf_position_manager_address"] != UNISWAP_NFT_MANAGER
        
        self.daily_prices = pd.read_csv(os.path.join(self.data_folder_path, "raw", 'daily_pool_agg_results.csv'))
        self.weekly_prices = pd.read_csv(os.path.join(self.data_folder_path, "raw", 'weekly_pool_agg_results.csv'))
        self.false_sc_list = pd.read_csv(os.path.join(self.data_folder_path, "raw", 'not_verified_sc_list.csv'))
        self.sc_verification_data = pd.read_csv(os.path.join(self.data_folder_path, "raw", 'sc_verified_data.csv'))
        
        self.res_df = pd.read_pickle(os.path.join(self.data_folder_path, 'raw', 'pkl', f"done_accounting_day_datas_{self.pool_addr}.pkl"))
        self.res_df["position_id"] = self.res_df["position_id"].astype(str)
        
        self.action_df = pd.read_pickle(os.path.join(self.data_folder_path, 'raw', 'pkl', f"data_{self.pool_addr}_0626_no_short.pkl"))
        self.action_df["position_id"] = self.action_df["position_id"].astype(str)
        
    def process_amounts(self) -> None:
        """Process amount calculations."""
        self.res_df = self.res_df.drop(columns=['open', 'high', 'low', 'close', 'high_tick', 'low_tick'])
        cols_to_change_type = ['amount0', 'amount1', 'fee0', 'fee1']
        self.res_df[cols_to_change_type] = self.res_df[cols_to_change_type].astype('float')
        
        daily_price = self.daily_prices[self.daily_prices["pool_address"] == self.pool_addr].copy()
        self.res_df = self.res_df.merge(daily_price, how='left', on='date')
        self.res_df.sort_values(by=["position_id", "date"], inplace=True)
        
        if self.pool_info.base_token0:
            self.res_df["amount"] = self.res_df["close"] * self.res_df["amount1"] + self.res_df["amount0"]
            self.res_df["fee"] = self.res_df["close"] * self.res_df["fee1"] + self.res_df["fee0"]
            self.res_df["amount_input"] = self.res_df["open"] * self.res_df["amount1_input"] + self.res_df["amount0_input"]
            self.res_df["amount_output"] = self.res_df["close"] * self.res_df["amount1_output"] + self.res_df["amount0_output"]
        else:
            self.res_df["amount"] = self.res_df["close"] * self.res_df["amount0"] + self.res_df["amount1"]
            self.res_df["fee"] = self.res_df["close"] * self.res_df["fee0"] + self.res_df["fee1"]
            self.res_df["amount_input"] = self.res_df["open"] * self.res_df["amount0_input"] + self.res_df["amount1_input"]
            self.res_df["amount_output"] = self.res_df["close"] * self.res_df["amount0_output"] + self.res_df["amount1_output"]
        
        self._process_amount_events()
        
    def _process_amount_events(self) -> None:
        """Process amount events for inputs and outputs."""
        for prefix in ['amount0', 'amount1', 'amount']:
            self.res_df[f"{prefix}_last"] = self.res_df.groupby(["position_id"])[prefix].shift(1).fillna(0)
            
            add_events = self.res_df[f"{prefix}_input"] > 0
            self.res_df.loc[add_events, f"{prefix}_last"] += self.res_df.loc[add_events, f"{prefix}_input"]
            
            remove_events = self.res_df[f"{prefix}_output"] > 0
            self.res_df.loc[remove_events, prefix] += self.res_df.loc[remove_events, f"{prefix}_output"]
        
        self.res_df["total_amount0"] = self.res_df["amount0"] + self.res_df["fee0"]
        self.res_df["total_amount1"] = self.res_df["amount1"] + self.res_df["fee1"]
        self.res_df["total_amount"] = self.res_df["amount"] + self.res_df["fee"]
        self.res_df = self.res_df[self.res_df["amount_last"] != 0].copy()
        
    def identify_sc_positions(self) -> None:
        """Identify smart contract positions."""
        sc_cond1 = self.action_df["nf_position_manager_address"] != UNISWAP_NFT_MANAGER
        self.position_id_sc = self.action_df[sc_cond1]["position_id"].unique()
        self.sc_observations = self.action_df[self.action_df["position_id"].isin(self.position_id_sc)].copy()
        
    def process_positions(self) -> pd.DataFrame:
        """Process all positions and return aggregated results."""
        self.load_data()
        self.process_amounts()
        self.identify_sc_positions()
        
        # Process SC positions
        sc_positions = self.res_df[self.res_df["position_id"].isin(self.position_id_sc)].copy()
        sc_positions_created_date = sc_positions.groupby(["position_id"])["date"].min().reset_index()
        sc_positions_created_date.rename(columns={"date": "sc_position_creation_date"}, inplace=True)
        sc_positions = sc_positions.merge(sc_positions_created_date, how='left', on='position_id')
        
        # Calculate weekly metrics
        weekly_obs = sc_positions.copy()
        weekly_obs["week"] = pd.to_datetime(weekly_obs["date"]).dt.to_period('W-SAT').dt.start_time
        weekly_obs["week"] = weekly_obs["week"].astype(str)
        
        # Process amounts for weekly aggregation
        weekly_obs["amount_last_temp"] = weekly_obs["amount_last"] - weekly_obs["amount_input"]
        weekly_obs["amount_temp"] = weekly_obs["amount"] - weekly_obs["amount_output"]
        
        # Group by LP, position, and week
        lp_position_id_week = ["liquidity_provider", "position_id", "week"]
        weekly_metrics = weekly_obs.groupby(lp_position_id_week).agg({
            "amount_last_temp": "first",
            "amount_temp": "last",
            "fee": "sum",
            "amount_input": "sum",
            "amount_output": "sum",
            "active_perc": "mean",
            "net_liquidity": "last"
        }).reset_index()
        
        # Calculate ROIs
        weekly_metrics["amount_last_new"] = weekly_metrics["amount_last_temp"] + weekly_metrics["amount_input"]
        weekly_metrics["amount_new"] = weekly_metrics["amount_temp"] + weekly_metrics["amount_output"]
        weekly_metrics["amt_roi"] = weekly_metrics["amount_new"] / weekly_metrics["amount_last_new"]
        weekly_metrics["fee_roi"] = weekly_metrics["fee"] / weekly_metrics["amount_last_new"]
        weekly_metrics["overall_roi"] = weekly_metrics["amt_roi"] + weekly_metrics["fee_roi"]
        
        if self.cumulative_mode:
            weekly_metrics = self._process_cumulative_metrics(weekly_metrics)
        
        return weekly_metrics
    
    def _process_cumulative_metrics(self, weekly_metrics: pd.DataFrame) -> pd.DataFrame:
        """Process cumulative asset metrics if in cumulative mode."""
        weekly_metrics["cumulative_amount_input"] = weekly_metrics.groupby(["liquidity_provider"])["amount_input"].cumsum()
        weekly_metrics["cumulative_amount_output"] = weekly_metrics.groupby(["liquidity_provider"])["amount_output"].cumsum()
        weekly_metrics["cumulative_fee"] = weekly_metrics.groupby(["liquidity_provider"])["fee"].cumsum()
        
        weekly_metrics["cumulative_roi"] = (
            (weekly_metrics["cumulative_amount_output"] + weekly_metrics["cumulative_fee"]) / 
            weekly_metrics["cumulative_amount_input"]
        )
        
        return weekly_metrics
