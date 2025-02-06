import os
from typing import Optional, Dict, Any
import pandas as pd
from abc import ABC, abstractmethod

from codes.shared_library.position_processor import BasePositionProcessor
from codes.shared_library.utils import POOL_INFO, get_parent
from codes.shared_library.position_utils import (
    get_relevant_position_ids_for_increase_only,
    get_relevant_position_ids_for_decrease,
    get_last_operation
)
from codes.shared_library.data_utils import (
    group_weighted_mean_factory,
    calculate_roi,
    cumsum_mpz
)
from codes.shared_library.utils import POOL_INFO

class PositionCreationTimeProcessor(BasePositionProcessor):
    def __init__(self, pool_addr: str, debug: bool = False):
        super().__init__(pool_addr, debug)
        self.pool_info = POOL_INFO[pool_addr]
        self.data_folder_path = os.path.join(get_parent(), "data")
        
    def process_positions(self) -> pd.DataFrame:
        """Process position data with time-based analysis."""
        # Load data
        data_df = pd.read_pickle(os.path.join(self.data_folder_path, 'raw', 'pkl', f"input_info_{self.pool_addr}.pkl"))
        res_df = pd.read_pickle(os.path.join(self.data_folder_path, 'raw', 'pkl', f"done_accounting_day_datas_{self.pool_addr}.pkl"))
        
        # Process amounts
        res_df = self.process_amount_events(res_df)
        
        # Calculate ROIs
        res_df = calculate_roi(res_df)
        
        # Filter short-lived positions
        res_df["date"] = pd.to_datetime(res_df["date"])
        position_dates = res_df.groupby("position_id").agg(
            max_date=("date", "max"),
            min_date=("date", "min")
        ).reset_index()
        position_dates["date_diff"] = position_dates["max_date"] - position_dates["min_date"]
        valid_positions = position_dates[position_dates["date_diff"] >= pd.Timedelta(days=7)]["position_id"]
        res_df = res_df[res_df["position_id"].isin(valid_positions)]
        
        # Calculate metrics
        res_df["week"] = pd.to_datetime(res_df["date"]).dt.to_period('W-SAT').dt.start_time
        res_df["daily_amount_roi_temp"] = res_df["amount"] / res_df["amount_last"]
        res_df["daily_overall_roi_temp"] = res_df["total_amount"] / res_df["amount_last"]
        
        # Group by week
        weekly_metrics = res_df.groupby(["position_id", "week"]).agg(
            daily_amount_roi=("daily_amount_roi_temp", "prod"),
            amount_last_temp=("amount_last", "first"),
            amount_temp=("amount", "last"),
            amount_output=("amount_output", "sum"),
            fee_total=("fee", "sum"),
            amount_input=("amount_input", "sum"),
            active_perc=("active_perc", "mean")
        ).reset_index()
        
        return weekly_metrics

class LPCreationProcessor(BasePositionProcessor):
    def __init__(self, pool_addr: str, debug: bool = False):
        super().__init__(pool_addr, debug)
        self.pool_info = POOL_INFO[pool_addr]
        self.data_folder_path = os.path.join(get_parent(), "data")
        
    def process_positions(self) -> pd.DataFrame:
        """Process LP position data with type analysis."""
        # Load data
        data_df = pd.read_pickle(os.path.join(self.data_folder_path, 'raw', 'pkl', f"input_info_{self.pool_addr}.pkl"))
        res_df = pd.read_pickle(os.path.join(self.data_folder_path, 'raw', 'pkl', f"done_accounting_day_datas_{self.pool_addr}.pkl"))
        action_df = pd.read_pickle(os.path.join(self.data_folder_path, 'raw', 'pkl', f"data_{self.pool_addr}_0626_no_short.pkl"))
        
        # Process position IDs
        data_df["position_id"] = data_df["position_id"].astype(str)
        res_df["position_id"] = res_df["position_id"].astype(str)
        action_df["position_id"] = action_df["position_id"].astype(str)
        
        # Identify SC positions
        sc_cond1 = action_df["nf_position_manager_address"] != UNISWAP_NFT_MANAGER
        multiple_owners = action_df.groupby("position_id")["liquidity_provider"].nunique()
        position_id_multiple = multiple_owners[multiple_owners > 1].index
        sc_cond2 = (action_df["position_id"].isin(position_id_multiple)) & \
                   (action_df["to_address"] != UNISWAP_NFT_MANAGER) & \
                   (action_df["to_address"] != UNISWAP_MIGRATOR)
        position_id_sc = action_df[(sc_cond1 | sc_cond2)]["position_id"].unique()
        
        # Process amounts and calculate ROIs
        res_df = self.process_amount_events(res_df)
        res_df = calculate_roi(res_df)
        
        # Add LP type information
        res_df["sc"] = res_df["position_id"].isin(position_id_sc)
        res_df = self._calculate_lp_metrics(res_df)
        
        return res_df
        
    def _calculate_lp_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate LP-specific metrics."""
        df["lp_type"] = "sc"
        df.loc[df["both_match"], "lp_type"] = "rec"
        df.loc[(~df["both_match"]) & (~df["sc"]), "lp_type"] = "manual"
        return df

class SCPositionProcessor(BasePositionProcessor):
    def __init__(self, pool_addr: str, debug: bool = False):
        super().__init__(pool_addr, debug)
        self.pool_info = POOL_INFO[pool_addr]
        self.data_folder_path = os.path.join(get_parent(), "data")
        
    def process_positions(self) -> pd.DataFrame:
        """Process smart contract position data."""
        # Load data
        data_df = pd.read_pickle(os.path.join(self.data_folder_path, 'raw', 'pkl', f"input_info_{self.pool_addr}.pkl"))
        res_df = pd.read_pickle(os.path.join(self.data_folder_path, 'raw', 'pkl', f"done_accounting_day_datas_{self.pool_addr}.pkl"))
        action_df = pd.read_pickle(os.path.join(self.data_folder_path, 'raw', 'pkl', f"data_{self.pool_addr}_0626_no_short.pkl"))
        
        # Process position IDs
        data_df["position_id"] = data_df["position_id"].astype(str)
        res_df["position_id"] = res_df["position_id"].astype(str)
        action_df["position_id"] = action_df["position_id"].astype(str)
        
        # Identify SC positions
        sc_cond1 = action_df["nf_position_manager_address"] != UNISWAP_NFT_MANAGER
        multiple_owners = action_df.groupby("position_id")["liquidity_provider"].nunique()
        position_id_multiple = multiple_owners[multiple_owners > 1].index
        sc_cond2 = (action_df["position_id"].isin(position_id_multiple)) & \
                   (action_df["to_address"] != UNISWAP_NFT_MANAGER) & \
                   (action_df["to_address"] != UNISWAP_MIGRATOR)
        position_id_sc = action_df[(sc_cond1 | sc_cond2)]["position_id"].unique()
        
        # Process amounts and calculate ROIs
        res_df = self.process_amount_events(res_df)
        res_df = calculate_roi(res_df)
        
        # Add SC information
        res_df["sc"] = res_df["position_id"].isin(position_id_sc)
        res_df = self._process_sc_positions(res_df)
        
        # Calculate SC metrics
        sc_positions = res_df[res_df["sc"]].copy()
        sc_positions_created_date = sc_positions.groupby("position_id")["date"].min().reset_index()
        sc_positions_created_date.rename(columns={"date": "sc_position_creation_date"}, inplace=True)
        res_df = res_df.merge(sc_positions_created_date, on="position_id", how="left")
        
        return res_df
        
    def _process_sc_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process smart contract specific metrics."""
        df["sc_type"] = "unknown"
        df.loc[df["sc"] & df["both_match"], "sc_type"] = "recommended"
        df.loc[df["sc"] & ~df["both_match"], "sc_type"] = "custom"
        return df
