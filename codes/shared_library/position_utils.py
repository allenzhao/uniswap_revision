from typing import List, Dict, Any
import pandas as pd
import numpy as np

def get_relevant_position_ids_for_increase_only(x: pd.Series, helper_df: pd.DataFrame) -> np.ndarray:
    helper_df = helper_df.copy()
    sc_addr = x["nf_position_manager_address"]
    date = x["block_timestamp"]
    cond1 = helper_df["nf_position_manager_address"] == sc_addr
    cond2 = helper_df["date_max"] >= date
    cond = cond1 & cond2
    return helper_df[cond]["position_id"].unique()

def get_relevant_position_ids_for_decrease(x: pd.Series, helper_df: pd.DataFrame) -> np.ndarray:
    helper_df = helper_df.copy()
    sc_addr = x["nf_position_manager_address"]
    start_date = x["block_timestamp_min"]
    end_date = x["block_timestamp_max"]
    cond1 = helper_df["nf_position_manager_address"] == sc_addr
    cond2 = helper_df["date_max"] >= start_date
    cond3 = helper_df["date_min"] <= end_date
    cond = cond1 & cond2 & cond3
    return helper_df[cond]["position_id"].unique()

def get_last_operation(x: pd.DataFrame) -> bool:
    all_actions = x["action"].values
    last_action = all_actions[-1] if len(all_actions) > 0 else 'NAN'
    return last_action == 'INCREASE_LIQUIDITY'
