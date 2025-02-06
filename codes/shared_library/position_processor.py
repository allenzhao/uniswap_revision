from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd

class BasePositionProcessor(ABC):
    def __init__(self, pool_addr: str, debug: bool = False):
        self.pool_addr = pool_addr
        self.debug = debug
        self.data: Optional[pd.DataFrame] = None
        
    @abstractmethod
    def process_positions(self) -> pd.DataFrame:
        pass
        
    def calculate_roi(self, df: pd.DataFrame) -> pd.DataFrame:
        df["daily_overall_roi"] = df["total_amount"] / df["amount_last"]
        df["daily_amount_roi"] = df["amount"] / df["amount_last"]
        df["daily_fee_roi"] = df["fee"] / df["amount_last"]
        return df
        
    def process_amount_events(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        amount0_add_events = df["amount0_input"] > 0
        amount1_add_events = df["amount1_input"] > 0
        amount_add_events = df["amount_input"] > 0
        
        df.loc[amount0_add_events, "amount0_last"] += df.loc[amount0_add_events, "amount0_input"]
        df.loc[amount1_add_events, "amount1_last"] += df.loc[amount1_add_events, "amount1_input"]
        df.loc[amount_add_events, "amount_last"] += df.loc[amount_add_events, "amount_input"]
        
        return df
