import os
from typing import Dict, Any

import pandas as pd
from scipy.stats.mstats import winsorize
from tqdm import tqdm

from codes.shared_library.utils import get_parent
from codes.01_create.position_processors import PositionCreationTimeProcessor

if __name__ == "__main__":
    result_df = pd.DataFrame()
    pool_addrs = [
        '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36',
        '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
    ]
    
    for pool_addr in tqdm(pool_addrs):
        processor = PositionCreationTimeProcessor(pool_addr, debug=True)
        result = processor.process_positions()
        
        # Winsorize extreme ROI values
        for col in ['daily_overall_roi', 'daily_amount_roi', 'daily_fee_roi']:
            result[f'{col}_w1'] = winsorize(result[col], limits=[0.01, 0.01])
            result[f'{col}_w5'] = winsorize(result[col], limits=[0.05, 0.05])
            result[f'{col}_w10'] = winsorize(result[col], limits=[0.10, 0.010])
        
        result_df = pd.concat([result_df, result], ignore_index=True)
    
    # Save results
    output_path = os.path.join(get_parent(), "data", "raw", "pkl", "position_creation_metrics.pkl")
    result_df.to_pickle(output_path)
