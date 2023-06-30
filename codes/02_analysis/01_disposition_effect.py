import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from codes.shared_library.utils import POOL_ADDR, POOL_INFO, get_parent, UNISWAP_NFT_MANAGER, UNISWAP_MIGRATOR
if __name__ == "__main__":
    data_folder_path = os.path.join(get_parent(), "data")
    res_dfs = []
    dfs = []
    results = []
    result_df = pd.DataFrame()
