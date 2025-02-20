import os

import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from pandarallel import pandarallel

from codes.shared_library.utils import POOL_ADDR, POOL_INFO, get_parent, UNISWAP_NFT_MANAGER, UNISWAP_MIGRATOR

if __name__ == "__main__":
    debug = True
    pandarallel.initialize(progress_bar=True, nb_workers=60)
    data_folder_path = os.path.join(get_parent(), "data")
    df = pd.read_pickle(os.path.join(data_folder_path, 'raw', 'pkl', "data_by_positions_0711.pkl"))
    pass
