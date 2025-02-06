import datetime
import logging
import os

from parallel_pandas import ParallelPandas

import math
from tqdm import tqdm

import numpy as np
import pandas as pd

from codes.shared_library.utils import TICK_BASE, POOL_INFO, UNISWAP_NFT_MANAGER, Q96, POOL_ADDR, get_parent, POOL_TICK_QUERY_AT_GIVEN_BLOCK, query_graphql



if __name__ == "__main__":
    result_df = pd.DataFrame()
    for pool_addr in tqdm(POOL_ADDR):
        all_blocks_df = pd.read_csv(f"all_blocks_{pool_addr}.csv")
        first_block = int(all_blocks_df["block_number"].min())
        to_save_data = []
        for block_num in tqdm(all_blocks_df["block_number"].unique()):
            variables = {"pool_id": pool_addr, "block_num": int(block_num)}
            ret_data = query_graphql(POOL_TICK_QUERY_AT_GIVEN_BLOCK, variables)
            row = {"block_number": block_num, "liquidity": ret_data['data']['pools'][0]['liquidity'],
                   "tick": ret_data['data']['pools'][0]['tick']}
            to_save_data.append(row)
        temp_df = pd.DataFrame(to_save_data)
        temp_df["pool_addr"] = pool_addr
        result_df = pd.concat([result_df, temp_df], ignore_index=True)
    result_df.to_csv("block_liquidity_and_tick.csv")
    print("Done")

