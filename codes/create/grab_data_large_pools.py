from flipside import Flipside
import pandas as pd
import numpy as np

# Initialize `Flipside` with your API Key and API Url
flipside = Flipside("4c211c59-f320-4808-bdd3-590ecbc8163b", "https://api-v2.flipsidecrypto.xyz")

POOL_ADDRS = [
    '0x11b815efb8f581194ae79006d24e0d814b7697f6',
    '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'
]

START_BLOCK = 12369854
END_BLOCK = 16308144
PARTS = 4

def get_block_ranges():
    block_range = END_BLOCK - START_BLOCK
    step = block_range // PARTS
    ranges = []
    for i in range(PARTS):
        start = START_BLOCK + (i * step)
        end = start + step if i < PARTS - 1 else END_BLOCK
        ranges.append((start, end))
    return ranges

def get_swap_data(pool, start_block, end_block):
    sql = f"""
    WITH swaps AS (
        SELECT *
        FROM ethereum.core.ez_decoded_event_logs
        WHERE 1 = 1
        AND BLOCK_NUMBER >= {start_block}
        AND BLOCK_NUMBER < {end_block}
        AND CONTRACT_ADDRESS = '{pool}'
        AND EVENT_NAME = 'Swap'
    ),
    tx AS (
        SELECT *
        FROM ethereum.core.fact_transactions
        WHERE 1 = 1
        AND BLOCK_NUMBER >= {start_block}
        AND BLOCK_NUMBER < {end_block}
    )
    SELECT
        el.BLOCK_TIMESTAMP as block_timestamp,
        el.BLOCK_NUMBER as block_number,
        el.TX_HASH as transaction_hash,
        el.EVENT_INDEX as log_index,
        el.decoded_log:sender::string as sender,
        el.decoded_log:recipient::string as recipient,
        el.decoded_log:amount0::string as amount0,
        el.decoded_log:amount1::string as amount1,
        el.decoded_log:sqrtPriceX96::string as sqrtPriceX96,
        el.decoded_log:liquidity::string as liquidity,
        el.decoded_log:tick::integer as tick,
        tx.FROM_ADDRESS as from_address,
        tx.TO_ADDRESS as to_address,
        tx.POSITION as transaction_index,
        CAST(tx.EFFECTIVE_GAS_PRICE as string) as gas_price,
        CAST(tx.GAS_USED as string) as gas_used
    FROM swaps el
    JOIN tx ON el.TX_HASH = tx.TX_HASH
    WHERE el.TX_SUCCEEDED = TRUE
    ORDER BY el.BLOCK_NUMBER, el.EVENT_INDEX ASC;
    """
    
    query_result_set = flipside.query(sql, page_number=1, page_size=1)
    current_page_number = 1
    page_size = 80000
    total_pages = 70
    all_rows = []

    while current_page_number <= total_pages:
        results = flipside.get_query_results(
            query_result_set.query_id,
            page_number=current_page_number,
            page_size=page_size
        )
        total_pages = results.page.totalPages
        if results.records:
            all_rows.extend(results.records)
        current_page_number += 1
    
    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    block_ranges = get_block_ranges()
    
    for pool in POOL_ADDRS:
        print(f"Processing pool: {pool}")
        all_data = []
        
        for i, (start, end) in enumerate(block_ranges):
            print(f"Processing block range {i+1}/{PARTS}: {start} to {end}")
            df = get_swap_data(pool, start, end)
            df['address'] = pool
            all_data.append(df)
        
        # Merge all parts and save
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(f'swap_data_{pool}.csv', index=False)
        print(f"Saved data for pool: {pool}")