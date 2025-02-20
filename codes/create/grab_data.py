from flipside import Flipside
import pandas as pd
# Initialize `Flipside` with your API Key and API Url
flipside = Flipside("4c211c59-f320-4808-bdd3-590ecbc8163b", "https://api-v2.flipsidecrypto.xyz")

POOL_ADDRS = [
      '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36',
      '0x69d91b94f0aaf8e8a2586909fa77a5c2c89818d5',
      '0x84383fb05f610222430f69727aa638f8fdbf5cc1',
      '0x99ac8ca7087fa4a2a1fb6357269965a2014abc35',
      '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8',
      '0xc2e9f25be6257c210d7adf0d4cd6e3e881ba25f8'
]
if __name__ == "__main__":
    for pool in POOL_ADDRS:
        sql = f"""
        WITH swaps AS (
            SELECT
            *
            FROM
            ethereum.core.ez_decoded_event_logs
            WHERE
            1 = 1
            AND BLOCK_NUMBER >= 12369854
            AND BLOCK_NUMBER <= 16308144
            AND CONTRACT_ADDRESS = '{pool}'
            AND EVENT_NAME = 'Swap'
        ),
        tx AS (
            SELECT
            *
            FROM
            ethereum.core.fact_transactions
            WHERE
            1 = 1
            AND BLOCK_NUMBER >= 12369854
            AND BLOCK_NUMBER <= 16308144
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
        FROM
            swaps el
            JOIN tx ON el.TX_HASH = tx.TX_HASH
        WHERE
            el.TX_SUCCEEDED = TRUE
        ORDER BY
            el.BLOCK_NUMBER,
            el.EVENT_INDEX ASC;
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
                all_rows = all_rows + results.records
            
            current_page_number += 1
        data = pd.DataFrame(all_rows)
        data['address'] = pool
        data.to_csv(f'swap_data_{pool}_part1.csv')