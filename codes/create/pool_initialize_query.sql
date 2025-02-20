WITH POOLS AS (
  SELECT
    *
  FROM
    ethereum.uniswapv3.ez_pools
  WHERE
    pool_address IN (
      '0x11b815efb8f581194ae79006d24e0d814b7697f6',
      '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36',
      '0x69d91b94f0aaf8e8a2586909fa77a5c2c89818d5',
      '0x84383fb05f610222430f69727aa638f8fdbf5cc1',
      '0x99ac8ca7087fa4a2a1fb6357269965a2014abc35',
      '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',
      '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8',
      '0xc2e9f25be6257c210d7adf0d4cd6e3e881ba25f8'
    )
),
el AS (
  SELECT
    *
  FROM
    ethereum.core.ez_decoded_event_logs
  WHERE
    BLOCK_NUMBER IN (
      SELECT
        DISTINCT BLOCK_NUMBER
      FROM
        POOLS
    )
),
tx AS (
  SELECT
    *
  FROM
    ethereum.core.fact_transactions
  WHERE
    BLOCK_NUMBER IN (
      SELECT
        DISTINCT BLOCK_NUMBER
      FROM
        POOLS
    )
)
SELECT
  'ethereum' as chain_name,
  el.contract_address as address,
  el.BLOCK_TIMESTAMP as block_timestamp,
  el.BLOCK_NUMBER as block_number,
  el.TX_HASH as transaction_hash,
  el.EVENT_INDEX as log_index,
  el.decoded_log:sqrtPriceX96 :: string as sqrtPriceX96,
  el.decoded_log:tick :: integer as tick,
  tx.TO_ADDRESS as to_address,
  tx.FROM_ADDRESS as from_address,
  tx.POSITION as transaction_index,
  CAST(tx.EFFECTIVE_GAS_PRICE as string) as gas_price,
  CAST(tx.GAS_USED as string) as gas_used,
FROM
  el
  JOIN POOLS p ON el.TX_HASH = p.TX_HASH
  AND el.contract_address = p.POOL_ADDRESS
  JOIN tx ON el.TX_HASH = tx.TX_HASH
WHERE
  el.TX_SUCCEEDED = TRUE
  AND el.EVENT_NAME = 'Initialize'
ORDER BY
  el.BLOCK_NUMBER,
  el.EVENT_INDEX ASC;