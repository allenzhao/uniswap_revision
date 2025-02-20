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
)
SELECT
  'ethereum' as chain_name,
  el.BLOCK_TIMESTAMP as block_timestamp,
  el.BLOCK_NUMBER as block_number,
  el.TX_HASH as transaction_hash,
  el.EVENT_INDEX as log_index,
  p.TOKEN0_ADDRESS as token0,
  p.TOKEN1_ADDRESS as token1,
  p.FEE as fee,
  p.TICK_SPACING as tickSpacing,
  p.POOL_ADDRESS as pool
FROM
  el
  JOIN POOLS p ON el.TX_HASH = p.TX_HASH
  AND el.TX_SUCCEEDED = TRUE
  AND el.contract_address = '0x1f98431c8ad98523631ae4a59f267346ea31f984'
ORDER BY
  el.BLOCK_NUMBER,
  el.EVENT_INDEX ASC;