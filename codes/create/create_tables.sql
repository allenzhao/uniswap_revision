CREATE TABLE factory_pool_created (
    chain_name VARCHAR(50),
    block_timestamp TIMESTAMP,
    block_number BIGINT,
    transaction_hash VARCHAR(66),  -- Ethereum tx hash is 66 chars (with 0x)
    log_index BIGINT,
    token0 VARCHAR(42),           -- Ethereum address is 42 chars (with 0x)
    token1 VARCHAR(42),
    fee VARCHAR(20),
    tickSpacing VARCHAR(20),
    pool VARCHAR(42)
);

CREATE TABLE pool_initialize_events (
    chain_name VARCHAR(50),
    address VARCHAR(42),          -- Pool address
    block_timestamp TIMESTAMP,
    block_number BIGINT,
    transaction_hash VARCHAR(66),
    log_index BIGINT,
    sqrtPriceX96 VARCHAR(78),     -- Large numbers need bigger size
    tick BIGINT,
    to_address VARCHAR(42),
    from_address VARCHAR(42),
    transaction_index BIGINT,
    gas_price VARCHAR(78),        -- Gas price can be large
    gas_used VARCHAR(78)          -- Gas used can be large
);

CREATE TABLE pool_mint_burn_events (
    chain_name VARCHAR(50),
    address VARCHAR(42),          -- Pool address
    block_timestamp TIMESTAMP,
    block_number BIGINT,
    transaction_hash VARCHAR(66),
    log_index BIGINT,
    amount VARCHAR(78),           -- Large numbers need bigger size
    amount0 VARCHAR(78),
    amount1 VARCHAR(78),
    owner VARCHAR(42),            -- Ethereum address
    tick_lower BIGINT,
    tick_upper BIGINT,
    type_of_event VARCHAR(10),    -- 'MINT' or 'BURN'
    to_address VARCHAR(42),
    from_address VARCHAR(42),
    transaction_index BIGINT,
    gas_price VARCHAR(78),
    gas_used VARCHAR(78),
    l1_fee VARCHAR(78)
);

CREATE TABLE pool_swap_events (
    chain_name VARCHAR(50),
    address VARCHAR(42),          -- Pool address
    block_timestamp TIMESTAMP,
    block_number BIGINT,
    transaction_hash VARCHAR(66),
    log_index BIGINT,
    sender VARCHAR(42),           -- Ethereum address
    recipient VARCHAR(42),        -- Ethereum address
    amount0 VARCHAR(78),          -- Large numbers need bigger size
    amount1 VARCHAR(78),
    sqrtPriceX96 VARCHAR(78),
    liquidity VARCHAR(78),
    tick BIGINT,
    from_address VARCHAR(42),
    to_address VARCHAR(42),
    transaction_index BIGINT,
    gas_price VARCHAR(78),
    gas_used VARCHAR(78),
    l1_fee VARCHAR(78)
); 

