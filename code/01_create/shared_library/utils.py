from math import floor
from math import log

from easydict import EasyDict

TICK_BASE = 1.0001
Q96 = 2**96

pool_addresses = ['0x11b815efb8f581194ae79006d24e0d814b7697f6',
                  '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36',
                  '0x69d91b94f0aaf8e8a2586909fa77a5c2c89818d5',
                  '0x84383fb05f610222430f69727aa638f8fdbf5cc1',
                  '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',
                  '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8',
                  '0x99ac8ca7087fa4a2a1fb6357269965a2014abc35',
                  '0xc2e9f25be6257c210d7adf0d4cd6e3e881ba25f8',
                  ]

pool_info_edict_list = [
    EasyDict({'pair_name': 'ETH-USDT', 'fee_tier': 500, 'decimal0': 18, 'decimal1': 6, 'base_token0': False}),
    EasyDict({'pair_name': 'ETH-USDT', 'fee_tier': 3000, 'decimal0': 18, 'decimal1': 6, 'base_token0': False}),
    EasyDict({'pair_name': 'HEX-USDC', 'fee_tier': 3000, 'decimal0': 8, 'decimal1': 6, 'base_token0': False}),
    EasyDict({'pair_name': 'MM-USDC', 'fee_tier': 10000, 'decimal0': 18, 'decimal1': 6, 'base_token0': False}),
    EasyDict({'pair_name': 'USDC-ETH', 'fee_tier': 500, 'decimal0': 6, 'decimal1': 18, 'base_token0': True}),
    EasyDict({'pair_name': 'USDC-ETH', 'fee_tier': 3000, 'decimal0': 6, 'decimal1': 18, 'base_token0': True}),
    EasyDict({'pair_name': 'WBTC-USDC', 'fee_tier': 3000, 'decimal0': 8, 'decimal1': 18, 'base_token0': False}),
    EasyDict({'pair_name': 'DAI-ETH', 'fee_tier': 3000, 'decimal0': 18, 'decimal1': 18, 'base_token0': True}),
]

POOL_INFO = dict(zip(pool_addresses, pool_info_edict_list))

UNISWAP_NFT_MANAGER = "0xc36442b4a4522e871399cd717abdd847ab11fe88"
