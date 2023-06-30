from math import floor
from math import log
import os
from easydict import EasyDict
import gmpy2

TICK_BASE = 1.0001
Q96 = 2 ** 96

POOL_ADDR = ['0x11b815efb8f581194ae79006d24e0d814b7697f6',
             '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36',
             '0x69d91b94f0aaf8e8a2586909fa77a5c2c89818d5',
             '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',
             '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8',
             '0x99ac8ca7087fa4a2a1fb6357269965a2014abc35',
             '0xc2e9f25be6257c210d7adf0d4cd6e3e881ba25f8',
             ]

pool_info_edict_list = [
    EasyDict({'pair_name': 'ETH-USDT', 'fee_tier': 500, 'decimal0': 18, 'decimal1': 6, 'base_token0': False}),
    EasyDict({'pair_name': 'ETH-USDT', 'fee_tier': 3000, 'decimal0': 18, 'decimal1': 6, 'base_token0': False}),
    EasyDict({'pair_name': 'HEX-USDC', 'fee_tier': 3000, 'decimal0': 8, 'decimal1': 6, 'base_token0': False}),
    EasyDict({'pair_name': 'USDC-ETH', 'fee_tier': 500, 'decimal0': 6, 'decimal1': 18, 'base_token0': True}),
    EasyDict({'pair_name': 'USDC-ETH', 'fee_tier': 3000, 'decimal0': 6, 'decimal1': 18, 'base_token0': True}),
    EasyDict({'pair_name': 'WBTC-USDC', 'fee_tier': 3000, 'decimal0': 8, 'decimal1': 6, 'base_token0': False}),
    EasyDict({'pair_name': 'DAI-ETH', 'fee_tier': 3000, 'decimal0': 18, 'decimal1': 18, 'base_token0': True}),
]

POOL_INFO = dict(zip(POOL_ADDR, pool_info_edict_list))
UNISWAP_NFT_MANAGER = "0xc36442b4a4522e871399cd717abdd847ab11fe88"
UNISWAP_MIGRATOR = '0xa5644e29708357803b5a882d272c41cc0df92b34'


def get_parent(path=os.getcwd(), levels=1):
    common = path
    # Using for loop for getting starting point required for
    for i in range(levels + 1):
        # Starting point
        common = os.path.dirname(common)
    return os.path.abspath(common)

def price_to_tick(price, d1, d0):
    return gmpy2.floor(gmpy2.log(10 ** (d1 - d0) / price) / gmpy2.log(1.0001))