from easydict import EasyDict

cfg = EasyDict()

# Data File Locations

cfg.DATA_BASE_DIR = "/workspace/data/"
cfg.factory_df_file = cfg.DATA_BASE_DIR + "Factory_call_createPool.csv"
cfg.info_df_file = cfg.DATA_BASE_DIR + "coin_info_all.csv"
cfg.erc20_token_file = cfg.DATA_BASE_DIR + "erc20_tokens.csv"
cfg.new_coin_info_df_file = cfg.DATA_BASE_DIR + "coin_info_0607.csv"

# Crawler settings
cfg.use_proxies = True
cfg.proxies = {
    "http": "http://cnallenzhao.gmail.com:6j3cj3@gate2.proxyfuel.com:2000",
    "https": "http://cnallenzhao.gmail.com:6j3cj3@gate2.proxyfuel.com:2000",
}
cfg.headers = {'user-agent':
                   'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)' +
                   ' Chrome/84.0.4147.105 Safari/537.36', }

