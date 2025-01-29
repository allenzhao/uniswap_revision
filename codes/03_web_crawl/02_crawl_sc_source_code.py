import os
# rate limit to 5 requests per second
import requests
import json
import pandas as pd
import time
import pickle

from codes.shared_library.utils import POOL_INFO, UNISWAP_NFT_MANAGER, get_parent, \
    UNISWAP_MIGRATOR, POOL_ADDR, ETHERSCAN_API_KEY



def query_sc_source_code(sc_addr):
    url = "https://api.etherscan.io/api"
    params = {
        "module": "contract",
        "action": "getsourcecode",
        "address": sc_addr,
        "apikey": ETHERSCAN_API_KEY  # Replace with your actual API key
    }

    # Send the GET request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Check if the response contains the expected data
        if data["status"] == "1" and data["message"] == "OK":
            # Extract the result
            result = data["result"]
            # Do something with the result, for example, print it
            #print(json.dumps(json.loads(result), indent=4))
            return result
        else:
            print("Error in response:", data.get("message", "Unknown error"))
            return "Error"
    else:
        print("Failed to fetch data, status code:", response.status_code)
        return "Error"



if __name__ == "__main__":
    result_df = pd.DataFrame()
    data_folder_path = os.path.join(get_parent(), "data")
    pickle_path = os.path.join(data_folder_path, 'raw', 'pkl')
    pool_addrs = ['0x11b815efb8f581194ae79006d24e0d814b7697f6',
                  '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36',
                  '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',
                  '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8']
    pool_addrs_usdt = ['0x11b815efb8f581194ae79006d24e0d814b7697f6',
                       '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36']
    pool_addrs_usdc = ['0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',
                       '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8']
    pool_addrs_4 = [
        '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36',
        '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8',
    ]  # these are the
    all_pool_addrs = POOL_ADDR
    data_folder_path = os.path.join(get_parent(), "data")
    res_dfs = []
    dfs = []
    results = []
    result_df = pd.DataFrame()
    daily_prices = pd.read_csv(os.path.join(data_folder_path, "raw", 'daily_pool_agg_results.csv'))
    weekly_prices = pd.read_csv(os.path.join(data_folder_path, "raw", 'weekly_pool_agg_results.csv'))
    ret_data = pd.DataFrame()
    action_by_lp = pd.DataFrame()
    amount_by_lp_actual = pd.DataFrame()
    res = []
    res_weekly_for_maggie = pd.DataFrame()
    sc_nft_managers = pd.DataFrame()
    # # collect all sc information from all pools
    # for pool_addr in all_pool_addrs:
    #     print(pool_addr)
    #     data_df = pd.read_pickle(os.path.join(pickle_path, f"input_info_{pool_addr}.pkl"))
    #     data_df["position_id"] = data_df["position_id"].astype(str)
    #     data_df["sc"] = (data_df["nf_position_manager_address"] != UNISWAP_NFT_MANAGER) & (data_df["nf_position_manager_address"] != UNISWAP_MIGRATOR)
    #     sc_data_df = data_df[data_df["sc"]].copy()
    #     # get the user count for each SC
    #     sc_count = sc_data_df.groupby(["nf_position_manager_address"])["position_id"].nunique().reset_index()
    #     sc_count["pool_address"] = pool_addr
    #     sc_nft_managers = pd.concat([sc_nft_managers, sc_count ], ignore_index=True)
    #     print("here")
    sc_nft_managers = pd.read_csv("sc_nft_managers.csv")
    # print("outside loop")
    nft_managers = sc_nft_managers["nf_position_manager_address"].unique().tolist()
    nft_ret_data = {}
    rate_limit = 1 / 6
    not_verified = pd.read_csv("not_verified_sc_list.csv")
    not_verified_sc_list = not_verified['0'].unique().tolist()
    not_source_code = []
    for nft_manager in nft_managers:
        if nft_manager not in not_verified_sc_list:
            query_res = query_sc_source_code(nft_manager)
            if query_res != "Error":
                nft_ret_data[nft_manager] = query_res
            else:
                #nft_ret_data[nft_manager] = "Error"
                print("Error on " + nft_manager)
                not_source_code.append(nft_manager)
            time.sleep(rate_limit)
    with open('nft_manager_source_code_saved_dictionary.pkl', 'wb') as f:
        pickle.dump(nft_ret_data, f)
    pd.DataFrame(not_source_code).to_csv("not_verified_sc_list_source_code.csv", index=False)
    print("Done crawling")


    # API endpoint and parameters



