import os

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats.mstats import winsorize

from codes.shared_library.utils import POOL_INFO, UNISWAP_NFT_MANAGER, get_parent, \
    UNISWAP_MIGRATOR


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
    data_folder_path = os.path.join(get_parent(), "data")
    res_dfs = []
    dfs = []
    results = []
    result_df = pd.DataFrame()
    sc_daily_obs_df = pd.DataFrame()
    non_sc_daily_obs_df = pd.DataFrame()
    daily_prices = pd.read_csv(os.path.join(data_folder_path, "raw", 'daily_pool_agg_results.csv'))
    weekly_prices = pd.read_csv(os.path.join(data_folder_path, "raw", 'weekly_pool_agg_results.csv'))
    ret_data = pd.DataFrame()
    action_by_lp = pd.DataFrame()
    amount_by_lp_actual = pd.DataFrame()
    res = []
    to_export_lp_data = pd.DataFrame()
    false_sc_list = pd.read_csv(os.path.join(data_folder_path, "raw", 'not_verified_sc_list.csv'))

    for pool_addr in pool_addrs:
        temp_res = {}
        temp_res["pool_addr"] = pool_addr

        print(pool_addr)
        data_df = pd.read_pickle(os.path.join(pickle_path, f"input_info_{pool_addr}.pkl"))
        data_df["position_id"] = data_df["position_id"].astype(str)
        data_df["sc"] = data_df["nf_position_manager_address"] != UNISWAP_NFT_MANAGER

        temp_res["unique_lps"] = data_df["liquidity_provider"].unique().shape[0]
        temp_res["positions"] = data_df["position_id"].unique().shape[0]
        print(data_df.shape)
        print(data_df["liquidity_provider"].unique().shape)
        # Figure out the upper or lower prices which should be used as base;
        # todo: needs updating to also consider the other pools?
        if pool_addr in pool_addrs_usdc:
            upper_col = 'price_upper_0_1'
            lower_col = 'price_lower_0_1'
            token_price_col = 'token1_price'
        else:
            upper_col = 'price_upper_1_0'
            lower_col = 'price_lower_1_0'
            token_price_col = 'token0_price'

        # check if using automated recommendation
        upper_perc = 2
        lower_perc = 0.5
        threshold = 0.01
        if pool_addr not in pool_addrs_4:
            upper_perc = 1.001
            lower_perc = 0.999
            threshold = 0.001
        upper_to_lower = upper_perc / lower_perc
        data_df["upper/current"] = data_df[upper_col] / data_df[token_price_col]
        data_df["lower/current"] = data_df[lower_col] / data_df[token_price_col]
        data_df["upper/lower"] = data_df[upper_col] / data_df[lower_col]
        # print("Upper to be around recommended")
        data_df["upper_match"] = np.abs(data_df["upper/current"] - upper_perc) <= threshold
        # print(temp_res["upper_match"])
        # print("Lower to be around recommended")
        data_df["lower_match"] = np.abs(data_df["lower/current"] - lower_perc) <= threshold
        # print(temp_res["lower_match"])
        # print("Both Match")
        data_df["both_match"] = (np.abs(data_df["lower/current"] - lower_perc) <= threshold) & (
                np.abs(data_df["upper/current"] - upper_perc) <= threshold)
        # print(temp_res["both_match"])
        # print("Alternative_perc Match")
        data_df["alternative_match_05"] = (data_df["tick_upper"] - data_df["tick_lower"] < 40) & (
                data_df["upper/current"] >= 1) & (data_df["lower/current"] <= 1)
        temp_res["alternative_match_05"] = sum(data_df["alternative_match_05"])
        # print(temp_res["alternative_match_05"])
        # print("Percentage to be around recommonded")
        # print(data_df["upper/lower"].describe())
        res.append(temp_res)
        ##################
        # get price data #
        ##################
        daily_price = daily_prices[daily_prices["pool_address"] == pool_addr].copy()
        weekly_price = weekly_prices[weekly_prices["pool_address"] == pool_addr].copy()
        res_df = pd.read_pickle(
            os.path.join(data_folder_path, 'raw', 'pkl', f"done_accounting_day_datas_{pool_addr}.pkl"))
        res_df["position_id"] = res_df["position_id"].astype(str)
        res_df = res_df.drop(columns=['open', 'high', 'low', 'close', 'high_tick', 'low_tick'])
        cols_to_change_type = ['amount0', 'amount1', 'fee0', 'fee1']
        res_df[cols_to_change_type] = res_df[cols_to_change_type].astype('float')
        pool_info = POOL_INFO[pool_addr]
        res_df = res_df.merge(daily_price, how='left', on='date')
        res_df.sort_values(by=["position_id", "date"], inplace=True)
        # given different base, calculate different amount and fee
        if pool_info.base_token0:
            res_df["amount"] = res_df["close"] * res_df["amount1"] + res_df["amount0"]
            res_df["fee"] = res_df["close"] * res_df["fee1"] + res_df["fee0"]
            res_df["amount_input"] = res_df["open"] * res_df["amount1_input"] + res_df[
                "amount0_input"]
            res_df["amount_output"] = res_df["close"] * res_df[
                "amount1_output"] + res_df["amount0_output"]
        else:
            res_df["amount"] = res_df["close"] * res_df["amount0"] + res_df["amount1"]
            res_df["fee"] = res_df["close"] * res_df["fee0"] + res_df["fee1"]
            res_df["amount_input"] = res_df["open"] * res_df["amount0_input"] + res_df[
                "amount1_input"]
            res_df["amount_output"] = res_df["close"] * res_df[
                "amount0_output"] + res_df["amount1_output"]

        res_df["amount0_last"] = res_df.groupby(["position_id"])["amount0"].shift(
            1).fillna(0)
        res_df["amount1_last"] = res_df.groupby(["position_id"])["amount1"].shift(
            1).fillna(0)
        res_df["amount_last"] = res_df.groupby(["position_id"])["amount"].shift(
            1).fillna(0)
        amount0_add_events = res_df["amount0_input"] > 0
        amount1_add_events = res_df["amount1_input"] > 0
        amount_add_events = res_df["amount_input"] > 0
        res_df.loc[amount0_add_events, "amount0_last"] += res_df.loc[
            amount0_add_events, "amount0_input"
        ]
        res_df.loc[amount1_add_events, "amount1_last"] += res_df.loc[
            amount1_add_events, "amount1_input"
        ]
        res_df.loc[amount_add_events, "amount_last"] += res_df.loc[
            amount_add_events, "amount_input"
        ]

        amount0_remove_events = res_df["amount0_output"] > 0
        amount1_remove_events = res_df["amount1_output"] > 0
        amount_remove_events = res_df["amount_output"] > 0
        res_df.loc[amount0_remove_events, "amount0"] += res_df.loc[
            amount0_remove_events, "amount0_output"
        ]
        res_df.loc[amount1_remove_events, "amount1"] += res_df.loc[
            amount1_remove_events, "amount1_output"
        ]
        res_df.loc[amount_remove_events, "amount"] += res_df.loc[
            amount_remove_events, "amount_output"
        ]
        res_df["total_amount0"] = res_df["amount0"] + res_df["fee0"]
        res_df["total_amount1"] = res_df["amount1"] + res_df["fee1"]
        res_df["total_amount"] = res_df["amount"] + res_df["fee"]
        res_df = res_df[res_df["amount_last"] != 0].copy()

        #######
        # Linking positions with LPs
        #######
        # read in action df
        action_df = pd.read_pickle(os.path.join(data_folder_path, 'raw', 'pkl', f"data_{pool_addr}_0626_no_short.pkl"))
        action_df["position_id"] = action_df["position_id"].astype(str)
        action_df_for_week = action_df.copy()
        # first sc condition: not using uniswap_nft_manager
        sc_cond1 = action_df["nf_position_manager_address"] != UNISWAP_NFT_MANAGER
        # second sc condition: multiple lp ids
        one_position_id_has_multiple_lp_id = action_df.groupby(["position_id"])["liquidity_provider"].nunique()
        position_id_multiple = one_position_id_has_multiple_lp_id[one_position_id_has_multiple_lp_id > 1].index
        one_position_id_has_multiple_lp_id = action_df.groupby(["position_id"])["liquidity_provider"].nunique()
        position_id_multiple = one_position_id_has_multiple_lp_id[one_position_id_has_multiple_lp_id > 1].index
        # has multiple users
        # didn't interact with the official uniswap ?
        # todo: commented out for now
        # sc_cond2_pre = action_df["position_id"].isin(position_id_multiple)
        # sc_cond2 = (action_df["to_address"] != UNISWAP_NFT_MANAGER) & (
        #         action_df["to_address"] != UNISWAP_MIGRATOR) & sc_cond2_pre
        # position_id_sc = action_df[(sc_cond1 | sc_cond2)]["position_id"].unique()
        position_id_sc = action_df[sc_cond1]["position_id"].unique()
        sc_observations = action_df[action_df["position_id"].isin(position_id_sc)].copy()

        res_df["date"] = pd.to_datetime(res_df["date"])

        sc_position_with_start_and_end_dates = res_df[res_df["position_id"].isin(position_id_sc)].groupby(
            ["position_id"]).agg({"date": ["min", "max"]})
        sc_position_with_start_and_end_dates.columns = ["_".join(col) for col in
                                                        sc_position_with_start_and_end_dates.columns]
        sc_position_with_start_and_end_dates.reset_index(inplace=True)
        # Then for each position, find the associated SC:
        position_id_sc_map = sc_observations[["position_id", "nf_position_manager_address"]].drop_duplicates()
        sc_position_with_start_and_end_dates = sc_position_with_start_and_end_dates.merge(position_id_sc_map,
                                                                                          on='position_id', how='left')
        ######
        # mapping lp and sc positions
        # note: skipped since we only care about positions now
        # the next part should now be
        ######
        #### position ids
        sc_daily_obs = res_df[res_df["position_id"].isin(position_id_sc)].copy()

        ####
        sc_daily_obs = sc_daily_obs.merge(position_id_sc_map, how='left', on="position_id")
        sc_daily_obs["is_verified_sc"] = ~sc_daily_obs["nf_position_manager_address"].isin(false_sc_list["0"])
        sc_daily_obs["pool_address"] = pool_addr
        sc_daily_obs["daily_overall_roi"] = sc_daily_obs["total_amount"] / sc_daily_obs["amount_last"]
        sc_daily_obs["daily_amount_roi"] = sc_daily_obs["amount"] / sc_daily_obs["amount_last"]
        sc_daily_obs["daily_fee_roi"] = sc_daily_obs["fee"] / sc_daily_obs["amount_last"]
        # do some simple filtering:
        multiple_positions_new_cond = action_df.groupby(["position_id"]).agg(
            max_date=("block_timestamp", "max"),
            min_date=("block_timestamp", "min"),
            mpz=("liquidity_mpz", "sum")
        ).reset_index()
        multiple_positions_new_cond["time_diff"] = (
                multiple_positions_new_cond["max_date"] - multiple_positions_new_cond["min_date"])
        new_cond_for_position = (multiple_positions_new_cond["time_diff"] <= '1 hours') & (
                multiple_positions_new_cond["mpz"] == 0)
        ok_ids_new = multiple_positions_new_cond[~new_cond_for_position]["position_id"].unique()
        sc_daily_obs = sc_daily_obs[sc_daily_obs["position_id"].isin(ok_ids_new)]
        lp_max_min_date_check = sc_daily_obs.groupby(["nf_position_manager_address"]).agg(
            max_date=("date", "max"),
            min_date=("date", "min")
        ).reset_index()
        lp_max_min_date_check["date_diff"] = lp_max_min_date_check["max_date"] - lp_max_min_date_check["min_date"]
        # check condition: at least one week in pool
        ok_lp_ids = lp_max_min_date_check[lp_max_min_date_check["date_diff"] >= '7 days']["nf_position_manager_address"].unique()
        sc_daily_obs = sc_daily_obs[
            sc_daily_obs["nf_position_manager_address"].isin(ok_lp_ids)]
        # filtering out small total_amount
        sc_daily_obs = sc_daily_obs[sc_daily_obs["total_amount"] >= 10].copy()
        sc_daily_obs_df = pd.concat([sc_daily_obs_df, sc_daily_obs], ignore_index=True)

        # non_sc obs
        non_sc_daily_obs = res_df[~res_df["position_id"].isin(position_id_sc)].copy()
        non_sc_daily_obs["pool_address"] = pool_addr
        non_sc_daily_obs["daily_overall_roi"] = non_sc_daily_obs["total_amount"] / non_sc_daily_obs["amount_last"]
        non_sc_daily_obs["daily_amount_roi"] = non_sc_daily_obs["amount"] / non_sc_daily_obs["amount_last"]
        non_sc_daily_obs["daily_fee_roi"] = non_sc_daily_obs["fee"] / non_sc_daily_obs["amount_last"]

        non_sc_daily_obs_df = pd.concat([non_sc_daily_obs_df, non_sc_daily_obs], ignore_index=True)



        ######
        # todo: agg to weekly?;
        # todo: data is a bit off, try and figure out why?
        ######
    non_sc_daily_obs_df["sc"] = False
    sc_daily_obs_df["sc"] = True
    non_sc_daily_obs_df.to_pickle("non_sc_only_daily_obs_20240228.pkl")

    verified_scs = sc_daily_obs_df[sc_daily_obs_df["is_verified_sc"]].copy()
    non_verified_scs = sc_daily_obs_df[~sc_daily_obs_df["is_verified_sc"]].copy()

    print("Here")