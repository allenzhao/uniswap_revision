import datetime
import logging
import os
from scipy.stats.mstats import winsorize

from parallel_pandas import ParallelPandas

import math
from tqdm import tqdm

import numpy as np
import pandas as pd

from codes.shared_library.utils import TICK_BASE, POOL_INFO, UNISWAP_NFT_MANAGER, Q96, POOL_ADDR, get_parent, \
    POOL_TICK_QUERY_AT_GIVEN_BLOCK, query_graphql

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
    daily_prices = pd.read_csv(os.path.join(data_folder_path, "raw", 'daily_pool_agg_results.csv'))
    weekly_prices = pd.read_csv(os.path.join(data_folder_path, "raw", 'weekly_pool_agg_results.csv'))
    ret_data = pd.DataFrame()
    action_by_lp = pd.DataFrame()
    amount_by_lp_actual = pd.DataFrame()
    res = []
    for pool_addr in pool_addrs_4:
        temp_res = {}
        temp_res["pool_addr"] = pool_addr

        print(pool_addr)
        data_df = pd.read_pickle(os.path.join(pickle_path, f"input_info_{pool_addr}.pkl"))
        data_df["sc"] = data_df["nf_position_manager_address"] != UNISWAP_NFT_MANAGER

        temp_res["unique_lps"] = data_df["liquidity_provider"].unique().shape[0]
        temp_res["positions"] = data_df["position_id"].unique().shape[0]
        print(data_df.shape)
        print(data_df["liquidity_provider"].unique().shape)
        if pool_addr in pool_addrs_usdc:
            upper_col = 'price_upper_0_1'
            lower_col = 'price_lower_0_1'
            token_price_col = 'token1_price'
        else:
            upper_col = 'price_upper_1_0'
            lower_col = 'price_lower_1_0'
            token_price_col = 'token0_price'
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
        print(f"Working on f{pool_addr}")
        daily_price = daily_prices[daily_prices["pool_address"] == pool_addr].copy()
        weekly_price = weekly_prices[weekly_prices["pool_address"] == pool_addr].copy()
        res_df = pd.read_pickle(
            os.path.join(data_folder_path, 'raw', 'pkl', f"done_accounting_day_datas_{pool_addr}.pkl"))
        res_df = res_df.drop(columns=['open', 'high', 'low', 'close', 'high_tick', 'low_tick'])
        cols_to_change_type = ['amount0', 'amount1', 'fee0', 'fee1']
        res_df[cols_to_change_type] = res_df[cols_to_change_type].astype('float')
        pool_info = POOL_INFO[pool_addr]
        res_df = res_df.merge(daily_price, how='left', on='date')
        res_df.sort_values(by=["position_id", "date"], inplace=True)
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
        cols_to_keep = [
            'position_id',
            'sc',
            'upper_match',
            'lower_match',
            'both_match',
        ]
        data_df_temp = data_df[cols_to_keep].copy()
        daily_obs = res_df.merge(data_df_temp, how='left', on="position_id")
        # daily_obs["filter_in"] = daily_obs["sc"] | daily_obs["both_match"]
        # daily_obs = daily_obs[daily_obs["filter_in"]]
        order_cond = ["position_id", "date"]
        daily_obs.sort_values(by=order_cond, inplace=True)
        # filter out the obs where it only lasted for less than a week:
        daily_obs["date"] = pd.to_datetime(daily_obs["date"])
        position_max_min_date_check = daily_obs.groupby(["position_id"]).agg(
            max_date=("date", "max"),
            min_date=("date", "min")
        ).reset_index()
        position_max_min_date_check["date_diff"] = position_max_min_date_check["max_date"] - position_max_min_date_check["min_date"]
        # at least one week in pool
        ok_pos_ids = position_max_min_date_check[position_max_min_date_check["date_diff"] >= '7 days']["position_id"].unique()
        # weekly_obs["week"] = pd.to_datetime(weekly_obs["date"]).dt.to_period('W-SAT').dt.start_time


        # weekly_obs["week"].astype(str)
        daily_obs = daily_obs[daily_obs["position_id"].isin(ok_pos_ids)].copy()

        position_amt_in_check = daily_obs.groupby(["position_id"]).agg(
            amount_input_max=("amount_input", "max"),
            amount_input_sum=("amount_input", "sum"),
        ).reset_index()
        ok_pos_ids = position_amt_in_check[position_amt_in_check["amount_input_sum"] > 10]["position_id"].unique()
        daily_obs = daily_obs[daily_obs["position_id"].isin(ok_pos_ids)].copy()
        agg_type = 'week'  # either daily (anything else), or daily_cumulative, or weekly
        if agg_type == 'week':
            # Switch back with amount input
            daily_obs["amount_last_temp"] = daily_obs["amount_last"] - daily_obs["amount_input"]
            daily_obs["amount_temp"] = daily_obs["amount"] - daily_obs["amount_output"]

            grp_by_cond = ["position_id", "week"]
            daily_obs["week"] = pd.to_datetime(daily_obs["date"]).dt.to_period('W-SAT').dt.start_time
            daily_obs["daily_amount_roi_temp"] = daily_obs["amount"] / daily_obs["amount_last"]
            daily_obs["daily_overall_roi_temp"] = daily_obs["total_amount"] / daily_obs["amount_last"]

            # money_in_pool_total=('amount_last', 'sum'),
            # money_in_pool_avg=('amount_last', 'mean'),
            # money_in_pool_median=('amount_last', 'median'),
            # daily_price=('close', 'mean'),
            # overall_earning_count=('overall_earning', 'sum'),
            # overall_in_range=('in_range', 'sum'),
            # age_avg=('age', 'mean'),
            # age_median=('age', 'median'),
            temp_res_weekly = daily_obs.groupby(grp_by_cond).agg(
                daily_amount_roi=("daily_amount_roi_temp", "prod"),
                amount_last_temp=("amount_last_temp", "first"),
                amount_temp=("amount_temp", "last"),
                amount_output=("amount_output", "sum"),
                fee_total=("fee", "sum"),
                temp_base=("amount_last", "last"),
                close=('close', 'mean'),
                amount_input=('amount_input', 'sum'),
                active_perc=("active_perc", 'mean'),
                both_match=('both_match', 'max'),
                sc=('sc', 'max'),
            ).reset_index()
            per_lp_per_position_amount_table = temp_res_weekly
            per_lp_per_position_amount_table["amount_last_new"] = (
                        per_lp_per_position_amount_table["amount_last_temp"] +
                        per_lp_per_position_amount_table["amount_input"])
            # add the amount output at the end of the week
            per_lp_per_position_amount_table["amount_new"] = (per_lp_per_position_amount_table["amount_temp"] +
                                                              per_lp_per_position_amount_table["amount_output"])
            per_lp_per_position_amount_table["amt_roi_new"] = (per_lp_per_position_amount_table["amount_new"] /
                                                               per_lp_per_position_amount_table["amount_last_new"])
            per_lp_per_position_amount_table["fee_roi_new"] = (per_lp_per_position_amount_table["fee_total"] /
                                                               per_lp_per_position_amount_table["amount_last_new"])
            #daily_obs["effective_base"] = daily_obs["amount"] / daily_obs["daily_amount_roi"]
            #daily_obs["fee_total"] = daily_obs.groupby(grp_by_cond)["fee"].cumsum()
            temp_res_weekly["daily_fee_roi"] = temp_res_weekly["fee_roi_new"]
            temp_res_weekly["daily_amt_roi"] = temp_res_weekly["amt_roi_new"]
            temp_res_weekly["daily_overall_roi"] = temp_res_weekly["daily_fee_roi"] + temp_res_weekly["daily_amount_roi"]
            temp_res_weekly["amount_last"] = temp_res_weekly["amount_last_new"]
            temp_res_weekly["date"] = temp_res_weekly["week"]
            #daily_obs_bakup = daily_obs.copy()
            daily_obs = temp_res_weekly
        elif agg_type == 'daily_cumulative':
            # daily cumulative roi
            daily_obs["daily_amount_roi_temp"] = daily_obs["amount"] / daily_obs["amount_last"]
            daily_obs["daily_overall_roi_temp"] = daily_obs["total_amount"] / daily_obs["amount_last"]
            daily_obs["daily_amount_roi"] = daily_obs.groupby(["position_id"])["daily_amount_roi_temp"].cumprod()
            daily_obs["effective_base"] = daily_obs["amount"] / daily_obs["daily_amount_roi"]
            daily_obs["fee_total"] = daily_obs.groupby(["position_id"])["fee"].cumsum()
            daily_obs["daily_fee_roi"] = daily_obs["fee_total"] / daily_obs["effective_base"]
            daily_obs["daily_overall_roi"] = daily_obs["daily_fee_roi"] + daily_obs["daily_amount_roi"]
        else: # by default daily
            daily_obs["daily_overall_roi"] = daily_obs["total_amount"] / daily_obs["amount_last"]
            daily_obs["daily_amount_roi"] = daily_obs["amount"] / daily_obs["amount_last"]
            daily_obs["daily_fee_roi"] = daily_obs["fee"] / daily_obs["amount_last"]



        # winsorize extreme values
        daily_obs['daily_overall_roi_w1'] = winsorize(daily_obs['daily_overall_roi'], limits=[0.01, 0.01])
        daily_obs['daily_overall_roi_w5'] = winsorize(daily_obs['daily_overall_roi'], limits=[0.05, 0.05])
        daily_obs['daily_overall_roi_w10'] = winsorize(daily_obs['daily_overall_roi'], limits=[0.10, 0.010])

        daily_obs['daily_amount_roi_w1'] = winsorize(daily_obs['daily_amount_roi'], limits=[0.01, 0.01])
        daily_obs['daily_amount_roi_w5'] = winsorize(daily_obs['daily_amount_roi'], limits=[0.05, 0.05])
        daily_obs['daily_amount_roi_w10'] = winsorize(daily_obs['daily_amount_roi'], limits=[0.10, 0.010])

        daily_obs['daily_fee_roi_w1'] = winsorize(daily_obs['daily_fee_roi'], limits=[0.01, 0.01])
        daily_obs['daily_fee_roi_w5'] = winsorize(daily_obs['daily_fee_roi'], limits=[0.05, 0.05])
        daily_obs['daily_fee_roi_w10'] = winsorize(daily_obs['daily_fee_roi'], limits=[0.10, 0.010])

        # see where the extreme values lie


        daily_obs["overall_earning"] = daily_obs["daily_overall_roi"] >= 1
        daily_obs["in_range"] = daily_obs["active_perc"] > 0

        def qtile_25(x):
            return x.quantile(0.25)


        def qtile_75(x):
            return x.quantile(0.75)

        daily_obs["age"] = daily_obs.groupby(["position_id"]).cumcount()

        # group by both_match and daily and then generate a new df.

        # check for mixed strategy:
        rec_group = daily_obs[daily_obs["both_match"]]
        default_group = daily_obs[(~daily_obs["both_match"]) & (~daily_obs["sc"])]
        rec_group_pos_ids = rec_group["position_id"].unique()
        default_group_pos_ids = default_group["position_id"].unique()
        rec_group_lp_ids = pd.Series(data_df[data_df["position_id"].isin(rec_group_pos_ids)]["liquidity_provider"].unique())
        manual_group_lp_ids = pd.Series(data_df[data_df["position_id"].isin(default_group_pos_ids)]["liquidity_provider"].unique())
        both_rec_and_default_lp_ids = rec_group_lp_ids[rec_group_lp_ids.isin(manual_group_lp_ids)]
        rec_group_only_lp_ids = rec_group_lp_ids[~rec_group_lp_ids.isin(manual_group_lp_ids)]
        manual_group_only_lp_ids = manual_group_lp_ids[~manual_group_lp_ids.isin(rec_group_lp_ids)]

        positions_by_mixed_lps = pd.Series(data_df[data_df["liquidity_provider"].isin(both_rec_and_default_lp_ids)]["position_id"].unique())
        position_by_rec_lps =  pd.Series(data_df[data_df["liquidity_provider"].isin(rec_group_only_lp_ids)]["position_id"].unique())
        position_by_manual_lps =  pd.Series(data_df[data_df["liquidity_provider"].isin(manual_group_only_lp_ids)]["position_id"].unique())

        daily_obs_mixed_lp_position = daily_obs["position_id"].isin(positions_by_mixed_lps)
        daily_obs_rec_lp_position = daily_obs["position_id"].isin(position_by_rec_lps)
        daily_obs_manual_lp_position = daily_obs["position_id"].isin(position_by_manual_lps)

        result_filtered_out = True

        lps_position_cnt = data_df.groupby(["liquidity_provider"])["position_id"].nunique()
        lps_more_than_one_pos = lps_position_cnt[lps_position_cnt > 1].index.unique()
        positions_by_multiple_lps = data_df[data_df["liquidity_provider"].isin(lps_more_than_one_pos)]["position_id"].unique()
        daily_obs_multiple_lp_position = daily_obs["position_id"].isin(positions_by_multiple_lps)

        daily_obs["more_than_one_pos"] = False
        daily_obs.loc[daily_obs_multiple_lp_position, "more_than_one_pos"] = True

        if result_filtered_out:
            daily_obs = daily_obs[daily_obs["more_than_one_pos"]].copy()

        daily_obs["lp_type"] = "sc"
        daily_obs.loc[daily_obs_mixed_lp_position, "lp_type"] = "mixed"
        daily_obs.loc[daily_obs_rec_lp_position, "lp_type"] = "rec"
        daily_obs.loc[daily_obs_manual_lp_position, "lp_type"] = "manual"

        #  mixed strategy, based on LP, could be better?
        final_grp_by_cond = ["lp_type" , "date", ]
        # final_grp_by_cond = ["both_match",
        #                      "sc",
        #                      "date",
        #                      ]

        daily_by_group = daily_obs.groupby(final_grp_by_cond).agg(
            daily_overall_avg=('daily_overall_roi', 'mean'),
            daily_overall_avg_w1=('daily_overall_roi_w1', 'mean'),
            daily_overall_avg_w5=('daily_overall_roi_w5', 'mean'),
            daily_overall_avg_w10=('daily_overall_roi_w10', 'mean'),
            daily_amt_avg=('daily_amount_roi', 'mean'),
            daily_amt_avg_w1=('daily_amount_roi_w1', 'mean'),
            daily_amt_avg_w5=('daily_amount_roi_w5', 'mean'),
            daily_amt_avg_w10=('daily_amount_roi_w10', 'mean'),
            daily_fee_avg=('daily_fee_roi', 'mean'),
            daily_fee_avg_w1=('daily_fee_roi_w1', 'mean'),
            daily_fee_avg_w5=('daily_fee_roi_w5', 'mean'),
            daily_fee_avg_w10=('daily_fee_roi_w10', 'mean'),
            daily_overall_median=('daily_overall_roi', 'median'),
            daily_overall_median_w1=('daily_overall_roi_w1', 'median'),
            daily_overall_median_w5=('daily_overall_roi_w5', 'median'),
            daily_overall_median_w10=('daily_overall_roi_w10', 'median'),
            daily_amt_median=('daily_amount_roi', 'median'),
            daily_amt_median_w1=('daily_amount_roi_w1', 'median'),
            daily_amt_median_w5=('daily_amount_roi_w5', 'median'),
            daily_amt_median_w10=('daily_amount_roi_w10', 'median'),
            daily_fee_median=('daily_fee_roi', 'median'),
            daily_fee_median_w1=('daily_fee_roi_w1', 'median'),
            daily_fee_median_w5=('daily_fee_roi_w5', 'median'),
            daily_fee_median_w10=('daily_fee_roi_w10', 'median'),
            daily_overall_min=('daily_overall_roi', 'min'),
            daily_overall_min_w1=('daily_overall_roi_w1', 'min'),
            daily_overall_min_w5=('daily_overall_roi_w5', 'min'),
            daily_overall_min_w10=('daily_overall_roi_w10', 'min'),
            daily_amt_min=('daily_amount_roi', 'min'),
            daily_amt_min_w1=('daily_amount_roi_w1', 'min'),
            daily_amt_min_w5=('daily_amount_roi_w5', 'min'),
            daily_amt_min_w10=('daily_amount_roi_w10', 'min'),
            daily_fee_min=('daily_fee_roi', 'min'),
            daily_fee_min_w1=('daily_fee_roi_w1', 'min'),
            daily_fee_min_w5=('daily_fee_roi_w5', 'min'),
            daily_fee_min_w10=('daily_fee_roi_w10', 'min'),
            daily_overall_max=('daily_overall_roi', 'max'),
            daily_overall_max_w1=('daily_overall_roi_w1', 'max'),
            daily_overall_max_w5=('daily_overall_roi_w5', 'max'),
            daily_overall_max_w10=('daily_overall_roi_w10', 'max'),
            daily_amt_max=('daily_amount_roi', 'max'),
            daily_amt_max_w1=('daily_amount_roi_w1', 'max'),
            daily_amt_max_w5=('daily_amount_roi_w5', 'max'),
            daily_amt_max_w10=('daily_amount_roi_w10', 'max'),
            daily_fee_max=('daily_fee_roi', 'max'),
            daily_fee_max_w1=('daily_fee_roi_w1', 'max'),
            daily_fee_max_w5=('daily_fee_roi_w5', 'max'),
            daily_fee_max_w10=('daily_fee_roi_w10', 'max'),
            daily_overall_std=('daily_overall_roi', 'std'),
            daily_overall_std_w1=('daily_overall_roi_w1', 'std'),
            daily_overall_std_w5=('daily_overall_roi_w5', 'std'),
            daily_overall_std_w10=('daily_overall_roi_w10', 'std'),
            daily_amt_std=('daily_amount_roi', 'std'),
            daily_amt_std_w1=('daily_amount_roi_w1', 'std'),
            daily_amt_std_w5=('daily_amount_roi_w5', 'std'),
            daily_amt_std_w10=('daily_amount_roi_w10', 'std'),
            daily_fee_std=('daily_fee_roi', 'std'),
            daily_fee_std_w1=('daily_fee_roi_w1', 'std'),
            daily_fee_std_w5=('daily_fee_roi_w5', 'std'),
            daily_fee_std_w10=('daily_fee_roi_w10', 'std'),
            daily_overall_q25=('daily_overall_roi', qtile_25),
            #daily_overall_q25_w1=('daily_overall_roi_w1', qtile_25),
            # daily_overall_q25_w5=('daily_overall_roi_w5', qtile_25),
            # daily_overall_q25_w10=('daily_overall_roi_w10', qtile_25),
            daily_amt_q25=('daily_amount_roi', qtile_25),
            # daily_amt_q25_w1=('daily_amount_roi_w1', qtile_25),
            # daily_amt_q25_w5=('daily_amount_roi_w5', qtile_25),
            # daily_amt_q25_w10=('daily_amount_roi_w10', qtile_25),
            daily_fee_q25=('daily_fee_roi', qtile_25),
            # daily_fee_q25_w1=('daily_fee_roi_w1', qtile_25),
            # daily_fee_q25_w5=('daily_fee_roi_w5', qtile_25),
            # daily_fee_q25_w10=('daily_fee_roi_w10', qtile_25),
            daily_overall_q75=('daily_overall_roi', qtile_75),
            # daily_overall_q75_w1=('daily_overall_roi_w1', qtile_75),
            # daily_overall_q75_w5=('daily_overall_roi_w5', qtile_75),
            # daily_overall_q75_w10=('daily_overall_roi_w10', qtile_75),
            daily_amt_q75=('daily_amount_roi', qtile_75),
            # daily_amt_q75_w1=('daily_amount_roi_w1', qtile_75),
            # daily_amt_q75_w5=('daily_amount_roi_w5', qtile_75),
            # daily_amt_q75_w10=('daily_amount_roi_w10', qtile_75),
            daily_fee_q75=('daily_fee_roi', qtile_75),
            # daily_fee_q75_w1=('daily_fee_roi_w1', qtile_75),
            # daily_fee_q75_w5=('daily_fee_roi_w5', qtile_75),
            # daily_fee_q75_w10=('daily_fee_roi_w10', qtile_75),
            alive_positions=('position_id', 'nunique'),
            amt_in_daily=('amount_input', 'sum'),
            money_in_pool_total=('amount_last', 'sum'),
            money_in_pool_avg=('amount_last', 'mean'),
            money_in_pool_median=('amount_last', 'median'),
            daily_price=('close', 'mean'),
            overall_earning_count=('overall_earning', 'sum'),
            overall_in_range=('in_range', 'sum'),
            age_avg=('age', 'mean'),
            age_median=('age', 'median'),
        ).reset_index()
        daily_by_group.to_csv("result_filtered" + str(result_filtered_out) + pool_addr + "_" + agg_type + "_" + "_test_other_info_1220.csv")
    # result_df.to_csv("block_liquidity_and_tick.csv")
    my_res = pd.DataFrame(res)
    print("Done")
