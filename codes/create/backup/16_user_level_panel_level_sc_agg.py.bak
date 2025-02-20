import os

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats.mstats import winsorize
from gmpy2 import mpz


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
    sc_verification_data = pd.read_csv(os.path.join(data_folder_path, "raw", 'sc_verified_data.csv'))

    for pool_addr in pool_addrs:
        temp_res = {}
        temp_res["pool_addr"] = pool_addr

        print(pool_addr)
        data_df = pd.read_pickle(os.path.join(pickle_path, f"input_info_{pool_addr}.pkl"))
        data_df["position_id"] = data_df["position_id"].astype(str)
        data_df["sc"] = data_df["nf_position_manager_address"] != UNISWAP_NFT_MANAGER
        this_pool_scs = sc_verification_data[sc_verification_data["pool_address"] == pool_addr].reset_index(drop=True)
        # probably only need to keep the scs?

        temp_res["unique_lps"] = data_df["liquidity_provider"].unique().shape[0]
        temp_res["positions"] = data_df["position_id"].unique().shape[0]
        print(data_df.shape)
        print(data_df["liquidity_provider"].unique().shape)
        # Figure out the upper or lower prices which should be used as base
        if pool_addr in pool_addrs_usdc:
            upper_col = 'price_upper_0_1'
            lower_col = 'price_lower_0_1'
            token_price_col = 'token1_price'
        else:
            upper_col = 'price_upper_1_0'
            lower_col = 'price_lower_1_0'
            token_price_col = 'token0_price'

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
        # has multiple users
        # didn't interact with the official uniswap ?
        #sc_cond2_pre = action_df["position_id"].isin(position_id_multiple)
        #sc_cond2 = (action_df["to_address"] != UNISWAP_NFT_MANAGER) & (
        #         action_df["to_address"] != UNISWAP_MIGRATOR) & sc_cond2_pre
        position_id_sc = action_df[(sc_cond1)]["position_id"].unique()

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
        # assert (position_id_sc_map.shape[0] == position_id_sc_map.shape[0])
        # Then for each SC, figure out who interacted with position and when,
        # to know when to map which SC to which position and during which period
        per_lp_sc_count = sc_observations.groupby(["nf_position_manager_address", "liquidity_provider", "action"],
                                                  observed=True).size().unstack(fill_value=0).reset_index()
        # 1. Only observe deposit events for a given LP for a given SC
        nft_and_lp = ["nf_position_manager_address", "liquidity_provider"]
        only_increase_cond1 = per_lp_sc_count["DECREASE_LIQUIDITY"] == 0  # Only increase


        def get_last_operation(x):
            all_actions = x["action"].values
            last_action = all_actions[-1] if len(all_actions) > 0 else 'NAN'
            return last_action == 'INCREASE_LIQUIDITY'


        only_increase_cond2_lp_ids = sc_observations.sort_values(
            ["nf_position_manager_address", "liquidity_provider", "block_timestamp", "action"]).groupby(
            nft_and_lp).apply(get_last_operation)
        only_increase_cond2 = only_increase_cond2_lp_ids[only_increase_cond2_lp_ids].index.to_frame(index=False)
        only_increase_lp = per_lp_sc_count[only_increase_cond1][nft_and_lp].copy()
        only_increase_lp = pd.concat([only_increase_lp, only_increase_cond2], ignore_index=True)
        # For these LP, we figure out their start date, i.e. the first interaction with SC
        only_increase_lp_init_date = \
            only_increase_lp.merge(sc_observations, on=nft_and_lp, how='left').groupby(nft_and_lp)[
                "block_timestamp"].min().reset_index()

        def get_relevant_position_ids_for_increase_only(x, helper_df=sc_position_with_start_and_end_dates):
            helper_df = helper_df.copy()
            sc_addr = x["nf_position_manager_address"]
            date = x["block_timestamp"]
            cond1 = helper_df["nf_position_manager_address"] == sc_addr
            cond2 = helper_df["date_max"] >= date
            # that position has an end date later than the date being put into SC
            cond = cond1 & cond2
            pos_ids = helper_df[cond]["position_id"].unique()
            return pos_ids


        only_increase_lp_init_date["pos_ids"] = only_increase_lp_init_date.apply(
            get_relevant_position_ids_for_increase_only, axis=1)

        # 2. Observe deposit & removals for a given LP for a given SC
        # In this case we use the last *active* date for a given LP for a given SC
        # as the last date for a position to belong to that particular LP
        # , i.e. we assume that whenever there is a last operation
        # also we require that the removal event is the last event (this is shown in increase only)
        # So we need the ids that are not in the only_increase_lp_init_date
        per_lp_sc_count["nft_and_lp"] = (
                per_lp_sc_count["nf_position_manager_address"]
                + "--"
                + per_lp_sc_count["liquidity_provider"]
        )
        only_increase_lp_init_date["nft_and_lp"] = (
                only_increase_lp_init_date["nf_position_manager_address"]
                + "--"
                + only_increase_lp_init_date["liquidity_provider"]
        )
        decreased_lp_cond = ~per_lp_sc_count["nft_and_lp"].isin(only_increase_lp_init_date["nft_and_lp"].unique())
        decreased_lps = per_lp_sc_count[decreased_lp_cond][nft_and_lp].copy()
        decreased_lps_with_dates = decreased_lps.merge(sc_observations, on=nft_and_lp, how='left').groupby(
            nft_and_lp).agg({"block_timestamp": ["min", "max"]})
        decreased_lps_with_dates.columns = ["_".join(col) for col in decreased_lps_with_dates.columns]
        decreased_lps_with_dates.reset_index(inplace=True)

        def get_relevant_position_ids_for_decrease(x, helper_df=sc_position_with_start_and_end_dates):
            helper_df = helper_df.copy()
            sc_addr = x["nf_position_manager_address"]
            start_date = x["block_timestamp_min"]
            end_date = x["block_timestamp_max"]

            cond1 = helper_df["nf_position_manager_address"] == sc_addr
            cond2 = helper_df["date_max"] >= start_date
            cond3 = helper_df["date_min"] <= end_date
            cond = cond1 & cond2 & cond3
            pos_ids = helper_df[cond]["position_id"].unique()
            return pos_ids


        decreased_lps_with_dates["pos_ids"] = decreased_lps_with_dates.apply(
            get_relevant_position_ids_for_decrease, axis=1)

        sc_lp_position_map_part_one = only_increase_lp_init_date[
            ["liquidity_provider", "block_timestamp", "pos_ids"]].explode("pos_ids").rename(
            columns={"pos_ids": "position_id", "block_timestamp": "insertion_date"})
        sc_lp_position_map_part_two = decreased_lps_with_dates[
            ["liquidity_provider", "block_timestamp_min", "block_timestamp_max", "pos_ids"]].explode("pos_ids").rename(
            columns={"pos_ids": "position_id", "block_timestamp_min": "insertion_date",
                     "block_timestamp_max": "leave_date"})
        ######
        # mapping lp and sc positions
        ######

        # Next we want to create the LP-position level data.
        # Note that we shall divide the SC LP positions based on the count
        # (i.e. how many are there each day), this should be taken into consideration

        sc_positions = res_df[res_df["position_id"].isin(position_id_sc)].copy()
        sc_positions_created_date = sc_positions.groupby(["position_id"])["date"].min().reset_index().rename(
            columns={"date": "sc_position_creation_date"})
        sc_positions_liquidity_amount_daily_mpz = sc_positions[["position_id", "date", "net_liquidity"]]
        sc_positions = sc_positions.merge(sc_positions_created_date, how='left', on='position_id')
        # Then for the ones that have wrong date range: remove those dates?
        sc_positions_obs_part_one = sc_lp_position_map_part_one.merge(sc_positions, how='left', on='position_id')
        sc_positions_obs_part_two = sc_lp_position_map_part_two.merge(sc_positions, how='left', on='position_id')
        # Filter out the ones that do not match the time
        # todo: this might cause trouble when doing the division it's better we take everything into account
        # todo: see sc_position_lp_count
        ## todo: this seemed to fail to consider when the LP would insert into a "exsiting" position?
        part_one_date_condition = sc_positions_obs_part_one["date"] >= sc_positions_obs_part_one[
            "insertion_date"].dt.date # CREATED AFTER INSERTION DATE?
        sc_positions_obs_part_one_final = sc_positions_obs_part_one[part_one_date_condition].copy()
        part_two_date_condition_1 = sc_positions_obs_part_two["date"] >= sc_positions_obs_part_two[
            "insertion_date"].dt.date
        part_two_date_condition_2 = sc_positions_obs_part_two["date"] <= sc_positions_obs_part_two["leave_date"].dt.date
        part_two_date_conditions = part_two_date_condition_1 & part_two_date_condition_2
        sc_positions_obs_part_two_final = sc_positions_obs_part_two[part_two_date_conditions].copy()
        # Calculate ROIs - two ways, both need to take amount change into consideration
        sc_positions_obs = pd.concat([sc_positions_obs_part_one_final, sc_positions_obs_part_two_final],
                                     ignore_index=True)
        # Then for each position-date, figure out how many LPs are in that position, then divide
        sc_position_lp_count = sc_positions_obs.groupby(["position_id"])[
            "liquidity_provider"].nunique().reset_index().rename(columns={"liquidity_provider": "lp_cnt"})
        sc_positions_obs = sc_positions_obs.merge(sc_position_lp_count, on=["position_id"], how='left')

        ## from the original action df, determine the lp's deposit amounts and then calculate as percentage?
        sc_observations_copy = sc_observations.copy()
        sc_observations_copy["amount_changed_total"] = sc_observations_copy["amount0_usd"] + sc_observations_copy["amount1_usd"]
        sc_obs_decrease_cond = sc_observations_copy["action"] == "DECREASE_LIQUIDITY"
        # todo: better calculation
        sc_observations_copy.loc[sc_obs_decrease_cond, "amount_changed_total"] *= 0
        # todo: better accounting for percentage?
        sc_observations_copy["date"] = sc_observations_copy["block_timestamp"].dt.date
        sc_total_input_for_percentage = sc_observations_copy.groupby(["nf_position_manager_address", "date"])["amount_changed_total"].sum().reset_index()
        sc_total_input_for_percentage_by_person = sc_observations_copy.groupby(["liquidity_provider", "nf_position_manager_address", "date"])["amount_changed_total"].sum().reset_index()
        sc_total_input_for_percentage_by_person.rename(columns={"amount_changed_total": "amount_changed_total_this_lp"}, inplace=True)
        test_merge = sc_total_input_for_percentage_by_person.merge(sc_total_input_for_percentage, on=["nf_position_manager_address", "date"], how="left")
        test_merge["percentage"] = test_merge["amount_changed_total_this_lp"] / test_merge["amount_changed_total"]

        cols_to_divide = [
            'amount0', 'amount1', 'fee0', 'fee1', 'amount0_input', 'amount1_input', 'amount0_output', 'amount1_output',
            'amount', 'fee', 'amount_input', 'amount_output', 'amount0_last', 'amount1_last', 'amount_last',
            'total_amount0', 'total_amount1', 'total_amount'
        ]
        more_than_one = sc_positions_obs["lp_cnt"] > 1
        # for col in cols_to_divide:
            # sc_positions_obs.loc[more_than_one, col] = sc_positions_obs.loc[more_than_one, col] / sc_positions_obs.loc[more_than_one, 'lp_cnt']

        ## todo: note: amount input needs to be re-considered
        ## todo: combine the amount input at last

        daily_obs = sc_positions_obs
        daily_obs.sort_values(by=["liquidity_provider", "position_id", "date"])
        daily_obs["daily_overall_roi"] = daily_obs["total_amount"] / daily_obs["amount_last"]
        daily_obs["daily_amount_roi"] = daily_obs["amount"] / daily_obs["amount_last"]
        daily_obs["daily_fee_roi"] = daily_obs["fee"] / daily_obs["amount_last"]
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
        daily_obs = daily_obs[daily_obs["position_id"].isin(ok_ids_new)]
        lp_max_min_date_check = daily_obs.groupby(["liquidity_provider"]).agg(
            max_date=("date", "max"),
            min_date=("date", "min")
        ).reset_index()
        lp_max_min_date_check["date_diff"] = lp_max_min_date_check["max_date"] - lp_max_min_date_check["min_date"]
        # at least one week in pool
        ok_lp_ids = lp_max_min_date_check[lp_max_min_date_check["date_diff"] >= '7 days']["liquidity_provider"].unique()
        daily_obs = daily_obs[
            daily_obs["liquidity_provider"].isin(ok_lp_ids)]  # remove the lps that have lived too short amount of time
        daily_obs["date"] = daily_obs["date"].astype(str)
        weekly_obs = daily_obs.copy()
        weekly_obs["week"] = pd.to_datetime(weekly_obs["date"]).dt.to_period('W-SAT').dt.start_time
        weekly_obs["week"].astype(str)
        # Switch back with amount input
        weekly_obs["amount_last_temp"] = weekly_obs["amount_last"] - weekly_obs["amount_input"]
        weekly_obs["amount_temp"] = weekly_obs["amount"] - weekly_obs["amount_output"]
        lp_position_id_week = ["liquidity_provider", "position_id", "week"]
        lp_position_week_groupby = weekly_obs.groupby(lp_position_id_week)
        per_lp_per_position_week_end = lp_position_week_groupby["amount_last"].first()
        per_lp_per_position_week_start = lp_position_week_groupby["amount"].last()
        weekly_obs["complete_removal"] = weekly_obs["net_liquidity"] < 1024
        weekly_obs["not_first"] = (weekly_obs["net_liquidity"] != weekly_obs["liquidity_mpz"])
        per_lp_per_position_amount_table = lp_position_week_groupby.agg(
            {"amount_last_temp": "first", "amount_temp": "last", "fee": "sum", "amount_input": "sum",
             "amount_output": "sum", "daily_amount_roi": "prod", "active_perc": "mean",
             "complete_removal": "mean", "not_first": "mean"}).reset_index()
        # add the amount input at the beginning of the week
        per_lp_per_position_amount_table["amount_last_new"] = (per_lp_per_position_amount_table["amount_last_temp"] +
                                                               per_lp_per_position_amount_table["amount_input"])
        # add the amount output at the end of the week
        per_lp_per_position_amount_table["amount_new"] = (per_lp_per_position_amount_table["amount_temp"] +
                                                          per_lp_per_position_amount_table["amount_output"])
        per_lp_per_position_amount_table["amt_roi_new"] = (per_lp_per_position_amount_table["amount_new"] /
                                                           per_lp_per_position_amount_table["amount_last_new"])
        per_lp_per_position_amount_table["fee_roi_new"] = (per_lp_per_position_amount_table["fee"] /
                                                           per_lp_per_position_amount_table["amount_last_new"])
        per_lp_per_position_amount_table["sold_held"] = per_lp_per_position_amount_table["amount_output"] / \
                                                        per_lp_per_position_amount_table["amount_last_new"]
        per_lp_per_position_amount_table["bought_held"] = per_lp_per_position_amount_table["amount_input"] / \
                                                          per_lp_per_position_amount_table["amount_new"]
        larger_than_one_cond = per_lp_per_position_amount_table["sold_held"] > 1
        per_lp_per_position_amount_table.loc[larger_than_one_cond, "sold_held"] = 1
        larger_than_one_cond = per_lp_per_position_amount_table["bought_held"] > 1
        per_lp_per_position_amount_table.loc[larger_than_one_cond, "bought_held"] = 1
        per_lp_per_position_amount_table["sold_held_weighed_avg"] = per_lp_per_position_amount_table["sold_held"] * \
                                                                    per_lp_per_position_amount_table["amount_last_new"]
        per_lp_per_position_amount_table["bought_held_weighed_avg"] = per_lp_per_position_amount_table["bought_held"] * \
                                                                      per_lp_per_position_amount_table[
                                                                          "amount_last_new"]
        per_lp_per_position_amount_table = per_lp_per_position_amount_table.merge(position_id_sc_map, on=["position_id"], how="left")
        test1 = per_lp_per_position_amount_table.groupby(["liquidity_provider", "nf_position_manager_address", "week"]).agg(
            {"amount_last_new": "sum", "amount_new": "sum", "fee": "sum",  "position_id": "nunique",
             "amount_input": "sum", "amount_output": "sum", "active_perc": "mean", "sold_held_weighed_avg": "sum",
             "bought_held_weighed_avg": "sum"}).reset_index()
        # Now we also do the LP level action count
        test1["amt_roi"] = test1["amount_new"] / test1["amount_last_new"]
        test1["fee_roi"] = test1["fee"] / test1["amount_last_new"]
        test1["overall_roi"] = test1["amt_roi"] + test1["fee_roi"]
        test1["sold_held_weighed_avg"] = test1["sold_held_weighed_avg"] / test1["amount_last_new"]
        test1["bought_held_weighed_avg"] = test1["bought_held_weighed_avg"] / test1["amount_last_new"]
        test1["pool_addr"] = pool_addr

        def cumsum_mpz(my_df):
            my_res_df = my_df.copy()
            mpz_arr = my_res_df["liquidity_mpz"].to_numpy().cumsum()
            my_res_df["net_liquidity"] = mpz_arr
            return my_res_df


        action_df_temp_copy = action_df[['liquidity_provider', 'nf_position_manager_address', 'liquidity', 'position_id', 'block_timestamp', 'action']].copy().reset_index(drop=True)
        action_df_temp_copy = action_df_temp_copy[action_df_temp_copy["position_id"].isin(position_id_sc)].copy().reset_index(drop=True)


        action_df_temp_copy["week"] = action_df_temp_copy["block_timestamp"].dt.to_period('W-SAT').dt.start_time

        action_df_temp_copy["liquidity_mpz"] = action_df_temp_copy["liquidity"].apply(lambda x: mpz(x))
        decrease_condition = action_df_temp_copy["action"] == "DECREASE_LIQUIDITY"
        action_df_temp_copy.loc[decrease_condition, "liquidity_mpz"] = action_df_temp_copy.loc[decrease_condition, "liquidity_mpz"] * mpz(-1)
        # create LP share accounting table
        action_df_temp_copy_for_lp_share_supply = action_df_temp_copy.copy()
        action_df_temp_copy_for_lp_share_supply.sort_values(by=["nf_position_manager_address", "block_timestamp", "action"], inplace=True)
        action_df_temp_copy_for_lp_share_supply["obs_count_this_nft_mgr"] = action_df_temp_copy_for_lp_share_supply.groupby(["nf_position_manager_address"]).nf_position_manager_address.cumcount()
        action_df_temp_copy_for_lp_share_supply["coins_issued"] = np.nan
        # todo: also need to make sure the first event is indeed increasing
        first_obs_cond = action_df_temp_copy_for_lp_share_supply["obs_count_this_nft_mgr"] == 0
        action_df_temp_copy_for_lp_share_supply.loc[first_obs_cond, "coins_issued"] = action_df_temp_copy_for_lp_share_supply.loc[first_obs_cond, "liquidity_mpz"]
        # when adding, we use the liquidity changed * current liquidity / total liquidity to
        action_df_temp_copy_for_lp_share_supply.sort_values(by=["position_id", "block_timestamp", "action"], inplace=True)
        action_df_temp_copy_for_lp_share_supply_cumsum_liquidity = action_df_temp_copy_for_lp_share_supply.groupby(["position_id"]).apply(cumsum_mpz)
        # the first total supply now changed to the correct amount
        action_df_temp_copy_for_lp_share_supply_cumsum_liquidity["net_liquidity_lag"] = action_df_temp_copy_for_lp_share_supply_cumsum_liquidity["net_liquidity"] - action_df_temp_copy_for_lp_share_supply_cumsum_liquidity["liquidity_mpz"]
        increase_liq_cond = action_df_temp_copy_for_lp_share_supply_cumsum_liquidity["action"] == "INCREASE_LIQUIDITY"
        increase_liq_cond_and_net_liqudity_is_zero = increase_liq_cond & (action_df_temp_copy_for_lp_share_supply_cumsum_liquidity["net_liquidity_lag"] == 0)
        action_df_temp_copy_for_lp_share_supply_cumsum_liquidity.loc[increase_liq_cond_and_net_liqudity_is_zero, "net_liquidity_lag"] = 1
        action_df_temp_copy_for_lp_share_supply_cumsum_liquidity.loc[increase_liq_cond, "coins_issued"] = action_df_temp_copy_for_lp_share_supply_cumsum_liquidity.loc[increase_liq_cond,
            "liquidity_mpz"] / action_df_temp_copy_for_lp_share_supply_cumsum_liquidity.loc[increase_liq_cond, "net_liquidity"]

        decrease_liq_cond = action_df_temp_copy_for_lp_share_supply_cumsum_liquidity["action"] == "DECREASE_LIQUIDITY"
        action_df_temp_copy_for_lp_share_supply_cumsum_liquidity.loc[decrease_liq_cond, "coins_issued"] = \
        action_df_temp_copy_for_lp_share_supply_cumsum_liquidity.loc[decrease_liq_cond,
        "liquidity_mpz"] / action_df_temp_copy_for_lp_share_supply_cumsum_liquidity.loc[
            decrease_liq_cond, "net_liquidity_lag"]
        action_df_temp_copy_for_lp_share_supply_cumsum_liquidity["coins_issued"] = action_df_temp_copy_for_lp_share_supply_cumsum_liquidity["coins_issued"].fillna(0)
        action_df_temp_copy_for_lp_share_supply_cumsum_liquidity["coins_issued_temp"] = action_df_temp_copy_for_lp_share_supply_cumsum_liquidity["coins_issued"].astype("float64")
        # action_df_temp_copy_for_lp_share_supply_cumsum_liquidity stores how much this LP owns of the current position
        # then for each position?

        consider_short_lived = action_df_temp_copy.groupby("position_id").agg(liquidity_sum=("liquidity_mpz", "sum"),
                                                                    min_time=("block_timestamp", "min"),
                                                                    max_time=("block_timestamp", "max")).reset_index()
        drop_cond = ((consider_short_lived["max_time"] - consider_short_lived["min_time"]).dt.days < 1) & (
                consider_short_lived["liquidity_sum"] == 0)
        to_drop_ids = consider_short_lived[drop_cond]['position_id'].unique()
        increase_condition = action_df_temp_copy["action"] == "INCREASE_LIQUIDITY"

        overall_liquidity = action_df_temp_copy.groupby("position_id")["liquidity_mpz"].sum()
        data_for_resample = action_df_temp_copy[
            ["position_id", "block_timestamp", "liquidity_mpz"]
        ].copy()
        # create fake for the ones that were not removed
        fake_data = pd.DataFrame(overall_liquidity[overall_liquidity > 0].index)
        fake_data["block_timestamp"] = action_df_temp_copy["block_timestamp"].max()
        fake_data["liquidity_mpz"] = "0"

        fake_data["liquidity_mpz"] = fake_data["liquidity_mpz"].apply(
            lambda x: mpz(x)
        )
        data_for_resample = pd.concat([data_for_resample, fake_data], axis=0, ignore_index=True)
        data_for_resample.set_index("block_timestamp", inplace=True)
        # data_for_resample = mpd.DataFrame(data_for_resample)
        resampled = (
            data_for_resample.groupby(["position_id"])
            .resample("D")["liquidity_mpz"]
            .sum()
        )
        resampled = resampled.reset_index()

        # action_df_temp_copy_by_week_sum = action_df_temp_copy[~action_df_temp_copy['position_id'].isin(to_drop_ids)].groupby(['liquidity_provider', 'nf_position_manager_address', 'position_id', 'week'])['liquidity_mpz'].sum().reset_index()



        test_resampled = resampled.groupby(["position_id"]).apply(cumsum_mpz)

        # Calculating "percentage" - each at its first insertion; then if deleted
        # note: can we confirm that some LPs remove more than they deposit?


        # resample on individual level; if removed more than what they have then just ignore?
        action_df_temp_copy_insertion_only = action_df_temp_copy[action_df_temp_copy["action"] == "INCREASE_LIQUIDITY"].copy()
        #action_df_temp_copy_insertion_only = action_df_temp_copy.copy()
        data_for_resample_ind = action_df_temp_copy_insertion_only[
            ["liquidity_provider", "nf_position_manager_address", "block_timestamp", "liquidity_mpz"]
        ].copy()
        data_for_resample_ind.set_index('block_timestamp', inplace=True)
        # at first insertion, determine the percentage of
        data_for_resample_ind_nf_position_manager =(
            data_for_resample_ind.groupby(["nf_position_manager_address"])
            .resample("D")["liquidity_mpz"]
            .sum()
        )
        data_for_resample_ind_nf_position_manager = data_for_resample_ind_nf_position_manager.reset_index()
        data_for_resample_ind_nf_position_manager = data_for_resample_ind_nf_position_manager.groupby(["nf_position_manager_address"]).apply(cumsum_mpz)

        data_for_resample_ind_individual =(
            data_for_resample_ind.groupby(["liquidity_provider", "nf_position_manager_address"])
            .resample("D")["liquidity_mpz"]
            .sum()
        )
        data_for_resample_ind_individual = data_for_resample_ind_individual.reset_index()
        data_for_resample_ind_individual = data_for_resample_ind_individual.groupby(["liquidity_provider", "nf_position_manager_address"]).apply(cumsum_mpz)
        testing_resample_ind = data_for_resample_ind_individual.merge(data_for_resample_ind_nf_position_manager, on=["nf_position_manager_address", 'block_timestamp'], how="left")
        # consider first insertion only for LP's actions to get the percentage?
        testing_resample_ind["percentage"] = testing_resample_ind["net_liquidity_x"] / testing_resample_ind["net_liquidity_y"]


        test_ind_liquidity_mpz = data_for_resample_ind.groupby(["liquidity_provider", "nf_position_manager_address"])["liquidity_mpz"].sum().reset_index()
        # create fake for the ones that were not removed
        fake_data = pd.DataFrame(overall_liquidity[overall_liquidity > 0].index)
        fake_data["block_timestamp"] = action_df_temp_copy["block_timestamp"].max()
        fake_data["liquidity_mpz"] = "0"

        fake_data["liquidity_mpz"] = fake_data["liquidity_mpz"].apply(
            lambda x: mpz(x)
        )
        data_for_resample = pd.concat([data_for_resample, fake_data], axis=0, ignore_index=True)
        data_for_resample.set_index("block_timestamp", inplace=True)
        # data_for_resample = mpd.DataFrame(data_for_resample)
        resampled = (
            data_for_resample.groupby(["position_id"])
            .resample("D")["liquidity_mpz"]
            .sum()
        )
        resampled = resampled.reset_index()


        # action_df_temp_copy_by_week_sum.sort_values(["liquidity_provider", "nf_position_manager_address", "week"], inplace=True)
        # action_df_by_lp_net_liquidity_mpz = action_df_temp_copy_by_week_sum.groupby(['liquidity_provider', 'nf_position_manager_address']).apply(cumsum_mpz)

        action_df_for_week = action_df_for_week[action_df_for_week["position_id"].isin(position_id_sc)].copy().reset_index(drop=True)
        action_df_for_week["week"] = action_df_for_week["block_timestamp"].dt.to_period('W-SAT').dt.start_time
        action_df_for_week_items = action_df_for_week["action"] == "DECREASE_LIQUIDITY"
        action_df_for_week["amount0_output_usd"] = 0
        action_df_for_week.loc[action_df_for_week_items, "amount0_output_usd"] = action_df_for_week.loc[
            action_df_for_week_items, "amount0_usd"]
        action_df_for_week["amount1_output_usd"] = 0
        action_df_for_week.loc[action_df_for_week_items, "amount1_output_usd"] = action_df_for_week.loc[
            action_df_for_week_items, "amount1_usd"]
        action_df_for_week["amount_output_usd"] = action_df_for_week["amount0_output_usd"] + action_df_for_week[
            "amount1_output_usd"]
        action_df_for_week_items = action_df_for_week["action"] == "INCREASE_LIQUIDITY"
        action_df_for_week["amount0_input_usd"] = 0
        action_df_for_week.loc[action_df_for_week_items, "amount0_input_usd"] = action_df_for_week.loc[
            action_df_for_week_items, "amount0_usd"]
        action_df_for_week["amount1_input_usd"] = 0
        action_df_for_week.loc[action_df_for_week_items, "amount1_input_usd"] = action_df_for_week.loc[
            action_df_for_week_items, "amount1_usd"]
        action_df_for_week["amount_input_usd"] = action_df_for_week["amount0_input_usd"] + action_df_for_week[
            "amount1_input_usd"]
        action_df_for_week["position_id"] = action_df_for_week["position_id"].astype(str)
        action_by_lp_position_week = action_df_for_week.groupby(["liquidity_provider", "position_id", "week", "action"],
                                                                observed=True).size().unstack(
            fill_value=0).reset_index().rename(
            columns={"INCREASE_LIQUIDITY": "lp_position_deposits", "DECREASE_LIQUIDITY": "lp_position_removals",
                     "FEE_COLLECTION": "lp_position_collect_fees"})
        action_df_input_output_by_week = action_df_for_week.groupby(["liquidity_provider", "week"])[
            ["amount_input_usd", "amount_output_usd"]].sum().reset_index()
        action_df_input_output_by_week_sc = action_df_for_week.groupby(["liquidity_provider", "nf_position_manager_address", "week"])[
            ["amount_input_usd", "amount_output_usd"]].sum().reset_index()

        action_by_lp_week = action_df_for_week.groupby(["liquidity_provider", "nf_position_manager_address","week", "action"],
                                                       observed=True).size().unstack(
            fill_value=0).reset_index().rename(
            columns={"INCREASE_LIQUIDITY": "lp_deposits", "DECREASE_LIQUIDITY": "lp_removals",
                     "FEE_COLLECTION": "lp_collect_fees"})
        test1 = test1.merge(action_by_lp_week, on=["liquidity_provider", "nf_position_manager_address", "week"], how='left')
        test1 = test1.merge(action_df_input_output_by_week_sc, on=["liquidity_provider", "nf_position_manager_address", "week"], how='left')
        cols_to_fix_na = ['lp_deposits', 'lp_removals', 'lp_collect_fees', 'amount_input_usd', 'amount_output_usd']
        for col in cols_to_fix_na:
            test1[col] = test1[col].fillna(0)
        # find all SCs' actions

        sc_all_input_output = action_df_for_week.groupby(["nf_position_manager_address", "week"])[["amount_input_usd", "amount_output_usd"]].sum().reset_index()
        sc_all_input_output.sort_values(by=["nf_position_manager_address", "week"], inplace=True)
        sc_all_input_output["amount_input_cumsum_this_nft"] = sc_all_input_output.groupby(["nf_position_manager_address"])["amount_input_usd"].cumsum()
        sc_all_input_output["amount_output_cumsum_this_nft"] = sc_all_input_output.groupby(["nf_position_manager_address"])["amount_output_usd"].cumsum()
        #test1.sort_values(by=["liquidity_provider", "nf_position_manager_address", "week"], inplace=True)
        sc_user_input_output = action_df_for_week.groupby(["nf_position_manager_address", "liquidity_provider", "week"])[["amount_input_usd", "amount_output_usd"]].sum().reset_index()
        #test1["amount_input_cumsum"] = test1.groupby(["liquidity_provider", "nf_position_manager_address"])["amount_input_usd"].cumsum()
        #test1["amount_output_cumsum"] = test1.groupby(["liquidity_provider", "nf_position_manager_address"])["amount_output_usd"].cumsum()
        #test1 = test1.merge(sc_all_input_output[])
        sc_user_input_output.sort_values(by=["liquidity_provider", "nf_position_manager_address", "week"], inplace=True)
        sc_user_input_output["amount_input_cumsum"] = sc_user_input_output.groupby(["liquidity_provider", "nf_position_manager_address"])["amount_input_usd"].cumsum()
        sc_user_input_output["amount_output_cumsum"] = sc_user_input_output.groupby(["liquidity_provider", "nf_position_manager_address"])["amount_output_usd"].cumsum()
        test2 = sc_user_input_output.merge(sc_all_input_output, on=["nf_position_manager_address", "week"], how='left')
        #test2 = test1.merge(sc_all_input_output[['nf_position_manager_address', 'week', 'amount_input_cumsum_this_nft', 'amount_output_cumsum_this_nft']], on=["nf_position_manager_address", "week"], how='left')
        test2["percentage"] = test2["amount_input_cumsum"] / test2["amount_input_cumsum_this_nft"]
        test3 = test2[['nf_position_manager_address', 'liquidity_provider', 'week', 'percentage']].copy().merge(sc_user_input_output , on=["nf_position_manager_address", "liquidity_provider", "week"], how='left')
        test3.rename(columns={"amount_input_usd": "amount_input_usd_this_week_this_lp", "amount_output_usd": "amount_output_usd_this_week_this_lp"}, inplace=True)
        test4 = test1.merge(test3, on=["nf_position_manager_address", 'liquidity_provider', "week"], how='left')
        test4["percentage"] = test4.groupby(["nf_position_manager_address", 'liquidity_provider'])["percentage"].ffill()

        action_by_position_week = action_df_for_week.groupby(["position_id", "week", "action"],
                                                             observed=True).size().unstack(
            fill_value=0).reset_index().rename(
            columns={"INCREASE_LIQUIDITY": "position_deposits", "DECREASE_LIQUIDITY": "position_removals",
                     "FEE_COLLECTION": "position_collect_fees"})
        # map position to LP
        per_lp_per_position_amount_table["position_id"] = per_lp_per_position_amount_table["position_id"].astype(str)
        per_lp_per_position_amount_table = per_lp_per_position_amount_table.merge(action_by_lp_position_week,
                                                                                  on=["liquidity_provider",
                                                                                      "position_id", "week"],
                                                                                  how='left')
        per_lp_per_position_amount_table = per_lp_per_position_amount_table.merge(action_by_lp_week,
                                                                                  on=["liquidity_provider", "week"],
                                                                                  how='left')
        per_lp_per_position_amount_table = per_lp_per_position_amount_table.merge(action_by_position_week,
                                                                                  on=["position_id", "week"],
                                                                                  how='left')
        per_lp_per_position_amount_table["pool_address"] = pool_addr
        ret_data = pd.concat([ret_data, test1], ignore_index=True)
        result_df = pd.concat([result_df, per_lp_per_position_amount_table], ignore_index=True)
        action_by_lp_week["pool_address"] = pool_addr
        action_by_lp = pd.concat([action_by_lp, action_by_lp_week], ignore_index=True)
        action_df_input_output_by_week["pool_address"] = pool_addr
        amount_by_lp_actual = pd.concat([amount_by_lp_actual, action_df_input_output_by_week], ignore_index=True)



        ######
        # todo: agg to weekly?;
        # todo: data is a bit off, try and figure out why?
        ######

    print("Done")