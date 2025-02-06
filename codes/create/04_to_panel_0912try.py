import os

import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from pandarallel import pandarallel

from codes.shared_library.utils import POOL_ADDR, POOL_INFO, get_parent, UNISWAP_NFT_MANAGER, UNISWAP_MIGRATOR

if __name__ == "__main__":
    debug = True
    pandarallel.initialize(progress_bar=True, nb_workers=60)
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
    for pool_addr in tqdm(POOL_ADDR):
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
        # Now we link positions with LPs - based on who carried out the action and what
        action_df = pd.read_pickle(os.path.join(data_folder_path, 'raw', 'pkl', f"data_{pool_addr}_0626_no_short.pkl"))
        action_df_for_week = action_df.copy()
        # Need a mapping like this:
        # LP_ID, PositionID, Date
        # We need a date range, i.e. when does that LP interacted with the position (special case for SC)
        # Get the SC:
        sc_cond1 = action_df["nf_position_manager_address"] != UNISWAP_NFT_MANAGER
        # There could also be whose to address is not UNISWAP_NFT_MANAGER
        # This case we identify that position as an SC, in addition
        one_position_id_has_multiple_lp_id = action_df.groupby(["position_id"])["liquidity_provider"].nunique()
        position_id_multiple = one_position_id_has_multiple_lp_id[one_position_id_has_multiple_lp_id > 1].index
        sc_cond2_pre = action_df["position_id"].isin(position_id_multiple)
        sc_cond2 = (action_df["to_address"] != UNISWAP_NFT_MANAGER) & (
                action_df["to_address"] != UNISWAP_MIGRATOR) & sc_cond2_pre
        position_id_sc = action_df[(sc_cond1 | sc_cond2)][
            "position_id"].unique()

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
        assert (position_id_sc_map.shape[0] == position_id_sc_map.shape[0])
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


        if debug:
            only_increase_cond2_lp_ids = sc_observations.sort_values(
                ["nf_position_manager_address", "liquidity_provider", "block_timestamp", "action"]).groupby(
                nft_and_lp).apply(get_last_operation)
        else:
            only_increase_cond2_lp_ids = sc_observations.sort_values(
                ["nf_position_manager_address", "liquidity_provider", "block_timestamp", "action"]).groupby(
                nft_and_lp).parallel_apply(get_last_operation)
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


        if debug:
            only_increase_lp_init_date["pos_ids"] = only_increase_lp_init_date.apply(
                get_relevant_position_ids_for_increase_only, axis=1)
        else:
            only_increase_lp_init_date["pos_ids"] = only_increase_lp_init_date.parallel_apply(
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


        if debug:
            decreased_lps_with_dates["pos_ids"] = decreased_lps_with_dates.apply(
                get_relevant_position_ids_for_decrease, axis=1)
        else:
            decreased_lps_with_dates["pos_ids"] = decreased_lps_with_dates.parallel_apply(
                get_relevant_position_ids_for_decrease, axis=1)

        sc_lp_position_map_part_one = only_increase_lp_init_date[
            ["liquidity_provider", "block_timestamp", "pos_ids"]].explode("pos_ids").rename(
            columns={"pos_ids": "position_id", "block_timestamp": "insertion_date"})
        sc_lp_position_map_part_two = decreased_lps_with_dates[
            ["liquidity_provider", "block_timestamp_min", "block_timestamp_max", "pos_ids"]].explode("pos_ids").rename(
            columns={"pos_ids": "position_id", "block_timestamp_min": "insertion_date",
                     "block_timestamp_max": "leave_date"})
        # Next we want to create the LP-position level data.
        # Note that we shall divide the SC LP positions based on the count
        # (i.e. how many are there each day), this should be taken into consideration
        sc_positions = res_df[res_df["position_id"].isin(position_id_sc)].copy()
        non_sc_positions = res_df[~res_df["position_id"].isin(position_id_sc)].copy()

        # Then our mapping should first duplicate the position observations
        sc_positions_created_date = sc_positions.groupby(["position_id"])["date"].min().reset_index().rename(
            columns={"date": "sc_position_creation_date"})
        sc_positions = sc_positions.merge(sc_positions_created_date, how='left', on='position_id')
        # Then for the ones that have wrong date range: remove those dates?
        sc_positions_obs_part_one = sc_lp_position_map_part_one.merge(sc_positions, how='left', on='position_id')
        sc_positions_obs_part_two = sc_lp_position_map_part_two.merge(sc_positions, how='left', on='position_id')
        # Filter out the ones that do not match the time
        # todo: this might cause trouble when doing the division it's better we take everything into account
        # todo: see sc_position_lp_count
        part_one_date_condition = sc_positions_obs_part_one["sc_position_creation_date"] >= sc_positions_obs_part_one[
            "insertion_date"].dt.date
        sc_positions_obs_part_one_final = sc_positions_obs_part_one[part_one_date_condition].copy()
        part_two_date_condition_1 = sc_positions_obs_part_two["sc_position_creation_date"] >= sc_positions_obs_part_two[
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
        cols_to_divide = [
            'amount0', 'amount1', 'fee0', 'fee1', 'amount0_input', 'amount1_input', 'amount0_output', 'amount1_output',
            'amount', 'fee', 'amount_input', 'amount_output', 'amount0_last', 'amount1_last', 'amount_last',
            'total_amount0', 'total_amount1', 'total_amount'
        ]
        more_than_one = sc_positions_obs["lp_cnt"] > 1
        for col in cols_to_divide:
            sc_positions_obs.loc[more_than_one, col] = sc_positions_obs.loc[more_than_one, col] / sc_positions_obs.loc[
                more_than_one, 'lp_cnt']
        sc_positions_obs["sc"] = True
        sc_positions_obs.drop(
            columns=['insertion_date', 'sqrtPrice', 'leave_date', 'lp_cnt'],
            inplace=True)
        non_sc_positions["sc"] = False
        # Map non-sc-positions with liquidity providers
        non_sc_actions = action_df[~action_df["position_id"].isin(position_id_sc)]
        # Then for these actions we have to assume
        date_range = res_df.groupby(["position_id"]).agg(
            {"date": ["min", "max"]}
        )

        multiple_position_id_map = non_sc_actions.groupby(["position_id"])["liquidity_provider"].nunique()
        # todo: remove these for now take care later
        multiple_position_id_to_remove = multiple_position_id_map[multiple_position_id_map > 1].index.to_list()
        non_sc_positions = non_sc_positions[
            ~non_sc_positions["position_id"].isin(multiple_position_id_to_remove)].copy()
        non_sc_actions = non_sc_actions[~non_sc_actions["position_id"].isin(multiple_position_id_to_remove)].copy()
        non_sc_position_lp_map = non_sc_actions[["position_id", "liquidity_provider"]].drop_duplicates()
        non_sc_positions = non_sc_positions.merge(non_sc_position_lp_map, on=["position_id"], how='left')
        non_sc_positions = non_sc_positions[['liquidity_provider', 'position_id', 'date', 'tick_lower', 'tick_upper',
                                             'current_tick', 'amount0', 'amount1', 'fee0', 'fee1', 'price_range',
                                             'active_perc', 'amount0_input', 'amount1_input', 'amount0_output',
                                             'amount1_output', 'low', 'high', 'open', 'close', 'low_tick',
                                             'high_tick', 'open_tick', 'close_tick', 'volume_crypto_abs',
                                             'volume_stable_abs', 'volume_crypto_net', 'volume_stable_net',
                                             'volume_usd', 'volume_crypto_net_usd', 'volume_stable_net_usd',
                                             'buying_crypto_trade_cnt', 'buying_stable_trade_cnt', 'pool_address',
                                             'amount', 'fee', 'amount_input', 'amount_output', 'amount0_last',
                                             'amount1_last', 'amount_last', 'total_amount0', 'total_amount1',
                                             'total_amount', 'sc', 'net_liquidity', 'liquidity_mpz']].copy()
        daily_obs = pd.concat([sc_positions_obs, non_sc_positions], ignore_index=True)
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
             "amount_output": "sum", "daily_amount_roi": "prod", "sc": "mean", "active_perc": "mean",
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

        test1 = per_lp_per_position_amount_table.groupby(["liquidity_provider", "week"]).agg(
            {"amount_last_new": "sum", "amount_new": "sum", "fee": "sum", "sc": "mean", "position_id": "nunique",
             "amount_input": "sum", "amount_output": "sum", "active_perc": "mean", "sold_held_weighed_avg": "sum",
             "bought_held_weighed_avg": "sum"}).reset_index()
        # Now we also do the LP level action count
        test1["amt_roi"] = test1["amount_new"] / test1["amount_last_new"]
        test1["fee_roi"] = test1["fee"] / test1["amount_last_new"]
        test1["overall_roi"] = test1["amt_roi"] + test1["fee_roi"]
        test1["sold_held_weighed_avg"] = test1["sold_held_weighed_avg"] / test1["amount_last_new"]
        test1["bought_held_weighed_avg"] = test1["bought_held_weighed_avg"] / test1["amount_last_new"]
        test1["pool_addr"] = pool_addr
        action_df_for_week["week"] = action_df_for_week["block_timestamp"].dt.to_period('W-SAT').dt.start_time
        action_df_for_week_items = action_df_for_week["action"] == "DECREASE_LIQUIDITY"
        action_df_for_week["amount0_output_usd"] = 0
        action_df_for_week.loc[action_df_for_week_items, "amount0_output_usd"] = action_df_for_week.loc[action_df_for_week_items, "amount0_usd"]
        action_df_for_week["amount1_output_usd"] = 0
        action_df_for_week.loc[action_df_for_week_items, "amount1_output_usd"] = action_df_for_week.loc[
            action_df_for_week_items, "amount1_usd"]
        action_df_for_week["amount_output_usd"] = action_df_for_week["amount0_output_usd"] +  action_df_for_week["amount1_output_usd"]
        action_df_for_week_items = action_df_for_week["action"] == "INCREASE_LIQUIDITY"
        action_df_for_week["amount0_input_usd"] = 0
        action_df_for_week.loc[action_df_for_week_items, "amount0_input_usd"] = action_df_for_week.loc[action_df_for_week_items, "amount0_usd"]
        action_df_for_week["amount1_input_usd"] = 0
        action_df_for_week.loc[action_df_for_week_items, "amount1_input_usd"] = action_df_for_week.loc[
            action_df_for_week_items, "amount1_usd"]
        action_df_for_week["amount_input_usd"] = action_df_for_week["amount0_input_usd"] +  action_df_for_week["amount1_input_usd"]
        action_df_for_week["position_id"] = action_df_for_week["position_id"].astype(str)
        action_by_lp_position_week = action_df_for_week.groupby(["liquidity_provider", "position_id", "week", "action"],
                                                                observed=True).size().unstack(
            fill_value=0).reset_index().rename(
            columns={"INCREASE_LIQUIDITY": "lp_position_deposits", "DECREASE_LIQUIDITY": "lp_position_removals",
                     "FEE_COLLECTION": "lp_position_collect_fees"})
        action_df_input_output_by_week = action_df_for_week.groupby(["liquidity_provider", "week"])[["amount_input_usd", "amount_output_usd"]].sum().reset_index()
        action_by_lp_week = action_df_for_week.groupby(["liquidity_provider", "week", "action"],
                                                       observed=True).size().unstack(
            fill_value=0).reset_index().rename(
            columns={"INCREASE_LIQUIDITY": "lp_deposits", "DECREASE_LIQUIDITY": "lp_removals",
                     "FEE_COLLECTION": "lp_collect_fees"})
        test1 = test1.merge(action_by_lp_week, on=["liquidity_provider", "week"], how='left')
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
