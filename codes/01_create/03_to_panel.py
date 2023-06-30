import os

import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel

from codes.shared_library.utils import POOL_ADDR, POOL_INFO, get_parent, UNISWAP_NFT_MANAGER

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
    for pool_addr in tqdm(POOL_ADDR):
        print(f"Working on f{pool_addr}")
        daily_price = daily_prices[daily_prices["pool_address"] == pool_addr].copy()
        weekly_price = weekly_prices[weekly_prices["pool_address"] == pool_addr].copy()
        res_df = pd.read_pickle(
            os.path.join(data_folder_path, 'raw', 'pkl', f"done_accounting_day_datas_{pool_addr}.pkl"))
        res_df = res_df.drop(columns=['open', 'high', 'low', 'close', 'high_tick', 'low_tick'])
        cols_to_change_type = ['liquidity_mpz', 'net_liquidity', 'amount0', 'amount1', 'fee0', 'fee1']
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
        # Need a mapping like this:
        # LP_ID, PositionID, Date
        # We need a date range, i.e. when does that LP interacted with the position (special case for SC)
        # Get the SC:
        sc_cond1 = action_df["nf_position_manager_address"] != UNISWAP_NFT_MANAGER
        # There could also be whose to address is not UNISWAP_NFT_MANAGER
        sc_cond2 = action_df["to_address"] != UNISWAP_NFT_MANAGER
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
        # Then for the ones that have wrong date range: remove those dates?
        sc_positions_obs_part_one = sc_lp_position_map_part_one.merge(sc_positions, how='left', on='position_id')
        sc_positions_obs_part_two = sc_lp_position_map_part_two.merge(sc_positions, how='left', on='position_id')
        # Filter out the ones that do not match the time
        # todo: this might cause trouble when doing the division it's better we take everything into account
        # todo: see sc_position_lp_count
        part_one_date_condition = sc_positions_obs_part_one["date"] >= sc_positions_obs_part_one["insertion_date"]
        sc_positions_obs_part_one_final = sc_positions_obs_part_one[part_one_date_condition].copy()
        part_two_date_condition_1 = sc_positions_obs_part_two["date"] >= sc_positions_obs_part_two["insertion_date"]
        part_two_date_condition_2 = sc_positions_obs_part_two["date"] <= sc_positions_obs_part_two["leave_date"]
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
            columns=['insertion_date', 'liquidity_mpz', 'net_liquidity', 'sqrtPrice', 'leave_date', 'lp_cnt'],
            inplace=True)
        non_sc_positions["sc"] = False
        # Map non-sc-positions with liquidity providers
        non_sc_actions = action_df[~action_df["position_id"].isin(position_id_sc)]
        # Then for these actions we have to assume

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
                                             'total_amount', 'sc']].copy()
        daily_obs = pd.concat([sc_positions_obs, non_sc_positions], ignore_index=True)
        daily_obs.sort_values(by=["liquidity_provider", "position_id", "date"])
        daily_obs["date"] = daily_obs["date"].astype(str)
        daily_obs["daily_overall_roi"] = daily_obs["total_amount"] / daily_obs["amount_last"]
        daily_obs["daily_amount_roi"] = daily_obs["amount"] / daily_obs["amount_last"]
        daily_obs["daily_fee_roi"] = daily_obs["fee"] / daily_obs["amount_last"]
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
        per_lp_per_position_amount_table = lp_position_week_groupby.agg({"amount_last_temp":"first", "amount_temp":"last", "fee":"sum", "amount_input":"sum", "amount_output":"sum", "daily_amount_roi": "prod"}).reset_index()
        # add the amount input at the beginning of the week
        per_lp_per_position_amount_table["amount_last_new"] = per_lp_per_position_amount_table["amount_last_temp"] +  per_lp_per_position_amount_table["amount_input"]
        # add the amount output at the end of the week
        per_lp_per_position_amount_table["amount_new"] = per_lp_per_position_amount_table["amount_temp"] +  per_lp_per_position_amount_table["amount_output"]
        per_lp_per_position_amount_table["amt_roi_new"] = per_lp_per_position_amount_table["amount_new"] / per_lp_per_position_amount_table["amount_last_new"]
        per_lp_per_position_amount_table["fee_roi_new"] = per_lp_per_position_amount_table["fee"] / per_lp_per_position_amount_table["amount_last_new"]
        test1 = per_lp_per_position_amount_table.groupby(["liquidity_provider", "week"]).agg({"amount_last_new":"sum", "amount_new":"sum", "fee":"sum"}).reset_index()
        test1["amt_roi"] = test1["amount_new"] / test1["amount_last_new"]
        continue

        def group_weighted_mean_factory(df: pd.DataFrame, weight_col_name: str):
            # Ref: https://stackoverflow.com/a/69787938/
            def group_weighted_mean(x):
                import numpy as np
                try:
                    return np.average(x, weights=df.loc[x.index, weight_col_name])
                except ZeroDivisionError:
                    return np.average(x)

            return group_weighted_mean


        group_weighted_mean_one = group_weighted_mean_factory(weekly_obs, 'amount_last')
        print("\n grouping and calculating agg")

        tik = time.time()


        def agg_daily_data_daily_average_test(x):
            import numpy as np
            import pandas as pd
            weights = x["amount_last"].values
            x1 = np.average(x["daily_overall_roi"].values)
            x2 = np.average(x["daily_amount_roi"].values)
            x3 = np.average(x["daily_fee_roi"].values)
            x4 = np.sum(x["amount_input"].values)
            x5 = np.sum(x["amount_output"].values)
            x6 = x["position_id"].nunique()
            x7 = np.average(x["sc"].values)
            try:
                x8 = np.average(x["daily_overall_roi"].values, weights=weights)
            except ZeroDivisionError:
                x8 = np.average(x["daily_overall_roi"].values)
            try:
                x9 = np.average(x["daily_amount_roi"].values, weights=weights)
            except ZeroDivisionError:
                x9 = np.average(x["daily_amount_roi"].values)
            try:
                x10 = np.average(x["daily_fee_roi"].values, weights=weights)
            except ZeroDivisionError:
                x10 = np.average(x["daily_fee_roi"].values)
            try:
                x11 = np.average(x["active_perc"].values, weights=weights)
            except ZeroDivisionError:
                x11 = np.average(x["active_perc"].values)
            x12 = np.average(x["active_perc"].values)
            return pd.Series([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12],
                             index=["overall_roi_daily_avg", "amount_roi_daily_avg", "fee_roi_daily_avg",
                                    "total_amount_input", "total_amount_output", "n_pos", "sc_pct",
                                    "overall_roi_daily_avg_weighted", "amount_roi_daily_avg_weighted",
                                    "fee_roi_daily_avg_weighted", "active_perc_avg_weighted", "active_perc_avg"])


        lp_daily_data_daily_averages = weekly_obs.groupby(["liquidity_provider", "week"]).parallel_apply(
            agg_daily_data_daily_average_test)
        print(f"{time.time() - tik} passed")
        tik = time.time()
        # Merge with action
        action_df_for_week = action_df.copy()
        action_df_for_week["week"] = action_df_for_week["block_timestamp"].dt.to_period('W-SAT').dt.start_time
        action_by_lp_week = action_df_for_week.groupby(["liquidity_provider", "week", "action"]).size().unstack(
            fill_value=0).reset_index()
        lp_daily_data_daily_averages = lp_daily_data_daily_averages.reset_index().merge(action_by_lp_week, how='left',
                                                                                        on=["liquidity_provider",
                                                                                            "week"])

        lp_daily_data_daily_averages["week"] = lp_daily_data_daily_averages["week"].astype(str)
        lp_daily_data_daily_averages = lp_daily_data_daily_averages.merge(weekly_price, how='left', on=["week"])
        lp_daily_data_daily_averages.to_pickle(f"{pool_addr}_lp_data_daily_03.pkl")
        print(f"{time.time() - tik} passed")
        tik = time.time()
        print("grouping and calculating agg (alternative method)")
        weekly_obs.sort_values(by=["liquidity_provider", "position_id", "week"], inplace=True)

        # Alternatively, we agg to position-week level then agg to position-lp level
        def agg_lp_position_weekly_test(x):
            import numpy as np
            import pandas as pd
            weights = x["amount_last"].values
            x1 = np.prod(x["daily_amount_roi"].values)
            x2 = np.sum(x["fee"].values)
            x3 = np.average(x["amount_last"].values)
            x4 = x["amount_last"].values[-1]
            x5 = x["amount"].values[-1]
            x6 = np.average(x["amount"].values)
            x7 = np.sum(x["amount_input"].values)
            x8 = np.sum(x["amount_output"].values)
            x9 = np.mean(x["sc"].values)
            x10 = np.mean(x["active_perc"].values)
            try:
                x11 = np.average(x["amount_last"].values, weights=weights)
            except ZeroDivisionError:
                x11 = np.average(x["amount_last"].values)
            try:
                x12 = np.average(x["amount"].values, weights=weights)
            except ZeroDivisionError:
                x12 = np.average(x["amount"].values)
            try:
                x13 = np.average(x["active_perc"].values, weights=weights)
            except ZeroDivisionError:
                x13 = np.average(x["active_perc"].values)
            return pd.Series([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13],
                             index=["amount_roi_weekly_prod", "fee_total", "amount_last_mean",
                                    "amount_last_last", "amount_end", "amount_end_mean", "total_amount_input",
                                    "total_amount_output", "sc_pct",
                                    "active_perc_avg", "amount_last_wm", "amount_end_wm", "active_perc_avg_weighted"])

        lp_position_weekly = groupby.parallel_apply(agg_lp_position_weekly_test).reset_index()
        print(f"{time.time() - tik} passed")
        tik = time.time()
        lp_position_weekly["amount_last"] = lp_position_weekly["amount_end"] / lp_position_weekly[
            'amount_roi_weekly_prod']
        lp_position_weekly["amount_last_by_mean"] = lp_position_weekly["amount_end_wm"] / lp_position_weekly[
            'amount_roi_weekly_prod']
        lp_position_weekly["amount_last_by_wm"] = lp_position_weekly["amount_end_mean"] / lp_position_weekly[
            'amount_roi_weekly_prod']

        # Then from position we agg to the lp level
        wm_for_lp_position_weekly_amt_last = group_weighted_mean_factory(lp_position_weekly, 'amount_last')
        wm_for_lp_position_weekly_amt_last_by_mean = group_weighted_mean_factory(lp_position_weekly,
                                                                                 'amount_last_by_mean')
        wm_for_lp_position_weekly_amt_last_by_wm = group_weighted_mean_factory(lp_position_weekly, 'amount_last_by_wm')
        print(f"{time.time() - tik} passed")
        tik = time.time()
        print("Finalizing result")
        lp_position_weekly.sort_values(
            by=["liquidity_provider", "position_id", "week"], inplace=True)


        def agg_lp_position_weekly_final_test(x):
            import numpy as np
            import pandas as pd
            wts_amt_last = x["amount_last"].values
            wts_amt_last_by_mean = x["amount_last_by_mean"].values
            wts_amt_last_by_wm = x["amount_last_by_wm"].values
            x5 = np.sum(x["amount_last"].values)
            x6 = np.sum(x["amount_last_by_mean"].values)
            x7 = np.sum(x["amount_last_by_wm"].values)
            x8 = np.sum(x["fee_total"].values)
            x9 = np.sum(x["total_amount_input"].values)
            x10 = np.sum(x["total_amount_output"].values)
            x11 = x["position_id"].nunique()
            x12 = np.mean(x["sc_pct"].values)
            x13 = np.mean(x["active_perc_avg"])
            try:
                x1 = np.average(x["amount_roi_weekly_prod"].values, weights=wts_amt_last)
            except ZeroDivisionError:
                x1 = np.average(x["amount_roi_weekly_prod"].values)
            try:
                x2 = np.average(x["amount_roi_weekly_prod"].values, weights=wts_amt_last_by_wm)
            except ZeroDivisionError:
                x2 = np.average(x["amount_roi_weekly_prod"].values)
            try:
                x3 = np.average(x["amount_roi_weekly_prod"].values, weights=wts_amt_last_by_mean)
            except ZeroDivisionError:
                x3 = np.average(x["amount_roi_weekly_prod"].values)
            try:
                x4 = np.average(x["active_perc_avg_weighted"].values, weights=wts_amt_last)
            except ZeroDivisionError:
                x4 = np.average(x["active_perc_avg_weighted"].values)
            return pd.Series([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13],
                             index=["amount_roi_weekly_prod_simple", "amount_roi_weekly_prod_wm",
                                    "amount_roi_weekly_prod_mean",
                                    "active_perc_wm", "amount_last_sum", "amount_last_by_mean_sum",
                                    "amount_last_by_wm_sum",
                                    "fee_total", "total_amount_input",
                                    "total_amount_output", "n_pos", "sc_pct", "active_perc_avg"])


        lp_position_weekly_final = lp_position_weekly.groupby(
            ["liquidity_provider", "week"]).parallel_apply(agg_lp_position_weekly_final_test).reset_index()
        print(f"{time.time() - tik} passed")
        tik = time.time()
        lp_position_weekly_final["fee_roi_simple"] = lp_position_weekly_final['fee_total'] / lp_position_weekly_final[
            'amount_last_sum']
        lp_position_weekly_final["fee_roi_mean"] = lp_position_weekly_final['fee_total'] / lp_position_weekly_final[
            'amount_last_by_mean_sum']
        lp_position_weekly_final["fee_roi_wm"] = lp_position_weekly_final['fee_total'] / lp_position_weekly_final[
            'amount_last_by_wm_sum']
        lp_position_weekly_final["overall_roi_simple"] = lp_position_weekly_final["fee_roi_simple"] + \
                                                         lp_position_weekly_final["amount_roi_weekly_prod_simple"]
        lp_position_weekly_final["overall_roi_mean"] = lp_position_weekly_final["fee_roi_mean"] + \
                                                       lp_position_weekly_final["amount_roi_weekly_prod_wm"]
        lp_position_weekly_final["overall_roi_wm"] = lp_position_weekly_final["fee_roi_wm"] + lp_position_weekly_final[
            "amount_roi_weekly_prod_mean"]
        # Calculate action counts - need to be from that LP directly!

        # merge with

        lp_position_weekly_final = lp_position_weekly_final.reset_index().merge(action_by_lp_week, how='left',
                                                                                on=["liquidity_provider", "week"])
        lp_position_weekly_final = lp_position_weekly_final.merge(action_by_lp_week, how='left',
                                                                  on=["liquidity_provider", "week"])
        lp_position_weekly_final["week"] = lp_position_weekly_final["week"].astype(str)
        lp_position_weekly_final = lp_position_weekly_final.merge(weekly_price, how='left', on=["week"])
        lp_position_weekly_final.to_pickle(f"{pool_addr}_lp_data_weekly_test.pkl")
