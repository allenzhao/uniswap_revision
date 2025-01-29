import os

import pandas as pd
from tqdm import tqdm
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
        
    amount_by_lp_actual.to_stata("amt_actual.dta", convert_dates={"week": "td"})
    ret_data["lp_id"] = pd.factorize(ret_data['liquidity_provider'])[0]
    ret_data["pool_id"] = pd.factorize(ret_data['pool_addr'])[0]
    ret_data.to_pickle(os.path.join(data_folder_path, 'raw', 'pkl', "data_by_lp_0808.pkl"))
    # Generate Data File
    ret_data["sc_pct"] = ret_data["sc"]
    ret_data["amount_roi"] = ret_data["amt_roi"]
    ret_data["position_cnt"] = ret_data["position_id"]
    result_df.to_pickle(os.path.join(data_folder_path, 'raw', 'pkl', "data_by_positions_0711.pkl"))
    ret_data.to_pickle(os.path.join(data_folder_path, 'raw', 'pkl', "ret_data_0808.pkl"))
    ret_data.to_stata("no_short_lived_latest_please_work_0808.dta", convert_dates={"week": "td"})
    ret_data["amount_roi"] = ret_data["amt_roi"]
    ret_data["sc_binary"] = ret_data["sc"] > 0
    ret_data["more_than_one_pos"] = ret_data["position_cnt"] > 1
    # todo: Calculate input output separately, from original file?
    ret_data["id_var"] = ret_data["lp_id"].astype(str) + "_" + ret_data["pool_id"].astype(str)
    ret_data.sort_values(by=["id_var", "week"], inplace=True)
    ret_data["cum_amt_in"] = ret_data.groupby(["id_var"])['amount_input'].cumsum()
    ret_data["cum_amt_out"] = ret_data.groupby(["id_var"])['amount_output'].cumsum()
    ret_data["cum_amt_out"] = ret_data.groupby(["id_var"])['amount_output'].cumsum()
    # deal with potential not continuous?
    ret_data["last_week"] = ret_data.groupby("id_var")["week"].shift(1)
    ret_data["not_continuous_for_lag"] = ret_data["week"] != ret_data["last_week"] + pd.DateOffset(days=7)
    cond_for_not_continuous = (~ret_data["last_week"].isna()) & (ret_data["not_continuous_for_lag"])
    not_continuous_id_vars = ret_data[cond_for_not_continuous]["id_var"].unique()
    ret_data["keep_id_vars_cond1"] = ~ret_data["id_var"].isin(not_continuous_id_vars)
    # remove large ROI?
    large_ROI_id_vars = ret_data[ret_data["overall_roi"] > 1.4]["id_var"].unique()
    ret_data["keep_id_vars_cond2"] = ~ret_data["id_var"].isin(large_ROI_id_vars)
    # remove small LPs
    ret_data_keep_id_vars_for_not_small_lp = ret_data[ret_data["cum_amt_in"] > 10]["id_var"].unique()
    ret_data["keep_id_vars_cond3"] = ret_data["id_var"].isin(ret_data_keep_id_vars_for_not_small_lp)
    # Now generate the ranks, after removing some variables
    ret_data_to_keep_cond = ret_data["keep_id_vars_cond1"] & ret_data["keep_id_vars_cond2"] & ret_data["keep_id_vars_cond3"]
    ret_data_kept = ret_data[ret_data_to_keep_cond].copy()
    ret_data_kept["overall_roi_rank"] = ret_data_kept.groupby(["pool_id", "week"])["overall_roi"].rank(method='max', pct=True)
    ret_data_kept["amount_roi_rank"] = ret_data_kept.groupby(["pool_id", "week"])["amount_roi"].rank(method='max', pct=True)
    ret_data_kept["fee_roi_rank"] = ret_data_kept.groupby(["pool_id", "week"])["fee_roi"].rank(method='max', pct=True)
    ret_data_kept["amount_in_rank"] = ret_data_kept.groupby(["pool_id", "week"])["cum_amt_in"].rank(method='max', pct=True)
    ret_data_kept.sort_values(by=["id_var", "week"], inplace=True)
    ret_data_kept["overall_roi_rank_lag"] = ret_data_kept.groupby(["id_var"])["overall_roi_rank"].shift(1)
    ret_data_kept["amount_roi_rank_lag"] = ret_data_kept.groupby(["id_var"])["amount_roi_rank"].shift(1)
    ret_data_kept["fee_roi_rank_lag"] = ret_data_kept.groupby(["id_var"])["fee_roi_rank"].shift(1)
    ret_data_kept["amount_in_rank_lag"] = ret_data_kept.groupby(["id_var"])["amount_in_rank"].shift(1)
    # todo: fix the nas
    # df['moving'] = df.groupby('object').rolling(10)['value'].mean().reset_index(drop=True)
    ret_data_kept["soph"] = ret_data_kept.groupby(["id_var"]).rolling(4)['amount_roi_rank_lag'].mean().reset_index(drop=True)
    ret_data_kept.to_stata("no_short_lived_latest_please_work_0808_testing.dta", convert_dates={"week": "td"})

    action_by_lp.to_stata("action_lp.dta", convert_dates={"week": "td"})
