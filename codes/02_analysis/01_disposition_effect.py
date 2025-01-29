import os

import pandas as pd
from codes.shared_library.utils import get_parent


def get_measures(my_df, group_by):
    my_res = my_df.groupby(group_by).agg(
        realized_gains=('realized_gains', 'sum'),
        realized_loses=('realized_loses', 'sum'),
        paper_gains=('paper_gains', 'sum'),
        paper_loses=('paper_loses', 'sum')
    ).reset_index()
    my_res["pgr"] = my_res["realized_gains"] / (my_res["realized_gains"] + my_res["paper_gains"])
    my_res["plr"] = my_res["realized_loses"] / (my_res["realized_loses"] + my_res["paper_loses"])
    my_res["de"] = my_res["pgr"] - my_res["plr"]
    return my_res.copy()

def get_measures_oc_1(my_df, group_by):
    my_res = my_df.groupby(group_by).agg(
        loss_last_week=('loss_last_week', 'sum'),
        add_money_event=('add_money_event', 'sum'),
        loss_this_week=('loss_this_week', 'sum'),
    ).reset_index()
    my_res["add_then_loss"] = my_res["realized_gains"] / (my_res["realized_gains"] + my_res["paper_gains"])
    my_res["plr"] = my_res["realized_loses"] / (my_res["realized_loses"] + my_res["paper_loses"])
    my_res["de"] = my_res["pgr"] - my_res["plr"]
    return my_res.copy()



def get_measures_odean(my_df, group_by):
    my_res = my_df.groupby(group_by).agg(
        loss_last_week=('loss_last_week', 'sum'),
        add_money_event=('add_money_event', 'sum'),
        loss_this_week=('loss_this_week', 'sum'),
    ).reset_index()
    my_res["add_then_loss"] = my_res["realized_gains"] / (my_res["realized_gains"] + my_res["paper_gains"])
    my_res["plr"] = my_res["realized_loses"] / (my_res["realized_loses"] + my_res["paper_loses"])
    my_res["de"] = my_res["pgr"] - my_res["plr"]
    return my_res.copy()


if __name__ == "__main__":
    data_folder_path = os.path.join(get_parent(), "data")
    res_dfs = []
    dfs = []
    results = []
    result_df = pd.DataFrame()
    lp_info = pd.read_stata("uniswap0826.dta")
    position_info = pd.read_pickle(os.path.join(data_folder_path, "raw", "pkl", "data_by_lp_0808.pkl"))
    position_info.rename(columns={"pool_addr":"pool_address"}, inplace=True)
    #position_info = pd.read_pickle(os.path.join(data_folder_path, "raw", "pkl", "data_by_positions.pkl"))
    # User-selection only?
    cols_to_fix_na = ['lp_deposits', 'lp_removals', 'lp_collect_fees']
    cols_to_fix_na_full = cols_to_fix_na + ['lp_position_deposits', 'lp_position_removals', 'lp_position_collect_fees',
                                            'position_deposits', 'position_removals', 'position_collect_fees']
    # for col in cols_to_fix_na_full:
    #     position_info[col] = position_info[col].fillna(0)
    # position_info["user_operations_on_this_position_cnt"] = (
    #         position_info['lp_position_deposits'] +
    #         position_info['lp_position_removals'] +
    #         position_info['lp_position_collect_fees'])
    position_info["user_operations_on_all_positions_cnt"] = (
            position_info['lp_deposits'] +
            position_info['lp_removals'] +
            position_info['lp_collect_fees'])
    # position_info["user_operations_focal_position"] = position_info["user_operations_on_this_position_cnt"] > 0
    # position_info["position_cnts"] = (
    #         position_info['position_deposits'] +
    #         position_info['position_removals'] +
    #         position_info['position_collect_fees'])
    for col in cols_to_fix_na:
        position_info[col] = position_info[col].fillna(0)
    #position_info["position_operated"] = position_info["position_cnts"] > 0
    position_info["realized"] = (position_info["lp_removals"] > 0)
    position_info["paper"] = (position_info["user_operations_on_all_positions_cnt"] > 0) & (~position_info["realized"])
    # Overall Gaining - by position level
    grp_by_cond = ["pool_address", "liquidity_provider", "week"]
    grp_by_cond_without_week = ["pool_address", "liquidity_provider"]
    position_info.sort_values(by=grp_by_cond, inplace=True)
    #position_info["amt_roi"] = position_info["amount_new"] / position_info["amount_last_new"]
    #position_info["fee_roi"] = position_info["fee"] / position_info["amount_last_new"]
    #position_info["overall_roi"] = position_info["amt_roi"] + position_info["fee_roi"]
    position_info["amt_roi_cum_prod"] = position_info.groupby(grp_by_cond_without_week)["amt_roi"].cumprod()
    position_info["fee_cumsum"] = position_info.groupby(grp_by_cond_without_week)["fee"].cumsum()
    position_info["amount_input_cumsum"] = position_info.groupby(grp_by_cond_without_week)["amount_input"].cumsum()
    position_info["amount_output_cumsum"] = position_info.groupby(grp_by_cond_without_week)["amount_output"].cumsum()
    position_info["amount_output_cumsum_lag"] = position_info.groupby(grp_by_cond_without_week)["amount_output_cumsum"].shift(1).fillna(0)
    position_info["amount_temp"] = position_info["amount_new"] - position_info["amount_output"]
    position_info["amount_last_temp"] = position_info["amount_last_new"] - position_info["amount_input"]
    position_info["gain_alter"] = (position_info["amount_temp"] + position_info["fee_cumsum"]) > (position_info["amount_input_cumsum"] - position_info[ "amount_output_cumsum"])
    position_info["loses_later"] = (position_info["amount_temp"] + position_info["fee_cumsum"]) < (position_info["amount_input_cumsum"] - position_info[ "amount_output_cumsum"])
    position_info["fee_roi_cum_prod"] = position_info.groupby(grp_by_cond_without_week)["fee_roi"].cumsum()
    position_info["overall_roi_cum_prod"] = position_info["amt_roi_cum_prod"] + position_info["fee_roi_cum_prod"]
    position_info["overall_roi_cum_prod_lag"] = position_info.groupby(grp_by_cond_without_week)["overall_roi_cum_prod"].shift(1)
    position_info["overall_roi"] = position_info.groupby(grp_by_cond_without_week)["overall_roi_cum_prod"].shift(1)
    # Then merge to our left table
    position_info["week"] = position_info["week"].astype(str)
    lp_info = lp_info[["liquidity_provider","pool_address","week","sophisitication"]].copy()
    lp_info["amt_fourweek_rank_rollmean"] = lp_info["sophisitication"]
    lp_info["week"] = lp_info["week"].astype(str)
    lp_info = lp_info.merge(position_info, on=grp_by_cond, how='left')
    lp_info["gains"] = lp_info["overall_roi_cum_prod_lag"] > 1
    lp_info["loses"] = lp_info["overall_roi_cum_prod_lag"] < 1
    lp_info["gains_thisweek"] = lp_info["overall_roi_cum_prod"] > 1
    lp_info["loses_thisweek"] = lp_info["overall_roi_cum_prod"] < 1
    lp_info["paper_gains"] = lp_info["paper"] & lp_info["gains_thisweek"]
    lp_info["realized_gains"] = lp_info["realized"] & lp_info["gains_thisweek"]
    lp_info["paper_loses"] = lp_info["paper"] & lp_info["loses_thisweek"]
    lp_info["realized_loses"] = lp_info["realized"] & lp_info["loses_thisweek"]
    #lp_info_temp = lp_info.drop_duplicates(subset=(["position_id", "week"]))
    lp_info["soph"] = lp_info["amt_fourweek_rank_rollmean"] > 0.5
    lp_info["soph_by_value_3bins"] = pd.cut(lp_info["amt_fourweek_rank_rollmean"], bins=3, labels=['Low Soph', 'Mid Soph', 'High Soph'])
    lp_info["soph_by_value_3bins_alter"] = pd.cut(lp_info["amt_fourweek_rank_rollmean"], bins=4, labels=['Low Soph', 'Mid Soph',  'Mid Soph', 'High Soph'], ordered=False)
    lp_info["soph_by_value_2bins"] = pd.cut(lp_info["amt_fourweek_rank_rollmean"], bins=2, labels=['Low Soph', 'High Soph'])
    lp_info["soph_by_value_4bins"] = pd.cut(lp_info["amt_fourweek_rank_rollmean"], bins=4, labels=['V','Low Soph', 'Mid Soph', 'High Soph'])
    lp_info["soph_by_equal_3bins"] = pd.qcut(lp_info["amt_fourweek_rank_rollmean"], q=3, labels=['Low Soph', 'Mid Soph', 'High Soph'])
    lp_info["soph_by_equal_3bins_alter"] = pd.qcut(lp_info["amt_fourweek_rank_rollmean"], q=[0, 0.25, 0.75, 1], labels=['Low Soph', 'Mid Soph', 'High Soph'])
    lp_info["soph_by_equal_2bins"] = pd.qcut(lp_info["amt_fourweek_rank_rollmean"], q=2, labels=['Low Soph', 'High Soph'])
    lp_info["soph_by_equal_4bins"] = pd.qcut(lp_info["amt_fourweek_rank_rollmean"], q=4, labels=['V','Low Soph', 'Mid Soph', 'High Soph'])
    t0 = get_measures(lp_info, ["soph_by_value_2bins"]).rename(columns={"soph_by_value_2bins": "By"})
    t1 = get_measures(lp_info, ["soph_by_value_3bins"]).rename(columns={"soph_by_value_3bins": "By"})
    t2 = get_measures(lp_info, ["soph_by_value_3bins_alter"]).rename(columns={"soph_by_value_3bins_alter": "By"})
    t3 = get_measures(lp_info, ["soph_by_value_4bins"]).rename(columns={"soph_by_value_4bins": "By"})
    t4 = get_measures(lp_info, ["soph_by_equal_2bins"]).rename(columns={"soph_by_equal_2bins": "By"})
    t5 = get_measures(lp_info, ["soph_by_equal_3bins"]).rename(columns={"soph_by_equal_3bins": "By"})
    t6 = get_measures(lp_info, ["soph_by_equal_3bins_alter"]).rename(columns={"soph_by_equal_3bins_alter": "By"})
    t7 = get_measures(lp_info, ["soph_by_equal_4bins"]).rename(columns={"soph_by_equal_4bins": "By"})
    pd.concat([t0, t1, t2, t3, t4, t5, t6, t7], ignore_index=True).to_csv("by_soph_alter.csv")
    lp_info["overall_roi_lag"] = lp_info.groupby(grp_by_cond_without_week)["overall_roi"].shift(1)
    lp_info["loss_last_week"] = lp_info["overall_roi_lag"] < 1
    lp_info["add_money"] = lp_info['lp_deposits'] > 0
    lp_info["add_money_and_loss"] = (lp_info["overall_roi"] < 1) & (lp_info['lp_deposits'] > 0)
    lp_info["loss_and_add_money_and_loss"] = lp_info["loss_last_week"] & lp_info["add_money_and_loss"]
    lp_info["earns_last_week"] = lp_info["overall_roi_lag"] > 1
    lp_info["removes"] = lp_info['lp_removals'] > 0
    lp_info["removes_and_earns"] = (lp_info["overall_roi"] > 1) & (lp_info['removes'] > 0)
    lp_info["earns_and_remove_money_and_earns"] = lp_info["earns_last_week"] & lp_info["removes_and_earns"]
    lp_info_temp_for_cnt = lp_info[lp_info["loss_last_week"]].copy()
    lp_info_temp_for_cnt_2 = lp_info[lp_info["earns_last_week"]].copy()
    lp_info["sc_binary"] =lp_info["sc"] > 0
    lp_info["soph"] = lp_info["amt_fourweek_rank_rollmean"] >= 0.668
    get_measures(lp_info, "sc_binary")
    lp_info["add_money_and_loss"] = (lp_info["not_first"] == 1) & (lp_info["deposits"] > 0) & (
            lp_info["overall_roi"] < 1)
    position_info["add_money_and_loss"] = (position_info["not_first"] == 1) & (position_info["deposits"] > 0) & (
            position_info["overall_roi"] < 1)
    print("Here")
