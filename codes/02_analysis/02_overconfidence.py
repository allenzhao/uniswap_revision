import os

import pandas as pd
from codes.shared_library.utils import get_parent

def get_measures_turnover(my_df, group_by):
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
    lp_info = pd.read_csv(os.path.join(data_folder_path, "raw", "lp_amt_fourweek_rank_rollmean.csv"))
    position_info = pd.read_pickle(os.path.join(data_folder_path, "raw", "pkl", "data_by_lp_0711.pkl"))
    position_info.rename(columns={"pool_addr":"pool_address"}, inplace=True)
    # User-selection only?
    cols_to_fix_na = ['lp_deposits', 'lp_removals', 'lp_collect_fees']
    for col in cols_to_fix_na:
        position_info[col] = position_info[col].fillna(0)

    position_info["user_operations_on_all_positions_cnt"] = (
                position_info['lp_deposits'] +
                position_info['lp_removals'] +
                position_info['lp_collect_fees'])
    grp_by_cond = ["pool_address", "liquidity_provider", "week"]
    position_info.sort_values(by=grp_by_cond, inplace=True)
    position_info["week"] = position_info["week"].astype(str)
    lp_info = lp_info.merge(position_info, on=grp_by_cond, how='left')
    #lp_info_temp = lp_info.drop_duplicates(subset=(["position_id", "week"]))
    lp_info["soph"] = lp_info["amt_fourweek_rank_rollmean"] > 0.5
    lp_info["sc_binary"] = lp_info["sc"] > 0
    lp_info["soph_by_value_3bins"] = pd.cut(lp_info["amt_fourweek_rank_rollmean"], bins=3, labels=['Low Soph', 'Mid Soph', 'High Soph'])
    lp_info["soph_by_value_3bins_alter"] = pd.cut(lp_info["amt_fourweek_rank_rollmean"], bins=[0, 0.254013, 0.751095, 1], labels=['Low Soph', 'Mid Soph', 'High Soph'], ordered=False)
    lp_info["soph_by_value_2bins"] = pd.cut(lp_info["amt_fourweek_rank_rollmean"], bins=2, labels=['Low Soph', 'High Soph'])
    lp_info["soph_by_value_4bins"] = pd.cut(lp_info["amt_fourweek_rank_rollmean"], bins=4, labels=['V','Low Soph', 'Mid Soph', 'High Soph'])
    lp_info["soph_by_equal_3bins"] = pd.qcut(lp_info["amt_fourweek_rank_rollmean"], q=3, labels=['Low Soph', 'Mid Soph', 'High Soph'])
    lp_info["soph_by_equal_3bins_alter"] = pd.qcut(lp_info["amt_fourweek_rank_rollmean"], q=[0, 0.25, 0.75, 1], labels=['Low Soph', 'Mid Soph', 'High Soph'])
    lp_info["soph_by_equal_2bins"] = pd.qcut(lp_info["amt_fourweek_rank_rollmean"], q=2, labels=['Low Soph', 'High Soph'])
    lp_info["soph_by_equal_4bins"] = pd.qcut(lp_info["amt_fourweek_rank_rollmean"], q=4, labels=['V','Low Soph', 'Mid Soph', 'High Soph'])
    lp_buy_action = lp_info["lp_deposits"] == 0
    lp_sell_action = lp_info["lp_removals"] == 0

    lp_info["sales_turnover"] = lp_info["sold_held_weighed_avg"]
    lp_info.loc[lp_sell_action, "sales_turnover"] = 0
    lp_info["buy_turnover"] = lp_info["bought_held_weighed_avg"]
    lp_info.loc[lp_buy_action, "buy_turnover"] = 0
    lp_info["turnover"] = (lp_info["sales_turnover"] + lp_info["buy_turnover"]) / 2
    t0 = get_measures_turnover(lp_info, ["soph_by_value_2bins"]).rename(columns={"soph_by_value_2bins": "By"})
    t1 = get_measures_turnover(lp_info, ["soph_by_value_3bins"]).rename(columns={"soph_by_value_3bins": "By"})
    t2 = get_measures_turnover(lp_info, ["soph_by_value_3bins_alter"]).rename(columns={"soph_by_value_3bins_alter": "By"})
    t3 = get_measures_turnover(lp_info, ["soph_by_value_4bins"]).rename(columns={"soph_by_value_4bins": "By"})
    t4 = get_measures_turnover(lp_info, ["soph_by_equal_2bins"]).rename(columns={"soph_by_equal_2bins": "By"})
    t5 = get_measures_turnover(lp_info, ["soph_by_equal_3bins"]).rename(columns={"soph_by_equal_3bins": "By"})
    t6 = get_measures_turnover(lp_info, ["soph_by_equal_3bins_alter"]).rename(columns={"soph_by_equal_3bins_alter": "By"})
    t7 = get_measures_turnover(lp_info, ["soph_by_equal_4bins"]).rename(columns={"soph_by_equal_4bins": "By"})
    lp_info["add_money"] = lp_info['lp_deposits'] > 0
    lp_info["removes"] = lp_info['lp_removals'] > 0
    get_measures_turnover(lp_info, "sc_binary")
    print("Here")
