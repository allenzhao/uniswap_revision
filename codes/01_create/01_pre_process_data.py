import datetime
import logging
import os

from pandas import DataFrame
from parallel_pandas import ParallelPandas

import gmpy2
from gmpy2 import mpz
from tqdm import tqdm

import numpy as np
import pandas as pd

from codes.shared_library.utils import TICK_BASE, POOL_INFO, UNISWAP_NFT_MANAGER, Q96, POOL_ADDR


class LPData:
    __slots__ = (
        "pool_address",
        "data_path",
        "pickle_path",
        "df",
        "data",
        "start_block",
        "NFT_MANAGER",
        "base_token0",
        "decimal0",
        "decimal1",
        "fee_tier",
        "pool_name",
        "LIQ_REMOVE_THRESHOLD",
        "accounting_table",
        "parallelize",
        "debug",
        "pool_info",
        "fake_data_for_accounting",
    )

    def __init__(self, pool_address, data_path=None, debug=True, parallelize_processing=False):
        # basic setups
        ParallelPandas.initialize(n_cpu=32, split_factor=4, disable_pr_bar=False)
        gmpy2.get_context().precision = 200
        tqdm.pandas()
        # create vars
        self.debug = debug
        self.pool_address = pool_address
        if data_path is not None:
            self.data_path = data_path
        else:
            self.data_path = os.path.join(self.get_parent(), "data", "raw")
        self.df = pd.DataFrame()
        self.accounting_table = pd.DataFrame()
        self.fake_data_for_accounting = pd.DataFrame()
        self.pool_info = POOL_INFO[pool_address]
        self.pool_name = "{} @ {}".format(
            self.pool_info.pair_name, self.pool_info.fee_tier / 10000
        )
        self.LIQ_REMOVE_THRESHOLD = 1024
        self.decimal0 = self.pool_info.decimal0
        self.decimal1 = self.pool_info.decimal1
        self.base_token0 = self.pool_info.base_token0
        self.fee_tier = self.pool_info.fee_tier
        if parallelize_processing is not None:
            self.parallelize = parallelize_processing
        else:
            self.parallelize = self.debug
        self.NFT_MANAGER = UNISWAP_NFT_MANAGER
        self.pickle_path = os.path.join(self.data_path, "pkl")
        self.data = self.read_file(data_path=os.path.join(self.data_path, self.pool_address + "_fixed.csv"))

        self.start_block = self.data["block_number"].min()
        logging.basicConfig(
            format="%(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler("./logs/" + self.curr_time() + ".log"),
                logging.StreamHandler(),
            ],
            level=logging.INFO,
        )
        logging.info("Finished reading in data for pool {0}%".format(self.pool_name))

    def preprocess(self):
        logging.info("Start processing")
        cols_to_drop = [
            "blockchain",
            "token0_address",
            "token1_address",
            "token0_symbol",
            "token1_symbol",
            "pool_address",
            "pool_name",
        ]
        self.data.drop(columns=cols_to_drop, inplace=True)
        logging.info(f"Starting with shape: {self.data.shape}")
        # Using MPZ to avoid large number overflowing
        self.data["liquidity_mpz"] = self.data["liquidity"].apply(lambda x: mpz(x))
        wrong_liq_condition = (
                (self.data["amount0_adjusted"] == 0.0)
                & (self.data["amount1_adjusted"] == 0.0)
                & (self.data["liquidity_mpz"] != 0.0)
        )
        logging.info(f"Change liquidity rows {sum(wrong_liq_condition)}")
        self.data.loc[wrong_liq_condition, "liquidity_mpz"] = gmpy2.mpz('0')
        fee_collect_cond = self.data["liquidity_mpz"] == 0
        logging.info(f"Fee collection rows {sum(fee_collect_cond)}")
        self.data.loc[fee_collect_cond, "action"] = "FEE_COLLECTION"
        # future: fee collection amount could be considered using in our data
        self.data["action"] = (
            self.data["action"]
            .astype("category")
            .cat.reorder_categories(
                ["INCREASE_LIQUIDITY", "DECREASE_LIQUIDITY", "FEE_COLLECTION"],
                ordered=True,
            )
        )
        decrease_condition = self.data["action"] == "DECREASE_LIQUIDITY"
        self.data.loc[
            decrease_condition,
            ["amount0_adjusted", "amount1_adjusted"],
        ] *= -1.0
        self.data.loc[decrease_condition, "liquidity_mpz"] = self.data.loc[
                                                                 decrease_condition, "liquidity_mpz"] * gmpy2.mpz(-1)
        # For the ID we're going to use the NF_TOKEN_ID, where available,
        # if not we use the token_id, else we create one based on the other information available to use
        # self.data.loc[self.data["token_id"] == -1, "token_id"] = np.nan
        # make sure the token_id is int, fill nas with -1
        logging.info("Start fixing id issues and create custom ids")
        self.data["token_id"] = self.data["token_id"].fillna(-1).astype(int)
        self.data.loc[self.data["nf_token_id"] == "None", "nf_token_id"] = np.nan
        self.data["position_id"] = (
            self.data["nf_token_id"].fillna(-1).astype(float).astype(int)
        )
        position_id_custom_condition = self.data["position_id"] == -1
        self.data.loc[position_id_custom_condition, "position_id"] = self.data.loc[
            position_id_custom_condition, "token_id"
        ]
        self.data["custom_id"] = (
                self.data["nf_position_manager_address"]
                + "--"
                + self.data["tick_lower"].astype("str")
                + "--"
                + self.data["tick_upper"].astype("str")
        )
        position_id_custom_condition = self.data["position_id"] == -1
        self.data.loc[position_id_custom_condition, "position_id"] = self.data.loc[
            position_id_custom_condition, "custom_id"
        ]
        sort_by_lst = ["position_id", "block_number", "block_timestamp", "action"]
        self.data = self.sort_df(sort_by_lst, self.data)
        # This should be how we identify a position interacting with SC or not;
        # But the to address could be alarming, so instead now we only use if the NFT manager is not official
        self.data["USING_UNI_STRICT"] = True
        self.data["USING_UNI_RELAXED"] = True
        # strict: if not interacting directly with nft manager
        not_using_uni_condition_strict = (self.data["nf_position_manager_address"] != self.NFT_MANAGER)
        # relaxed: if it doesn't have a token_id
        not_using_uni_condition_relaxed = (self.data["token_id"] == -1)
        self.data.loc[not_using_uni_condition_strict, "USING_UNI_STRICT"] = False
        self.data.loc[not_using_uni_condition_relaxed, "USING_UNI_RELAXED"] = False
        self.data = self.sort_df(sort_by_lst, self.data)
        first_obs = self.data.drop_duplicates(subset="position_id", keep="first").copy()
        other_first = first_obs[first_obs["action"] != "INCREASE_LIQUIDITY"]
        if not other_first.empty:
            wrong_pos = other_first["position_id"].unique()
            logging.info(f"Remove wrong pos ids ({wrong_pos.shape} affected)")
            logging.info(f"Before dropping: {self.data.shape}")
            self.data[self.data["position_id"].isin(wrong_pos)].to_csv(
                f"wrong_pos_{self.pool_address}.csv"
            )
            self.data = self.data[
                ~self.data["position_id"].isin(wrong_pos)
            ].reset_index(drop=True)
            logging.info(f"After dropping: {self.data.shape}")
            self.data = self.sort_df(sort_by_lst, self.data)
        # Drop short-lived liquidity (liquidity that lasted only within a block)
        consider_short_lived = self.data.groupby("position_id").agg(liquidity_sum=("liquidity_mpz", "sum"),
                                                                    min_time=("block_timestamp", "min"),
                                                                    max_time=("block_timestamp", "max")).reset_index()
        drop_cond = (consider_short_lived["max_time"] == consider_short_lived["min_time"]) & (
                consider_short_lived["liquidity_sum"] == 0)
        to_drop_ids = consider_short_lived[drop_cond]['position_id']
        logging.info(f"To drop: {to_drop_ids.nunique()} of {self.data['position_id'].nunique()}")
        self.data = self.data[~self.data["position_id"].isin(to_drop_ids)].copy()
        logging.info("Start to generate accounting table")
        cols_for_accounting = [
            "position_id",
            "liquidity_provider",
            "block_number",
            "block_timestamp",
            "nf_position_manager_address",
            "amount0_adjusted",
            "amount1_adjusted",
            "token0_price",
            "token1_price",
            "liquidity_mpz",
            "action",
        ]
        self.accounting_table = self.data[cols_for_accounting].copy()
        # remove the ones that were only fee collection
        self.accounting_table = self.accounting_table[
            self.accounting_table["liquidity_mpz"] != 0
            ]
        logging.info("Start on resampling to daily")
        if self.debug:
            first_positions = self.data["position_id"].unique()[:20]
            self.data = self.data[self.data["position_id"].isin(first_positions)]
        overall_liquidity = self.data.groupby("position_id")["liquidity_mpz"].sum()
        data_for_resample = self.data[
            ["position_id", "block_timestamp", "liquidity_mpz"]
        ].copy()
        # create fake for the ones that were not removed
        logging.info("Create fake data for resampling")
        fake_data = pd.DataFrame(overall_liquidity[overall_liquidity > 0].index)
        fake_data["block_timestamp"] = self.data["block_timestamp"].max()
        fake_data["liquidity_mpz"] = "0"
        if self.parallelize:
            fake_data["liquidity_mpz"] = fake_data["liquidity_mpz"].p_apply(
                lambda x: gmpy2.mpz(x)
            )
        else:
            fake_data["liquidity_mpz"] = fake_data["liquidity_mpz"].apply(
                lambda x: gmpy2.mpz(x)
            )
        data_for_resample = pd.concat([data_for_resample, fake_data], axis=0, ignore_index=True)
        data_for_resample.set_index("block_timestamp", inplace=True)
        # data_for_resample = mpd.DataFrame(data_for_resample)
        logging.info("Resampling")
        resampled = (
            data_for_resample.groupby(["position_id"])
            .resample("D")["liquidity_mpz"]
            .sum()
        )
        resampled = resampled.reset_index()
        logging.info("Resampling done, start to do cumulative sum")

        def cumsum_mpz(my_df):
            my_res_df = my_df.copy()
            mpz_arr = my_res_df["liquidity_mpz"].to_numpy().cumsum()
            my_res_df["net_liquidity"] = mpz_arr
            return my_res_df

        if self.parallelize:
            self.df = resampled.groupby(["position_id"]).p_apply(cumsum_mpz)
        else:
            self.df = resampled.groupby(["position_id"]).apply(cumsum_mpz)
        logging.info("Preprocessing done, saving df to pickle")

        self.df.to_pickle(os.path.join(self.pickle_path, f"preprocessed_day_datas_{self.pool_address}.pkl"))

    def balance_calculation(self):
        logging.info("Calculating balances based on net liquidity...")
        data = (
            self.data[["position_id", "tick_lower", "tick_upper"]]
            .drop_duplicates(subset="position_id", keep="first")
            .copy()
        )
        data["sa"] = TICK_BASE ** (data["tick_lower"] / 2)
        data["sb"] = TICK_BASE ** (data["tick_upper"] / 2)
        self.df.reset_index(drop=True, inplace=True)
        self.df = self.df.merge(data, on="position_id", how="left")
        # merge with pool daily data
        pool_day_data = pd.read_csv(
            os.path.join(self.data_path, 'pool_data', "pool_day_data_{}.csv".format(self.pool_address))
        )
        self.df.rename(columns={"block_timestamp": "date"}, inplace=True)
        self.df["date"] = self.df["date"].astype(str)
        self.df = self.df.merge(pool_day_data, on="date", how="left")
        self.df["sp"] = self.df["sqrtPrice"].apply(lambda x: mpz(x)) / Q96
        self.df["new_sp"] = self.df[['sp', 'sb']].min(axis=1)
        self.df["sp_for_calculation"] = self.df[['new_sp', 'sa']].max(axis=1)
        self.df["amount0"] = self.df["net_liquidity"] * (self.df["sb"] - self.df["sp_for_calculation"]) / (
                self.df["sb"] * self.df["sp_for_calculation"])
        self.df["amount1"] = self.df["net_liquidity"] * (self.df["sp_for_calculation"] - self.df["sa"])

        self.df["amount0"] = self.df["amount0"] / (10 ** self.decimal0)
        self.df["amount1"] = self.df["amount1"] / (10 ** self.decimal1)

        self.df.drop(
            columns=[
                "sa",
                "sb",
                "sp",
                "new_sp",
                "sp_for_calculation"
            ],
            inplace=True,
        )
        self.df.to_pickle(os.path.join(self.pickle_path, f"balance_day_datas_{self.pool_address}.pkl"))

    def fee_calculation(self):
        logging.info("Calculating fees...")
        # need pool open high low close data
        pool_close_data = pd.read_csv(
            os.path.join(self.data_path, 'pool_data', "pool_close_data_{}.csv".format(self.pool_address))
        )
        temp_df = self.df[
            ["position_id", "date", "tick_upper", "tick_lower", "net_liquidity"]
        ].copy()
        merged: DataFrame = temp_df.merge(pool_close_data, on="date", how="left")
        # noinspection PyTypeChecker
        change_direction_condition: pd.Series = merged["high_tick"] > merged["low_tick"]
        if change_direction_condition.sum() > 10:
            pass
        else:
            merged.rename(columns={"high_tick": "temp1", "low_tick": "temp2"}, inplace=True)
            merged.rename(columns={"temp1": "low_tick", "temp2": "high_tick"}, inplace=True)
        merged["price_range"] = merged["high_tick"] - merged["low_tick"]
        my_ones = np.ones(merged["price_range"].shape)
        condition_0_0 = (
                merged["high_tick"] == merged["low_tick"]
        )  # tick did not change within the day, no range, only
        condition_0_1 = (merged["tick_lower"] <= merged["high_tick"]) & (
                merged["tick_upper"] >= merged["high_tick"]
        )
        condition_tick_do_not_change = condition_0_0 & condition_0_1
        condition_intersection = (merged["tick_lower"] <= merged["high_tick"]) & (
                merged["tick_upper"] >= merged["low_tick"]
        )
        # If sometimes in range, we need to find the intersection of the range and the position
        merged["intersection_range"] = merged[["high_tick", "tick_upper"]].min(
            axis=1
        ) - merged[["tick_lower", "low_tick"]].max(axis=1)
        merged["temp_percentage"] = (
                merged["intersection_range"] / merged["price_range"]
        )

        choices_lst = [my_ones, merged["temp_percentage"]]
        conditions_list = [condition_tick_do_not_change, condition_intersection]
        merged["active_perc"] = np.select(conditions_list, choices_lst, default=0)
        merged["fee0"] = (
                merged["fee0token"] * merged["active_perc"] * merged["net_liquidity"]
        )
        merged["fee1"] = (
                merged["fee1token"] * merged["active_perc"] * merged["net_liquidity"]
        )
        merged = merged[
            [
                "position_id",
                "date",
                "tick_upper",
                "tick_lower",
                "fee0",
                "fee1",
                "high_tick",
                "low_tick",
                "price_range",
                "active_perc",
            ]
        ].copy()
        self.df = self.df.merge(
            merged, on=["position_id", "date", "tick_upper", "tick_lower"], how="left"
        )
        self.df.to_pickle(os.path.join(self.pickle_path, f"fee_calculated_{self.pool_address}.pkl"))

    def accounting_calculation(self):
        self.accounting_table["input"] = 0
        self.accounting_table["output"] = 0
        input_condition = self.accounting_table["action"] == "INCREASE_LIQUIDITY"
        self.accounting_table.loc[input_condition, "input"] = 1
        self.accounting_table.loc[~input_condition, "output"] = 1
        self.accounting_table["amount0_input"] = (
                self.accounting_table["amount0_adjusted"] * self.accounting_table["input"]
        ).abs()
        self.accounting_table["amount1_input"] = (
                self.accounting_table["amount1_adjusted"] * self.accounting_table["input"]
        ).abs()
        self.accounting_table["amount0_output"] = (
                self.accounting_table["amount0_adjusted"] * self.accounting_table["output"]
        ).abs()
        self.accounting_table["amount1_output"] = (
                self.accounting_table["amount1_adjusted"] * self.accounting_table["output"]
        ).abs()
        self.accounting_table["date"] = self.accounting_table["block_timestamp"].dt.date.astype(str)
        self.accounting_table.sort_values(by=["position_id", "date"], inplace=True)
        accounting_table = self.accounting_table.groupby(["position_id", "date"])[
            ["amount0_input", "amount1_input", "amount0_output", "amount1_output"]].sum().reset_index()
        self.df = self.df.merge(accounting_table, on=["position_id", "date"], how="left")
        self.df["amount0_input"] = self.df["amount0_input"].fillna(0)
        self.df["amount1_input"] = self.df["amount1_input"].fillna(0)
        self.df["amount0_output"] = self.df["amount0_output"].fillna(0)
        self.df["amount1_output"] = self.df["amount1_output"].fillna(0)

        self.df.to_pickle(os.path.join(self.pickle_path, f"done_accounting_day_datas_{self.pool_address}.pkl"))

    def main_process(self):
        self.preprocess()
        self.balance_calculation()
        self.fee_calculation()
        self.accounting_calculation()
        print("Done!")

    @staticmethod
    def read_file(data_path: str, parse_dates: object = None, dtypes: object = None) -> pd.DataFrame:
        if parse_dates is None:
            parse_dates = ["block_timestamp"]

        return pd.read_csv(
            data_path,
            parse_dates=parse_dates,
            low_memory=False,
            dtype=dtypes,
        )

    @staticmethod
    def sort_df(sort_by_lst: list, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by=sort_by_lst, ignore_index=True)

    @staticmethod
    def curr_time() -> str:
        return f"{datetime.datetime.now():%Y%m%d-%H%M%S.%f}"

    @staticmethod
    def get_parent(path=os.getcwd(), levels=1):
        common = path
        # Using for loop for getting starting point required for
        for i in range(levels + 1):
            # Starting point
            common = os.path.dirname(common)
        return os.path.abspath(common)


if __name__ == "__main__":
    # POOL_ADDR = ['0x84383fb05f610222430f69727aa638f8fdbf5cc1']
    # Issue with this: some days there are no trades; so have to ignore this for now.
    for pool_addr in POOL_ADDR:
        lp_data = LPData(pool_addr, debug=False, parallelize_processing=False)
        lp_data.main_process()
        lp_data.data.to_pickle(f"data_{pool_addr}_0626_no_short.pkl")
