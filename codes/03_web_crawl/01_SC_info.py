# Collect the large SC information
# definition of large: many users
# need the SC lists

# -*- coding: utf-8 -*-
import re
import threading

import requests
from bs4 import BeautifulSoup
import json
from queue import Queue
import pandas as pd
import numpy as np
import helpers
from config import cfg


class TokenDetail:
    def __init__(self):
        self.headers = cfg.headers
        self.factory_df_file = cfg.factory_df_file
        self.factory_df = None
        self.info_df_file = cfg.info_df_file
        self.info_df = None
        self.erc20_token_file = cfg.erc20_token_file
        self.erc20_token_df = None
        self.all_token_addr = None
        self.crawl_queue = Queue()
        self.missing_list = []
        self.html_list_link = []
        self.result = pd.DataFrame()
        self.use_proxies = cfg.use_proxies
        self.proxies = cfg.proxies
        self.BASE_URL = "https://etherscan.io/token/"
        self.new_result_df_file = cfg.new_coin_info_df_file
        self.new_result_df = None
        self.prepare_data()

    def prepare_data(self):
        factory_df_cols = ["tokenA", "tokenB"]
        self.factory_df = (
            pd.read_csv(self.factory_df_file)
            .pipe(helpers.keep_cols, factory_df_cols)
            .pipe(helpers.fix_address_df, *factory_df_cols)
        )
        self.all_token_addr = pd.concat(
            [self.factory_df.tokenA, self.factory_df.tokenB], ignore_index=True
        ).unique()
        self.erc20_token_df = pd.read_csv(self.erc20_token_file).pipe(
            helpers.fix_address_df, "contract_address"
        )
        # First compare using Dune's table
        self.missing_list = self.all_token_addr[
            ~np.isin(self.all_token_addr, self.erc20_token_df.contract_address.unique())
        ]

        info_df_cols = [
            "contract_addr",
            "decimals",
            "holders",
            "max_total_supply",
            "name",
            "symbol",
        ]
        self.info_df = (
            pd.read_csv(self.info_df_file)
            .pipe(helpers.keep_cols, info_df_cols)
            .pipe(helpers.col_to_lower, "contract_addr")
        )
        # Now compare using our table
        self.missing_list = self.missing_list[
            ~np.isin(self.missing_list, self.info_df.contract_addr.unique())
        ]

        self.html_list_link = [self.BASE_URL + addr for addr in self.missing_list]

    def recrawling_processor(self):
        self.new_result_df = pd.read_csv(self.new_result_df_file)
        correct_last_time = self.new_result_df.dropna()["contract_addr"].unique()
        self.missing_list = self.missing_list[
            ~np.isin(self.missing_list, correct_last_time)
        ]
        self.html_list_link = [self.BASE_URL + addr for addr in self.missing_list]


class SCInfo:
    def __init__(self):
        self.headers = cfg.headers
        self.factory_df_file = cfg.factory_df_file
        self.factory_df = None
        self.info_df_file = cfg.info_df_file
        self.info_df = None
        self.erc20_token_file = cfg.erc20_token_file
        self.erc20_token_df = None
        self.all_token_addr = None
        self.crawl_queue = Queue()
        self.missing_list = []
        self.html_list_link = []
        self.result = pd.DataFrame()
        self.use_proxies = cfg.use_proxies
        self.proxies = cfg.proxies
        self.BASE_URL = "https://etherscan.io/token/"
        self.new_result_df_file = cfg.new_coin_info_df_file
        self.new_result_df = None
        self.prepare_data()

    def prepare_data(self):
        pass

    def recrawling_processor(self):
        pass


class CrawlerThread(threading.Thread):

    def __init__(self, thread_id, queue):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.queue = queue

    def run(self):
        # print("Enable Crawler Thread", self.thread_id)
        self.crawl_spider()
        # print("Exit Crawler Thread", self.thread_id)

    def crawl_spider(self):
        while True:
            if self.queue.empty():
                break
            else:
                page = self.queue.get()
                # print("Current thread:", self.thread_id, ", working on", page)
                try:
                    content = requests.get(url=page, headers=cfg.headers, proxies=cfg.proxies, timeout=60)
                    data_queue.put(content)
                except Exception as e:
                    print('URL error on', page, e)


class ParserThread(threading.Thread):
    def __init__(self, thread_id, queue, file):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.queue = queue
        self.file = file

    def run(self):
        # print("Enable Parse Thread", self.thread_id)
        while not flag:
            try:
                item = self.queue.get(False)
                if not item:
                    pass
                self.parse_data(item)
                self.queue.task_done()
            except Exception as e:
                pass
        # print("Exit Parse Thread", self.thread_id)

    def parse_data(self, item):
        soup = BeautifulSoup(item.text, 'html.parser')
        url = item.url
        res = {"contract_addr": re.findall(r'0x.*', url)[0]}
        try:
            res["name"] = soup.find_all("span", class_="text-secondary small")[0].text
        except Exception as e:
            print("Cannot find name for " + url + " " + str(e))
        try:
            max_total_supply = soup.find_all("div", class_="col-md-8 font-weight-medium")[0].text.split()
            res["max_total_supply"] = max_total_supply[0]
            res["symbol"] = max_total_supply[1]
        except Exception as e:
            print("Cannot find max total supply and symbol for " + url + " " + str(e))
        try:
            res["decimals"] = soup.find(text="Decimals:").find_next('div').text.replace('\n', '')
        except Exception as e:
            print("Cannot find decimals for " + url + " " + str(e))
        try:
            res['holders'] = soup.find_all("div", class_="mr-3")[0].text.split()[0]
        except Exception as e:
            print("Cannot find holders for " + url + " " + str(e))
        json.dump(res, fp=self.file, ensure_ascii=False)
        self.file.write(",\n")


if __name__ == "__main__":
    td = TokenDetail()
    # td.recrawling_processor()
    data_queue = Queue()
    pages_queue = Queue()
    output = open('result_append0607.json', 'a', encoding='utf-8')
    output.write("[\n")
    flag = False
    for link in td.html_list_link:
        pages_queue.put(link)
    crawl_threads = []
    crawl_name_list = [i for i in range(3)]  # 30 crawler threads
    for thread_id in crawl_name_list:
        thread = CrawlerThread(thread_id, pages_queue)  # enable crawler threads
        thread.start()  # start thread
        crawl_threads.append(thread)

    parse_thread = []
    parser_name_list = [i for i in range(3)]
    for thread_id in parser_name_list:  #
        thread = ParserThread(thread_id, data_queue, output)
        thread.start()  # start thread
        parse_thread.append(thread)

    while not pages_queue.empty():
        pass

    for t in crawl_threads:
        t.join()

    while not data_queue.empty():
        pass
    flag = True
    for t in parse_thread:
        t.join()
    output.write("]")
    output.close()
    print("Exit main;)")
