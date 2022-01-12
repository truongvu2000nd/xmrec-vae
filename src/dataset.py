from typing import Union

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import (random, 
                          coo_matrix,
                          csr_matrix, 
                          vstack)
from tqdm import tqdm


def load_n_items(tgt_file, run_valid_file, src_files=None):
    id_index_bank = Central_ID_Bank()
    tgt_data = pd.read_csv(tgt_file, sep="\t")
    for item_id in tgt_data["itemId"]:
        id_index_bank.query_item_index(item_id)
    for user_id in tgt_data["userId"]:
        user_index = id_index_bank.query_user_index(user_id)
        id_index_bank.query_user_market(user_index, market=0)

    if src_files is not None:
        market = 1
        for src_file in src_files:
            src_data = pd.read_csv(src_file, sep="\t")
            for item_id in src_data["itemId"]:
                id_index_bank.query_item_index(item_id)
            for user_id in src_data["userId"]:
                user_index = id_index_bank.query_user_index(user_id)
                id_index_bank.query_user_market(user_index, market=market)
            market += 1

    run_files = [run_valid_file]
    for run_file in run_files:
        with open(run_file, "r") as f:
            for line in f:
                linetoks = line.split("\t")
                item_ids = linetoks[1].strip().split(",")
                for cindex, item_id in enumerate(item_ids):
                    id_index_bank.query_item_index(item_id)
    return id_index_bank


class Central_ID_Bank(object):
    """
    Central for all cross-market user and items original id and their corrosponding index values
    """

    def __init__(self):
        self.user_id_index = {}
        self.item_id_index = {}
        self.user_market = {}
        self.last_user_index = 0
        self.last_item_index = 0

    def query_user_index(self, user_id):
        if user_id not in self.user_id_index:
            self.user_id_index[user_id] = self.last_user_index
            self.last_user_index += 1
        return self.user_id_index[user_id]

    def query_item_index(self, item_id):
        if item_id not in self.item_id_index:
            self.item_id_index[item_id] = self.last_item_index
            self.last_item_index += 1
        return self.item_id_index[item_id]

    def query_user_id(self, user_index):
        user_index_id = {v: k for k, v in self.user_id_index.items()}
        if user_index in user_index_id:
            return user_index_id[user_index]
        else:
            print(f"USER index {user_index} is not valid!")
            return "xxxxx"

    def query_item_id(self, item_index):
        item_index_id = {v: k for k, v in self.item_id_index.items()}
        if item_index in item_index_id:
            return item_index_id[item_index]
        else:
            print(f"ITEM index {item_index} is not valid!")
            return "yyyyy"

    def query_user_market(self, user_index, market=0):
        if user_index not in self.user_market:
            self.user_market[user_index] = market
        return self.user_market[user_index]


class MarketTrainDataset(Dataset):
    def __init__(self, tgt_file, id_index_bank, src_files=None) -> None:
        self.id_index_bank = id_index_bank
        self.ratings = pd.read_csv(tgt_file, sep="\t")
        
        if src_files is not None:
            for src_file in src_files:
                src_ratings = pd.read_csv(src_file, sep="\t")
                self.ratings = self.ratings.append(src_ratings)

        # replace ids with corrosponding index for both users and items
        # self.market = self.ratings["userId"].apply(
        #     lambda x: self.id_index_bank.query_user_index(x)
        # ).values

        self.ratings["userId"] = self.ratings["userId"].apply(
            lambda x: self.id_index_bank.query_user_index(x)
        )
        self.ratings["itemId"] = self.ratings["itemId"].apply(
            lambda x: self.id_index_bank.query_item_index(x)
        )
        self.n_items = id_index_bank.last_item_index
        self.n_users = id_index_bank.last_user_index
        self.market = np.array([self.id_index_bank.query_user_market(x) for x in range(self.n_users)]).reshape(-1, 1)

        rows, cols = self.ratings["userId"], self.ratings["itemId"]
        self.data = csr_matrix((np.ones_like(rows),
                        (rows, cols)), dtype='float32',
                        shape=(self.n_users, self.n_items)).toarray()

    def __getitem__(self, index:int):
        return (self.data[index], self.market[index])

    def __len__(self):
        return self.n_users


class MarketRunDataset(Dataset):
    def __init__(self, train_data, run_file, id_index_bank) -> None:
        self.id_index_bank = id_index_bank
        self.run_file = run_file
        self.train_data = train_data

        users, items = [], []
        with open(run_file, "r") as f:
            for line in f:
                linetoks = line.split("\t")
                user_id = linetoks[0]
                item_ids = linetoks[1].strip().split(",")
                items_line = []
                for cindex, item_id in enumerate(item_ids):
                    if item_id not in self.id_index_bank.item_id_index:
                        print(item_id)
                        exit(0)
                    items_line.append(self.id_index_bank.query_item_index(item_id))
                users.append(self.id_index_bank.query_user_index(user_id))
                items.append(items_line)

        self.users = np.array(users)
        self.items = np.array(items)

    def __getitem__(self, index):
        return (*self.train_data[self.users[index]], self.users[index], self.items[index])

    def __len__(self):
        return self.users.shape[0]


def sparse_coo_to_tensor(coo: coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)


def train_batch_collate(batch):
    x, y = zip(*batch)
    x = torch.FloatTensor(vstack(x).toarray())
    print(y)
    y = torch.LongTensor(vstack(y))
    return batch
    

def run_batch_collate(batch: list): 
    data_batch, market_class, users_batch, items_batch = zip(*batch)
    data_batch = torch.FloatTensor(vstack(data_batch).toarray())
    market_class = torch.LongTensor(vstack(market_class))

    users_batch = torch.LongTensor(np.array(users_batch))
    items_batch = torch.LongTensor(np.array(items_batch))
    return data_batch, market_class, users_batch, items_batch


if __name__ == '__main__':
    id_index_bank = Central_ID_Bank()
    train_data = MarketTrainDataset("../DATA/t1/train_5core.tsv", id_index_bank)
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True, collate_fn=train_batch_collate)
    for i in tqdm(range(200)):
        for batch in train_loader:
            pass

    # run_data = MarketRunDataset(train_data, "..//DATA/t1/valid_run.tsv", id_index_bank)
    # run_loader = DataLoader(run_data, batch_size=2, shuffle=False, collate_fn=run_batch_collate)
    # x, y = next(iter(run_loader))

    # import torch.nn.functional as F

    # x = F.normalize(x, dim=1)
    # print(x)
