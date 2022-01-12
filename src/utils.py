"""
    Some handy functions for pytroch model training ...
"""
import torch
import numpy as np
import os
import errno
import sys
import math
import pandas as pd
import random
from .evaluation import Evaluator


def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot


def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_evaluations_final(run_mf, test):
    metrics = {"ndcg_cut_10"}
    eval_obj = Evaluator(metrics)
    indiv_res = eval_obj.evaluate(run_mf, test)
    overall_res = eval_obj.show_all()
    return overall_res, indiv_res


def read_qrel_file(qrel_file, my_id_bank):
    qrel = {}
    df_qrel = pd.read_csv(qrel_file, sep="\t")
    for row in df_qrel.itertuples():
        cur_user_qrel = qrel.get(str(my_id_bank.query_user_index(row.userId)), {})
        cur_user_qrel[str(my_id_bank.query_item_index(row.itemId))] = int(row.rating)
        qrel[str(my_id_bank.query_user_index(row.userId))] = cur_user_qrel
    return qrel


def write_run_file(rankings, model_output_run):
    if not os.path.exists(os.path.dirname(model_output_run)):
        try:
            os.makedirs(os.path.dirname(model_output_run))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(model_output_run, "w") as f:
        f.write(f"userId\titemId\tscore\n")
        for userid, cranks in rankings.items():
            for itemid, score in cranks.items():
                f.write(f"{userid}\t{itemid}\t{score}\n")


def use_optimizer(network, params):
    if params["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, network.parameters()),
            lr=params["sgd_lr"],
            momentum=params["sgd_momentum"],
            weight_decay=params["l2_regularization"],
        )

    elif params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, network.parameters()),
            lr=params["adam_lr"],
            weight_decay=params["l2_regularization"],
        )
        
    elif params["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, network.parameters()),
            lr=params["rmsprop_lr"],
            alpha=params["rmsprop_alpha"],
            momentum=params["rmsprop_momentum"],
        )
    return optimizer


def get_run_mf(rec_list, unq_users, my_id_bank):
    ranking = {}
    for cuser in unq_users:
        cuser_ids = np.where(rec_list[:, 0] == cuser)[0]
        user_ratings = rec_list[cuser_ids, :]
        user_ratings = user_ratings[user_ratings[:, 2].argsort()[::-1]]
        ranking[cuser] = user_ratings

    run_mf = {}
    for k, v in ranking.items():
        cur_rank = {}
        for item in v:
            citem_ind = int(item[1])
            citem_id = my_id_bank.query_item_id(citem_ind)
            cur_rank[citem_id] = item[2]
        cuser_ind = int(k)
        cuser_id = my_id_bank.query_user_id(cuser_ind)
        run_mf[cuser_id] = cur_rank
    return run_mf


def get_run_mf_index(rec_list, unq_users):
    ranking = {}
    for cuser in unq_users:
        cuser_ids = np.where(rec_list[:, 0] == cuser)[0]
        user_ratings = rec_list[cuser_ids, :]
        user_ratings = user_ratings[user_ratings[:, 2].argsort()[::-1]]
        ranking[str(cuser)] = dict(zip(user_ratings[:, 1].astype('int').astype('str'), user_ratings[:, 2].astype('float')))

    return ranking
