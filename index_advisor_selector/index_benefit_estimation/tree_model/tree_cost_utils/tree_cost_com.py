# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: tree_cost_com
# @Author: Wei Zhou
# @Time: 2022/8/17 15:45

import numpy as np

import logging
import argparse

tf_step = 0
summary_writer = None


def get_parser():
    parser = argparse.ArgumentParser(
        description="A MODEL for cost refinement.")

    parser.add_argument("--exp_id", type=str, default="new_exp_opt")
    parser.add_argument("--num_rounds", type=int, default=5000)
    parser.add_argument("--model_type", type=str, default="LightGBM",
                        choices=["XGBoost", "LightGBM", "RandomForest"])

    parser.add_argument("--plan_num", type=int, default=1)
    parser.add_argument("--feat_chan", type=str, default="cost_row",
                        choices=["cost", "row", "cost_row"])
    parser.add_argument("--label_type", type=str, default="raw",
                        choices=["ratio", "diff_ratio", "cla", "raw"])

    parser.add_argument("--feat_conn", type=str, default="concat")
    parser.add_argument("--task_type", type=str, default="reg")
    parser.add_argument("--cla_min_ratio", type=float, default=0.2)

    # src, tgt

    # 1. tpch
    parser.add_argument("--train_data_load", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/tree_model/data/tpch/tree_tpch_cost_data_tgt_train.json")
    parser.add_argument("--valid_data_load", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/tree_model/data/tpch/tree_tpch_cost_data_tgt_valid.json")
    parser.add_argument("--test_data_load", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/tree_model/data/tpch/tree_tpch_cost_data_tgt_test.json")

    # parser.add_argument("--model_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpch_tgt_round5k/model/reg_xgb_cost.xgb.model")
    # parser.add_argument("--scale_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpch_tgt_round5k/data/train_scale_data.pt")

    parser.add_argument("--model_load", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_lgb_tpch_tgt_round5k/model/reg_lgb_cost.lgb.model")
    parser.add_argument("--scale_load", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_lgb_tpch_tgt_round5k/data/train_scale_data.pt")

    # 2. tpcds
    # parser.add_argument("--train_data_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/tree_model/data/tpcds/tree_tpcds_cost_data_tgt_train.json")
    # parser.add_argument("--valid_data_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/tree_model/data/tpcds/tree_tpcds_cost_data_tgt_valid.json")
    # parser.add_argument("--test_data_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/tree_model/data/tpcds/tree_tpcds_cost_data_tgt_test.json")
    #
    # parser.add_argument("--model_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpcds_tgt_round5k/model/reg_xgb_cost.xgb.model")
    # parser.add_argument("--scale_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpcds_tgt_round5k/data/train_scale_data.pt")

    # 3. job
    # parser.add_argument("--train_data_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/tree_model/data/job/tree_job_cost_data_tgt_train.json")
    # parser.add_argument("--valid_data_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/tree_model/data/job/tree_job_cost_data_tgt_valid.json")
    # parser.add_argument("--test_data_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/tree_model/data/job/tree_job_cost_data_tgt_test.json")

    # parser.add_argument("--model_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_job_tgt_round5k/model/reg_xgb_cost.xgb.model")
    # parser.add_argument("--scale_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_job_tgt_round5k/data/train_scale_data.pt")

    parser.add_argument("--data_save", type=str,
                        default="./cost_exp_res/{}/data/{}_data.pt")

    parser.add_argument("--model_save_gap", type=int, default=1)
    parser.add_argument("--model_save", type=str,
                        default="./cost_exp_res/{}/model/cost_{}.pt")
    parser.add_argument("--model_save_dir", type=str,
                        default="./cost_exp_res/{}/model/{}")

    # : 1. common setting.
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--logdir", type=str,
                        default="./cost_exp_res/{}/logdir/")
    parser.add_argument("--runlog", type=str,
                        default="./cost_exp_res/{}/exp_runtime.log")

    # : hyper parameter.
    parser.add_argument("--num_round", type=int, default=5000)

    return parser


def set_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # log to file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def add_summary_value(key, value, step=None):
    if step is None:
        summary_writer.add_scalar(key, value, tf_step)
    else:
        summary_writer.add_scalar(key, value, step)


class Normalizer:
    def __init__(self, mini=None, maxi=None):
        self.mini = mini
        self.maxi = maxi

    def normalize_labels(self, labels, reset_min_max=False):
        # added 0.001 for numerical stability
        labels = np.array([np.log(float(l) + 0.001) for l in labels])
        if self.mini is None or reset_min_max:
            self.mini = labels.min()
            print("min log(label): {}".format(self.mini))
        if self.maxi is None or reset_min_max:
            self.maxi = labels.max()
            print("max log(label): {}".format(self.maxi))

        labels_norm = (labels - self.mini) / (self.maxi - self.mini)

        # Threshold labels <-- but why...
        labels_norm = np.minimum(labels_norm, 1)
        labels_norm = np.maximum(labels_norm, 0.001)

        return labels_norm

    def unnormalize_labels(self, labels_norm):
        labels_norm = np.array(labels_norm, dtype=np.float32)
        labels = (labels_norm * (self.maxi - self.mini)) + self.mini

        #         return np.array(np.round(np.exp(labels) - 0.001), dtype=np.int64)
        return np.array(np.exp(labels) - 0.001)
