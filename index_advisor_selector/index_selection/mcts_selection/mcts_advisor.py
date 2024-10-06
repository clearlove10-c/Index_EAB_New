# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: mcts_advisor
# @Author: Wei Zhou
# @Time: 2023/7/21 17:31

import os
import copy
import json
import time
import logging
import configparser
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from index_advisor_selector.index_selection.heu_selection.heu_utils.heu_com import get_utilized_indexes, get_columns_from_schema
from index_advisor_selector.index_selection.heu_selection.heu_utils.candidate_generation import candidates_per_query, \
    syntactically_relevant_indexes_dqn_rule, \
    syntactically_relevant_indexes_openGauss

from index_advisor_selector.index_selection.mcts_selection.mcts_model import State, Node, MCTS

from index_advisor_selector.index_selection.mcts_selection.mcts_utils.cost_evaluation import CostEvaluation
from index_advisor_selector.index_selection.mcts_selection.mcts_utils.postgres_dbms import PostgresDatabaseConnector
from index_advisor_selector.index_selection.mcts_selection.mcts_utils.mcts_com import syntactically_relevant_indexes, get_parser, pre_work, plot_report, mb_to_b
from index_advisor_selector.index_selection.mcts_selection.mcts_utils.mcts_workload import Column, Table, Index, Workload


class MCTSAdvisor:
    def __init__(self, database_connector, parameters, process=False):
        self.did_run = False

        self.parameters = parameters

        self.database_connector = database_connector
        self.database_connector.drop_indexes()
        self.cost_evaluation = CostEvaluation(database_connector)

        self.mcts_tree = None

        # : newly added. for process visualization.
        self.process = process
        self.step = {"selected": list()}

    def calculate_best_indexes(self, workload, overhead=False):
        assert self.did_run is False, "Selection algorithm can only run once."
        self.did_run = True

        # cost_estimations 表示 db_connector 调用数据库代价估计器的总次数（在 PG 中是调用 explain <SQL>）
        # cost_estimation_duration 表示 db_connector 调用数据库代价估计器的总时间
        estimation_num_bef = self.database_connector.cost_estimations
        estimation_duration_bef = self.database_connector.cost_estimation_duration

        # simulated_indexes 表示 db_connector 调用虚拟索引创建方法的总次数
        # （在 PG 中是调用 select * from hypopg_create_index('create index on table_name (column1, column2)')）
        simulation_num_bef = self.database_connector.simulated_indexes
        simulation_duration_bef = self.database_connector.index_simulation_duration

        time_start = time.time()
        indexes = self._calculate_best_indexes(workload)
        time_end = time.time()

        estimation_duration_aft = self.database_connector.cost_estimation_duration
        estimation_num_aft = self.database_connector.cost_estimations

        simulation_num_aft = self.database_connector.simulated_indexes
        simulation_duration_aft = self.database_connector.index_simulation_duration

        # : newly added. for selection runtime
        cache_hits = self.cost_evaluation.cache_hits
        cost_requests = self.cost_evaluation.cost_requests

        self.cost_evaluation.complete_cost_estimation()

        # : newly added.
        if self.process:
            if overhead:
                return indexes, {"step": self.step, "cache_hits": cache_hits,
                                 "cost_requests": cost_requests, "time_duration": time_end - time_start,
                                 "estimation_num": estimation_num_aft - estimation_num_bef,
                                 "estimation_duration": estimation_duration_aft - estimation_duration_bef,
                                 "simulation_num": simulation_num_aft - simulation_num_bef,
                                 "simulation_duration": simulation_duration_aft - simulation_duration_bef}
            else:
                return indexes, {"step": self.step, "cache_hits": cache_hits, "cost_requests": cost_requests}
        elif overhead:
            return indexes, {"cache_hits": cache_hits, "cost_requests": cost_requests,
                             "time_duration": time_end - time_start,
                             "estimation_num": estimation_num_aft - estimation_num_bef,
                             "estimation_duration": estimation_duration_aft - estimation_duration_bef,
                             "simulation_num": simulation_num_aft - simulation_num_bef,
                             "simulation_duration": simulation_duration_aft - simulation_duration_bef}
        else:
            return indexes, ""

    def _calculate_best_indexes(self, workload):
        """
        :param workload:
        :return:
        """
        logging.info("Calculating the best indexes by MCTS")

        # 1. synthesize the potential index candidate set.
        # potential_index = syntactically_relevant_indexes(workload, self.parameters.max_index_width)

        # Generate syntactically relevant candidates
        # (0917): newly added.
        # 这里为每条查询分别生成了候选索引，生成的 potential_index 每一个元素是一个列表，表示每条查询的候选索引
        if self.parameters.cand_gen is None or self.parameters.cand_gen == "permutation":
            potential_index = candidates_per_query(
                Workload(workload),
                self.parameters.max_index_width,
                candidate_generator=syntactically_relevant_indexes,
            )

        elif self.parameters.cand_gen == "dqn_rule":
            db_conf = configparser.ConfigParser()
            db_conf.read(self.parameters.db_file)

            _, columns = get_columns_from_schema(self.parameters.schema_file)

            potential_index = [syntactically_relevant_indexes_dqn_rule(db_conf, [query.text], columns,
                                                                       self.parameters.max_index_width) for query in
                               workload]

        elif self.parameters.cand_gen == "openGauss":
            db_conf = configparser.ConfigParser()
            db_conf.read(self.parameters.db_file)

            _, columns = get_columns_from_schema(self.parameters.schema_file)

            potential_index = [syntactically_relevant_indexes_openGauss(db_conf, [query.text], columns,
                                                                        self.parameters.max_index_width) for query in
                               workload]

        # (0918): newly modified.
        if self.parameters.cand_gen is None or self.parameters.is_utilized:
            # Obtain the utilized indexes considering every single query
            potential_index, _ = get_utilized_indexes(Workload(workload), potential_index, self.cost_evaluation)
        else:
            cand_set = list()
            for cand in potential_index:
                cand_set.extend(cand)
            candidates = set(cand_set)

            potential_index = copy.deepcopy(candidates)
            _ = self.cost_evaluation.calculate_cost(
                Workload(workload), potential_index
                , store_size=True  # newly added.
            )

        potential_index = sorted(potential_index)

        # 1. synthesize the potential index candidate set.
        # if not self.parameters.is_utilized:
        #     potential_index = syntactically_relevant_indexes(workload, self.parameters.max_index_width)
        # else:
        #     candidates = candidates_per_query(
        #         Workload(workload),
        #         self.parameters.max_index_width,
        #         candidate_generator=syntactically_relevant_indexes,
        #     )
        #
        #     potential_index, _ = get_utilized_indexes(
        #         Workload(workload), candidates, self.cost_evaluation, True
        #     )
        #
        #     potential_index = sorted(potential_index)

        # _ = self.cost_evaluation.calculate_cost(Workload(workload), potential_index, store_size=True)

        # (0805): newly added. for `storage`.
        if self.parameters.constraint == "storage":
            # _ = self.cost_evaluation.calculate_cost(Workload(workload), potential_index, store_size=True)

            potential_index_filter = list()
            for index in potential_index:
                if index.estimated_size <= mb_to_b(self.parameters.storage):
                    potential_index_filter.append(index)
            potential_index = copy.deepcopy(potential_index_filter)

            # potential_index_pre = list()
            # for index in potential_index:
            #     tbl, col = index.split("#")
            #     col = [Column(c, Table(tbl)) for c in col.split(",")]
            #     potential_index_pre.append(Index(col))
            # _ = self.cost_evaluation.calculate_cost(Workload(workload), potential_index_pre, store_size=True)

            # potential_index_pre_filter = list()
            # for index1, index2 in zip(potential_index, potential_index_pre):
            #     if index2.estimated_size <= mb_to_b(self.parameters.storage):
            #         potential_index_pre_filter.append(index1)
            # potential_index = copy.deepcopy(potential_index_pre_filter)

        # 2. index selection based on MCTS.
        current_index = list()

        root = Node(State(current_index, potential_index, self.parameters.constraint,
                          self.parameters.cardinality, self.parameters.storage))
        # (0818): newly added.
        if self.mcts_tree is None:
            self.mcts_tree = MCTS(self.parameters, workload, potential_index,
                                  self.database_connector, self.cost_evaluation, self.process)
        final_conf, final_reward = self.mcts_tree.mcts_search(self.parameters.budget, root)

        # (0818): newly added.
        save_dir = os.path.dirname(self.parameters.log_file.format(self.parameters.exp_id))
        # plot_report(save_dir, self.mcts_tree.measure)

        if self.process:
            self.step = copy.deepcopy(self.mcts_tree.step)

        return final_conf


if __name__ == "__main__":
    # 解析参数
    parser = get_parser()
    args = parser.parse_args()
    # 初始化数据库连接
    db_conf = configparser.ConfigParser()
    db_conf.read(args.db_file)
    database_connector = PostgresDatabaseConnector(db_conf, autocommit=True)
    # 读取工作负载
    if args.work_file.endswith(".sql"):
        with open(args.work_file, "r") as rf:
            work_list = rf.readlines()
    elif args.work_file.endswith(".json"):
        with open(args.work_file, "r") as rf:
            work_list = json.load(rf)
            work_list = work_list[:10]
    # 预处理工作负载
    workloads = list()
    for workload in work_list:
        workload = pre_work(workload, args.schema_file)
        workloads.append(workload)

    # 计算最佳索引
    for workload in workloads:
        # 初始化蒙特卡洛树索引推荐算法
        advisor = MCTSAdvisor(database_connector, args, args.process)
        # 通过蒙特卡洛树索引推荐算法计算最佳索引
        indexes, sel_infos = advisor.calculate_best_indexes(workload, overhead=args.overhead)
        # 输出最佳索引
        indexes_pre = list()
        for index in indexes:
            index_pre = f"{index.columns[0].table.name}#{','.join([col.name for col in index.columns])}"
            indexes_pre.append(index_pre)
        indexes_pre.sort()
        print(indexes_pre)
        # 输出索引优化成本
        print("--- workload %d---" % workloads.index(workload))
        for sql in workload:
            origin_sql_cost = database_connector.get_ind_cost(sql.text, "", mode="hypo")
            tuned_sql_cost = database_connector.get_ind_cost(sql.text, indexes, mode="hypo")
            print(f"origin_sql_cost: {origin_sql_cost}, tuned_sql_cost: {tuned_sql_cost}")
