from collections import defaultdict
import torch
import numpy as np
import pandas as pd
import dgl
from tqdm import tqdm
import os
import torch.utils.data as dataloader
import scipy.sparse as sp
import random
import pickle
from TimeLogger import log
import math
# import prctl
from collections import Counter
from collections import deque
from collections import defaultdict
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from Config_TKGR import args
import setproctitle

Data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('03_RobustTKG')[0]+'Data')

"""Load raw data"""



"""Add random noise"""
def generate_gaussian_noise():
    return


"""TKG datahandler"""
class DataHandler():
    def __init__(self, data_name=None, device=None, num_e=None, num_r=None) -> None:
        self.data_name = data_name
        self.device = device
        self.num_e = num_e
        self.num_r = num_r


    def load_file(self, file_path, num_r=0, inverse=True):
        with open(file_path, 'r') as fr:
            quadrupleList = []
            times = set()
            for line in fr:
                line_split = line.split()
                s = int(line_split[0])
                o = int(line_split[2])
                r = int(line_split[1])
                T = int(line_split[3])
                quadrupleList.append((s, r, o, T, 1))
                if inverse:
                    quadrupleList.append((o, r+num_r, s, T, 1))
                times.add(T)
        times = list(times)
        times.sort()
        return np.asarray(quadrupleList), np.asarray(times)

    def get_total_number(self, file_path):
        with open(file_path, 'r') as fr:
            for line in fr:
                line_split = line.split()
                num_e, num_r =  int(line_split[0]), int(line_split[1])
                return num_e, num_r

    def get_data_with_time(self, data, tim):
        facts = [(fact[0], fact[1], fact[2], fact[3], fact[4]) for fact in data if fact[3] == tim]
        return np.array(facts)

    """Below there are functions for graph process"""  

    def build_dgl_graph(self, num_e, num_r, facts, weights=None, inverse=False):
        # log('build_dgl_graph 1')
        def comp_deg_norm(g):
            in_deg = g.in_degrees(range(g.number_of_nodes())).float()
            in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
            norm = 1.0 / in_deg
            return norm
        
        def r2e(facts):
            src, rel, dst, T, labels = facts.transpose()
            # get all relations
            uniq_r = np.unique(rel)
            uniq_r = np.concatenate((uniq_r, uniq_r+self.num_r))
            # generate r2e
            r_to_e = defaultdict(set)
            for j, (src, rel, dst, T, labels) in enumerate(facts):
                r_to_e[rel].add(src)
                r_to_e[rel].add(dst)
                r_to_e[rel+self.num_r].add(src)
                r_to_e[rel+self.num_r].add(dst)
            r_len = []
            e_idx = []
            idx = 0
            for r in uniq_r:
                r_len.append((idx,idx+len(r_to_e[r]))) #  [(idx_start, idx_end), ...]
                e_idx.extend(list(r_to_e[r]))
                idx += len(r_to_e[r])
            return uniq_r, r_len, e_idx

        src, rel, dst, T, labels = facts.transpose()
        # if inverse:  
        #     src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        #     rel = np.concatenate((rel, rel + num_r))  #  Add inverse facts
        g = dgl.graph((src, dst),num_nodes= num_e)
        # g.add_nodes(num_e)
        # g.add_edges(src, dst)
        # log('build_dgl_graph 2')
        norm = comp_deg_norm(g)
        node_id = torch.arange(0, num_e, dtype=torch.long).view(-1, 1)
        g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
        g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        g.edata['type'] = torch.LongTensor(rel)
        
        if weights is not None:
            g.edata['weight'] = torch.cat((weights, weights))
        
        uniq_r, r_len, r_to_eList = r2e(facts)
        g.uniq_r = uniq_r
        g.r_to_eList = r_to_eList
        g.r_len = r_len
        g.r_to_eList = torch.from_numpy(np.array(r_to_eList))
        # log('build_dgl_graph: g have {} facts'.format(src.shape[0]))
        return g

    def build_adj_matrix(self, num_e, num_r, facts):
        src, rel, dst, T, labels = facts.transpose()
        data = np.ones(len(src) * 2 + num_e)  # 每个边都需要存两次 (src, dst) 和 (dst, src)，加上对角线元素
        row = np.concatenate((src, dst, np.arange(num_e)))
        col = np.concatenate((dst, src, np.arange(num_e)))
        coo_mat = sp.coo_matrix((data, (row, col)), shape=(num_e, num_e))
        # # delete replicated edges 
        # src, rel, dst = triples.transpose()
        # edges = set()
        # for s, d in zip(src, dst):
        #     if s != d:
        #         edges.add((s, d))
        #         edges.add((d, s))
        # unique_src, unique_dst = zip(*edges)
        # data = np.ones(len(unique_src) + num_e)
        # row = np.concatenate((unique_src, np.arange(num_e)))
        # col = np.concatenate((unique_dst, np.arange(num_e)))
        return coo_mat


    def corrupt_data(self, data: np.array, ratio=0.0, strategy=None, data_occur=None):
        # log('corrupt_data')
        def r_to_e_cooccur(facts):
            co_occur_head = defaultdict(set)
            co_occur_tail = defaultdict(set)
            for j, (src, rel, dst, T, label) in enumerate(facts):
                co_occur_head[rel].add(src)
                co_occur_tail[rel].add(dst)
                co_occur_head[rel+self.num_r].add(src)
                co_occur_tail[rel+self.num_r].add(dst)
            return co_occur_head, co_occur_tail        
        
        neg_count = 0
        neg_amount = int(len(data) * ratio)
        # replace_idx_list = random.sample(range(data.shape[0]), neg_amount) # 不放回采样
        # log('Sample without replacement')

        replace_idx_list = random.choices(range(data.shape[0]), k=neg_amount) # 放回采样
        # log('Sample with replacement')

        T = data[0,:][3]
        data_set_visited = {tuple(row[0:3]) for row in data}
        data_pos_neg = [(item[0], item[1], item[2], T, 1) for item in data_set_visited]
        
        '''替换头、尾实体，随机替换'''
        if strategy == 0:
            assert data_occur is None, 'Error: strategy not compatible with data_occur'
            # log('Inject random noise with ratio {:.2f}'.format(ratio))
            for idx in replace_idx_list:
                item = list(data[idx, :])
                for position in random.sample([0,2], 2): # 随机替换头实体或者尾实体
                    candidate_list = list(range(self.num_e))
                    random.shuffle(candidate_list)
                    for entity_neg in candidate_list:
                        item[position] = entity_neg
                        if (item[0], item[1], item[2]) not in data_set_visited:
                            data_pos_neg.append((item[0], item[1], item[2], T, 0))
                            data_set_visited.add((item[0], item[1], item[2]))
                            neg_count+=1
                            success = True
                            break
                    if success:
                        break
            
        '''替换头、尾实体，要求替换的依据是和关系有过共现'''
        if (strategy == 1) or (strategy == 2):
            # log('Inject co-occur noise with ratio {:2f}'.format(ratio))
            if strategy == 1:
                assert data_occur.shape[0]==data.shape[0], 'Error: data_occur not compatible' # 只考虑同时刻的实体-关系共现
            elif strategy == 2:
                assert data_occur.shape[0]>data.shape[0], 'Error: data_occur not compatible' # '考虑所有时刻的实体-关系共现

            co_occur_head, co_occur_tail = r_to_e_cooccur(data_occur)

            for idx in replace_idx_list:
                item = list(data[idx, :])
                success = False 
                for position in random.sample([0,2], 2): # 随机替换头实体或者尾实体
                    co_occur=co_occur_head if position==0 else co_occur_tail
                    candidate_list = list(co_occur.get(item[1]))
                    random.shuffle(candidate_list)
                    for entity_neg in candidate_list:
                        item[position] = entity_neg
                        if (item[0], item[1], item[2]) not in data_set_visited:
                            data_pos_neg.append((item[0], item[1], item[2], T, 0))
                            data_set_visited.add((item[0], item[1], item[2]))
                            neg_count+=1
                            success = True
                            break
                    if success:
                        break
                        
        if strategy == 3:  # 替换关系，而不是替换实体
            for idx in replace_idx_list:
                item = list(data[idx, :])
                success = False 
                r_candidate_list = list(range(self.num_r*2))
                random.shuffle(r_candidate_list)
                for relation_neg in r_candidate_list:
                    item[1] = relation_neg
                    if (item[0], item[1], item[2]) not in data_set_visited:
                        data_pos_neg.append((item[0], item[1], item[2], T, 0))
                        data_set_visited.add((item[0], item[1], item[2]))
                        neg_count+=1
                        success = True
                        break
                if success:
                    break

        # log("Noise achievement : {} / {}".format(neg_count, neg_amount))
        return np.array(data_pos_neg)


    '''提前处理路径，降低训练和推理的时间'''

    def bfs_bypass_paths(self, g: dgl.graph, triple, L=2):
        (start_node, relation_target, end_node, _, _) = triple
        queue = deque([(start_node, [], [start_node])])
        paths = []

        while queue:
            current_node, current_path, nodes_in_path = queue.popleft()
            if (len(current_path) /2) >= L:
                continue
            
            out_edges = g.out_edges(current_node, form='all')
            if out_edges[0].shape[0]==0:
                continue
            out_edges = list(zip(out_edges[1].tolist(), out_edges[2].tolist()))
            random.shuffle(out_edges)
            for down_node, edge_id in out_edges:
                if down_node in nodes_in_path:
                    continue
                relation = g.edata['type'][edge_id].item()
                if (current_node == start_node) and (relation == relation_target) and (down_node == end_node):
                    continue
                new_path = current_path+ [current_node] + [relation] 
                if down_node == end_node:
                    new_path = new_path + [down_node]  # es, r1, e1, r2, e2, ..., et
                    paths.append(new_path)
                else:
                    queue.append((down_node, new_path, nodes_in_path + [down_node]))
        return paths
    
    def bfs_upstream_paths(self, g: dgl.graph, start_node, L=2, exlude_node = None):
            """ 查找从start_node开始的上游L个步长范围内的所有路径"""
            queue = deque([(start_node, [], [start_node, exlude_node])])
            paths = []

            while queue:
                if len(paths)>=5:
                    break
                current_node, current_path, nodes_in_path = queue.popleft()
                if (len(current_path)/2) >= L:
                    current_path = current_path + [start_node]  #  e2, r2, e1, r1, e_s
                    paths.append(current_path)
                    continue

                in_edges = g.in_edges(current_node, form='all')
                if in_edges[0].shape[0]==0:
                    if (len(current_path)/2) > 0:
                        current_path = current_path + [start_node]  #  e3, r3, ...,e1, r1, e_s
                        paths.append(current_path)
                    continue

                in_edges = list(zip(in_edges[0].tolist(), in_edges[2].tolist()))
                random.shuffle(in_edges)
                for up_node, edge_id in in_edges:
                    if up_node in nodes_in_path:
                        continue
                    rel = g.edata['type'][edge_id].item()
                    new_path = [up_node] + [rel]+ current_path
                    queue.append((up_node, new_path, nodes_in_path + [up_node]))
                
                if not queue:
                    if (len(current_path)/2) > 0:
                        current_path = current_path + [start_node]  #  e3, r3, ...,e1, r1, e_s
                        paths.append(current_path)
            return paths
    
    def PCRA(self, g: dgl.graph, path, initial_resource=1.0, num_e=0):
        resources = {node: initial_resource for node in range(num_e)}
        for i in range(0, len(path), 2): # (es,r1,e1,r2, et)
            current_entity = path[i]
            # previous_entity = path[i - 2]
            predecessors = g.in_edges(current_entity, form='all')[0].cpu().tolist()
            
            resources[current_entity] = sum(resources[pred] / len(g.out_edges(pred, form='all')[1].cpu().tolist()) for pred in predecessors) + resources[current_entity]
        tail_entity = path[-1]
        reliability = resources.get(tail_entity, 0.0)
        
        return reliability
    

    def process_fact(self, args):
        # log('process_fact')
        g, fact = args
        conf_info = dict()
        bypass = self.bfs_bypass_paths(g, fact)
        # s_context = self.bfs_upstream_paths(g, fact[0], 2, fact[1])
        # o_context = self.bfs_upstream_paths(g, fact[1], 2, fact[0])
        conf_info['bypass'] = bypass
        # conf_info['s_context'] = s_context
        # conf_info['o_context'] = o_context

        pcra_bypass = [self.PCRA(g, p, initial_resource=1.0, num_e=self.num_e) for p in bypass]
        conf_info['bypass_PCRA'] = pcra_bypass
        # pcra_s_context = [self.PCRA(g, p, initial_resource=1.0, num_e=self.num_e) for p in s_context]
        # conf_info['s_context_PCRA'] = pcra_s_context
        # pcra_o_context = [self.PCRA(g, p, initial_resource=1.0, num_e=self.num_e) for p in o_context]
        # conf_info['o_context_PCRA'] = pcra_o_context
        
        return (tuple((fact[0], fact[1], fact[2], fact[3], fact[4])), conf_info)
    

    def process_kg_data(self, kg_data, ratio, Noise_strategy, data_occur):
        # log('1')
        if Noise_strategy == 1:
            data_occur = kg_data
        kg_data_pos_neg = self.corrupt_data(kg_data, ratio, Noise_strategy, data_occur)
        # log('2')
        g = self.build_dgl_graph(self.num_e, self.num_r, kg_data_pos_neg, weights=None, inverse=False)
        kg_data_dict = dict()

        with ThreadPoolExecutor(max_workers=20) as executor:
            facts_with_graph = [(g, fact) for fact in kg_data_pos_neg]
            count = 0
            all_count = len(facts_with_graph)
            for result in executor.map(self.process_fact, facts_with_graph):
                fact_tuple, conf_info = result
                kg_data_dict[fact_tuple] = conf_info
                # count+=1
                # log('Finish snapshot: {}/{}'.format(count, all_count), save=False, oneline=False)
            log('Finish one snapshot')

        # facts_with_graph = [(g, fact) for fact in kg_data_pos_neg]
        # count = 0
        # all_count = len(facts_with_graph)
        # for triple_args in facts_with_graph:
        #     fact_tuple, conf_info = self.process_fact(triple_args)
        #     kg_data_dict[fact_tuple] = conf_info
        #     count+=1
        #     log('Finish snapshot: {}/{}'.format(count, all_count), save=False, oneline=True)
        # log('Finish one snapshot')

        return kg_data_dict


    def load_data(self, ratio, Noise_strategy=1):
        self.num_e, self.num_r = self.get_total_number(os.path.join(Data_file_path,self.data_name)+'/stat.txt')
        trnData, trnTime = self.load_file(os.path.join(Data_file_path,self.data_name)+'/train.txt', num_r=self.num_r)
        valData, valTime = self.load_file(os.path.join(Data_file_path,self.data_name)+'/valid.txt', num_r=self.num_r)
        tstData, tstTime = self.load_file(os.path.join(Data_file_path,self.data_name)+'/test.txt', num_r=self.num_r)
        self.allTime = np.concatenate((trnTime, valTime, tstTime)).tolist()

        self.trnList = [self.get_data_with_time(trnData, T_idx) for T_idx in trnTime]
        self.valList = [self.get_data_with_time(valData, T_idx) for T_idx in valTime]
        self.tstList = [self.get_data_with_time(tstData, T_idx) for T_idx in tstTime]

        """多进程和多线程"""
        # if Noise_strategy==0:
        #     data_occur = None
        # elif Noise_strategy==1:
        #     data_occur = None
        if Noise_strategy==2:
            data_occur = np.concatenate((trnData, valData), axis=0)  # 从训练集和验证集上去寻找关系-实体共现
        else:
            data_occur = None

        handler_params = (self.data_name, self.device, self.num_e, self.num_r)
        self.trnListDirt = []
        kg_data_args = [(handler_params, kg_data, ratio, Noise_strategy, data_occur) for kg_data in self.trnList]
        with multiprocessing.Pool(processes=8) as pool:
            for result in pool.map(process_kg_data_wrapper, kg_data_args):
                self.trnListDirt.append(result)

        self.valListDirt = []
        kg_data_args = [(handler_params, kg_data, ratio, Noise_strategy, data_occur) for kg_data in self.valList]
        with multiprocessing.Pool(processes=8) as pool:
            for result in pool.map(process_kg_data_wrapper, kg_data_args):
                self.valListDirt.append(result)

        self.tstListDirt = []
        kg_data_args = [(handler_params, kg_data, ratio, Noise_strategy, data_occur) for kg_data in self.tstList]
        with multiprocessing.Pool(processes=8) as pool:
            for result in pool.map(process_kg_data_wrapper, kg_data_args):
                self.tstListDirt.append(result)

        # kg_data_args = [(handler_params, kg_data, ratio, Noise_strategy, data_occur) for kg_data in self.trnList]
        # all_count = len(kg_data_args)
        # with ThreadPoolExecutor(max_workers=20) as executor:
        #     for result in executor.map(process_kg_data_wrapper, kg_data_args):
        #         self.trnListDirt.append(result)
        #         # count+=1
        #         # log('TKG process: {}/{}'.format(count, all_count), save=False, oneline=False)


        # kg_data_args = [(handler_params, kg_data, ratio, Noise_strategy, data_occur) for kg_data in self.valList]
        # all_count = len(kg_data_args)
        # with ThreadPoolExecutor(max_workers=20) as executor:
        #     for result in executor.map(process_kg_data_wrapper, kg_data_args):
        #         self.valListDirt.append(result)
        #         # count+=1
        #         # log('TKG process: {}/{}'.format(count, all_count), save=False, oneline=False)

        # kg_data_args = [(handler_params, kg_data, ratio, Noise_strategy, data_occur) for kg_data in self.tstList]
        # all_count = len(kg_data_args)
        # with ThreadPoolExecutor(max_workers=20) as executor:
        #     for result in executor.map(process_kg_data_wrapper, kg_data_args):
        #         self.tstListDirt.append(result)
        #         # count+=1
        #         # log('TKG process: {}/{}'.format(count, all_count), save=False, oneline=False)





        # self.trnDglList = [self.build_dgl_graph(self.num_e, self.num_r, facts) for facts in self.trnList]
        # self.valDglList = [self.build_dgl_graph(self.num_e, self.num_r, facts) for facts in self.valList]
        # self.tstDglList = [self.build_dgl_graph(self.num_e, self.num_r, facts) for facts in self.tstList]

        log('')
    
    def calculate_history_freq(self, ):
        num_e, num_r = self.get_total_number(os.path.join(Data_file_path,self.data_name)+'/stat.txt')
        trnData, trnTime = self.load_file(os.path.join(Data_file_path,self.data_name)+'/train.txt', num_r=self.num_r)
        valData, valTime = self.load_file(os.path.join(Data_file_path,self.data_name)+'/valid.txt', num_r=self.num_r)
        tstData, tstTime = self.load_file(os.path.join(Data_file_path,self.data_name)+'/test.txt', num_r=self.num_r)
        all_times = np.concatenate((trnTime, valTime, tstTime)).tolist()
        all_data = np.concatenate((trnData, valData, tstData), axis=0)
        # self.trnList = [self.get_data_with_time(trnData, T_idx) for T_idx in trnTime]
        # self.valList = [self.get_data_with_time(valData, T_idx) for T_idx in valTime]
        # self.tstList = [self.get_data_with_time(tstData, T_idx) for T_idx in tstTime]

        all_data_set = set((s,r,o,t) for s,r,o,t in all_data)
        save_dir_obj = data_dir + '/history_seq'
        if not os.path.exists(save_dir_obj):
            os.makedirs(save_dir_obj)

        for idx, tim in tqdm(enumerate(all_times)):
            test_new_data = np.array([[quad[0], quad[1], quad[2], quad[3], quad[4]] for quad in all_data if quad[3] == tim])
            row = test_new_data[:, 0] * num_r * 2 + test_new_data[:, 1] 
            col = test_new_data[:, 2]
            d = np.ones(len(row))
            tail_seq = sp.csr_matrix((d, (row, col)), shape=(num_e * num_r * 2, num_e)) # 历史矩阵的压缩存储
            sp.save_npz(save_dir_obj + '/h_r_history_seq_{}.npz'.format(idx), tail_seq)
    

    def load_data_cut_relation(self,ratio, Noise_strategy):
        if 'cut' in args.dataset:
            self.data_name = args.dataset.split('cut')[0]
        self.num_e, self.num_r = self.get_total_number(os.path.join(Data_file_path,self.data_name)+'/stat.txt')
        trnData, trnTime = self.load_file(os.path.join(Data_file_path,self.data_name)+'/train.txt', num_r=self.num_r)
        valData, valTime = self.load_file(os.path.join(Data_file_path,self.data_name)+'/valid.txt', num_r=self.num_r)
        tstData, tstTime = self.load_file(os.path.join(Data_file_path,self.data_name)+'/test.txt', num_r=self.num_r)
        if 'cut' in args.dataset:
            self.data_name = args.dataset
        
        self.num_r = 50 # 只提取前50种关系
        trnData =  np.array([(fact[0], fact[1], fact[2], fact[3], fact[4]) for fact in trnData if fact[1] < self.num_r])
        valData =  np.array([(fact[0], fact[1], fact[2], fact[3], fact[4]) for fact in valData if fact[1] < self.num_r])
        tstData =  np.array([(fact[0], fact[1], fact[2], fact[3], fact[4]) for fact in tstData if fact[1] < self.num_r])


        self.allTime = np.concatenate((trnTime, valTime, tstTime)).tolist()

        self.trnList = [self.get_data_with_time(trnData, T_idx) for T_idx in trnTime]
        self.valList = [self.get_data_with_time(valData, T_idx) for T_idx in valTime]
        self.tstList = [self.get_data_with_time(tstData, T_idx) for T_idx in tstTime]

        """多进程和多线程"""
        if Noise_strategy==2:
            data_occur = np.concatenate((trnData, valData), axis=0)  # 从训练集和验证集上去寻找关系-实体共现
        else:
            data_occur = None

        handler_params = (self.data_name, self.device, self.num_e, self.num_r)
        self.trnListDirt = []
        kg_data_args = [(handler_params, kg_data, ratio, Noise_strategy, data_occur) for kg_data in self.trnList]
        with multiprocessing.Pool(processes=8) as pool:
            for result in pool.map(process_kg_data_wrapper, kg_data_args):
                self.trnListDirt.append(result)

        self.valListDirt = []
        kg_data_args = [(handler_params, kg_data, ratio, Noise_strategy, data_occur) for kg_data in self.valList]
        with multiprocessing.Pool(processes=8) as pool:
            for result in pool.map(process_kg_data_wrapper, kg_data_args):
                self.valListDirt.append(result)

        self.tstListDirt = []
        kg_data_args = [(handler_params, kg_data, ratio, Noise_strategy, data_occur) for kg_data in self.tstList]
        with multiprocessing.Pool(processes=8) as pool:
            for result in pool.map(process_kg_data_wrapper, kg_data_args):
                self.tstListDirt.append(result)


    
def process_kg_data_wrapper(args):
    handler_params, kg_data, ratio, Noise_strategy, data_occur = args
    handler = DataHandler(*handler_params)
    return handler.process_kg_data(kg_data, ratio, Noise_strategy, data_occur)


if __name__ == '__main__':
    proctitle = 'DataHandler process: {}'.format(args.dataset)
    setproctitle.setproctitle(proctitle) # 读取环境变量并设置进程名称
    # args.dataset = 'ICEWS14cut{}'.format(args.max_relations)
    
    # prctl.set_name('Handler_{}'.format(args.dataset))
    # args.dataset = "GDELT"
    log('Start preprocess {}'.format(args.dataset))
    os.chdir(os.getcwd().split('/MyWork')[0]+'/MyWork')
    data_dir = './SavedData/{}'.format(args.dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for ratio in Ratios[3:4]: # GDELT 只生成0.3的，因为太耗时间了
        handler = DataHandler(args.dataset, args.device)
        Noise_strategy = args.noise_strategy  # 0,1,2,分别代表不同的噪声策略
        handler.load_data(ratio, Noise_strategy)
        # handler.load_data_cut_relation(ratio, Noise_strategy)  # 删减关系
        data_file = './SavedData/{}/{}_handler_ratio{}_strategy{}.pkl'.format(args.dataset, args.dataset, ratio, Noise_strategy)
        with open(data_file, 'wb') as f:
            pickle.dump(handler, f)
            log('Finished: {}_handler_ratio{}_strategy{}.pkl'.format(args.dataset, ratio, Noise_strategy))

    log('Finished: {}'.format(args.dataset))



    # # 重新存储噪声数据集到txt文件中
    # os.chdir(os.getcwd().split('/MyWork')[0]+'/MyWork')
    # data_dir = './SavedData/{}'.format(args.dataset)
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir)
    # import itertools
    # for neg_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     noise_data_dir = os.getcwd().split('RobustTKG')[0]+'/Data/{}-N{}'.format(args.dataset, int(neg_ratio*10))
    #     if not os.path.exists(noise_data_dir):
    #         os.makedirs(noise_data_dir)
        
    #     data_file = './SavedData/{}/{}_handler_ratio{}_strategy{}.pkl'.format(args.dataset, args.dataset, neg_ratio, args.noise_strategy)
    #     if os.path.exists(data_file):
    #         with open(data_file, 'rb') as f:
    #             handler = pickle.load(f)
    #     # trndata = np.concatenate(handler.trnListDirt)[:,0:4]
    #     trndata = []
    #     for kg in handler.trnListDirt:
    #         tmp_array = np.array(list(kg.keys()))
    #         filtered_array = tmp_array[tmp_array[:, 1] < handler.num_r]
    #         trndata.append(filtered_array)
    #     trndata = np.vstack(trndata)[:,0:5]

    #     trndata = '\n'.join(['\t'.join(map(str, row)) for row in trndata])
    #     with open(noise_data_dir+'/train.txt', 'w') as file:
    #         file.write(trndata + '\n')

    #     # valdata = np.concatenate(handler.valListDirt)[:,0:4]
    #     valdata = []
    #     for kg in handler.valListDirt:
    #         tmp_array = np.array(list(kg.keys()))
    #         filtered_array = tmp_array[tmp_array[:, 1] < handler.num_r]
    #         valdata.append(filtered_array)
    #     valdata = np.vstack(valdata)[:,0:5]
    #     valdata = '\n'.join(['\t'.join(map(str, row)) for row in valdata])
    #     with open(noise_data_dir+'/valid.txt', 'w') as file:
    #         file.write(valdata + '\n')

    #     # tstdata = np.concatenate(handler.tstListDirt)[:,0:4]
    #     tstdata = []
    #     for kg in handler.tstListDirt:
    #         tmp_array = np.array(list(kg.keys()))
    #         filtered_array = tmp_array[tmp_array[:, 1] < handler.num_r]
    #         tstdata.append(filtered_array)
    #     tstdata = np.vstack(tstdata)[:,0:5]
    #     tstdata = '\n'.join(['\t'.join(map(str, row)) for row in tstdata])
    #     with open(noise_data_dir+'/test.txt', 'w') as file:
    #         file.write(tstdata + '\n')
    #     log("Save {} neg ratio {}".format(args.dataset, neg_ratio))
    #     # break
        
    
    