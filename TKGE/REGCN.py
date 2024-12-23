# import dgl.network
import torch.nn as nn
import torch
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import numpy as np
from Utils import build_dgl_graph
from collections import deque
from TKGE.layers import PathEncoder, RGCN, GCN, GKAN
import networkx as nx
import random
from TimeLogger import log
from concurrent.futures import ThreadPoolExecutor
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from Utils import calc_mrr_hits, get_ranks
import copy

import GPUtil


"""
TKG embedding model 
1) encoder
2) decoder
"""
class ConvTransE(nn.Module):
    """Here, we implement classical ConvTransE as decoder"""
    def __init__(self, num_entities, embed_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):
        super(ConvTransE, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embed_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embed_dim * channels, embed_dim)
        self.bn3 = torch.nn.BatchNorm1d(embed_dim)
        self.bn_init = torch.nn.BatchNorm1d(embed_dim)

    def forward(self, eEmbeds, rEmbeds, triplets, all_candidate=True):
        eEmbeds = F.tanh(eEmbeds)
        batch_size = len(triplets)
        e1_embedded = eEmbeds[triplets[:, 0]].unsqueeze(1)
        # e1_embedded = 0.3*query_Embeds.unsqueeze(1) + 0.7*e1_embedded
        rel_embedded = rEmbeds[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)

        # x = F.normalize(x)
        
        if all_candidate:
            scores = torch.mm(x, eEmbeds.transpose(1, 0))
        else:
            e2_embedded = eEmbeds[triplets[:, 2]]
            scores = torch.sum(x * e2_embedded, dim=-1)
        scores = F.relu(scores)
        return scores
    
    def calculate_h_r(self, eEmbeds, rEmbeds, triplets):
        with torch.no_grad():
            eEmbeds = F.tanh(eEmbeds)
            batch_size = len(triplets)
            e1_embedded = eEmbeds[triplets[:, 0]].unsqueeze(1)
            rel_embedded = rEmbeds[triplets[:, 1]].unsqueeze(1)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
            stacked_inputs = self.bn0(stacked_inputs)
            x = self.inp_drop(stacked_inputs)
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = x.view(batch_size, -1)
            x = self.fc(x)
            x = self.hidden_drop(x)
            if batch_size > 1:
                x = self.bn2(x)
            x = F.relu(x)
            return x

    def calculate_score_with_embed(self, e1_embed, e2_embed, r_embed):
        with torch.no_grad():
            e1_embed = F.tanh(e1_embed)
            e2_embed = F.tanh(e2_embed)

            batch_size = e1_embed.shape[0]
            e1_embed = e1_embed.unsqueeze(1)
            r_embed = r_embed.unsqueeze(1)
            stacked_inputs = torch.cat([e1_embed, r_embed], 1)
            stacked_inputs = self.bn0(stacked_inputs)
            x = self.inp_drop(stacked_inputs)
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = x.view(batch_size, -1)
            x = self.fc(x)
            x = self.hidden_drop(x)
            if batch_size > 1:
                x = self.bn2(x)
            x = F.relu(x)
            scores = torch.sum(x * e2_embed, dim=-1)
            scores = F.relu(scores)
            return scores


class REGCN(nn.Module): # encoder
    def __init__(self, num_e, num_r, dim, layer_n, layer_norm, encoder, drop_rate, device, init_method, init_std, lambda1, lambda2, lambda3):
        super(REGCN, self).__init__()
        self.num_e = num_e
        self.num_r = num_r
        self.dim = dim
        self.device = device

        self.layer_norm = layer_norm

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
        self.init_method=init_method

        if init_method == 'normal':
            self.eEmbeds = nn.Parameter(nn.init.normal_(torch.empty(num_e, dim), std=init_std))
            self.rEmbeds = nn.Parameter(nn.init.normal_(torch.empty(num_r*2, dim), std=init_std))
        elif init_method == 'xavier':
            self.eEmbeds = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_e, dim)))
            self.rEmbeds = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_e, dim))) 
        elif init_method == 'normal+xavier':
            self.eEmbeds = nn.Parameter(nn.init.normal_(torch.empty(num_e, dim), std=init_std))
            self.rEmbeds = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_e, dim)))
        elif init_method == 'xavier+normal':
            self.eEmbeds = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_e, dim)))
            self.rEmbeds = nn.Parameter(nn.init.normal_(torch.empty(num_e, dim), std=init_std))    
        elif init_method == 'kaiming':
            self.eEmbeds = nn.Parameter(nn.init.kaiming_normal_(torch.empty(num_e, dim)))
            self.rEmbeds = nn.Parameter(nn.init.kaiming_normal_(torch.empty(num_e, dim))) 

        self.rEmbeds = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_r*2, dim)))


        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, dim))
        self.w1 = nn.Linear(self.dim*2, self.dim)
        self.w2 = nn.Linear(self.dim, self.dim)
        self.w3 = nn.Linear(self.dim, self.dim)

        self.time_gate_weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(dim, dim), gain=nn.init.calculate_gain('relu')))
        self.time_gate_bias = nn.Parameter(nn.init.zeros_(torch.empty(dim)))

        self.path_encoder = PathEncoder(dim, dim)

        if encoder == 'RGCN':
            self.eEmbeds_layers = nn.ModuleList([RGCN(dim, dim, drop_rate) for i in range(layer_n)])  # Entity embedding update
        elif encoder == 'GCN':
            self.eEmbeds_layers = nn.ModuleList([GCN(dim, dim, drop_rate) for i in range(layer_n)])
        elif encoder == 'GKAN':
            self.eEmbeds_layers = nn.ModuleList([GKAN(dim, dim, drop_rate) for i in range(layer_n)])

        self.rEmbeds_GRU = nn.GRUCell(2*dim, dim) # Relation embedding update
        self.eEmbeds_GRU = nn.GRUCell(dim, dim) # Entity embedding update

        self.decoder = ConvTransE(num_e, dim, input_dropout=drop_rate, hidden_dropout=drop_rate, feature_map_dropout=drop_rate)
        # self.loss_func = torch.nn.CrossEntropyLoss()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')


    def calculate_loss(self, kg_dict, facts_neg=None, confidence=0):
        eEmbeds, rEmbeds = self.eEmbeds_pred, self.rEmbeds_pred
        # query_Embeds = self.query_Embeds
        facts, weights = self.calculate_confidence(eEmbeds, rEmbeds, kg_dict, confidence)
        facts = torch.LongTensor(facts).to(self.device)
        scores = self.decoder(eEmbeds, rEmbeds, facts)
        loss_all = self.loss_func(scores, facts[:,2])
        loss = loss_all * weights.to(self.device)
        loss = torch.mean(loss)
        scores_neg = None
        return loss, scores, scores_neg

    def predict(self, facts: torch.LongTensor):
        eEmbeds, rEmbeds = self.eEmbeds_pred, self.rEmbeds_pred
        # query_Embeds = self.query_Embeds
        assert facts.shape[0] > 0
        scores = self.decoder(eEmbeds, rEmbeds, facts)
        # loss_all = self.loss_func(scores, facts[:,2])
        return scores
    

    def forward_old(self, data_history_all, copy_vocabulary=None, confidence=0):  # run as REGCN
        eEmbeds_pred = F.normalize(self.eEmbeds) if self.layer_norm else self.eEmbeds
        rEmbeds_pred = F.normalize(self.rEmbeds) if self.layer_norm else self.rEmbeds

        for i in range(0, len(data_history_all)):
            kg_data_dict = data_history_all[i]
            facts, weights = self.calculate_confidence(eEmbeds_pred, rEmbeds_pred, kg_data_dict, confidence=0)
            g = build_dgl_graph(self.num_e, self.num_r, facts, weights, inverse=False).to(self.device)

            r_to_eEmbeds = eEmbeds_pred[g.r_to_eList]
            r_GRUInput = torch.zeros(self.num_r * 2, self.dim).float().to(self.device)
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = r_to_eEmbeds[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                r_GRUInput[r_idx] = x_mean
            r_GRUInput = torch.cat((rEmbeds_pred, r_GRUInput), dim=1)
            rEmbeds_pred = self.rEmbeds_GRU(r_GRUInput, rEmbeds_pred)
            rEmbeds_pred = F.normalize(rEmbeds_pred) if self.layer_norm else rEmbeds_pred

            # eEmbedding update
            eEmbeds_tmp = eEmbeds_pred
            for e_layer in self.eEmbeds_layers:
                e_layer(g, eEmbeds_tmp, rEmbeds_pred)
                eEmbeds_tmp = g.ndata.pop('h')
            time_weight = F.sigmoid(torch.mm(eEmbeds_pred, self.time_gate_weight) + self.time_gate_bias)
            eEmbeds_pred = time_weight * eEmbeds_tmp + (1 - time_weight) * eEmbeds_pred

            # eEmbeds_pred = self.eEmbeds_GRU(eEmbeds_tmp, eEmbeds_pred)  # 该用GRU来建模实体表示的时间动态
            # eEmbeds_pred = F.normalize(eEmbeds_pred) if self.layer_norm else eEmbeds_pred
            # 建模注意力

        
        self.eEmbeds_pred = F.normalize(eEmbeds_pred) if self.layer_norm else eEmbeds_pred
        self.rEmbeds_pred = rEmbeds_pred
        return eEmbeds_pred, rEmbeds_pred

    def forward(self, data_history_all, kg_dict_pred, copy_vocabulary=None, confidence=0):
        eEmbeds_pred = F.normalize(self.eEmbeds) if self.layer_norm else self.eEmbeds
        rEmbeds_pred = F.normalize(self.rEmbeds) if self.layer_norm else self.rEmbeds

        # with torch.no_grad():
        #     for i in range(0, len(data_history_all)):
        #         kg_data_dict = data_history_all[i]
        #         facts, weights = self.calculate_confidence(eEmbeds_pred, rEmbeds_pred, kg_data_dict, confidence)
        #         g = build_dgl_graph(self.num_e, self.num_r, facts, weights, inverse=False).to(self.device)

        #         r_to_eEmbeds = eEmbeds_pred[g.r_to_eList]
        #         r_GRUInput = torch.zeros(self.num_r * 2, self.dim).float().to(self.device)
        #         for span, r_idx in zip(g.r_len, g.uniq_r):
        #             x = r_to_eEmbeds[span[0]:span[1], :]
        #             x_mean = torch.mean(x, dim=0, keepdim=True)
        #             r_GRUInput[r_idx] = x_mean
        #         r_GRUInput = torch.cat((rEmbeds_pred, r_GRUInput), dim=1)
        #         rEmbeds_pred = self.rEmbeds_GRU(r_GRUInput, rEmbeds_pred)
        #         rEmbeds_pred = F.normalize(rEmbeds_pred) if self.layer_norm else rEmbeds_pred

        #         # eEmbedding update
        #         eEmbeds_tmp = eEmbeds_pred
        #         for e_layer in self.eEmbeds_layers:
        #             e_layer(g, eEmbeds_tmp, rEmbeds_pred)
        #             eEmbeds_tmp = g.ndata.pop('h')
        #         # time_weight = F.sigmoid(torch.mm(eEmbeds_pred, self.time_gate_weight) + self.time_gate_bias)
        #         # eEmbeds_pred = time_weight * eEmbeds_tmp + (1 - time_weight) * eEmbeds_pred

        #         eEmbeds_pred = self.eEmbeds_GRU(eEmbeds_tmp, eEmbeds_pred)  # 该用GRU来建模实体表示的时间动态
        #         eEmbeds_pred = F.normalize(eEmbeds_pred) if self.layer_norm else eEmbeds_pred
        
        # 建模注意力, 不对所有邻居信息等效地聚合传递，而是有注意力的聚合信息；抓取有助于最终问题的信息，而不是追求真实的信息, 学习所有真实信息并不一定都有助于预测

        query_mask = self.get_query_Embeds(kg_dict_pred)  # (num_e, dim)
        query_Embeds = self.get_query_Embeds_split(kg_dict_pred)  # (facts.shape[0], dim)
        attn_eEmbeds_list = []
        attn_eEmbeds_list_split = []
        for i in range(0, len(data_history_all)):

            # t2 = len(data_history_all)-i+1
            # h_t = torch.cos(self.weight_t2 * t2 + self.bias_t2).repeat(self.num_e,1)  # 时间位置编码
            # eEmbeds_pred =self.w4(torch.concat([eEmbeds_pred, h_t],dim=1))  # 通过一个参数矩阵，修改维度

            kg_data_dict = data_history_all[i]
            facts, weights = self.calculate_confidence(eEmbeds_pred, rEmbeds_pred, kg_data_dict, confidence)
            g = build_dgl_graph(self.num_e, self.num_r, facts, weights, inverse=False).to(self.device)

            r_to_eEmbeds = eEmbeds_pred[g.r_to_eList]
            r_GRUInput = torch.zeros(self.num_r * 2, self.dim).float().to(self.device)
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = r_to_eEmbeds[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                r_GRUInput[r_idx] = x_mean
            r_GRUInput = torch.cat((rEmbeds_pred, r_GRUInput), dim=1)
            rEmbeds_pred = self.rEmbeds_GRU(r_GRUInput, rEmbeds_pred)
            rEmbeds_pred = F.normalize(rEmbeds_pred) if self.layer_norm else rEmbeds_pred

            # eEmbedding update
            eEmbeds_tmp = eEmbeds_pred
            for e_layer in self.eEmbeds_layers:
                e_layer(g, eEmbeds_tmp, rEmbeds_pred)
                eEmbeds_tmp = g.ndata.pop('h')
            eEmbeds_pred = self.eEmbeds_GRU(eEmbeds_tmp, eEmbeds_pred)  # 该用GRU来建模实体表示的时间动态
            eEmbeds_pred = F.normalize(eEmbeds_pred) if self.layer_norm else eEmbeds_pred
            # eEmbeds_pred = eEmbeds_tmp  # 尝试不经过GRU，让实体表示更新的幅度更佳明显，凸显动态性

            # att_e = F.softmax(self.w2(query_mask+eEmbeds_tmp),dim=1)  # 按照dim=1进行softmax # （1）softmax操作
            # att_e = query_mask+eEmbeds_tmp # （2）去除softmax操作->影响不大，甚至可以更快地完成训练，会有潜在问题，无法强化有效信息和弱化无效信息，需要softmax来完成信息的
            # att_e = F.softmax(query_mask)  # (3) 单纯用mask进行指导信息提取->影响不大，甚至可以更快地完成训练，但是会有潜在的问题，可能未被涉及到的实体表示会变为0
            att_e = query_mask
            # att_e = eEmbeds_tmp  # (4) 跟query完全无关，只是当前更新后表示的自注意力->优越性能的失效，变为和传统结构一样的效果；说明mask操作非常有用，为什么？

            # att_emb = att_e*eEmbeds_pred
            att_emb = att_e*eEmbeds_tmp  # 预测性能更好，更好地捕获动态性

            attn_eEmbeds_list.append(att_emb.unsqueeze(0))

            # att_q = self.w3(query_Embeds @ eEmbeds_tmp.transpose(1,0))  # (facts.shape[0], num_e)
            # att_emb_q = att_q @ eEmbeds_pred  # (facts.shape[0], dim)

            # time_weight = F.sigmoid(torch.mm(eEmbeds_pred, self.time_gate_weight) + self.time_gate_bias)
            # eEmbeds_pred = time_weight * eEmbeds_tmp + (1 - time_weight) * eEmbeds_pred

        attn_eEmbeds = torch.mean(torch.concat(attn_eEmbeds_list, dim=0), dim=0)
        attn_eEmbeds = F.normalize(attn_eEmbeds)
        # self.query_Embeds = attn_eEmbeds
        # eEmbeds_pred = eEmbeds_pred + attn_eEmbeds

        self.eEmbeds_pred = F.normalize(attn_eEmbeds) if self.layer_norm else attn_eEmbeds
        # self.eEmbeds_pred = F.normalize(eEmbeds_pred) if self.layer_norm else eEmbeds_pred  # 不使用mask操作，传统RNN+GNN框架
        self.rEmbeds_pred = rEmbeds_pred
        return eEmbeds_pred, rEmbeds_pred

    def forward_observe(self, data_history_all, kg_dict_pred, copy_vocabulary=None, confidence=0, dataset=None):
        eEmbeds_pred = F.normalize(self.eEmbeds) if self.layer_norm else self.eEmbeds
        rEmbeds_pred = F.normalize(self.rEmbeds) if self.layer_norm else self.rEmbeds

        query_mask = self.get_query_Embeds(kg_dict_pred)  # (num_e, dim)
        # query_Embeds = self.get_query_Embeds_split(kg_dict_pred)  # (facts.shape[0], dim)
        attn_eEmbeds_list = []
        attn_eEmbeds_list_split = []

        self.observe_query_mask = query_mask
        self.observe_eEmbeds_list = []
        self.observe_attn_eEmbeds_list = []

        for i in range(0, len(data_history_all)):
            kg_data_dict = data_history_all[i]
            facts, weights = self.calculate_confidence(eEmbeds_pred, rEmbeds_pred, kg_data_dict, confidence)
            g = build_dgl_graph(self.num_e, self.num_r, facts, weights, inverse=False).to(self.device)

            r_to_eEmbeds = eEmbeds_pred[g.r_to_eList]
            r_GRUInput = torch.zeros(self.num_r * 2, self.dim).float().to(self.device)
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = r_to_eEmbeds[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                r_GRUInput[r_idx] = x_mean
            r_GRUInput = torch.cat((rEmbeds_pred, r_GRUInput), dim=1)
            rEmbeds_pred = self.rEmbeds_GRU(r_GRUInput, rEmbeds_pred)
            rEmbeds_pred = F.normalize(rEmbeds_pred) if self.layer_norm else rEmbeds_pred

            # eEmbedding update
            eEmbeds_tmp = eEmbeds_pred
            for e_layer in self.eEmbeds_layers:
                e_layer(g, eEmbeds_tmp, rEmbeds_pred)
                eEmbeds_tmp = g.ndata.pop('h')
            eEmbeds_pred = self.eEmbeds_GRU(eEmbeds_tmp, eEmbeds_pred)  # 该用GRU来建模实体表示的时间动态
            eEmbeds_pred = F.normalize(eEmbeds_pred) if self.layer_norm else eEmbeds_pred

            att_e = query_mask
            att_emb = att_e*eEmbeds_tmp  # 预测性能更好，更好地捕获动态性

            attn_eEmbeds_list.append(att_emb.unsqueeze(0))


            self.observe_eEmbeds_list.append(eEmbeds_pred)
            self.observe_attn_eEmbeds_list.append(att_emb.unsqueeze(0))

        attn_eEmbeds = torch.mean(torch.concat(attn_eEmbeds_list, dim=0), dim=0)
        attn_eEmbeds = F.normalize(attn_eEmbeds)
        # self.query_Embeds = attn_eEmbeds
        # eEmbeds_pred = eEmbeds_pred + attn_eEmbeds

        with open('./SavedObserve/{}/observe.pkl'.format(dataset), 'wb') as f:
            observe = {}
            observe['observe_query_mask'] = self.observe_query_mask
            observe['observe_eEmbeds_list'] = self.observe_eEmbeds_list
            observe['observe_attn_eEmbeds_list'] = self.observe_attn_eEmbeds_list
            pickle.dump(observe, f)

        self.eEmbeds_pred = F.normalize(attn_eEmbeds) if self.layer_norm else attn_eEmbeds
        # self.eEmbeds_pred = F.normalize(eEmbeds_pred) if self.layer_norm else eEmbeds_pred  # 不使用mask操作，传统RNN+GNN框架
        self.rEmbeds_pred = rEmbeds_pred
        return eEmbeds_pred, rEmbeds_pred



    def get_query_Embeds(self, kg_dict_pred):
        """
        获取query的嵌入表示 指导信息的获取
        """
        query_mask = torch.ones((self.num_e, self.dim)).to(self.device)  # ones操作，会导致弱化mask的作用，从而噪声测试性能的下降（66->60）； 但如果采用ones替代eEmbeds_tmp，则可以实现效果些微提升，因为避免了实体的表达不断和自己比较
        # query_mask = torch.zeros((self.num_e, self.dim)).to(self.device)
        facts = torch.LongTensor(list(kg_dict_pred.keys())).to(self.device)
        s_Embeds = self.eEmbeds[facts[:, 0]].squeeze()
        r_Embeds = self.rEmbeds[facts[:, 1]].squeeze()
        
        query_Embeds = self.w1(torch.concat([s_Embeds, r_Embeds], dim=1))  # query的表示
        # query_Embeds = s_Embeds  # 单纯query中头实体店表示，性能显著下降

        query_mask.index_add_(0, facts[:, 0], query_Embeds)  #  按照 头实体s 的索引，将query的表示添加到mask上
        count_mask = torch.zeros((self.num_e, 1)).to(self.device)
        count_mask.index_add_(0, facts[:, 0], torch.ones_like(facts[:, 0], dtype=torch.float).unsqueeze(1))
        query_Embeds = query_mask / count_mask.clamp(min=1)  # (num_e, dim)
        
        # query_Embeds = torch.where(query_Embeds>0, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))
        return query_Embeds

    def get_query_Embeds_split(self, kg_dict_pred):
        """
        获取query的嵌入表示 指导信息的获取
        """
        query_mask = torch.zeros((self.num_e, self.dim)).to(self.device)
        facts = torch.LongTensor(list(kg_dict_pred.keys())).to(self.device)
        s_Embeds = self.eEmbeds[facts[:, 0]].squeeze()
        r_Embeds = self.rEmbeds[facts[:, 1]].squeeze()
        query_Embeds = self.w1(torch.concat([s_Embeds, r_Embeds], dim=1))  # （facts.shape[0], dim）

        # query_mask.index_add_(0, facts[:, 0], query_Embeds)
        # count_mask = torch.zeros((self.num_e, 1)).to(self.device)
        # count_mask.index_add_(0, facts[:, 0], torch.ones_like(facts[:, 0], dtype=torch.float).unsqueeze(1))
        # query_Embeds = query_mask / count_mask.clamp(min=1)  # (num_e, dim)
        return query_Embeds

    def forward_check(self, data_history_all, copy_vocabulary=None, confidence=0):
        with torch.no_grad():
            eEmbeds_pred = F.normalize(self.eEmbeds) if self.layer_norm else self.eEmbeds
            rEmbeds_pred = F.normalize(self.rEmbeds) if self.layer_norm else self.rEmbeds
            data_input = copy.deepcopy(data_history_all[-3:])

            for i in range(0, 3):
                kg_data_dict_list = data_history_all[i:i+3]
                for kg_data_dict in kg_data_dict_list:
                    facts, weights = self.calculate_confidence(eEmbeds_pred, rEmbeds_pred, kg_data_dict, confidence=0)
                    g = build_dgl_graph(self.num_e, self.num_r, facts, weights, inverse=False).to(self.device)

                    r_to_eEmbeds = eEmbeds_pred[g.r_to_eList]
                    r_GRUInput = torch.zeros(self.num_r * 2, self.dim).float().to(self.device)
                    for span, r_idx in zip(g.r_len, g.uniq_r):
                        x = r_to_eEmbeds[span[0]:span[1], :]
                        x_mean = torch.mean(x, dim=0, keepdim=True)
                        r_GRUInput[r_idx] = x_mean
                    r_GRUInput = torch.cat((rEmbeds_pred, r_GRUInput), dim=1)
                    rEmbeds_pred = self.rEmbeds_GRU(r_GRUInput, rEmbeds_pred)
                    rEmbeds_pred = F.normalize(rEmbeds_pred) if self.layer_norm else rEmbeds_pred

                    eEmbeds_now = eEmbeds_pred
                    for e_layer in self.eEmbeds_layers:
                        e_layer(g, eEmbeds_now, rEmbeds_pred)
                        eEmbeds_now = g.ndata.pop('h')

                    # if self.init_method=='normal' or self.init_method=='normal+xavier' :
                    #     eEmbeds_now = F.normalize(eEmbeds_now) if self.layer_norm else eEmbeds_now
                    time_weight = F.sigmoid(torch.mm(eEmbeds_pred, self.time_gate_weight) + self.time_gate_bias)
                    eEmbeds_pred = time_weight * eEmbeds_now + (1 - time_weight) * eEmbeds_pred
                    kg_dict_input = data_input[i]
                    facts = torch.LongTensor(list(kg_dict_input.keys())).to(self.device)
                    scores = self.decoder(eEmbeds_pred, rEmbeds_pred, facts)
                    ranks = torch.tensor(get_ranks(facts, scores, batch_size=-1, filter=0, num_e=self.num_e))
                    neg_idx = torch.nonzero(ranks>80).to(self.device)
                    facts_to_remove = facts[neg_idx].view(-1,facts.shape[-1])
                    facts_to_remove = [tuple(item.tolist()) for item in facts_to_remove]
                    kg_dict_input_new = {key: value for key, value in kg_dict_input.items() if key not in facts_to_remove}
                    data_input[i] = kg_dict_input_new
        
        self.forward(data_input) # 用新的数据进行学习训练
        return eEmbeds_pred, rEmbeds_pred
    
    def calculate_loss_check(self, kg_dict, facts_neg=None, confidence=0, conf_threshold=0):
        eEmbeds, rEmbeds = self.eEmbeds_pred, self.rEmbeds_pred
        facts = torch.LongTensor(list(kg_dict.keys())).to(self.device)
        scores = self.decoder(eEmbeds, rEmbeds, facts)
        # ranks = torch.tensor(get_ranks(facts, scores, batch_size=-1, filter=0, num_e=self.num_e))

        facts_raw, weights = self.calculate_confidence(eEmbeds, rEmbeds, kg_dict, confidence)
        weights.detach()
        noise_idx = torch.nonzero(weights<conf_threshold).to(self.device).squeeze()
        weights[noise_idx] = 0.0
        keep_idx = torch.nonzero(weights>=conf_threshold).to(self.device).squeeze()
        weights[keep_idx] = 1.0

        # pos_idx = torch.nonzero(ranks<=80).to(self.device).squeeze()
        # weights[pos_idx] = 2.0
        # pos_idx = torch.nonzero(ranks<=50).to(self.device).squeeze()
        # weights[pos_idx] = 1.0
        # pos_idx = torch.nonzero(ranks<=10).to(self.device).squeeze()
        # weights[pos_idx] = 3.0
        # neg_idx = torch.nonzero(ranks>100).to(self.device).squeeze()
        # weights[neg_idx] = 0.0
        # facts_to_learn = facts[neg_idx].view(-1,facts.shape[-1])
        # scores_to_learn = scores[neg_idx].view(-1, self.num_e)

        loss_all = self.loss_func(scores, facts[:,2])
        loss = loss_all * weights.to(self.device)
        loss = torch.mean(loss)

        # facts_pos = facts[facts[:,4]==1]
        # scores_pos = self.decoder(eEmbeds, rEmbeds, facts_pos)[torch.arange(facts_pos.size(0)), facts_pos[:,2].squeeze()]
        # Y = scores_pos.tolist()
        # bins = [0, 2, 4, 6, 8, np.inf]
        # counts, _ = np.histogram(Y, bins=bins)
        # percentages = counts / len(Y) * 100
        # labels = ['0-2', '2-4', '4-6', '6-8', '8-inf'] # 可视化分数段的占比
        # plt.figure(figsize=(10, 6))
        # plt.bar(labels, percentages, color='skyblue')
        # plt.title('Percentage of Scores in Different Ranges')
        # plt.xlabel('Score Range')
        # plt.ylabel('Percentage (%)')
        # for i, v in enumerate(percentages):
        #     plt.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=10)
        # plt.savefig("./scores_pos.png")
        
        # facts_neg = facts[facts[:,4]==0]
        # scores_neg = self.decoder(eEmbeds, rEmbeds, facts_neg)[torch.arange(facts_neg.size(0)), facts_neg[:,2].squeeze()]
        # Y = scores_neg.tolist()
        # bins = [0, 2, 4, 6, 8, np.inf]
        # counts, _ = np.histogram(Y, bins=bins)
        # percentages = counts / len(Y) * 100
        # labels = ['0-2', '2-4', '4-6', '6-8', '8-inf'] # 可视化分数段的占比
        # plt.figure(figsize=(10, 6))
        # plt.bar(labels, percentages, color='skyblue')
        # plt.title('Percentage of Scores in Different Ranges')
        # plt.xlabel('Score Range')
        # plt.ylabel('Percentage (%)')
        # for i, v in enumerate(percentages):
        #     plt.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=10)
        # plt.savefig("./scores_negs.png")

        return loss, scores, None


    def calculate_confidence(self, eEmbeds_tmp, rEmbeds_tmp, kg_data_dict=None, confidence=0):
        triples = list(kg_data_dict.keys())  # [(s,r,o,T,label),...]
        if not confidence:
            return np.asarray(triples), torch.ones(len(triples)).float()
        
        assert (self.lambda1 + self.lambda2 + self.lambda3)>0, "Please specify types of confidence"

        # with torch.no_grad():
        if True:
            conf = torch.zeros([len(triples)], dtype=torch.float32).to(self.device)
            conf1, conf2, conf3 = 0,0,0
            if self.lambda1:
                conf1 =self.get_predict_conf(eEmbeds_tmp, rEmbeds_tmp, np.asarray(triples))
                conf1 = self.lambda1 *conf1
                # conf = torch.max(conf, conf1)
            if self.lambda2: 
                bypass = [kg_data_dict[fact]['bypass'] for fact in triples]
                bypass_PCRA = [kg_data_dict[fact]['bypass_PCRA'] for fact in triples]
                conf2 = self.lambda2 * self.get_bypass_context_conf(eEmbeds_tmp, rEmbeds_tmp, triples, bypass, bypass_PCRA)
                # conf = conf + conf2
            if self.lambda3:
                # s_context = [kg_data_dict[fact]['s_context'] for fact in triples]
                # o_context = [kg_data_dict[fact]['o_context'] for fact in triples]
                # s_context_PCRA = [kg_data_dict[fact]['s_context_PCRA'] for fact in triples]
                # o_context_PCRA = [kg_data_dict[fact]['o_context_PCRA'] for fact in triples]
                # conf3 = self.get_semantic_context_conf(eEmbeds, rEmbeds, triples, s_context, o_context, s_context_PCRA, o_context_PCRA)  # semantic environment confidence
  
                df = pd.DataFrame(triples, columns=['s', 'r', 'o', 't', 'label'])
                df_unique = df.drop_duplicates(subset=['s', 'o'])
                facts = np.asarray(df_unique.values.tolist())
                g_tmp = build_dgl_graph(self.num_e, self.num_r, facts, None, inverse=False).to(self.device)  # 处理后，每两个节点之间有且仅有一条边
                # g_tmp = build_dgl_graph(self.num_e, self.num_r, np.asarray(triples), None, inverse=False).to(self.device)
                conf3 = self.lambda3 * self.get_community_conf(np.asarray(triples), g_tmp, eEmbeds_tmp, rEmbeds_tmp)

                # conf = torch.max(conf, conf3)
            conf = conf1+ conf2+ conf3
            conf = torch.sigmoid(conf)


            # conf = conf/(self.lambda1 + self.lambda2 + self.lambda3)
            # Y = conf.tolist()
            # bins = [0, 2, 4, 6, 8, np.inf]
            # counts, _ = np.histogram(Y, bins=bins)
            # percentages = counts / len(Y) * 100
            # labels = ['0-2', '2-4', '4-6', '6-8', '8-inf'] # 可视化分数段的占比
            # plt.figure(figsize=(10, 6))
            # plt.bar(labels, percentages, color='skyblue')
            # plt.title('Percentage of Scores in Different Ranges')
            # plt.xlabel('Score Range')
            # plt.ylabel('Percentage (%)')
            # for i, v in enumerate(percentages):
            #     plt.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=10)
            # plt.savefig("./conf.png")


            # facts = torch.LongTensor(np.asarray(triples)).to(self.device)
            # facts_pos = facts[facts[:,4]==1]
            # facts_neg = facts[facts[:,4]==0]
            # scores_pos = conf[facts_pos[2]]
            # Y = scores_pos.tolist()
            # bins = [0, 2, 4, 6, 8, np.inf]
            # counts, _ = np.histogram(Y, bins=bins)
            # percentages = counts / len(Y) * 100
            # labels = ['0-2', '2-4', '4-6', '6-8', '8-inf'] # 可视化分数段的占比
            # plt.figure(figsize=(10, 6))
            # plt.bar(labels, percentages, color='skyblue')
            # plt.title('Percentage of Scores in Different Ranges')
            # plt.xlabel('Score Range')
            # plt.ylabel('Percentage (%)')
            # for i, v in enumerate(percentages):
            #     plt.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=10)
            # plt.savefig("./conf_pos.png")

            # scores_neg = conf[facts_neg[2]]
            # Y = scores_neg.tolist()
            # bins = [0, 2, 4, 6, 8, np.inf]
            # counts, _ = np.histogram(Y, bins=bins)
            # percentages = counts / len(Y) * 100
            # labels = ['0-2', '2-4', '4-6', '6-8', '8-inf'] # 可视化分数段的占比
            # plt.figure(figsize=(10, 6))
            # plt.bar(labels, percentages, color='skyblue')
            # plt.title('Percentage of Scores in Different Ranges')
            # plt.xlabel('Score Range')
            # plt.ylabel('Percentage (%)')
            # for i, v in enumerate(percentages):
            #     plt.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=10)
            # plt.savefig("./conf_negs.png")

            return np.asarray(triples), conf.cpu()


    def get_predict_conf(self, eEmbeds, rEmbeds, facts=None, temperature=2.0):
        # with torch.no_grad():  # 通过历史知识进行预测，度量当前时刻的样本的置信度，体现了时间趋势的置信度
        #     conf = self.decoder(eEmbeds, rEmbeds, facts, all_candidate=False) + 1e-5
        #     # conf = torch.sigmoid(conf)
        #     return conf

        conf = self.decoder(eEmbeds, rEmbeds, facts, all_candidate=False) + 1e-5  # 尝试让置信度参与回归
        # conf = torch.sigmoid(conf)
        return conf
    
    def get_community_conf(self, facts, g, eEmbeds, rEmbeds):
        with torch.no_grad():
            for e_layer in self.eEmbeds_layers[:1]:
                eEmbeds_context_facts_s, eEmbeds_context_facts_o = e_layer.forward_context(g, eEmbeds, rEmbeds, facts)

            conf = self.decoder.calculate_score_with_embed(eEmbeds_context_facts_s, eEmbeds_context_facts_o, rEmbeds[facts[:,1]].squeeze()) + 1e-5
            # conf = torch.sigmoid(conf)
            return conf
    
    def get_bypass_context_conf(self, eEmbeds, rEmbeds, triples, bypass, bypass_PCRA):
        with torch.no_grad():
            '''结合嵌入表示，用多跳旁路对当前时刻的样本进行置信度的度量'''
            def quality(path, triple, eEmbeds, rEmbeds):
                (s,r,o) = triple
                s_r_target = self.decoder.calculate_h_r(eEmbeds, rEmbeds, np.array([(s, r, -1)]))

                s_r_path = []
                for i in range(1, len(path)-1, 2): # (es,r1,e1,r2,e2,...,rl, et)
                    entity = path[i - 1]
                    relation = path[i]
                    s_r = self.decoder.calculate_h_r(eEmbeds, rEmbeds, np.array([(entity, relation, -1)]))
                    s_r_path.append(s_r)
                s_r_path = torch.stack(s_r_path, dim=1)
                path_embed = self.path_encoder(s_r_path)
                Quality = torch.sum(path_embed * s_r_target)
                # Quality = torch.sigmoid(Quality)
                return Quality

            def get_similarity(triple, Paths, pcras):
                '''计算item(s,r,o)在paths这个上下文环境下的置信度 首先需要对path的可靠性进行计算'''
                '''PCRA度量路径的可靠性'''
                if len(Paths)==0:  #  如果s、o之间没有其他路径，置信度为0
                    return torch.tensor(0.0).to(self.device)
                res = []
                for j,p in enumerate(Paths):
                    R_p = pcras[j]
                    Q_p = quality(p, triple, eEmbeds, rEmbeds)
                    res.append(R_p * Q_p) 
                    # res.append(Q_p)  

                res = torch.stack(res)
                assert torch.isnan(res).any, 'Nan'
                res = torch.mean(res)
                res = F.sigmoid(res)
                return res
            
            bypass_conf = []
            # log('')
            for i,(s,r,o,_,_) in enumerate(triples):
                '''对每个三元组，单独计算置信度'''
                conf = get_similarity((s,r,o), bypass[i], bypass_PCRA[i])
                bypass_conf.append(conf)
            # log('')
            bypass_conf = torch.stack(bypass_conf)

            assert not torch.isnan(bypass_conf).any(),  "NaN detected in bypass_conf"
            # print(torch.max(bypass_conf))
            return bypass_conf


def check_gpu_memory():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        memory_used = gpu.memoryUsed  # 已使用显存（MB）
        memory_total = gpu.memoryTotal  # 总显存（MB）
        memory_used_gb = memory_used / 1024
        memory_total_gb = memory_total / 1024
        
        if memory_used_gb > 8:
            print(f"GPU {gpu.id} 显存占用超过 8GB: {memory_used_gb:.2f}GB / {memory_total_gb:.2f}GB")