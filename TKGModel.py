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
from MyWork.src_DGAD.layers_old import PathEncoder, RGCN, GCN, GKAN
import networkx as nx
import random
from TimeLogger import log
from concurrent.futures import ThreadPoolExecutor
import concurrent
import pandas as pd

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
        if all_candidate:
            scores = torch.mm(x, eEmbeds.transpose(1, 0))
        else:
            e2_embedded = eEmbeds[triplets[:, 2]]
            scores = torch.sum(x * e2_embedded, dim=-1)
        scores = F.relu(scores) + 1e-5
        return scores
    
    # def calculate_scores(self, eEmbeds, rEmbeds, triplets):
    #     eEmbeds = F.tanh(eEmbeds)
    #     batch_size = len(triplets)
    #     e1_embedded = eEmbeds[triplets[:, 0]].unsqueeze(1)
    #     rel_embedded = rEmbeds[triplets[:, 1]].unsqueeze(1)
    #     stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
    #     stacked_inputs = self.bn0(stacked_inputs)
    #     x = self.inp_drop(stacked_inputs)
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = F.relu(x)
    #     x = self.feature_map_drop(x)
    #     x = x.view(batch_size, -1)
    #     x = self.fc(x)
    #     x = self.hidden_drop(x)
    #     if batch_size > 1:
    #         x = self.bn2(x)
    #     x = F.relu(x)
    #     e2_embedded = eEmbeds[triplets[:, 0]]
        
        # scores = F.relu(scores)
        # return scores
    
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
            scores = F.relu(scores) + 1e-5
            return scores


class TKGModel(nn.Module): # encoder
    def __init__(self, num_e, num_r, dim, layer_n, encoder, drop_rate, device, lambda1, lambda2, lambda3):
        super(TKGModel, self).__init__()
        self.num_e = num_e
        self.num_r = num_r
        self.dim = dim
        self.device = device

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.eEmbeds = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_e, dim)))
        self.rEmbeds = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_r*2, dim)))
        self.time_gate_weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(dim, dim), gain=nn.init.calculate_gain('relu')))
        self.time_gate_bias = nn.Parameter(nn.init.zeros_(torch.empty(dim)))

        self.path_encoder = PathEncoder(dim, dim)

        if encoder == 'RGCN':
            self.RGCN_layers = nn.ModuleList([RGCN(dim, dim, drop_rate) for i in range(layer_n)])  # Entity embedding update
        elif encoder == 'GCN':
            self.RGCN_layers = nn.ModuleList([GCN(dim, dim, drop_rate) for i in range(layer_n)])
        elif encoder == 'GKAN':
            self.RGCN_layers = nn.ModuleList([GKAN(dim, dim, drop_rate) for i in range(layer_n)])

        self.rEmbeds_layers = nn.ModuleList([nn.GRUCell(2*dim, dim)])  # Relation embedding update

        self.decoder = ConvTransE(num_e, dim, input_dropout=drop_rate, hidden_dropout=drop_rate, feature_map_dropout=drop_rate)
        self.loss_func = torch.nn.CrossEntropyLoss()

        self.connected = []

    def calculate_loss(self, eEmbeds, rEmbeds, data_pos, data_neg=None):
        scores_pos = self.decoder(eEmbeds, rEmbeds, data_pos)
        loss = self.loss_func(scores_pos, data_pos[:,2])
        score_neg = None
        if data_neg is not None:
            score_neg = self.decoder(eEmbeds, rEmbeds, data_neg)
            loss_neg = - self.loss_func(score_neg, data_neg[:,2])
            loss = torch.mean(loss*data_pos.shape[0]+loss_neg*data_neg.shape[0])
        return loss, scores_pos, score_neg

    def forward(self, data_history_all):
        eEmbeds_pred = self.eEmbeds
        rEmbeds_pred = self.rEmbeds
        assert not torch.isnan(eEmbeds_pred).any(), 'Nan in eEmbeds_pred'
        assert not torch.isnan(rEmbeds_pred).any(), 'Nan in rEmbeds_pred'

        for i in range(0, len(data_history_all)):
            # calculate engergy for each triples
            # triples = data_history[i]

            kg_data_dict = data_history_all[i]
            facts, weights = self.calculate_confidence(eEmbeds_pred, rEmbeds_pred,  kg_data_dict)

            assert not torch.isnan(eEmbeds_pred).any(), 'Nan in eEmbeds_pred'
            assert not torch.isnan(rEmbeds_pred).any(), 'Nan in rEmbeds_pred'

            g = build_dgl_graph(self.num_e, self.num_r, facts, weights, inverse=False).to(self.device)

            for r_layer in self.rEmbeds_layers:
                r_to_eEmbeds = eEmbeds_pred[g.r_to_eList]
                r_GRUInput = torch.zeros(self.num_r * 2, self.dim).float().to(self.device)
                for span, r_idx in zip(g.r_len, g.uniq_r):
                    x = r_to_eEmbeds[span[0]:span[1], :]
                    x_mean = torch.mean(x, dim=0, keepdim=True)
                    r_GRUInput[r_idx] = x_mean
                r_GRUInput = torch.cat((rEmbeds_pred, r_GRUInput), dim=1)
                rEmbeds_pred = r_layer(r_GRUInput, rEmbeds_pred)
                rEmbeds_pred = F.normalize(rEmbeds_pred)

            assert not torch.isnan(eEmbeds_pred).any(), 'Nan in eEmbeds_pred'
            assert not torch.isnan(rEmbeds_pred).any(), 'Nan in rEmbeds_pred'

            # RGCN
            # node_id = g.ndata['id'].squeeze()
            # g.ndata['h'] = eEmbeds_pred[node_id]
            for e_layer in self.RGCN_layers:
                e_layer(g, eEmbeds_pred, rEmbeds_pred)
            eEmbeds_now = g.ndata.pop('h')

            eEmbeds_now = F.normalize(eEmbeds_now)
            time_weight = F.sigmoid(torch.mm(eEmbeds_pred, self.time_gate_weight) + self.time_gate_bias)
            eEmbeds_pred = time_weight * eEmbeds_now + (1 - time_weight) * eEmbeds_pred

            assert not torch.isnan(eEmbeds_pred).any(), 'Nan in eEmbeds_pred'
            assert not torch.isnan(rEmbeds_pred).any(), 'Nan in rEmbeds_pred'
            # eEmbeds_pred = eEmbeds_now
        # self.eEmbeds_pred = eEmbeds_pred
        # self.rEmbeds_pred = rEmbeds_pred
        return eEmbeds_pred, rEmbeds_pred

    def calculate_confidence(self, eEmbeds_tmp, rEmbeds_tmp, kg_data_dict=None):
        triples = list(kg_data_dict.keys())  # [(s,r,o,T,label),...]

        if (not self.lambda1) & (not self.lambda2) & (not self.lambda3) :
            return np.asarray(triples), None

        with torch.no_grad():
            conf = torch.zeros([len(triples)], dtype=torch.float32).to(self.device)
            if self.lambda1:
                conf1 = self.lambda1 * self.get_temp_conf(eEmbeds_tmp, rEmbeds_tmp, np.asarray(triples))
                conf = conf + conf1
            if self.lambda2:
                bypass = [kg_data_dict[fact]['bypass'] for fact in triples]
                bypass_PCRA = [kg_data_dict[fact]['bypass_PCRA'] for fact in triples]
                conf2 = self.lambda2 * self.get_bypass_conf(eEmbeds_tmp, rEmbeds_tmp, triples, bypass, bypass_PCRA)
                conf = conf + conf2
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
                conf3 = self.lambda3 * self.get_context_conf(np.asarray(triples), g_tmp, eEmbeds_tmp, rEmbeds_tmp)
                conf = conf + conf3

                nx_g = g_tmp.cpu().to_networkx()
                num_weakly_connected_components = nx.number_weakly_connected_components(nx_g)
                self.connected.append(num_weakly_connected_components)

                # conf = conf * conf3
            return np.asarray(triples), conf.cpu()



    def get_temp_conf(self, eEmbeds, rEmbeds, facts=None, ):
        with torch.no_grad():
            '''通过历史知识进行预测，度量当前时刻的样本的置信度，体现了时间趋势的置信度'''
            predict_conf = self.decoder(eEmbeds, rEmbeds, facts, all_candidate=False)
            return predict_conf
    
    def get_bypass_conf(self, eEmbeds, rEmbeds, triples, bypass, bypass_PCRA):
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
            for i,(s,r,o,_,_) in enumerate(triples):
                '''对每个三元组，单独计算置信度'''
                conf = get_similarity((s,r,o), bypass[i], bypass_PCRA[i])
                bypass_conf.append(conf)
            bypass_conf = torch.stack(bypass_conf)
            assert not torch.isnan(bypass_conf).any(),  "NaN detected in bypass_conf"
            return bypass_conf


    def get_context_conf(self, facts, g, eEmbeds, rEmbeds):
        with torch.no_grad():
            for e_layer in self.RGCN_layers:
                eEmbeds_context_facts_s, eEmbeds_context_facts_o = e_layer.forward_context(g, eEmbeds, rEmbeds, facts)

            conf = self.decoder.calculate_score_with_embed(eEmbeds_context_facts_s, eEmbeds_context_facts_o, rEmbeds[facts[:,1]].squeeze())
            return conf


    # def get_semantic_context_conf(self, eEmbeds, rEmbeds, triples, s_context, o_context, s_context_PCRA, o_context_PCRA):
        with torch.no_grad():
            '''结合嵌入表示,提取s和o各自的语义环境。假设思想是,如果s和o是一对正样本,那么s的上游信息构成替代s'和o的上游信息构成的替代o',也可以是一对正样本'''
            def get_path_embed(path_tmp, eEmbeds_tmp, rEmbeds_tmp):
                s_r_path = []
                for i in range(1, len(path_tmp)-1, 2): # (es,r1,e1,r2,e2,...,rl, et)
                    entity = path_tmp[i - 1]
                    relation = path_tmp[i]
                    s_r = self.decoder.calculate_h_r(eEmbeds_tmp, rEmbeds_tmp, np.array([(entity, relation, -1)]))
                    s_r_path.append(s_r)
                s_r_path = torch.stack(s_r_path, dim=1)
                path_embed = self.path_encoder(s_r_path)
                return path_embed

            def get_similarity_thread(triple_args):
                triple, s_paths, o_paths, s_paths_pcras, o_paths_pcras = triple_args           
                s_context = get_upstream_embed(s_paths, s_paths_pcras)
                o_context = get_upstream_embed(o_paths, o_paths_pcras)
                r_embed = rEmbeds[triple[1]].unsqueeze(0)
                conf = self.decoder.calculate_score_with_embed(s_context, o_context, r_embed)
                return conf
            

            def get_upstream_embed(paths_tmp, pcras_tmp):
                if len(paths_tmp)==0:  #  如果s、o之间没有其他路径，置信度为0
                    return torch.zeros((1,self.dim)).to(self.device)
                if len(paths_tmp)>=3:  #  如果s、o之间没有其他路径，置信度为0
                    paths_tmp = paths_tmp[:3]

                res = []
                for j,p in enumerate(paths_tmp):
                    R_p = pcras_tmp[j]
                    path_embed = get_path_embed(p, eEmbeds, rEmbeds)
                    res.append(R_p * path_embed)

                res = torch.vstack(res)
                assert torch.isnan(res).any, 'Nan'
                res = torch.mean(res, dim=0, keepdim=True)
                return res

            def get_similarity(triple, s_paths, o_paths, s_paths_pcras, o_paths_pcras):            
                s_context = get_upstream_embed(s_paths, s_paths_pcras)
                o_context = get_upstream_embed(o_paths, o_paths_pcras)
                r_embed = rEmbeds[triple[1]].unsqueeze(0)
                conf = self.decoder.calculate_score_with_embed(s_context, o_context, r_embed)
                return conf


            # log('')
            # context_conf = []
            # for i,(s,r,o,_,_) in enumerate(triples):
            #     '''对每个三元组，单独计算置信度'''  
            #     conf = get_similarity((s,r,o), s_context[i], o_context[i], s_context_PCRA[i], o_context_PCRA[i])
            #     context_conf.append(conf)
            # log('--')

            # log('')
            # context_conf = []
            # triples_args = []
            # for i,(s,r,o,_,_) in enumerate(triples):
            #     triples_args.append(((s,r,o), s_context[i], o_context[i], s_context_PCRA[i], o_context_PCRA[i]))
            # with ThreadPoolExecutor(max_workers=10) as executor:
            #     for conf in executor.map(get_similarity_thread, triples_args):
            #         context_conf.append(conf)
            # log('--')



            def process_batch(batch, s_context_batch, o_context_batch, s_context_PCRA_batch, o_context_PCRA_batch):
                batch_conf = []
                for i, triple in enumerate(batch):
                    s, r, o, _, _ = triple
                    conf = get_similarity((s, r, o), s_context_batch[i], o_context_batch[i], s_context_PCRA_batch[i], o_context_PCRA_batch[i])
                    batch_conf.append(conf)
                return batch_conf
            batchsize = 800
            context_conf = []
            # batches = [triples[i:i + batchsize] for i in range(0, len(triples), batchsize)]
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range(0, len(triples), batchsize):
                    batch = triples[i:min(i + batchsize, len(triples))]
                    s_context_batch = s_context[i:min(i + batchsize, len(triples))]
                    o_context_batch = o_context[i:min(i + batchsize, len(triples))]
                    s_context_PCRA_batch = s_context_PCRA[i:min(i + batchsize, len(triples))]
                    o_context_PCRA_batch = o_context_PCRA[i:min(i + batchsize, len(triples))]
                    futures.append(executor.submit(process_batch, batch, s_context_batch, o_context_batch, s_context_PCRA_batch, o_context_PCRA_batch))
                for future in futures:
                    context_conf.extend(future.result())
            # log('--')

            # s_context_path_embed = []
            # s_path_all_idx = []
            # o_context_path_embed = []
            # o_path_all_idx = []
            # log('')
            # for i,(s,r,o,_,_) in enumerate(triples):
            #     for p, R_p in zip(s_context[i], s_context_PCRA[i]):
            #         path_embed = get_path_embed(p, eEmbeds, rEmbeds)
            #         s_context_path_embed.append(R_p * path_embed)
            #         s_path_all_idx.append(i)
            #     for p, R_p in zip(o_context[i], o_context_PCRA[i]):
            #         path_embed = get_path_embed(p, eEmbeds, rEmbeds)
            #         o_context_path_embed.append(R_p * path_embed)
            #         o_path_all_idx.append(i)
            # log('--')
            # s_context_path_embed = torch.vstack(s_context_path_embed).to(self.device)
            # o_context_path_embed = torch.vstack(o_context_path_embed).to(self.device)
            # s_path_all_idx = torch.tensor(s_path_all_idx).unsqueeze(1).to(self.device)
            # o_path_all_idx = torch.tensor(o_path_all_idx).unsqueeze(1).to(self.device)

            # group_sums = torch.zeros(len(triples), self.dim).to(self.device).scatter_add(0, s_path_all_idx, s_context_path_embed)
            # group_counts = torch.zeros(len(triples), 1).to(self.device).scatter_add(0, s_path_all_idx, torch.ones_like(s_path_all_idx, dtype=torch.float32))
            # group_counts = group_counts.masked_fill(group_counts == 0, 1)
            # s_context_embed = F.sigmoid((group_sums / group_counts).squeeze(1))

            # group_sums = torch.zeros(len(triples), self.dim).to(self.device).scatter_add(0, o_path_all_idx, o_context_path_embed)
            # group_counts = torch.zeros(len(triples), 1).to(self.device).scatter_add(0, o_path_all_idx, torch.ones_like(o_path_all_idx, dtype=torch.float32))
            # group_counts = group_counts.masked_fill(group_counts == 0, 1)
            # o_context_embed = F.sigmoid((group_sums / group_counts).squeeze(1))

            # triples = np.asarray(triples)
            # r_embed_all = rEmbeds[triples[:,1]].unsqueeze(0)
            # context_conf = self.decoder.calculate_score_with_embed(s_context_embed, o_context_embed, r_embed_all)

            context_conf = torch.hstack(context_conf)
            assert not torch.isnan(context_conf).any(),  "NaN detected in context_conf"
            return context_conf

    