import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import numpy as np


class PathEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(PathEncoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # self.hidden_dim = hidden_dim

    def forward(self, path_embeddings):
        lstm_out, _ = self.lstm(path_embeddings)
        return lstm_out[:, -1, :]

class RGCN(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate) -> None:
        super(RGCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))
        self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))
        # self.skip_connect_weight = nn.Parameter()
        # self.skip_connect_bias = nn.Parameter()
        self.activation = F.rrelu
        self.dropout = nn.Dropout(drop_rate)

        
    def propagate(self, g, rEmbeds):
        g.update_all(lambda x: self.msg_func(x, rEmbeds), fn.sum(msg='msg', out='h'), self.apply_func)

    def msg_func(self, edges, rEmbeds):
        relation = rEmbeds.index_select(0, edges.data['type']).view(-1, self.out_feat)
        node = edges.src['h'].view(-1, self.out_feat)

        if 'weight' in edges.data:
            edge_weight = edges.data['weight'].view(-1,1)
        else:
            edge_weight = torch.ones(node.shape[0], 1).to(rEmbeds.device)
        
        # msg = (node + relation) * edge_weight
        msg = node * edge_weight+ relation

        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}
    
    def forward(self, g, eEmbeds, rEmbeds):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = eEmbeds[node_id]
        # self.rEmbeds = rEmbeds
        masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).to(g.device),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
        loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)  # 独立的节点，周围没有邻居，通过 evolve直接更新
        loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
        self.propagate(g, rEmbeds)
        eEmbeds_next = g.ndata['h']
        eEmbeds_next = eEmbeds_next+loop_message
        eEmbeds_next = self.activation(eEmbeds_next)
        eEmbeds_next = self.dropout(eEmbeds_next)
        eEmbeds_next = F.normalize(eEmbeds_next)
        g.ndata['h'] = eEmbeds_next
    


    """
    往下是置信度计算后的GNN传播
    """
    def propagate_context(self, g, rEmbeds):
        g.update_all(lambda x: self.msg_func_context(x, rEmbeds), fn.sum(msg='msg', out='h'))

    def msg_func_context(self, edges, rEmbeds):
        # relation = rEmbeds.index_select(0, edges.data['type']).view(-1, self.out_feat)
        node = edges.src['h'].view(-1, self.out_feat)
        msg = node 
        # msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def forward_context(self, g, eEmbeds, rEmbeds, facts):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = eEmbeds[node_id]

        # masked_index = torch.masked_select(torch.arange(0, g.number_of_nodes(), dtype=torch.long).to(g.device), (g.in_degrees(range(g.number_of_nodes())) > 0))
        # loop_message = g.ndata['h']
        # loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]

        self.propagate_context(g, rEmbeds)
        eEmbeds_context = g.ndata['h']
        eEmbeds_context = self.activation(eEmbeds_context)
        eEmbeds_context = self.dropout(eEmbeds_context)

        norm_s = g.ndata['norm'][facts[:,0]]
        eEmbeds_context_facts_s = (eEmbeds_context[facts[:, 0]].squeeze() - eEmbeds[facts[:, 2]].squeeze() + eEmbeds[facts[:, 0]].squeeze())/(norm_s+1)
        eEmbeds_context_facts_s = F.normalize(eEmbeds_context_facts_s)
        
        norm_o = g.ndata['norm'][facts[:,2]]
        eEmbeds_context_facts_o = (eEmbeds_context[facts[:, 2]].squeeze() - eEmbeds[facts[:, 0]].squeeze() + eEmbeds[facts[:, 2]].squeeze())/(norm_o+1)
        eEmbeds_context_facts_o = F.normalize(eEmbeds_context_facts_o)

        return eEmbeds_context_facts_s, eEmbeds_context_facts_o




class GCN(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate) -> None:
        super(GCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))
        self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))
        # self.skip_connect_weight = nn.Parameter()
        # self.skip_connect_bias = nn.Parameter()
        self.activation = F.leaky_relu
        self.dropout = nn.Dropout(drop_rate)

        
    def propagate(self, g, rEmbeds):
        g.update_all(lambda x: self.msg_func(x, rEmbeds), fn.sum(msg='msg', out='h'), self.apply_func)

    def msg_func(self, edges, rEmbeds):
        # relation = rEmbeds.index_select(0, edges.data['type']).view(-1, self.out_feat)
        node = edges.src['h'].view(-1, self.out_feat)

        if 'weight' in edges.data:
            edge_weight = edges.data['weight'].view(-1,1)
        else:
            edge_weight = torch.ones(node.shape[0], 1).to(rEmbeds.device)
        
        # msg = (node + relation) * edge_weight
        msg = node * edge_weight

        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}
    
    def forward(self, g, rEmbeds):
        # self.rEmbeds = rEmbeds
        masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).to(g.device),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
        loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
        loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
        self.propagate(g, rEmbeds)
        eEmbeds_next = g.ndata['h']
        eEmbeds_next = eEmbeds_next+loop_message
        eEmbeds_next = self.activation(eEmbeds_next)
        eEmbeds_next = self.dropout(eEmbeds_next)
        g.ndata['h'] = eEmbeds_next



class RGAT_confidence(nn.Module):
    pass

class RGDC(nn.Module):  # Graph diffusion convolution layer
    def __init__(self, in_feat, out_feat, num_r, diffusion_steps=1, drop_rate=0.0) -> None:
        super(RGDC, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_r = num_r
        self.diffusion_steps = diffusion_steps


        self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        # 关系特异的权重
        self.relation_weights = nn.Parameter(torch.Tensor(num_r * 2, in_feat, out_feat))
        nn.init.xavier_uniform_(self.relation_weights, gain=nn.init.calculate_gain('relu'))

        # 循环连接权重
        self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(drop_rate)
        self.activation = F.relu

    def propagate(self, g, rFeatures):
        for _ in range(self.diffusion_steps):
            # g.update_all(lambda edges: {'msg': self.message_func(edges, rFeatures)}, fn.sum('msg', 'h'), self.apply_func )
            g.update_all(lambda x: self.message_func(x, rFeatures), fn.sum('msg', 'h'), self.apply_func )

    def message_func(self, edges, rFeatures):
        # 根据边的类型选择相应的关系权重
        relation = rFeatures.index_select(0, edges.data['type']).view(-1, self.out_feat)
        # node = eFeatures[edges.src['id']].view(-1, self.out_feat)
        node = edges.src['h'].view(-1, self.out_feat)

        # weight = self.relation_weights.index_select(0, edges.data['type']).view(-1, self.out_feat)
        # msg = node * weight + relation

        weight = self.relation_weights[edges.data['type']]
        msg = torch.bmm(node.unsqueeze(1), weight).squeeze(1) + relation
        return {'msg': msg}
    
    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}
    
    def forward(self, g, rFeatures):
        self.propagate(g, rFeatures)
        # 应用循环权重
        loop_message = torch.mm(g.ndata['h'], self.loop_weight)

        # 更新节点特征
        g.ndata['h'] = g.ndata['h'] + loop_message
        g.ndata['h'] = self.activation(g.ndata['h'])
        g.ndata['h'] = self.dropout(g.ndata['h'])
    

        # masked_index = torch.masked_select(
        #         torch.arange(0, g.number_of_nodes(), dtype=torch.long).to(rFeatures.device),
        #         (g.in_degrees(range(g.number_of_nodes())) > 0))
        # loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
        # loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
        # self.propagate(g, rFeatures)
        # eEmbeds_next = g.ndata['h']
        # eEmbeds_next = eEmbeds_next+loop_message
        # eEmbeds_next = self.activation(eEmbeds_next)
        # eEmbeds_next = self.dropout(eEmbeds_next)
        # g.ndata['h'] = eEmbeds_next


"""GNN with diffusion idea"""
class RHDC(nn.Module):
    def __init__(self, in_feats, out_feats, num_r, diffusion_time, drop_rate=0.0):
        super(RHDC, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_r = num_r
        self.diffusion_time = diffusion_time

        # 关系特异的权重
        self.relation_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_feats, out_feats))
            for _ in range(num_r * 2)
        ])
        for weight in self.relation_weights:
            nn.init.xavier_uniform_(weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(drop_rate)
        self.activation = F.relu

    def forward(self, g, features):
        g = dgl.add_self_loop(g)
        
        # 为每种关系类型创建子图
        outputs = torch.zeros_like(features)
        for rel_type in range(self.num_r * 2):
            edges = (g.edata['type'] == rel_type).nonzero(as_tuple=True)[0]
            if len(edges) > 0:
                subgraph = g.edge_subgraph(edges, preserve_nodes=True)
                sub_features = features.clone()

                # 执行模拟的热核扩散过程
                for _ in range(int(self.diffusion_time)):
                    subgraph.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'h'))
                    sub_features = subgraph.ndata['h']

                # 应用关系特异的权重
                weight = self.relation_weights[rel_type]
                sub_features = torch.mm(sub_features, weight)
                outputs += sub_features

        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        g.ndata['h'] = outputs


"""GNN with KAN idea"""
class GKAN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, grid_feat, num_layers, use_bias=False):
        super().__init__()
        self.num_layers = num_layers
        self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.lins = torch.nn.ModuleList()
        for i in range(num_layers):
            self.lins.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
        self.lins.append(nn.Linear(hidden_feat, out_feat, bias=False))
 
    def forward(self, x, adj):
        x = self.lin_in(x)
        for layer in self.lins[:self.num_layers - 1]:
            x = layer(torch.spmm(adj, x))
        x = self.lins[-1](x)
        return x.log_softmax(dim=-1)
     
class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=300, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                        (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        y = y.view(outshape)
        return y


