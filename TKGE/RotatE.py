import math
import torch
from torch.nn import Parameter, init
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class RotatE(nn.Module):
    def __init__(self, num_e ,num_r, hidden_dim, gamma=12.0) -> None:
        super(RotatE, self).__init__()
        self.num_e = num_e
        self.num_r = num_r
        self.embedding_dim = hidden_dim
        # self.eEmbeds = nn.Parameter(torch.Tensor(num_e, hidden_dim))
        # self.rEmbeds = nn.Parameter(torch.Tensor(num_r*2, hidden_dim))
        # nn.init.xavier_uniform_(self.eEmbeds, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.rEmbeds, gain=nn.init.calculate_gain('relu'))

        self.epsilon = 2.0
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        self.entity_dim = hidden_dim*2 
        self.relation_dim = hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(num_e, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(num_r*2, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.loss_func = torch.nn.CrossEntropyLoss()
    
    def calculate_loss(self, facts_pos, facts_neg=None, confidence=0):
        scores_pos = self.predict(facts_pos)   
        loss = self.loss_func(scores_pos, facts_pos[:,2])
        return loss, scores_pos, None
    
    def predict(self, facts, confidence=0):
        head = self.entity_embedding[facts[:, 0]]  # (batch_size, embedding_dim)
        relation = self.relation_embedding[facts[:, 1]]  # (batch_size, embedding_dim)
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=1)

        # 获取所有尾实体的嵌入
        all_tail = self.entity_embedding  # (num_entities, embedding_dim)
        re_tail, im_tail = torch.chunk(all_tail, 2, dim=1)

        # 将关系相位映射到 [-π, π] 区间
        phase_relation = relation / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        # 批量计算 (head, relation) 与所有实体的 score
        re_score = re_head.unsqueeze(1) * re_relation.unsqueeze(1) - im_head.unsqueeze(1) * im_relation.unsqueeze(1)
        im_score = re_head.unsqueeze(1) * im_relation.unsqueeze(1) + im_head.unsqueeze(1) * re_relation.unsqueeze(1)
        
        # 计算得分差异
        re_score = re_score - re_tail.unsqueeze(0)
        im_score = im_score - im_tail.unsqueeze(0)

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        scores = self.gamma.item() - score.sum(dim = 2)

        return scores

    def forward(self, data_history_all, kg_dict_pred=None, copy_vocabulary=None, confidence=0):
        assert len(data_history_all) == 1,  "Wrong history length for TransE"
        # kg_data_dict = data_history_all[0]
        # facts = np.asarray(kg_data_dict.keys())
        # do nothing
        pass