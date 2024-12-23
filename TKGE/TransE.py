import math
import torch
from torch.nn import Parameter, init
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class TransE(nn.Module):
    def __init__(self, num_e ,num_r, h_dim) -> None:
        super(TransE, self).__init__()
        self.num_e = num_e
        self.num_r = num_r
        self.eEmbeds = nn.Parameter(torch.Tensor(num_e, h_dim))
        self.rEmbeds = nn.Parameter(torch.Tensor(num_r*2, h_dim))
        nn.init.xavier_uniform_(self.eEmbeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rEmbeds, gain=nn.init.calculate_gain('relu'))
        self.loss_func = torch.nn.CrossEntropyLoss()
    
    def calculate_loss(self, facts_pos, facts_neg=None, confidence=0):
        s_embeds = self.eEmbeds[facts_pos[:,0]]
        r_embeds = self.eEmbeds[facts_pos[:,1]]
        s_r_embeds = s_embeds + r_embeds
        scores_pos = torch.sum(torch.abs(s_r_embeds.unsqueeze(1)  - self.eEmbeds.unsqueeze(0) ), dim=-1).view(-1, self.num_e)

        loss = self.loss_func(scores_pos, facts_pos[:,2])

        return loss, scores_pos, None
    
    def predict(self, facts, confidence=0):
        s_embeds = self.eEmbeds[facts[:,0]]
        r_embeds = self.eEmbeds[facts[:,1]]
        s_r_embeds = s_embeds + r_embeds
        scores = torch.sum(torch.abs(s_r_embeds.unsqueeze(1)  - self.eEmbeds.unsqueeze(0) ), dim=-1).view(-1, self.num_e)
        return scores

    def forward(self, data_history_all, kg_dict_pred=None, copy_vocabulary=None, confidence=0):
        assert len(data_history_all) == 1,  "Wrong history length for TransE"
        # kg_data_dict = data_history_all[0]
        # facts = np.asarray(kg_data_dict.keys())
        # do nothing
        pass