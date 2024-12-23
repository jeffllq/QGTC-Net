import math
import torch
from torch.nn import Parameter, init
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class DisMult(nn.Module):
    def __init__(self, num_e ,num_r, h_dim) -> None:
        super(DisMult, self).__init__()
        self.num_e = num_e
        self.num_r = num_r
        self.eEmbeds = nn.Parameter(torch.Tensor(num_e, h_dim))
        self.rEmbeds = nn.Parameter(torch.Tensor(num_r*2, h_dim))
        nn.init.xavier_uniform_(self.eEmbeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rEmbeds, gain=nn.init.calculate_gain('relu'))
        self.loss_func = torch.nn.CrossEntropyLoss()
    
    def calculate_loss(self, facts_pos, facts_neg=None, confidence=0):
        if type(facts_pos)==dict:
            facts_pos = torch.LongTensor(list(facts_pos.keys())).to(self.eEmbeds.device)
        s_embeds = self.eEmbeds[facts_pos[:,0]]
        r_embeds = self.eEmbeds[facts_pos[:,1]]
        s_r_embeds = s_embeds * r_embeds
        # score_pos = torch.sum(s_r_embeds * self.eEmbeds.t(), -1)
        score_pos = torch.mm(s_r_embeds , self.eEmbeds.t())
        loss = self.loss_func(score_pos, facts_pos[:,2])

        score_neg = None
        if facts_neg is not None:
            facts_pos = torch.LongTensor(list(facts_neg.keys())).to(self.eEmbeds.device)
            s_embeds = self.eEmbeds[facts_neg[:,0]]
            r_embeds = self.eEmbeds[facts_neg[:,1]]
            s_r_embeds = s_embeds * r_embeds
            # score_neg = torch.sum(s_r_embeds * self.eEmbeds.t(), -1)
            score_neg = torch.mm(s_r_embeds , self.eEmbeds.t())
            loss_neg = - self.loss_func(score_neg, facts_neg[:,2])
            loss = torch.mean(loss*facts_pos.shape[0]+loss_neg*facts_neg.shape[0])
        return loss, score_pos, score_neg

    def predict(self, facts):
        if type(facts)==dict:
            facts = torch.LongTensor(list(facts.keys())).to(self.eEmbeds.device)
        s_embeds = self.eEmbeds[facts[:,0]]
        r_embeds = self.eEmbeds[facts[:,1]]
        s_r_embeds = s_embeds * r_embeds
        scores = torch.mm(s_r_embeds , self.eEmbeds.t())
        return scores

    def forward(self, data_history_all, kg_dict_pred=None, copy_vocabulary=None, confidence=0):
        assert len(data_history_all) == 1,  "Wrong history length for TransE"
        # kg_data_dict = data_history_all[0]
        # facts = np.asarray(kg_data_dict.keys())
        # do nothing
        pass