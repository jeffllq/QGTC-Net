import math
import torch
from torch.nn import Parameter, init
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp




class Copy_mode(nn.Module):
    def __init__(self, hidden_dim, num_e): #  
        super(Copy_mode, self).__init__()
        self.hidden_dim = hidden_dim
        self.tanh = nn.Tanh()
        self.W_s = nn.Linear(hidden_dim * 3, num_e)

    def forward(self, ent_embed, rel_embed, time_embed, copy_vocabulary):
        m_t = torch.cat((ent_embed, rel_embed, time_embed), dim=1)
        q_s = self.tanh(self.W_s(m_t))
        mask_array = np.array(copy_vocabulary == 0, dtype=float) * (-100)
        encoded_mask = torch.tensor(mask_array, dtype=torch.float32).to(q_s.device)
        score_c = q_s + encoded_mask
        return F.softmax(score_c, dim=1)

class Generate_mode(nn.Module):
    def __init__(self, input_dim, hidden_size, num_e):
        super(Generate_mode, self).__init__()
        self.W_mlp = nn.Linear(hidden_size * 3, num_e)

    def forward(self, ent_embed, rel_embed, tim_embed):
        m_t = torch.cat((ent_embed, rel_embed, tim_embed), dim=1)
        score_g = self.W_mlp(m_t)
        return F.softmax(score_g, dim=1)


class CyGNet(nn.Module):
    def __init__(self, num_e ,num_r, num_times, h_dim, time_stamp):
        super(CyGNet, self).__init__()
        self.num_e = num_e
        self.h_dim = h_dim
        self.num_r = num_r
        self.num_times = num_times + 10

        self.eEmbeds = nn.Parameter(torch.Tensor(num_e, h_dim))
        self.w_relation = nn.Parameter(torch.Tensor(num_r*2, h_dim))
        self.tim_init_embeds = nn.Parameter(torch.Tensor(1, h_dim))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.generate_mode = Generate_mode(h_dim, h_dim, self.num_e)
        self.copy_mode = Copy_mode(self.h_dim, self.num_e)
        self.reset_parameters()

        self.alpha = 0.5
        self.time_stamp = time_stamp
        self.loss_func = torch.nn.CrossEntropyLoss()
        # self.loss_func = torch.nn.functional.nll_loss()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.eEmbeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.tim_init_embeds, gain=nn.init.calculate_gain('relu'))

    def get_facts_embeds(self, quadrupleList):
        s_idx = quadrupleList[:, 0]
        r_idx = quadrupleList[:, 1]
        t_idx = quadrupleList[:, 2]
        s_embeds = self.eEmbeds[s_idx]
        r_embeds = self.w_relation[r_idx]

        T_idx = (quadrupleList[:, 3] / self.time_stamp).int()
        init_tim = torch.Tensor(self.num_times, self.h_dim)
        for i in range(self.num_times):
            init_tim[i] = torch.Tensor(self.tim_init_embeds.cpu().detach().numpy().reshape(self.h_dim)) * (i + 1)
        init_tim = init_tim.to(self.eEmbeds.device)
        T_embeds = init_tim[T_idx]
        return s_embeds, r_embeds, T_embeds

    def regularization_loss(self, reg_param):
        regularization_loss = torch.mean(self.w_relation.pow(2)) + torch.mean(self.eEmbeds.pow(2)) + torch.mean(
            self.tim_init_embeds.pow(2))
        return regularization_loss * reg_param

    def calculate_loss(self, facts_pos, facts_neg=None, confidence=None):
        s_embeds, r_embeds, T_embeds = self.get_facts_embeds(facts_pos)
        score_c=0
        score_g=0
        if self.copy:
            score_c = self.copy_mode(s_embeds, r_embeds, T_embeds, self.copy_vocabulary) + 1e-6
        if self.generate:
            score_g = self.generate_mode(s_embeds, r_embeds, T_embeds) + 1e-6
        score_pos = score_c * self.alpha + score_g * (1 - self.alpha)
        # loss = self.loss_func(score_pos, facts_pos[:,2]) 
        score_pos = torch.log(score_pos)
        loss = F.nll_loss(score_pos, facts_pos[:,2]) + self.regularization_loss(reg_param = 0.01)

        score_neg = None
        if facts_neg is not None:
            score_g = self.generate_mode(s_embeds, r_embeds, T_embeds)
            score_c = self.copy_mode(s_embeds, r_embeds, T_embeds, self.copy_vocabulary)
            score_neg = score_c * self.alpha + score_g * (1 - self.alpha)
            loss_neg = - self.loss_func(score_neg, facts_neg[:,2])
            loss = torch.mean(loss*facts_pos.shape[0]+loss_neg*facts_neg.shape[0])
        return loss, score_pos, score_neg

    def forward(self, data_history_all, kg_dict_pred=None, copy_vocabulary=None, confidence=None):
        assert len(data_history_all) == 1,  "Wrong history length for CyGNet"
        # kg_data_dict = data_history_all[0]
        # facts = np.asarray(kg_data_dict.keys())

        # s_embeds, r_embeds, T_embeds = self.get_facts_embeds(facts)
        self.copy_vocabulary = copy_vocabulary
        # score_g = self.generate_mode(s_embeds, r_embeds, T_embeds)
        # score_c = self.copy_mode(s_embeds, r_embeds, T_embeds, self.copy_vocabulary)
        # score = score_c * self.alpha + score_g * (1 - self.alpha)
        pass



    def predict(self, facts):
        s_embeds, r_embeds, T_embeds = self.get_facts_embeds(facts)
        scores_c=0
        scores_g=0
        if self.copy:
            scores_c = self.copy_mode(s_embeds, r_embeds, T_embeds, self.copy_vocabulary) + 1e-6
        if self.generate:
            scores_g = self.generate_mode(s_embeds, r_embeds, T_embeds)
        # scores_pos = scores_c * self.alpha + scores_g * (1 - self.alpha) + 1e-6
        scores = scores_c * self.alpha + scores_g * (1 - self.alpha)
        scores = torch.log(scores)
        return scores

    def get_temp_conf(self,):        
        pass

    def get_bypass_conf(self,):
        pass

    def get_context_conf(self,):
        pass
