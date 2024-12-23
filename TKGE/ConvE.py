import math
import torch
from torch.nn import Parameter, init
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class ConvE(nn.Module):
    def __init__(self, num_e ,num_r, h_dim, channels=50, kernel_size=3) -> None:
        super(ConvE, self).__init__()
        self.num_e = num_e
        self.num_r = num_r
        self.dropout = 0.2

        self.eEmbeds = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_e, h_dim)))
        self.rEmbeds = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_r*2, h_dim)))

        self.inp_drop = torch.nn.Dropout(self.dropout)
        self.hidden_drop = torch.nn.Dropout(self.dropout)
        self.feature_map_drop = torch.nn.Dropout(self.dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(h_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_e)))
        self.fc = torch.nn.Linear(h_dim * channels, h_dim)

        self.loss_func = torch.nn.CrossEntropyLoss()
    
    def calculate_loss(self, facts_pos, facts_neg=None, confidence=0):
        scores_pos = self.predict(facts_pos)

        loss = self.loss_func(scores_pos, facts_pos[:,2])

        return loss, scores_pos, None
    
    def predict(self, facts, confidence=0):
        eEmbeds = F.tanh(self.eEmbeds)
        rEmbeds = self.rEmbeds
        batch_size = facts.shape[0]
        e1_embedded = eEmbeds[facts[:, 0]].unsqueeze(1)
        rel_embedded = rEmbeds[facts[:, 1]].unsqueeze(1)
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
        scores = torch.mm(x, eEmbeds.transpose(1, 0))
        scores = F.relu(scores) + 1e-5
        return scores

    def forward(self, data_history_all, kg_dict_pred=None, copy_vocabulary=None, confidence=0):
        assert len(data_history_all) == 1,  "Wrong history length for ConvE"
        self.eEmbeds = F.normalize(self.eEmbeds)
        self.rEmbeds = F.normalize(self.rEmbeds)
        # kg_data_dict = data_history_all[0]
        # facts = np.asarray(kg_data_dict.keys())
        # do nothing
        pass