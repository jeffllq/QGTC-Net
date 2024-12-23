import math
import torch
from torch.nn import Parameter, init
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class RENET(nn.Module):
    def __init__(self) -> None:
        super(RENET, self).__init__()
        pass
    
    def calculate_loss(self, facts_pos, facts_neg=None):
        pass

    def forward(self, data_history_all, copy_vocabulary=None):
        assert len(data_history_all) == 1,  "Wrong history length for CyGNet"
        kg_data_dict = data_history_all[0]
        facts = np.asarray(kg_data_dict.keys())
        pass