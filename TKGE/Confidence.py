import numpy as np
import pandas as pd
import torch
import dgl
from ..Utils import build_dgl_graph
import networkx as nx

class Confidence():
    def __init__(self) -> None:
        pass

    def temp_conf(self):
        pass

    def bypass_conf(self):
        pass

    def context_conf(self):
        with torch.no_grad():
            for e_layer in self.RGCN_layers:
                eEmbeds_context_facts_s, eEmbeds_context_facts_o = e_layer.forward_context(g, eEmbeds, rEmbeds, facts)

            conf = self.decoder.calculate_score_with_embed(eEmbeds_context_facts_s, eEmbeds_context_facts_o, rEmbeds[facts[:,1]].squeeze())
            return conf
    
    def forward(self, eEmbeds, rEmbeds, kg_data_dict=None):
        facts = list(kg_data_dict.keys())  # [(s,r,o,T,label),...]

        if (not self.lambda1) & (not self.lambda2) & (not self.lambda3) :
            return np.asarray(facts), None

        with torch.no_grad():
            conf = torch.zeros([len(facts)], dtype=torch.float32).to(self.device)
            if self.lambda1:
                conf1 = self.lambda1 * self.get_predict_conf(eEmbeds, rEmbeds, np.asarray(facts))
                conf = conf + conf1
            if self.lambda2:
                bypass = [kg_data_dict[fact]['bypass'] for fact in facts]
                bypass_PCRA = [kg_data_dict[fact]['bypass_PCRA'] for fact in facts]
                conf2 = self.lambda2 * self.get_bypass_context_conf(eEmbeds, rEmbeds, facts, bypass, bypass_PCRA)
                conf = conf + conf2
            if self.lambda3:  
                df = pd.DataFrame(facts, columns=['s', 'r', 'o', 't', 'label'])
                df_unique = df.drop_duplicates(subset=['s', 'o'])
                facts = np.asarray(df_unique.values.tolist())
                g_tmp = build_dgl_graph(self.num_e, self.num_r, facts, None, inverse=False).to(self.device)  # 处理后，每两个节点之间有且仅有一条边
                conf3 = self.lambda3 * self.get_community_conf(np.asarray(facts), g_tmp, eEmbeds, rEmbeds)
                conf = conf + conf3

            return np.asarray(facts), conf.cpu()
        

