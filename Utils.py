import torch
import dgl
import numpy as np
from collections import defaultdict

"""Define evaluation metrics."""

def get_ranks(test_facts, scores, batch_size=1000, filter=0, num_e=0, facts_filter=None):
    ranks = []
    if filter:
        test_facts = test_facts.to('cpu').numpy()
        assert facts_filter is not None, "facts_filter should not be none"
        filter_facts_set = set((item[0], item[1], item[2]) for item in facts_filter)

        for idx in range(len(test_facts)):
            tmp_fact = test_facts[idx]
            score = scores[idx]
            filter_t = []
            for t_ in range(num_e):
                if (tmp_fact[0], tmp_fact[1], t_) in filter_facts_set:
                    filter_t.append(t_)
            if tmp_fact[2] in filter_t:
                filter_t.remove(tmp_fact[2])
            score[np.asarray(filter_t)] = -np.inf

            _, indices = torch.sort(score, descending=True)
            indice = torch.nonzero(indices == tmp_fact[2]).item()

            ranks.append(indice)
        ranks = torch.tensor(ranks)
        ranks += 1
    else:
        targets = test_facts[:, 2]
        ranks = sort_and_rank(scores, targets)
        ranks += 1
    return ranks



def sort_and_rank(scores, targets):
    _, indices = torch.sort(scores, dim=1, descending=True)  # bigger score is better
    indices = torch.nonzero(indices == targets.view(-1, 1))
    indices = indices[:, 1].view(-1)
    indices.tolist()
    # print(indices)
    return indices

# return MRR (raw), and Hits @ (1, 3, 10)
def calc_mrr_hits(ranks, hits=[1,3,10]):
    with torch.no_grad():
        if not torch.is_tensor(ranks):
            ranks = torch.tensor(ranks)
        mrr = torch.mean(1.0 / ranks.float())
        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())
    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

def F1(scores, labels):
    
    pass



"""Below there are functions for graph process"""
def r2e(num_e, num_r,facts,inverse=True):
    src, rel, dst,_,_= facts.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    if inverse:
        uniq_r = np.concatenate((uniq_r, uniq_r+num_r))
    
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst,_,_) in enumerate(facts):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        if inverse:
            r_to_e[rel+num_r].add(src)
            r_to_e[rel+num_r].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r]))) #  [(idx_start, idx_end), ...]
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx

def build_dgl_graph(num_e, num_r, facts, weights=None, inverse=False):
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    src, rel, dst,_,_= facts.transpose()
    # if inverse:
    #     src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    #     rel = np.concatenate((rel, rel + num_r))  #  Add inverse facts
    g = dgl.graph((src, dst),num_nodes= num_e)
    # g.add_nodes(num_e)
    # g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_e, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)
    
    if weights is not None:
        if inverse == True:
            g.edata['weight'] = torch.cat((weights, weights))
        else:
            g.edata['weight'] = weights

    uniq_r, r_len, r_to_eList = r2e(num_e, num_r, facts, inverse)
    g.uniq_r = uniq_r
    g.r_to_eList = r_to_eList
    g.r_len = r_len
    g.r_to_eList = torch.from_numpy(np.array(r_to_eList))
    return g