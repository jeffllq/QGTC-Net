import torch
import pynvml
import torch.nn.functional as F
from TimeLogger import log
import TimeLogger as logger
from MyWork.src_DGAD.Config import args
import numpy as np
import os
import random
from DataHandler import DataHandler
import json
from tqdm import tqdm
from Utils import calc_mrr_hits, get_ranks
import pickle
import setproctitle
from TKGModel import TKGModel
import scipy.sparse as sp

import warnings
warnings.filterwarnings("ignore")



class Coach:
    def __init__(self, handler, device, neg_ratio) -> None:
        self.handler = handler
        # self.gpu = args.gpu
        self.device = device
        self.neg_ratio = neg_ratio
        if self.neg_ratio>0:
            log('Train on noisy data. Noisy ratio {:2d}%'.format(int(self.neg_ratio * 100)))

    '''Pretrain TKG embedding model'''
    def run(self):
        '''加载freq信息，用于filtered metrics 计算'''
        facts_filter = self.handler.trnList + self.handler.valList + self.handler.tstList
        self.facts_filter = np.concatenate(facts_filter)
        # freq_path = './SavedData/{}/history_seq'.format(args.dataset)
        # if not os.path.exists(freq_path):
        #     os.makedirs(freq_path)
        # time_tail_seq_list = []
        # time_tail_freq_list = []
        # for idx in range(len(self.handler.allTime)):
        #     time_tail_seq = sp.load_npz(freq_path +'/h_r_history_seq_{}.npz'.format(args.dataset, idx))
        #     time_tail_seq_list.append(time_tail_seq)

        # for idx in range(len(self.handler.allTime)):
        #     if idx == 0:
        #         time_tail_freq = sp.csr_matrix(([], ([], [])), shape=(self.handler.num_e * (self.handler.num_r * 2), self.handler.num_e))
        #         time_tail_freq_list.append(time_tail_freq)
        #         continue
        #     else:
        #         time_tail_freq = time_tail_seq_list[idx - 1] + time_tail_freq_list[idx - 1]
        #         time_tail_freq_list.append(time_tail_freq)
 

        '''TKG embedding model, use learned embeddings to reason future facts'''
        self.model = TKGModel(self.handler.num_e, self.handler.num_r, args.dim, args.layer_n, 
                              args.encoder, args.drop_rate, self.device, args.lambda1, args.lambda2, 
                              args.lambda3).to(self.device) # TKG embedding model
        self.opt = torch.optim.Adam(self.model.parameters(), lr = args.lr, weight_decay=0)

        if args.mode == 'test':
            log('Test: directly test well-trained model...')
            if os.path.exists(model_state_file):
                checkpoint_pretrain = torch.load(model_state_file)
                self.model.load_state_dict(checkpoint_pretrain['model'])
            else:
                log('Error: Model not exist!')

        # log('Start')
        early_stop = 0
        Best_MRR = 0
        for ep in range(1, args.epoch):
            if args.mode == 'test':
                break
            tstFlag = (ep % args.tstEpoch == 0)
            self.model.connected=[]
            self.trainEpoch() # train epoch
            # print("Connected",np.mean(self.model.connected)/self.handler.num_e)
            
            if True:
                MRR, Hits1, Hits3, Hits10 = self.testEpoch(mode='valid')
                MRR = int(MRR*10000)/10000
                if MRR>Best_MRR:
                    Best_MRR = MRR
                    torch.save({'model':self.model.state_dict(), 'epoch':ep}, model_state_file)
                    early_stop = 0
                else:
                    early_stop+=1
            if early_stop == 3:
                break
            log('Train: Epochs %d/%d: Valid=> Best_MRR = %.4f, MRR = %.4f, Hits1 = %.4f, Hits3 = %.4f, Hits10 = %.4f' % (ep, args.epoch, Best_MRR, MRR, Hits1, Hits3, Hits10), save=False, oneline=False)
        
        MRR, Hits1, Hits3, Hits10 = self.testEpoch(mode='test')
        log('DGAD_noise{}_histlen{}_temp{}_bypass{}_context{}'.format(args.neg_ratio, args.history_len, args.lambda1, args.lambda2, args.lambda3))
        log('Test: MRR = %.4f, Hits1 = %.4f, Hits3 = %.4f, Hits10 = %.4f' % (MRR, Hits1, Hits3, Hits10), save=False, oneline=False)
        # log('Model pretrain finished!')

    def trainEpoch(self):
        self.model.train()
        idx_list = [_ for _ in range(len(self.handler.trnList))]
        random.shuffle(idx_list)
        for idx in idx_list:
            if idx < args.history_len:
                continue
            # data_history = self.handler.trnList[max(0, idx - args.history_len): idx]
            # # dgl_history = self.handler.trnDglList[max(0, idx - args.history_len): idx]
            # if self.neg_idx > 0:  # add noise data
            #     data_history_dirt = self.handler.trnListDirt_ratios[self.neg_idx-1][max(0, idx - args.history_len): idx]# Noise Input
            #     data_history = [ np.concatenate((data_history[k],data_history_dirt[k]), axis=0) for k in range(len(data_history))]
            #     # dgl_history = [self.handler.build_dgl_graph(self.handler.num_e, self.handler.num_r, quads).to(args.gpu) for quads in data_history]
            
             
            # dgl_pred = self.handler.trnDglList[idx]

            data_history_all = self.handler.trnListDirt[max(0, idx - args.history_len): idx]

            data_pred = torch.LongTensor(self.handler.trnList[idx]).to(self.device) # calculate loss on raw data
 
            self.opt.zero_grad()
            eEmbeds_pred, rEmbeds_pred= self.model(data_history_all)
            assert not torch.isnan(eEmbeds_pred).any(), 'Nan in eEmbeds_pred'
            assert not torch.isnan(rEmbeds_pred).any(), 'Nan in rEmbeds_pred'
            loss,_,_ = self.model.calculate_loss(eEmbeds_pred, rEmbeds_pred, data_pred)
            # log("loss: %.6f" %(loss.item()), save=False, oneline=True)
            loss.backward()
            assert not torch.isnan(self.model.eEmbeds.grad).any(), 'Nan in eEmbeds_pred'
            assert not torch.isnan(self.model.rEmbeds.grad).any(), 'Nan in rEmbeds_pred'

            self.opt.step()
            assert not torch.isnan(self.model.eEmbeds).any(), 'Nan in eEmbeds_pred'
            assert not torch.isnan(self.model.rEmbeds).any(), 'Nan in rEmbeds_pred'

    def testEpoch(self, mode='test'):
        self.model.eval()
        Ranks = []
        with torch.no_grad():
            if mode=='valid':
                idx_list = [_ for _ in range(len(self.handler.valList))]
            elif mode == 'test':
                idx_list = [_ for _ in range(len(self.handler.tstList))]

            for idx in idx_list:
                if mode == 'valid':
                    if idx >= args.history_len:
                        data_history_all = self.handler.valListDirt[max(0, idx - args.history_len): idx]
                    else:
                        data_history_all = self.handler.trnListDirt[ (idx - args.history_len): ] + self.handler.valListDirt[max(0, idx - args.history_len): idx] 
                    data_pred = torch.LongTensor(self.handler.valList[idx]).to(self.device)
                elif mode == 'test':
                    if idx >= args.history_len:
                        data_history_all = self.handler.tstListDirt[max(0, idx - args.history_len): idx]# Noise Input
                    else:
                        data_history_all = (self.handler.trnListDirt + self.handler.valListDirt)[ (idx - args.history_len): ] + self.handler.tstListDirt[max(0, idx - args.history_len): idx]
                    data_pred = torch.LongTensor(self.handler.tstList[idx]).to(self.device)

                self.opt.zero_grad()
                eEmbeds_pred, rEmbeds_pred= self.model(data_history_all)
                _, score_pred, _ = self.model.calculate_loss(eEmbeds_pred, rEmbeds_pred, data_pred)
                # ranks = sort_and_rank(score_pred.cpu(), data_pred[:,2].cpu())
                if mode =='test':
                    ranks = get_ranks(data_pred, score_pred, batch_size=1000, filter=args.filter, num_e=self.handler.num_e, facts_filter=self.facts_filter)
                else:
                    ranks = get_ranks(data_pred, score_pred, batch_size=1000, filter=0, num_e=self.handler.num_e, facts_filter=self.facts_filter)


                Ranks.extend(ranks)
            MRR, Hits1, Hits3, Hits10 = calc_mrr_hits(torch.tensor(Ranks))
        return MRR, Hits1, Hits3, Hits10


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

      
if __name__ == '__main__':
    print('\n')
    setproctitle.setproctitle('{}_{}_{}_{}_{}_{}_{}'.format(args.mode, args.encoder, args.dataset, args.history_len, args.lambda1, args.lambda2, args.lambda3)) # 读取环境变量并设置进程名称
    log('Start {}'.format(args.dataset))
    os.chdir(os.getcwd().split('/MyWork')[0]+'/MyWork')
    model_dir = './SavedModel/{}'.format(args.dataset)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)


    if args.encoder == 'RGCN':
        model_state_file = model_dir+'/DGAD_noise{}_{}_histlen{}_temp{}_bypass{}_context{}'.format(args.neg_ratio, args.noise_strategy, args.history_len, args.lambda1, args.lambda2, args.lambda3)
    else:
        log('Pleas define model state file name')
    seed_it(args.seed)
    log('Device: {}'.format(args.device))
    logger.saveDefault = True

    # log('Start')
    data_file = './SavedData/{}/{}_handler_ratio{}_strategy{}.pkl'.format(args.dataset, args.dataset, args.neg_ratio, args.noise_strategy)
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            handler = pickle.load(f)
        log("Load saved data!")
    log("Load {} :  num_e:{}  num_r:{}".format(args.dataset, handler.num_e, handler.num_r))
    
    # '''TKG Link prediction task'''
    coach = Coach(handler, args.device, args.neg_ratio)
    coach.run()
    log('Finish')


