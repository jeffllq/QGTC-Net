import torch
import pynvml
import torch.nn.functional as F
from TimeLogger import log
import TimeLogger as logger
import numpy as np
import os
import random
from DataHandler_TKGR import DataHandler
import json
from tqdm import tqdm
from Utils import calc_mrr_hits, get_ranks
import pickle
import setproctitle
import scipy.sparse as sp
from Config_TKGR import args


from TKGE.REGCN import REGCN
from TKGE.CyGNet import CyGNet
from TKGE.DisMult import DisMult
from TKGE.TransE import TransE
from TKGE.ConvE import ConvE
from TKGE.RotatE import RotatE


import warnings
warnings.filterwarnings("ignore")



class Coach:
    def __init__(self, handler_raw, handler, device, neg_ratio) -> None:
        self.handler_raw = handler_raw
        self.handler = handler
        self.batch_size  = 256 if (args.model).upper() != ('ROTATE').upper() else 128
        # self.gpu = args.gpus
        self.device = device
        self.neg_ratio = neg_ratio
        if self.neg_ratio>0:
            log('Train on noisy data. Noisy ratio {:2d}%'.format(int(self.neg_ratio * 100)))

    def run(self):
        '''拼接所有的facts 用于filtered metrics 计算'''
        facts_filter = self.handler.trnList + self.handler.valList + self.handler.tstList
        self.facts_filter = np.concatenate(facts_filter)

        '''TKG embedding model, use learned embeddings to reason future facts'''
        self.loss_check=False
        if (args.model).upper() == ('REGCN').upper():  
            self.model = REGCN(self.handler.num_e, self.handler.num_r, args.dim, args.layer_n, args.layer_norm,
                                args.encoder, args.drop_rate, self.device, 
                                args.init_method,
                                args.init_std,
                                args.lambda1, args.lambda2, args.lambda3).to(self.device) # TKG embedding model
            # self.loss_check = True  # 是否筛选预测的标签，训练标签的改变，来引导模型的学习
        elif (args.model).upper() == ('CYGNET').upper(): 
            time_stamp = {'YAGO': 1,'WIKI': 1,'ICEWS14': 1,'ICEWS14cut10': 1,'ICEWS14cut20': 1,'ICEWS14cut50': 1,'ICEWS14s': 1}
            self.model = CyGNet(self.handler.num_e, self.handler.num_r, len(self.handler.allTime), args.dim, time_stamp[args.dataset]).to(self.device)
            freq_path = './SavedData/{}/history_seq{}'.format(args.dataset, args.neg_ratio)
            if not os.path.exists(freq_path):
                os.makedirs(freq_path)
            time_tail_seq_list = []
            self.time_tail_freq_list = []
            for idx in range(len(self.handler.allTime)):
                time_tail_seq = sp.load_npz(freq_path +'/h_r_history_seq_{}.npz'.format(idx))
                time_tail_seq_list.append(time_tail_seq)
            for idx in range(len(self.handler.allTime)):
                if idx == 0:
                    time_tail_freq = sp.csr_matrix(([], ([], [])), shape=(self.handler.num_e * (self.handler.num_r * 2), self.handler.num_e))
                    self.time_tail_freq_list.append(time_tail_freq)
                    continue
                else:
                    time_tail_freq = time_tail_seq_list[idx - 1] + self.time_tail_freq_list[idx - 1]
                    self.time_tail_freq_list.append(time_tail_freq)

            self.model.copy=True
            self.model.generate=True
            if self.model.copy & self.model.generate:
                log("Copy and generation")
            if self.model.copy & (not self.model.generate):
                log("only Copy")
            if (not self.model.copy) & self.model.generate:
                log("only Generation")

        elif (args.model).upper() == ('DISMULT').upper():
            self.model = DisMult(self.handler.num_e, self.handler.num_r, args.dim).to(self.device)
        elif (args.model).upper() == ('TRANSE').upper(): 
            self.model = TransE(self.handler.num_e, self.handler.num_r, args.dim).to(self.device)
        elif (args.model).upper() == ('ROTATE').upper(): 
            self.model = RotatE(self.handler.num_e, self.handler.num_r, args.dim).to(self.device)
        elif (args.model).upper() == ('CONVE').upper(): 
            self.model = ConvE(self.handler.num_e, self.handler.num_r, args.dim).to(self.device)


        self.opt = torch.optim.Adam(self.model.parameters(), lr = args.lr, weight_decay=0)
        self.conf_threshold = args.conf_threshold

        if args.mode == 'test':
            log('Test: directly test well-trained model...')
            if os.path.exists(model_state_file):
                MRR, Hits1, Hits3, Hits10 = self.testEpoch(mode='test', model_state_file=model_state_file)
                log('Test: MRR = %.2f, Hits1 = %.2f, Hits3 = %.2f, Hits10 = %.2f' % (MRR*100, Hits1*100, Hits3*100, Hits10*100), save=False, oneline=False)
            else:
                log("Error: no well-trained model is found!")
        elif args.mode == 'train':
            early_stop = 0
            Best_MRR = 0

            for ep in range(0, args.max_epochs): 
                self.trainEpoch() # train epoch
                if True:  
                    MRR, Hits1, Hits3, Hits10 = self.testEpoch(mode='valid', model_state_file=None)
                    MRR = int(MRR*10000)/10000
                    if MRR>Best_MRR:
                        Best_MRR = MRR
                        torch.save({'model':self.model.state_dict(), 'epoch':ep}, model_state_file)
                        early_stop = 0
                        # self.conf_threshold = self.conf_threshold * 0.9  # 如果效果提升，说明样本是有效的，可以降低阈值，尝试接纳更多样本
                        # log("Lower down the conf-threshold {}".format(self.conf_threshold))
                    else:
                        early_stop+=1
                        # self.conf_threshold = self.conf_threshold + 0.1  # 如果性能没有提升，说明样本包含了噪声，提高阈值
                        # log("Raise up the conf-threshold {}".format(self.conf_threshold))
                    log('Train: Epochs %2d/%d: Valid=> Best_MRR = %.2f, MRR = %.2f, Hits1 = %.2f, Hits3 = %.2f, Hits10 = %.2f' % (ep, args.max_epochs, Best_MRR*100, MRR*100, Hits1*100, Hits3*100, Hits10*100), save=False, oneline=False)

                    # self.conf_threshold = 0.99 if self.conf_threshold >0.99 else self.conf_threshold 
                    # if self.conf_threshold>=0.99:
                    #     early_stop+=1
                if early_stop==3:
                    break        
            MRR, Hits1, Hits3, Hits10 = self.testEpoch(mode='test', model_state_file=model_state_file)
            log('Final Test: MRR = %.2f, Hits1 = %.2f, Hits3 = %.2f, Hits10 = %.2f' % (MRR*100, Hits1*100, Hits3*100, Hits10*100), save=False, oneline=False)
        

    def trainEpoch(self):
        self.model.train()
        idx_list = [_ for _ in range(len(self.handler.trnList))]
        random.shuffle(idx_list)

        for idx in idx_list:
            if idx < args.history_len:
                continue

            if args.attack == 'poison':
                data_history_all = self.handler.trnListDirt[max(0, idx - args.history_len): idx]
                data_pred = self.handler.trnListDirt[idx]   # 噪声标签
            elif args.attack == 'evasion':  # 逃逸攻击设定下，训练数据是原始的
                data_history_all = self.handler_raw.trnListDirt[max(0, idx - args.history_len): idx]
                data_pred = self.handler_raw.trnListDirt[idx]
            elif args.attack == 'poison+evasion':
                data_history_all = self.handler.trnListDirt[max(0, idx - args.history_len): idx]
                data_pred = self.handler.trnListDirt[idx]
            else: 
                data_history_all = self.handler_raw.trnListDirt[max(0, idx - args.history_len): idx]
                data_pred = self.handler_raw.trnListDirt[idx] # 没有攻击的原始数据
            data_output_dict = self.handler_raw.trnListDirt
            kg_dict_pred = data_output_dict[idx]

            one_hot_tail_seq = None

            self.model.forward_old(data_history_all, one_hot_tail_seq, confidence=0)
            # self.model.forward(data_history_all, kg_dict_pred, one_hot_tail_seq, confidence=0)
            if self.loss_check:
                loss,_,_ = self.model.calculate_loss_check(data_pred, confidence=1, conf_threshold=self.conf_threshold)
            else:
                loss,_,_ = self.model.calculate_loss(data_pred, confidence=0) 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_norm)  # clip gradients
            self.opt.step()
            self.opt.zero_grad()

    def testEpoch(self, mode='test', model_state_file=None):
        Ranks, Ranks_noise = [], []
        if mode=='test':
            assert model_state_file is not None, "please specify check_point"
            checkpoint_pretrain = torch.load(model_state_file)
            self.model.load_state_dict(checkpoint_pretrain['model'])
        self.model.eval()
        with torch.no_grad():
            if mode=='valid':
                idx_list = [_ for _ in range(len(self.handler.valList))]
            elif mode == 'test':
                idx_list = [_ for _ in range(len(self.handler.tstList))]

            if args.attack == 'poison':
                data_input_prev = self.handler_raw.valListDirt if mode=='test' else self.handler.trnListDirt
                data_input = self.handler_raw.tstListDirt if mode=='test' else self.handler_raw.valListDirt
            elif args.attack == 'evasion':
                data_input_prev = self.handler.valListDirt if mode=='test' else self.handler_raw.trnListDirt
                data_input = self.handler.tstListDirt if mode=='test' else self.handler.valListDirt
            elif args.attack == 'poison+evasion':
                data_input_prev = self.handler.valListDirt if mode=='test' else self.handler.trnListDirt
                data_input = self.handler.tstListDirt if mode=='test' else self.handler.valListDirt
            else:
                data_input_prev = self.handler_raw.valListDirt if mode=='test' else self.handler.trnListDirt
                data_input = self.handler_raw.tstListDirt if mode=='test' else self.handler_raw.valListDirt

            data_output = self.handler_raw.tstList if mode=='test' else self.handler_raw.valList # 无论是哪种攻击，都是对原始样本进行预测评价
            data_output_Dirt = self.handler.tstListDirt if mode=='test' else self.handler.valListDirt
            data_output_Dirt = [list(kg.keys()) for kg in data_output_Dirt]

            data_output_dict = self.handler_raw.tstListDirt if mode=='test' else self.handler_raw.valListDirt


            for idx in idx_list:
                if idx >= args.history_len:
                    data_history_all = data_input[max(0, idx - args.history_len): idx]
                else:
                    data_history_all = data_input_prev[ (idx - args.history_len): ] + data_input[max(0, idx - args.history_len): idx] 
                data_pred = torch.LongTensor(data_output[idx]).to(self.device)  # 原始标签
                data_pred_noise =  torch.LongTensor(data_output_Dirt[idx]).to(self.device)
                data_pred_noise = data_pred_noise[data_pred_noise[:,4]==0]  # 单独查看噪声样本的预测效果
                kg_dict_pred = data_output_dict[idx]

                one_hot_tail_seq = None

                self.model.forward_old(data_history_all, one_hot_tail_seq, confidence=0)
                # self.model.forward(data_history_all, kg_dict_pred, one_hot_tail_seq, confidence=0)
                score_pred = self.model.predict(data_pred)
                # ranks = sort_and_rank(score_pred.cpu(), data_pred[:,2].cpu())
                if mode =='test':
                    ranks = get_ranks(data_pred, score_pred, batch_size=-1, filter=args.filter, num_e=self.handler.num_e, facts_filter=self.facts_filter)
                else:
                    ranks = get_ranks(data_pred, score_pred, batch_size=-1, filter=0, num_e=self.handler.num_e, facts_filter=self.facts_filter)
                Ranks.extend(ranks)

                # if data_pred_noise.shape[0]>0:
                #     score_pred_noise = self.model.predict(data_pred_noise)
                #     if mode =='test':
                #         ranks_noise = get_ranks(data_pred_noise, score_pred_noise, batch_size=-1, filter=args.filter, num_e=self.handler.num_e, facts_filter=self.facts_filter)
                #     else:
                #         ranks_noise = get_ranks(data_pred_noise, score_pred_noise, batch_size=-1, filter=0, num_e=self.handler.num_e, facts_filter=self.facts_filter)
                #     Ranks_noise.extend(ranks_noise)

            MRR, Hits1, Hits3, Hits10 = calc_mrr_hits(torch.tensor(Ranks))
            # MRR_noise, Hits1_noise, Hits3_noise, Hits10_noise = calc_mrr_hits(torch.tensor(Ranks_noise))
            # log("Noise samples: MRR_noise = %.2f, Hits1_noise = %.2f, Hits3_noise = %.2f, Hits10_noise = %.2f" % ( MRR_noise*100, Hits1_noise*100, Hits3_noise*100, Hits10_noise*100))
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
    print('\n\n')
    proctitle = '{}_{}_{}_{}_{}_{}_{}'.format(args.mode, args.model, args.dataset, args.history_len, args.lambda1, args.lambda2, args.lambda3)
    setproctitle.setproctitle(proctitle) # 读取环境变量并设置进程名称

    log('Start {}'.format(args.dataset))
    os.chdir(os.getcwd().split('/03_RobustTKG')[0]+'/03_RobustTKG/MyWork')
    model_dir = './SavedModel/{}'.format(args.dataset)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print(args)

    # if args.encoder == 'RGCN':
    #     model_state_file = model_dir+'/{}_noise{}_{}_histlen{}_temp{}_bypass{}_context{}'.format(args.model, args.neg_ratio, args.noise_strategy, args.history_len, args.lambda1, args.lambda2, args.lambda3)
    # else:
    #     log('Pleas define model state file name')

    model_state_file = model_dir+'/{}_init{}_attack{}_noise{}+{}_lr{}_histlen{}_temp{}_bypass{}_context{}'.format(args.model, args.init_method, args.attack, args.neg_ratio, args.noise_strategy, args.lr, args.history_len, args.lambda1, args.lambda2, args.lambda3)

    seed_it(args.seed)
    log('Device: {}'.format(args.device))
    logger.saveDefault = True

    log('Load raw data for model test')
    data_file = './SavedData/{}/{}_handler_ratio{}_strategy{}.pkl'.format(args.dataset, args.dataset, 0.0, args.noise_strategy)
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            handler_raw = pickle.load(f)

    # log("Load {} :  num_e:{}  num_r:{}".format(args.dataset, handler_raw.num_e, handler_raw.num_r))
    
    log('Load noise data for model training')
    data_file = './SavedData/{}/{}_handler_ratio{}_strategy{}.pkl'.format(args.dataset, args.dataset, args.neg_ratio, args.noise_strategy)
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            handler = pickle.load(f)

    # log("Load {} :  num_e:{}  num_r:{}".format(args.dataset, handler.num_e, handler.num_r))

    # log('Attack strategy: {}'.format(args.attack))
    # '''TKG Link prediction task'''
    coach = Coach(handler_raw, handler, args.device, args.neg_ratio)
    coach.run()
    log('Finish')


