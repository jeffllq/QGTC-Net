import argparse
import torch
import os



parser = argparse.ArgumentParser(description='Model Params')
parser.add_argument('--attack', default='poison+evasion', type=str, help='attack strategy', choices=['poison', 'evasion', 'poison+evasion', 'raw'])  # poison, evasion, poison+evasion, raw
parser.add_argument('--mode', default='test', type=str, help='main python file state')
parser.add_argument('--model', default='REGCN', type=str, help='model name')

parser.add_argument('--use-query', default=1, type=int, help='use query-guided strategy')
parser.add_argument('--use-conf', default=1, type=int, help='use confidence-guided strategy')
parser.add_argument('--conf-threshold', type=float, default=0.5)

parser.add_argument('--dataset', '-d', default='ICEWS14', type=str, help='name of dataset')
parser.add_argument('--gpu', "-g", default = 0, type=int, help='which gpu to use')
parser.add_argument('--history-len', '-hl', default=2, type=int, help='history window length')


# parser.add_argument('--train-conf', type=int, default=0)
# parser.add_argument('--test-conf', type=int, default=0)
# parser.add_argument('--loss-check', type=int, default=1)

parser.add_argument('--lambda1', type=float, default=1.0)
parser.add_argument('--lambda2', type=float, default=0.0)  
parser.add_argument('--lambda3', type=float, default=1.0)

parser.add_argument('--neg-ratio', '-nr', type=float, default=0.3, help='noise dataset with different ratios,  0 means no noise, 1~8 means 0.1~0.8')
parser.add_argument('--noise-strategy', '-ns', type=int, default=1, help='noise currupt strategy')

parser.add_argument('--init-method', type=str, default='normal', choices=['normal','xavier','normal+xavier','xavier+normal', 'kaiming'])  # 如何初始化实体和关系的表示
parser.add_argument('--init-std', type=float, default=1.0, help="initialization std setting")  # 是否选用filter指标

parser.add_argument('--layer-norm', default=True, type=bool, help='layers norm or not')

parser.add_argument('--drop-rate', default=0.2, type=float, help='dropout rate')

parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")


# Settings for diffusion model
# parser.add_argument('--dim-DF', type=str, default='[1000]')
# parser.add_argument('--emb_size', type=int, default=10)
# parser.add_argument('--norm', type=bool, default=True)
# parser.add_argument('--diff-steps', type=int, default=2)

parser.add_argument('--filter', type=int, default=1)  # 是否选用filter指标


parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
# parser.add_argument('--batch', default=1024, type=int, help='batch size')


parser.add_argument('--encoder', default='RGCN', type=str, help='methods of encoder')
parser.add_argument('--layer-n', default=1, type=int, help='layers amount of RGCN')

parser.add_argument('--batchsize', default=64, type=int, help='batchsize')


parser.add_argument("--seed", type=int, default=421, help="random seed")
parser.add_argument('--tstEpoch', default=2, type=int, help='number of epoch to test while training:')
parser.add_argument('--max_epochs', default=40, type=int, help='number of epochs')
parser.add_argument('--change-epoch', default=30, type=int, help='number of epochs')

parser.add_argument('--dim', type=int, default=200)


args = parser.parse_args()

if args.history_len <= 0:
    if args.dataset == 'YAGO':
        args.history_len = 1
    elif args.dataset == 'WIKI':
        args.history_len = 2
    elif args.dataset == 'ICEWS14':
        args.history_len = 4
    elif args.dataset == 'ICEWS18':
        args.history_len = 6
    else:
        args.history_len = 3


os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda:0')

args.device = device


# python MyWork/src_DGAD/main_TKGR.py -d YAGO --neg-ratio 0.3 --lambda1 0.1
