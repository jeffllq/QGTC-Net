import argparse
import torch
import os



parser = argparse.ArgumentParser(description='Model Params')
parser.add_argument('--max-relations', type=int, default=0)  # 是否选用filter指标


parser.add_argument('--dataset', '-d', default='WIKI', type=str, help='name of dataset')
parser.add_argument('--history-len', '-hl', default=3, type=int, help='history window length')
parser.add_argument('--drop-rate', default=0.2, type=float, help='dropout rate')

parser.add_argument('--mode', default='train', type=str, help='main.py state')


# parser.add_argument('--retrain-flag', default=1, type=int, help='Retrain setting for TKG model')
# parser.add_argument('--lambda1', default=0, type=float, help='Weight for denoise process')

parser.add_argument('--neg-ratio', '-nr', type=float, default=0.3, help='noise dataset with different ratios,  0 means no noise, 1~8 means 0.1~0.8')
parser.add_argument('--noise-strategy', '-ns', type=int, default=1, help='noise currupt strategy')


# parser.add_argument('--denoise', default='tucker', type=str, help='method for denoise process')

# parser.add_argument('--threshold', type=float, default=0.3, help='noise detect threshold')

# Settings for diffusion model
parser.add_argument('--dim-DF', type=str, default='[1000]')
parser.add_argument('--emb_size', type=int, default=10)
parser.add_argument('--norm', type=bool, default=True)

# # settings for tucker model
# parser.add_argument('--num-feat', type=int, default=100)
# parser.add_argument('--num-hid', type=int, default=300)
# parser.add_argument('--rank', type=int, default=50)
# parser.add_argument('--alpha', type=float, default=1e-4)
# parser.add_argument('--beta', type=float, default=1e-2)

# settings for RGDC
parser.add_argument('--diff-steps', type=int, default=2)
parser.add_argument('--lambda1', type=float, default=1.0)
parser.add_argument('--lambda2', type=float, default=0.4)  
parser.add_argument('--lambda3', type=float, default=0.0)  # 必须放弃这个conf，因为计算复杂度太高，而且体现不出区别

parser.add_argument('--filter', type=int, default=0)  # 是否选用filter指标


parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch', default=1024, type=int, help='batch size')


parser.add_argument('--encoder', default='RGCN', type=str, help='methods of encoder')
parser.add_argument('--layer-n', default=1, type=int, help='layers amount of RGCN')

parser.add_argument('--batchsize', default=64, type=int, help='batchsize')


parser.add_argument("--seed", type=int, default=421, help="random seed")
parser.add_argument('--gpu', "-g", default = 1, type=int, help='which gpu to use')
parser.add_argument('--tstEpoch', default=2, type=int, help='number of epoch to test while training:')
parser.add_argument('--epoch', default=30, type=int, help='number of epochs')
parser.add_argument('--dim', type=int, default=200)

# parser.add_argument('--steps', type=int, default=5, help='steps for diffusion model')
# parser.add_argument('--noise_scale', type=float, default=0.1)
# parser.add_argument('--noise_min', type=float, default=0.0001)
# parser.add_argument('--noise_max', type=float, default=0.02)

args = parser.parse_args()

if args.history_len == 0:
    if args.dataset == 'YAGO':
        args.history_len = 1
    elif args.dataset == 'WIKI':
        args.history_len = 2
    elif args.dataset == 'ICEWS14':
        args.history_len = 4
    elif args.dataset == 'ICEWS18':
        args.history_len = 6




# def select_gpu_device(threshold_gb=0, gpu=0):
#     pynvml.nvmlInit()
#     device_count = pynvml.nvmlDeviceGetCount()
#     for i in range(device_count):
#         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#         info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         free_memory_gb = info.free / 1024 ** 3  # 转换为 GB
#         if (free_memory_gb > threshold_gb) and (gpu >= 0):
#             pynvml.nvmlShutdown()  # 释放 NVML 资源
#             os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
#             device = torch.device('cuda:0')
#             return device
#     pynvml.nvmlShutdown()  # 如果没有找到合适的 GPU，则返回 None
#     return torch.device('cpu')
# device = select_gpu_device(threshold_gb=8, gpu=args.gpu)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda:0')
args.device = device


# python /home/llq/exp_proj/RobustTKG/MyWork/src_DGAD/main.py -d ICEWS14 --neg-ratio 0.0 --lambda3 0.0
