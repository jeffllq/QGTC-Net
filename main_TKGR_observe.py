import pickle
import setproctitle
import os
from Config_TKGR import args
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
from TimeLogger import log
from DataHandler_TKGR import DataHandler
import matplotlib.colors as mcolors




from collections import Counter
plt.rc('font',family='Times New Roman')
plt.rcParams['font.size'] = 16

if __name__=='__main__':

    proctitle = '{}_{}_{}_{}_{}_{}_{}'.format(args.mode, args.model, args.dataset, args.history_len, args.lambda1, args.lambda2, args.lambda3)
    setproctitle.setproctitle(proctitle) # 读取环境变量并设置进程名称
    print(args)
    os.chdir(os.getcwd().split('/03_RobustTKG')[0]+'/03_RobustTKG/MyWork')


    log('Load raw data for model test')
    data_file = './SavedData/{}/{}_handler_ratio{}_strategy{}.pkl'.format(args.dataset, args.dataset, 0.0, args.noise_strategy)
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            handler_raw = pickle.load(f)
    
    log('Load noise data for model training')
    data_file = './SavedData/{}/{}_handler_ratio{}_strategy{}.pkl'.format(args.dataset, args.dataset, args.neg_ratio, args.noise_strategy)
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            handler = pickle.load(f)

    idx_list = [_ for _ in range(len(handler.tstList))]
    idx = idx_list[0]
    mode = 'test'
    if args.attack == 'poison':
        data_input_prev = handler_raw.valListDirt if mode=='test' else handler.trnListDirt
        data_input = handler_raw.tstListDirt if mode=='test' else handler_raw.valListDirt
    elif args.attack == 'evasion':
        data_input_prev = handler.valListDirt if mode=='test' else handler_raw.trnListDirt
        data_input = handler.tstListDirt if mode=='test' else handler.valListDirt
    elif args.attack =='poison+evasion':
        data_input_prev = handler.valListDirt if mode=='test' else handler.trnListDirt
        data_input = handler.tstListDirt if mode=='test' else handler.valListDirt
    else:
        data_input_prev = handler_raw.valListDirt if mode=='test' else handler.trnListDirt
        data_input = handler_raw.tstListDirt if mode=='test' else handler_raw.valListDirt

    data_output = handler_raw.tstList if mode=='test' else handler_raw.valList # 无论是哪种攻击，都是对原始样本进行预测评价
    data_output_Dirt = handler.tstListDirt if mode=='test' else handler.valListDirt
    data_output_Dirt = [list(kg.keys()) for kg in data_output_Dirt]

    data_output_dict = handler_raw.tstListDirt if mode=='test' else handler_raw.valListDirt

    data_pred = torch.LongTensor(data_output[idx])  # 原始标签

    queries = [(fact[0], fact[1]) for fact in data_pred]
    head_frequency = Counter([head for head, _ in queries])
    

    select_how_many = 5
    top_50_heads = head_frequency.most_common(select_how_many)
    # rows_idx = torch.tensor([s_id.item() for s_id,_ in top_50_heads])



    with open('./SavedObserve/{}/observe.pkl'.format(args.dataset), 'rb') as f:
        observe = pickle.load(f)

    observe_query_mask = observe['observe_query_mask']
    observe_eEmbeds_list = observe['observe_eEmbeds_list']
    observe_attn_eEmbeds_list = observe['observe_attn_eEmbeds_list']

    # # 将所有的表示进行normalize，方便可视化展示
    # observe_query_mask = F.normalize(observe_query_mask)
    # for i in range(len(observe_eEmbeds_list)):
    #     observe_eEmbeds_list[i] = F.normalize(observe_eEmbeds_list[i])
    for i in range(len(observe_attn_eEmbeds_list)):
        observe_attn_eEmbeds_list[i] = F.normalize(observe_attn_eEmbeds_list[i].squeeze())


    # rows_idx = torch.randint(0, handler.num_e, (select_how_many,))

    if args.dataset == 'YAGO':
        rows_idx = torch.tensor([742, 4366,  697,  2258,7359])  # YAGO
    if args.dataset == 'WIKI':
        rows_idx = torch.tensor([ 250,  241, 1951, 1952, 8812])  # WIKI
    if args.dataset == 'ICEWS14':
        rows_idx = torch.tensor([30,  18,  77,   8,  44])  # ICEWS14
    if args.dataset == 'ICEWS18':
        rows_idx = torch.tensor([ 459, 4935, 2820,   64,   42])  # ICEWS18

    print(rows_idx)

    # vmin, vmax = 0.0, 1.0



    # selected_rows = observe_query_mask[rows_idx]
    # selected_rows_np = selected_rows.cpu().numpy()
    # # 创建热力图
    # plt.figure(figsize=(10, 3))
    # plt.imshow(selected_rows_np, aspect='auto', cmap='Reds', vmin=2)
    # # 添加颜色条和标签
    # plt.colorbar(label='Feature Value Intensity')
    # plt.title('Heatmap of Randomly Selected 10 Rows')
    # plt.xlabel('Features')
    # plt.ylabel('Sample Index')
    # # 显示热力图
    # plt.show()
    # plt.savefig('./SavedObserve/{}/observe_query_mask.png'.format(args.dataset))


    # for timestamp in range(len(observe_eEmbeds_list)):
    #     selected_rows = observe_eEmbeds_list[timestamp][rows_idx]
    #     selected_rows_np = selected_rows.cpu().numpy()

    #     plt.figure(figsize=(10, 3))
    #     plt.imshow(selected_rows_np, aspect='auto', cmap='Reds',vmin=0, vmax=1)

    #     # 添加颜色条和标签
    #     plt.colorbar(label='Feature Value Intensity')
    #     plt.title('Heatmap of Randomly Selected 10 Rows')
    #     plt.xlabel('Features')
    #     plt.ylabel('Sample Index')
    #     plt.tight_layout()
    #     # 显示热力图
    #     plt.show()
    #     plt.savefig('./SavedObserve/{}/observe_eEmbeds_list_{}.png'.format(args.dataset, timestamp))
    

    # for timestamp in range(len(observe_attn_eEmbeds_list)):
    #     selected_rows = observe_attn_eEmbeds_list[timestamp].squeeze()[rows_idx]
    #     selected_rows_np = selected_rows.cpu().numpy()

    #     plt.figure(figsize=(10, 3))
    #     plt.imshow(selected_rows_np, aspect='auto', cmap='Reds',vmin=0, vmax=1)
    #     plt.tight_layout()
    #     # 添加颜色条和标签
    #     plt.colorbar(label='Feature Value Intensity')
    #     plt.title('Heatmap of Randomly Selected 10 Rows')
    #     plt.xlabel('Features')
    #     plt.ylabel('Sample Index')
    #     plt.tight_layout()
    #     # 显示热力图
    #     plt.show()
    #     plt.savefig('./SavedObserve/{}/observe_attn_eEmbeds_list_{}.png'.format(args.dataset, timestamp))





    selected_rows = observe_query_mask[rows_idx]
    selected_rows_np = selected_rows.cpu().numpy()

    cmap = mcolors.ListedColormap(['#fffacd', '#ff4500', '#880000'])  # 0-0.5是淡黄色, 0.5-0.8是橙色, 0.8-1是红色
    bounds = [0, 4, 4.5, 5]  # 定义区间的边界
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    # 创建热力图
    plt.figure(figsize=(10, 3))
    # 使用自定义颜色映射和分段
    plt.imshow(selected_rows_np, aspect='auto', cmap=cmap, norm=norm)
    plt.tight_layout()
    
    # 添加颜色条和标签
    cbar = plt.colorbar(label='Feature Value Intensity')
    cbar.set_ticks([2, 4.25, 4.75])  # 在颜色条上设置标签
    cbar.set_ticklabels(['0-3','3-4', '4-5'])  # 颜色条标签
    
    plt.title('Feature Heatmap of Randomly Selected 5 Query Entities')
    plt.xlabel('Features')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    plt.savefig('./SavedObserve/{}/{}_observe_query_mask.png'.format(args.dataset, args.dataset,))



    # 定义颜色和对应的区间
    cmap = mcolors.ListedColormap(['#fffacd', '#ffa500', '#ff4500', '#880000'])  # 0-0.5是淡黄色, 0.5-0.8是橙色, 0.8-1是红色
    bounds = [0, 0.2, 0.5, 0.8, 1.0]  # 定义区间的边界
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 循环遍历时间戳的观测嵌入列表
    for timestamp in range(len(observe_eEmbeds_list)):
        selected_rows = observe_eEmbeds_list[timestamp].squeeze()[rows_idx]
        selected_rows_np = selected_rows.cpu().numpy()

        plt.figure(figsize=(10, 3))
        # 使用自定义颜色映射和分段
        plt.imshow(selected_rows_np, aspect='auto', cmap=cmap, norm=norm)
        plt.tight_layout()
        
        # 添加颜色条和标签
        cbar = plt.colorbar(label='Feature Value Intensity')
        cbar.set_ticks([0.1, 0.35, 0.65, 0.9])  # 在颜色条上设置标签
        cbar.set_ticklabels(['0-0.2', '0.2-0.5','0.5-0.8', '0.8-1.0'])  # 颜色条标签
        
        plt.title('General: Feature Heatmap of Randomly Selected 5 Entities')
        plt.xlabel('Features')
        plt.ylabel('Sample Index')
        plt.tight_layout()

        # 显示热力图并保存
        plt.savefig('./SavedObserve/{}/{}_observe_eEmbeds_list_{}.png'.format(args.dataset,args.dataset, timestamp))
        plt.show()

    # 循环遍历时间戳的观测嵌入列表
    for timestamp in range(len(observe_attn_eEmbeds_list)):
        selected_rows = observe_attn_eEmbeds_list[timestamp].squeeze()[rows_idx]
        selected_rows_np = selected_rows.cpu().numpy()

        plt.figure(figsize=(10, 3))
        # 使用自定义颜色映射和分段
        plt.imshow(selected_rows_np, aspect='auto', cmap=cmap, norm=norm)
        plt.tight_layout()
        
        # 添加颜色条和标签
        cbar = plt.colorbar(label='Feature Value Intensity')
        cbar.set_ticks([0.1, 0.35, 0.65, 0.9])  # 在颜色条上设置标签
        cbar.set_ticklabels(['0-0.2', '0.2-0.5','0.5-0.8', '0.8-1.0'])  # 颜色条标签
        
        plt.title('QGLM: Feature Heatmap of Randomly Selected 5 Entities')
        plt.xlabel('Features')
        plt.ylabel('Sample Index')
        plt.tight_layout()

        # 显示热力图并保存
        plt.savefig('./SavedObserve/{}/{}_observe_attn_eEmbeds_list_{}.png'.format(args.dataset, args.dataset, timestamp))
        plt.show()










    # # 间隔参数，可以根据需要调整
    # norm = mcolors.BoundaryNorm(bounds, cmap.N)
    # row_spacing = 0.5

    # for timestamp in range(len(observe_attn_eEmbeds_list)):
    #     selected_rows = observe_attn_eEmbeds_list[timestamp].squeeze()[rows_idx]
    #     selected_rows_np = selected_rows.cpu().numpy()

    #     # 获取行数和列数
    #     num_rows, num_cols = selected_rows_np.shape

    #     plt.figure(figsize=(10, 3))

    #     # 使用 extent 调整每行的间隔
    #     plt.imshow(selected_rows_np, aspect='auto', cmap=cmap, norm=norm, extent=[0, num_cols, 0, num_rows * (1 + row_spacing)])

    #     # 手动设置y轴的刻度位置，以调整间隔
    #     yticks = np.arange(0.5, num_rows * (1 + row_spacing), 1 + row_spacing)
    #     plt.yticks(yticks, np.arange(num_rows))  # 保持原始行索引

    #     # 添加颜色条和标签
    #     cbar = plt.colorbar(label='Feature Value Intensity')
    #     cbar.set_ticks([0.25, 0.65, 0.9])  # 设置颜色条的刻度
    #     cbar.set_ticklabels(['0-0.5', '0.5-0.8', '0.8-1.0'])  # 设置颜色条的标签

    #     plt.title('Heatmap of Randomly Selected 10 Rows')
    #     plt.xlabel('Features')
    #     plt.ylabel('Sample Index')
    #     plt.tight_layout()

    #     # 显示热力图并保存
    #     plt.savefig('./SavedObserve/{}/observe_attn_eEmbeds_list_{}.png'.format(args.dataset, timestamp))