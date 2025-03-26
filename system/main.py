#!/usr/bin/env python
import os
device_id="1"
os.environ["CUDA_VISIBLE_DEVICES"] = device_id

import copy
import torch
import argparse
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serveravg import FedAvg

from flcore.trainmodel.models import *

from flcore.trainmodel import CLIPWithAdapter

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

# 初始化logger
# logger = logging.getLogger()
# logger.setLevel(logging.ERROR)
# warnings.simplefilter("ignore")

# 设置随机种子
torch.manual_seed(0)

def run(args):

    """
    初始化记录器
    """
    time_list = []  #记录时间
    reporter = MemReporter()    #记录内存


    for i in range(args.prev, args.times):  #重复实验
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        """
        初始化模型
        """
        #根据数据集分类情况初始化模型，将args.model由用户输入的模型名字设置为模型instance，并把模型放到设备上
        model_str = args.model 
        if model_str == "CLIPWithAdapter":  #这里用自己定义的模型，写在flcore.trainmodel.models.py中
            args.model = CLIPWithAdapter(clip_model_name='ViT-L/14@336px').to(args.device)
        else:
            raise NotImplementedError

        #打印模型的结构信息
        print(args.model)

        """
        设置联邦学习聚合方法
        """
        # 选择聚合方法，初始化Server类
        if args.algorithm == "FedAvg":
            # if model_str == "CLIPWithAdapter":
            #     args.head = copy.deepcopy(args.model.fc)
            #     args.model.fc = nn.Identity()
            #     args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":

    """
    开始设置参数
    """
    parser = argparse.ArgumentParser()

    #设备配置
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])  #设备
    parser.add_argument('-did', "--device_id", type=str, default=device_id)                           #设备id
    
    #全局实验配置，每次重复实验会重新创建server和client
    parser.add_argument('-pv', "--prev", type=int, default=0,                       #前一次重复实验标号
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,                       #运行次数
                        help="Running times")
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)            #全局轮数, 每一轮各个客户端都会训练local_epochs次

    #本地基础训练配置
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,                               #评估间隔
                        help="Rounds gap for evaluation")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)                           #批大小
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,                           #本地轮数
                        help="Multiple update steps in one local epoch.")    
    parser.add_argument('-logd', "--log_dir", type=str, default="")                           #本地步数                   
    
    #学习率配置
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=3e-5,              #本地学习率
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=True)               #学习率是否衰减

    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)        #学习率衰减gamma
    
    #自动停止配置
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)                        #是否自动停止
    parser.add_argument('-tc', "--top_cnt", type=int, default=100)                              #停止条件
    
    #保存模型配置
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")                                     #目标，用于保存结果时候命名
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')                 #保存文件夹名

    #联邦学习配置
    parser.add_argument('-m', "--model", type=str, default="CLIPWithAdapter")                  #模型
    parser.add_argument('-ncl', "--num_classes", type=int, default=6)               #分类数
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")         #聚合算法

    #数据集配置
    parser.add_argument('-data', "--dataset", type=str, default="DRnpz")            #数据集
    parser.add_argument('-dr', "--data_root", type=str, default="data")              #数据集路径

    #客户端配置
    parser.add_argument('-nc', "--num_clients", type=int, default=5, 
                        help="Total number of clients")                             #总客户端数量
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")            #训练但退出的客户端比例
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False)    #是否只随机选择一定比例客户端加入全局训练
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, 
                        help="Ratio of clients per round")                          #每轮参与训练的客户端比例
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)           #新客户端数量
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)    #新客户端微调轮数
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,       #在训练时很慢的客户端比例
                        help="The rate for slow clients when training locally")     
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,        #在发送时很慢的客户端比例
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")              #丢弃慢客户端的阈值
        
    #对话评估配置
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)             #是否进行对话评估
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)                #对话评估间隔
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)     #每个客户端的批次数

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    """
    开始训练
    """
    run(args)

