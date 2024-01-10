from __future__ import print_function, division
import torch
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from models.CDCNs import Conv2d_cd, CDCN_3modality2

from Loadtemporal_valtest_3modality import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from utils import AvgrageMeter, accuracy, performances



# Dataset root       
#image_dir = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/'  
   
image_dir = '/data/usr/lhr/FakeAVCeleb-main/Multimodal/Multimodal-2/CHN_DF_1W/' 
#test_list =  '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@1_test_res.txt'
test_list =  '/data/usr/lhr/FakeAVCeleb-main/Multimodal/Multimodal-2/CHN_DF_1W/' 
#test_list =  '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@2_test_res.txt'
#test_list =  '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@3_test_res.txt'


# main function
def train_test():


    print("test:\n ")

     
    #model = CDCN_3modality2( basic_conv=Conv2d_cd, theta=0.7)
    model = CDCN_3modality2( basic_conv=Conv2d_cd, theta=args.theta)
    
    model.load_state_dict(torch.load('../CDCN_3modality2_P1/CDCN_3modality2_P1_5.pkl'))


    model = model.cuda()

    print(model) 
    


    model.eval()
    
    with torch.no_grad():
        ###########################################
        '''                val             '''
        ###########################################
        # val for threshold
        val_data = Spoofing_valtest(test_list,transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        #val_data = Spoofing_valtest(test_list, image_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
        
        map_score_list = []
        real_count = []
        fake_count = []
        totall = 0
        for i, sample_batched in enumerate(dataloader_val):
            # get the inputs
            inputs = sample_batched['image_x'].cuda()
            inputs_ir = sample_batched['image_ir'].cuda()
            string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cuda()
        
            #optimizer.zero_grad()
                    
                    
            map_score = 0.0
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs, inputs_ir)

            for mapp in range(map_x.shape[0]):
                #map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:], inputs_depth[:,frame_t,:,:,:])
                score_norm = torch.sum(map_x[mapp,:,:])/torch.sum(binary_mask[mapp,:,:])
                map_score = score_norm
                #print(score_norm)
                if map_score>1:
                    score_norm = 1
                else:
                    score_norm = 0 
                if score_norm == 0 :
                    fake_count.append(1)
                    if string_name[mapp] > 0:
                        real_count.append(1)
                    else:
                        real_count.append(0)
                                
                if score_norm == 0 :
                    fake_count.append(0)
                    if string_name[mapp] < 1:
                        real_count.append(0)
                    else:
                        real_count.append(1)
                totall+=1
                #print(string_name[mapp],score_norm)
            map_score = map_score/map_x.shape[0]    
            map_score_list.append('{} {}\n'.format( string_name[0], map_score ))
            #print(string_name[0], map_score )
        map_score_val_filename = "/data/usr/lhr/FakeAVCeleb-main/Multimodal/CDCN/res.txt"
        with open(map_score_val_filename, 'w') as file:
            file.writelines(map_score_list)
        # 要写入的文件路径
        file_path = "/data/usr/lhr/FakeAVCeleb-main/Multimodal/CDCN/temp_bianliang.txt"
        # 使用 'w' 模式打开文件进行写入
        with open(file_path, 'w') as file:
            # 将 fake_count 列表转换为字符串并写入文件
            fake_count_str = ' '.join(map(str, fake_count))
            file.write("Fake Count: " + fake_count_str + "\n")

            # 将 real_count 列表转换为字符串并写入文件
            real_count_str = ' '.join(map(str, real_count))
            file.write("Real Count: " + real_count_str + "\n")
                
        #print("Accuracy: ",((real_count+fake_count)*100)/totall)
        print(f1_score( fake_count, real_count, average="binary"))
        print(precision_score(fake_count, real_count, average="binary"))
        print(recall_score(fake_count, real_count, average="binary"))                     
        precision, recall, fscore, support = score(fake_count, real_count)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
                

    print('Finished testing')
  

  
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  #default=0.0001
    parser.add_argument('--batchsize', type=int, default=9, help='initial batchsize')  #default=7  
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCN_3modality2_P1", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
	
    args = parser.parse_args()
    train_test()
