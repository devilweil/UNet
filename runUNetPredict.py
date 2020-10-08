# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:34:47 2020

@author: LW
"""
###导入包
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from torch import optim
from tqdm import tqdm

import numpy as np
from PIL import Image
import datetime

from runUNetModel import UNet
from runUNetModel import BasicDataset
from runUNetModel import DoubleConv
from runUNetModel import Down
from runUNetModel import Up
from runUNetModel import OutConv

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
def loaddata():

    root_img_dir=r'H:\gansu\wuwei\DataSet\AllImageSampleSingleImage64\20190423\Test\Grape'
    root_mask_dir=r'H:\gansu\wuwei\DataSet\AllImageSampleSingleImage64\20190423\mask\Test'
    
    trainDataset=BasicDataset(root_img_dir,root_mask_dir)
    
    return trainDataset,len(trainDataset)#,testDataset,len(testDataset)

def predict():
    ###测试
    print('Predict....')

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == "__main__":
    
    ###定义参数
    savePath=r'H:\gansu\wuwei\DataSet\AllImageSampleSingleImage64\20190423\预测结果'
    ###定义网络
    net = UNet(n_channels=3, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load('H:\gansu\wuwei\CNN\Model_Trained\weights_20201007_0.pth', map_location=device))
    
    train_Dataset,train_Dataset_len=loaddata()
    train_loader = DataLoader(train_Dataset, batch_size=1)
    
    for batch in train_loader:
        batch_img,batch_mask=batch['image'],batch['mask']
        print(batch_img.shape)
        batch_img = batch_img.to(device=device, dtype=torch.float32)
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        batch_mask = batch_mask.to(device=device, dtype=mask_type)
        output = net(batch_img)
        print(output.shape)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(batch_img.shape[2]),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.to(device=device))
        full_mask = probs.squeeze().to(device=device).numpy()
        ####保存图片
        name=batch['name'][0]
        result = mask_to_image(full_mask)
        result.save(savePath+'\\'+name)
