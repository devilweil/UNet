# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:49:57 2020

@author: LW
"""
###导入包
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch import optim
from tqdm import tqdm

import numpy as np
from PIL import Image
import datetime

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

###定义全局变量和导入数据
batch_size=8
epochs=1
lr=0.001

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        super(BasicDataset, self).__init__()
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.imgs_list=os.listdir(self.imgs_dir)
        self.masks_list=os.listdir(self.masks_dir)
    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self,idx):
        img_path=os.path.join(self.imgs_dir,self.imgs_list[idx])
        mask_path=os.path.join(self.masks_dir,self.masks_list[idx])
        img = Image.open(img_path)
        mask= Image.open(mask_path)
        img=np.array(img)
        mask=np.array(mask)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        # HWC to CHW
        img = img.transpose((2, 0, 1))
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)

        # HWC to CHW
        mask = mask.transpose((2, 0, 1))
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'name': self.imgs_list[idx]
        }  

def loaddata():
#     root_img_dir=r'H:\gansu\wuwei\DataSet\AllImageSampleSingleImage52\20190423\Train\Grape'
#     root_mask_dir=r'H:\gansu\wuwei\DataSet\AllImageSampleSingleImage52\20190423\mask\Train'
    
#     trainDataset=BasicDataset(root_img_dir,root_mask_dir)
    
#     root_img_val_dir=r'H:\gansu\wuwei\DataSet\AllImageSampleSingleImage52\20190423\Test\Grape'
#     root_mask_val_dir=r'H:\gansu\wuwei\DataSet\AllImageSampleSingleImage52\20190423\mask\Test'
    
#     testDataset=BasicDataset(root_img_val_dir,root_mask_val_dir)
    
    root_img_dir=r'H:\gansu\wuwei\DataSet\AllImageSampleSingleImage64\20190423\Train\Grape'
    root_mask_dir=r'H:\gansu\wuwei\DataSet\AllImageSampleSingleImage64\20190423\mask\train'
    
    trainDataset=BasicDataset(root_img_dir,root_mask_dir)
    
    return trainDataset,len(trainDataset)#,testDataset,len(testDataset)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def train():
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    train_Dataset,train_Dataset_len=loaddata()
    train_loader = DataLoader(train_Dataset, batch_size=batch_size) ####, shuffle=True
    print('TrainSet is',train_Dataset,' and TrainSet Length is ',train_Dataset_len)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        print('epoch is ',epoch)
        for batch in train_loader:
            batch_img,batch_mask=batch['image'],batch['mask']
#             print(batch_img,batch_mask)
            batch_img = batch_img.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            batch_mask = batch_mask.to(device=device, dtype=mask_type)
            
            #print(batch_img.shape)
            masks_pred = net(batch_img)
            loss = criterion(masks_pred, batch_mask)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('train loss is {:.6f}'.format(epoch_loss/train_Dataset_len))
    torch.save(net.state_dict(), 'H:\gansu\wuwei\CNN\Model_Trained\weights_20201007_%d.pth' % epoch)
    torch.save(net, 'H:\gansu\wuwei\CNN\Model_Trained\model_20201007_%d.pth' % epoch)
    

if __name__ == "__main__":     
    
    #初始化工作空间
    starttime = datetime.datetime.now()
    train()
    endtime = datetime.datetime.now()
    print('Time spend  ', endtime - starttime)


