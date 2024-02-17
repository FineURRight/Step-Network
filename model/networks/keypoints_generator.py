import torch.nn as nn
import torch
import functools
import torch.nn.functional as F
from torch import transpose
import random


class KPModel(nn.Module):
    def __init__(self, n_keypoints=18, BS=6, random_shift=30, random_corp=0.5):
        super(KPModel, self).__init__()
        self.n_keypoints = n_keypoints
        self.BS = BS
        self.random_shift = random_shift
        self.random_corp = random_corp
        self.conv_1 = nn.Conv1d(
            n_keypoints, 10, kernel_size=1, padding=1, stride=1)
        # (BS, 10, 4)
        self.BN1 = nn.BatchNorm1d(10)
        self.conv_2 = nn.Conv1d(10, 20, kernel_size=2, padding=0, stride=1)
        # (BS, 20, 3)
        self.BN2 = nn.BatchNorm1d(20)
        self.conv_3 = nn.Conv1d(20, 30, kernel_size=2, padding=0, stride=1)
        # (BS, 30, 2)
        self.BN3 = nn.BatchNorm1d(30)
        self.fc1 = nn.Linear(60, 100)
        self.fc2 = nn.Linear(100, 36)
        self.BN4 = nn.BatchNorm1d(100)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def keypoint_input(self, input_cor, data_agumentation=True):
        # move to GPU and change data types
        # input_cor = input['BP1_cor']
        # self.input_cor = input_cor.cuda().float()

        B, n, cor = input_cor.shape
        self.input_cor = torch.cuda.FloatTensor(B, n, cor).fill_(0)
        self.input_cor[:, :, :] = input_cor[:, :, :]

        # 标记原数据缺失的地方
        self.miss_mask = self.input_cor != -1
        miss_mask_x = self.input_cor[:, :, 0] != -1
        miss_mask_y = self.input_cor[:, :, 1] != -1

        # 数据增强
        if data_agumentation:
            # 数据增强
            # 位移
            self.input_cor[miss_mask_x] = self.input_cor[miss_mask_x] + \
                random.random()*self.random_shift - random.random()*self.random_shift
            self.input_cor[miss_mask_y] = self.input_cor[miss_mask_y] + \
                random.random()*self.random_shift - random.random()*self.random_shift
            # 缩放
            zoom_x = random.randint(100, 155)
            zoom_y = random.randint(100, 155)
            zoom_per = random.random()*self.random_corp
            self.input_cor[miss_mask_x] = zoom_x + \
                (1+zoom_per) * (self.input_cor[miss_mask_x] - zoom_x)
            self.input_cor[miss_mask_y] = zoom_y + \
                (1+zoom_per) * (self.input_cor[miss_mask_y] - zoom_y)

        # 生成ground truth
        self.gt = torch.cuda.FloatTensor(B, self.n_keypoints*2).fill_(0)
        self.gt[:, :] = self.input_cor.view(B, -1)[:, :]
        # print(self.gt)
        # print(self.input_cor)
        train_matrix = torch.cuda.ByteTensor(B, self.n_keypoints, 2).fill_(0)

        if data_agumentation:
            # 随机隐藏30%的keypoints，进行预测
            for j in range(B):
                for i in range(18):
                    if random.random() >= 0.7 and self.input_cor[j, i, 0] != -1:
                        self.input_cor[j, i, :] = -1
                        train_matrix[j, i, :] = 1
        # 预测的地方为true
        self.train_mask = train_matrix.view(B, -1) == 1

    def forward(self):
        B, n, cor = self.input_cor.shape
        x = self.input_cor.float()
        x = self.conv_1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.BN3(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # x = self.BN4(x)
        x = self.relu(x)
        x = self.fc2(x)
        kp_loss = torch.mean(
            torch.pow(x[self.train_mask]-self.gt[self.train_mask], 2))
        # print(x[self.train_mask])
        # print(self.gt[self.train_mask])
        x = x.view(B, n, cor)
        # 距離損失
        head_loss = 0
        h_n = 0
        righthand_loss = 0
        rh_n = 0
        lefthand_loss = 0
        lh_n = 0
        rightleg_loss = 0
        rl_n = 0
        leftleg_loss = 0
        ll_n = 0
        for idx in range(B):
            if self.input_cor[idx, 0, 0] != -1 and self.input_cor[idx, 14, 0] != -1 and self.input_cor[idx, 15, 0] != -1:
                head_dis = (torch.pow((x[idx, 0, 0]-x[idx, 14, 0])**2+(x[idx, 0, 1]-x[idx, 14, 1])**2, 0.5)+torch.pow(
                    (x[idx, 0, 0]-x[idx, 15, 0])**2+(x[idx, 0, 1]-x[idx, 15, 1])**2, 0.5))/2
                gt_head_dis = (torch.pow((self.input_cor[idx, 0, 0]-self.input_cor[idx, 14, 0])**2+(self.input_cor[idx, 0, 1]-self.input_cor[idx, 14, 1])**2, 0.5)+torch.pow(
                    (self.input_cor[idx, 0, 0]-x[idx, 15, 0])**2+(self.input_cor[idx, 0, 1]-self.input_cor[idx, 15, 1])**2, 0.5))/2
                head_loss += torch.pow(head_dis-gt_head_dis, 2)
                h_n += 1

            if self.input_cor[idx, 2, 0] != -1 and self.input_cor[idx, 3, 0] != -1 and self.input_cor[idx, 4, 0] != -1:
                lefthand_dis = (torch.pow((x[idx, 2, 0]-x[idx, 3, 0])**2+(x[idx, 2, 1]-x[idx, 3, 1])**2, 0.5)+torch.pow(
                    (x[idx, 3, 0]-x[idx, 4, 0])**2+(x[idx, 3, 1]-x[idx, 4, 1])**2, 0.5))/2
                gt_lefthand_dis = (torch.pow((self.input_cor[idx, 2, 0]-self.input_cor[idx, 3, 0])**2+(self.input_cor[idx, 2, 1]-self.input_cor[idx, 3, 1])**2, 0.5)+torch.pow(
                    (self.input_cor[idx, 3, 0]-x[idx, 4, 0])**2+(self.input_cor[idx, 3, 1]-self.input_cor[idx, 4, 1])**2, 0.5))/2
                lefthand_loss += torch.pow(lefthand_dis-gt_lefthand_dis, 2)
                lh_n += 1

            if self.input_cor[idx, 5, 0] != -1 and self.input_cor[idx, 6, 0] != -1 and self.input_cor[idx, 7, 0] != -1:
                righthand_dis = (torch.pow((x[idx, 5, 0]-x[idx, 6, 0])**2+(x[idx, 5, 1]-x[idx, 6, 1])**2, 0.5)+torch.pow(
                    (x[idx, 6, 0]-x[idx, 7, 0])**2+(x[idx, 6, 1]-x[idx, 7, 1])**2, 0.5))/2
                gt_righthand_dis = (torch.pow((self.input_cor[idx, 5, 0]-self.input_cor[idx, 6, 0])**2+(self.input_cor[idx, 5, 1]-self.input_cor[idx, 6, 1])**2, 0.5)+torch.pow(
                    (self.input_cor[idx, 6, 0]-x[idx, 7, 0])**2+(self.input_cor[idx, 6, 1]-self.input_cor[idx, 7, 1])**2, 0.5))/2
                righthand_loss += torch.pow(righthand_dis-gt_righthand_dis, 2)
                rh_n += 1

            if self.input_cor[idx, 8, 0] != -1 and self.input_cor[idx, 9, 0] != -1 and self.input_cor[idx, 10, 0] != -1:
                leftleg_dis = (torch.pow((x[idx, 8, 0]-x[idx, 9, 0])**2+(x[idx, 8, 1]-x[idx, 9, 1])**2, 0.5)+torch.pow(
                    (x[idx, 9, 0]-x[idx, 10, 0])**2+(x[idx, 9, 1]-x[idx, 10, 1])**2, 0.5))/2
                gt_leftleg_dis = (torch.pow((self.input_cor[idx, 8, 0]-self.input_cor[idx, 9, 0])**2+(self.input_cor[idx, 8, 1]-self.input_cor[idx, 9, 1])**2, 0.5)+torch.pow(
                    (self.input_cor[idx, 9, 0]-x[idx, 10, 0])**2+(self.input_cor[idx, 9, 1]-self.input_cor[idx, 10, 1])**2, 0.5))/2
                leftleg_loss += torch.pow(leftleg_dis-gt_leftleg_dis, 2)
                ll_n += 1

            if self.input_cor[idx, 11, 0] != -1 and self.input_cor[idx, 12, 0] != -1 and self.input_cor[idx, 13, 0] != -1:
                rightleg_dis = (torch.pow((x[idx, 11, 0]-x[idx, 12, 0])**2+(x[idx, 11, 1]-x[idx, 12, 1])**2, 0.5)+torch.pow(
                    (x[idx, 12, 0]-x[idx, 13, 0])**2+(x[idx, 12, 1]-x[idx, 13, 1])**2, 0.5))/2
                gt_rightleg_dis = (torch.pow((self.input_cor[idx, 11, 0]-self.input_cor[idx, 12, 0])**2+(self.input_cor[idx, 11, 1]-self.input_cor[idx, 12, 1])**2, 0.5)+torch.pow(
                    (self.input_cor[idx, 12, 0]-x[idx, 13, 0])**2+(self.input_cor[idx, 12, 1]-self.input_cor[idx, 13, 1])**2, 0.5))/2
                rightleg_loss += torch.pow(rightleg_dis-gt_rightleg_dis, 2)
                rl_n += 1

        if h_n != 0:
            head_loss = head_loss/h_n
        if lh_n != 0:
            lefthand_loss = lefthand_loss/lh_n
        if rh_n != 0:
            righthand_loss = righthand_loss/rh_n
        if ll_n != 0:
            leftleg_loss = leftleg_loss/ll_n
        if rl_n != 0:
            rightleg_loss = rightleg_loss/rl_n

        kp_loss = kp_loss+head_loss+lefthand_loss + \
            righthand_loss+leftleg_loss+rightleg_loss

        return x, kp_loss, self.miss_mask
