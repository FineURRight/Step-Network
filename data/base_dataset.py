import os
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import torchvision.transforms.functional as F
from util import pose_utils
from PIL import Image
import pandas as pd
import torch
import math
import numbers

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()


    def initialize(self, opt):
        self.opt = opt
        self.image_dir, self.bone_file, self.name_pairs, self.par_dir, self.bbox_dir = self.get_paths(opt)
        size = len(self.name_pairs)
        self.dataset_size = size
        self.class_num = 8

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size


        transform_list=[]
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list) 

        self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        par_paths = []
        bbox_paths=[]
        assert False, "A subclass of MarkovAttnDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths, par_paths, bbox_paths

    def __getitem__(self, index):
        P1_name, P2_name = self.name_pairs[index]
        P3_name, _ =self.name_pairs[np.random.choice(range(0, len(self.name_pairs)))]
        P1_path = os.path.join(self.image_dir, P1_name) # person 1
        P2_path = os.path.join(self.image_dir, P2_name) # person 2
        P3_path = os.path.join(self.image_dir, P3_name) # person 3

        P1_SP=P1_name.replace('/','')
        P2_SP=P2_name.replace('/','')
        P3_SP=P3_name.replace('/','')

        P1_FB_path = P1_SP.replace('jpg', 'txt')
        P2_FB_path = P2_SP.replace('jpg', 'txt')
        P3_FB_path = P3_SP.replace('jpg', 'txt')

        P1_SP=P1_SP.replace('jpg','png')
        P2_SP=P2_SP.replace('jpg','png')
        P3_SP=P3_SP.replace('jpg','png')
        
        P1_SP,P2_SP,P3_SP = P1_SP.replace('_back', 'back'),P2_SP.replace('_back', 'back'),P3_SP.replace('_back', 'back')
        P1_SP,P2_SP,P3_SP  = P1_SP.replace('_front', 'front'),P2_SP.replace('_front', 'front'),P3_SP.replace('_front', 'front')
        P1_SP,P2_SP,P3_SP  = P1_SP.replace('_side', 'side'),P2_SP.replace('_side', 'side'),P3_SP.replace('_side', 'side')
        P1_SP,P2_SP,P3_SP  = P1_SP.replace('_additional', 'additional'),P2_SP.replace('_additional', 'additional'),P3_SP.replace('_additional', 'additional')
        P1_SP,P2_SP,P3_SP  = P1_SP.replace('_full', 'full'),P2_SP.replace('_full', 'full'),P3_SP.replace('_full', 'full')
        P1_SP,P2_SP,P3_SP  = P1_SP.replace('_flat', 'flat'),P2_SP.replace('_flat', 'flat'),P3_SP.replace('_flat', 'flat')
        P1_SP,P2_SP,P3_SP  = P1_SP.replace('id_0', 'id0'),P2_SP.replace('id_0', 'id0'),P3_SP.replace('id_0', 'id0')
        
        SPL1_path = os.path.join(self.par_dir, P1_SP)
        SPL2_path = os.path.join(self.par_dir, P2_SP)
        SPL3_path = os.path.join(self.par_dir, P3_SP)

        regions = (40,0,216,256)
        P1_img = Image.open(P1_path).convert('RGB')#.crop(regions)
        P2_img = Image.open(P2_path).convert('RGB')#.crop(regions)
        P3_img = Image.open(P3_path).convert('RGB')#.crop(regions)
        SPL1_img = Image.open(SPL1_path)#.crop(regions)
        SPL2_img = Image.open(SPL2_path)#.crop(regions)
        SPL3_img = Image.open(SPL3_path)#.crop(regions)

        P1_FB_path = os.path.join(self.bbox_dir, P1_FB_path)
        P2_FB_path = os.path.join(self.bbox_dir, P2_FB_path)
        P3_FB_path = os.path.join(self.bbox_dir, P3_FB_path)

        #FACE BOX
        FB1 = self.obtain_box(P1_FB_path)
        FB2 = self.obtain_box(P2_FB_path)
        FB3 = self.obtain_box(P3_FB_path)
        # FB1 = FB1.permute(2,0,1)
        # FB2 = FB2.permute(2,0,1)

        s1np = np.expand_dims(np.array(SPL1_img),-1)
        s2np = np.expand_dims(np.array(SPL2_img), -1)
        s3np = np.expand_dims(np.array(SPL3_img), -1)
        s1np = np.concatenate([s1np,s1np,s1np], -1)
        s2np = np.concatenate([s2np,s2np,s2np], -1)
        s3np = np.concatenate([s3np,s3np,s3np], -1)
        
    
        SPL1_img = Image.fromarray(np.uint8(s1np))
        SPL2_img = Image.fromarray(np.uint8(s2np))
        SPL3_img = Image.fromarray(np.uint8(s3np))


        BP1_path=os.path.join(self.image_dir, P1_name+'.txt')
        BP1,BP1_cor = self.obtain_bone(BP1_path)
        P1 = self.trans(P1_img)

        BP2_path=os.path.join(self.image_dir, P2_name+'.txt')
        BP2,BP2_cor = self.obtain_bone(BP2_path)
        P2 = self.trans(P2_img)

        P3 = self.trans(P3_img)

        SPL1_img = np.expand_dims(np.array(SPL1_img)[:,:,0],0)#[:,:,40:-40] # 1*256*176
        SPL2_img = np.expand_dims(np.array(SPL2_img)[:,:,0],0)#[:,:,40:-40]
        SPL3_img = np.expand_dims(np.array(SPL3_img)[:,:,0],0)#[:,:,40:-40]

        #print(SPL1_img.shape)
       # SPL1_img = SPL1_img.transpose(2,0)
       # SPL2_img = SPL2_img.transpose(2,0)
        _, h, w = SPL2_img.shape
       # print(SPL2_img.shape,SPL1_img.shape)
        num_class = self.class_num
        tmp = torch.from_numpy(SPL2_img).view( -1).long()
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        SPL2_onehot = ones.view([h,w, num_class])
        #print(SPL2_onehot.shape)
        SPL2_onehot = SPL2_onehot.permute(2,0,1)


        tmp = torch.from_numpy(SPL1_img).view( -1).long()
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        SPL1_onehot = ones.view([h,w, num_class])
        #print(SPL2_onehot.shape)
        SPL1_onehot = SPL1_onehot.permute(2,0,1)

        tmp = torch.from_numpy(SPL3_img).view( -1).long()
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        SPL3_onehot = ones.view([h,w, num_class])
        #print(SPL2_onehot.shape)
        SPL3_onehot = SPL3_onehot.permute(2,0,1)

        #print(SPL1.shape)
        SPL2 = torch.from_numpy(SPL2_img).long()


        return {'P1': P1, 'BP1': BP1, 'BP1_cor': BP1_cor, 'P2': P2, 'BP2': BP2, 'BP2_cor': BP2_cor, 'SPL1': SPL1_onehot, 'SPL2':SPL2_onehot, 'FB1': FB1, 'FB2': FB2, 'label_P2': SPL2,
                'P1_path': P1_name, 'P2_path': P2_name, 'P3': P3, 'FB3': FB3, 'SPL3': SPL3_onehot}


    def obtain_bone(self, path):
        pose,pose_cor = pose_utils.load_pose_cords(path)
        pose = np.transpose(pose,(2, 0, 1))
        pose = torch.Tensor(pose)
        return pose,pose_cor

    def obtain_box(self, path):
        txt = open(path, 'r')
        box_data = torch.IntTensor(2,2)
        # box = torch.zeros([256, 256, 1])
        for idx, line in enumerate(txt):
            cords = line.split(",")
            box_data[0][0] = float(cords[0])
            box_data[0][1] = float(cords[1])
            box_data[1][0] = float(cords[2])
            box_data[1][1] = float(cords[3])
        #     if int(float(cords[0]))!=-1:
        #         box[int(float(cords[0])), int(float(cords[1])-0.5), 0] = 1
        #         box[int(float(cords[0])), int(float(cords[3])-0.5), 0] = 1
        #         box[int(float(cords[2])), int(float(cords[1])-0.5), 0] = 1
        #         box[int(float(cords[2])), int(float(cords[3])-0.5), 0] = 1
        # box = torch.Tensor(box)
        return box_data

   

    def __len__(self):
        return self.dataset_size

    def name(self):
        assert False, "A subclass of BaseDataset must override self.name"

    def getRandomAffineParam(self):
        if self.opt.is_angle is not False:
            angle = np.random.uniform(low=0, high=30)
        else:
            angle = 0
        if self.opt.is_scale is not False:
            scale = np.random.uniform(low=0.6, high=1)
        else:
            scale=1
        if self.opt.is_shift is not False:
            shift_x = np.random.uniform(low=0.7, high=1)
            shift_y = np.random.uniform(low=0.7, high=1)
        else:
            shift_x=0
            shift_y=0
        return angle, (shift_x,shift_y), scale

    def get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        # code from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#affine
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RSS is rotation with scale and shear matrix
        #       RSS(a, scale, shear) = [ cos(a + shear_y)*scale    -sin(a + shear_x)*scale     0]
        #                              [ sin(a + shear_y)*scale    cos(a + shear_x)*scale     0]
        #                              [     0                  0          1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1


        angle = math.radians(angle)
        if isinstance(shear, (tuple, list)) and len(shear) == 2:
            shear = [math.radians(s) for s in shear]
        elif isinstance(shear, numbers.Number):
            shear = math.radians(shear)
            shear = [shear, 0]
        else:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " +
                "two values. Got {}".format(shear))
        scale = 1.0 / scale

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
            math.sin(angle + shear[0]) * math.sin(angle + shear[1])
        matrix = [
            math.cos(angle + shear[0]), math.sin(angle + shear[0]), 0,
            -math.sin(angle + shear[1]), math.cos(angle + shear[1]), 0
        ]
        matrix = [scale / d * m for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]
        return matrix


 
