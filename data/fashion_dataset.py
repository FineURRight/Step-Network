import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import pandas as pd
from util import pose_utils
import numpy as np
import torch

class FashionDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(old_size=(256, 256))
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)
        return parser



    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
        pairLst = os.path.join(root, 'pair_and_bone/fasion-resize-pairs-%s.csv' %phase)
        # pairLst = os.path.join(root, 'pair_and_bone/generate_more2.csv')
        name_pairs = self.init_categories(pairLst)

        image_dir = os.path.join(root)
        bonesLst = os.path.join(root, 'pair_and_bone/fasion-resize-annotation-%s.csv' %phase)
        par_dir = os.path.join(root, '%sSPL8' %phase)
        # par_dir = os.path.join(root, '%sSPL8' %phase)
        bbox_dir = os.path.join(root,'fashion_face_bboxes')
        return image_dir, bonesLst, name_pairs, par_dir, bbox_dir


    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')  
        return pairs    

    def name(self):
        return "FashionDataset"

                
