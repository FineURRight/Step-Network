import torch
import torch.nn as nn
from model.base_model import BaseModel
from model.networks import base_function, LOSS, facenet
import model.networks as network
from util import task, utils
import itertools
import data as Dataset
import numpy as np
import random
import lpips
import os
from util.utils import get_face_crop
from util import pose_utils
import torchvision.transforms as transforms
from model.networks.discriminator import *


import torch.nn.functional as F

class Painet(BaseModel):
    def name(self):
        return "parsing and inpaint network"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--netG', type=str, default='pose', help='The name of net Generator')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_crs_rec', type=float, default=5.0, help='weight for coarse image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--lambda_style_g', type=float, default=2.0, help='weight for style generation loss')
        parser.add_argument('--lambda_crs_g', type=float, default=2.0, help='weight for coarse generation loss')
        #parser.add_argument('--lambda_correct', type=float, default=5.0, help='weight for the Sampling Correctness loss')
        parser.add_argument('--lambda_style', type=float, default=200.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--lambda_regularization', type=float, default=30.0, help='weight for the affine regularization loss')
        parser.add_argument('--lambda_face', type=float, default=2.0, help='weight for the affine face')
        parser.add_argument('--lambda_kp', type=float, default=0.00001, help='weight for the keypoints')
        # parser.add_argument('--lambda_constraint', type=float, default=200.0, help='w')
        parser.add_argument('--lambda_lpips', type=float, default=3.0, help='weight for the lpips')

        parser.add_argument('--use_spect_g', action='store_false', help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false', help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # self.loss_names = [ 'app_gen', 'content_gen', 'style_gen', 'reg_gen', #'constraint',
        #                    'ad_gen','ad_coarse_gen','dis_img_gen', 'kp', 'par', 'par1', 'lpips', 'face']
        self.loss_names = [ 'app_gen', 'content_gen', 'style_gen', 'reg_gen', #'constraint',
                           'ad_gen','ad_coarse_gen','ad_style','dis_img_gen', 'kp', 'par', 'par1', 'lpips', 'face']

        self.visual_names = ['input_P1','input_P2', 'img_gen']
        self.model_names = ['G','D']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor

        # define the generator
        self.net_G = network.define_g(opt, image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64,
                                 use_spect=opt.use_spect_g, norm='instance', activation='LeakyReLU')

        # define the discriminator 
        if self.opt.dataset_mode == 'fashion':
            self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
        net_SD=Style_Discriminator()
        net_SD.cuda()
        self.net_SD=net_SD

        trained_list = ['parnet']
        for k,v in self.net_G.named_parameters():
            flag = False
            for i in trained_list:
                if i in k:
                    flag = True
            if flag:
                #v.requires_grad = False
                print(k)
                
        if self.isTrain:
            # define the loss functions
            self.GANloss = LOSS.AdversarialLoss(opt.gan_mode).to(opt.device)
            self.L1loss = torch.nn.L1Loss()
            self.S_GANLoss=LOSS.Style_AdversarialLoss()
            self.Mixloss = LOSS.MS_SSIM_L1_LOSS().to(opt.device)
            self.Vggloss = LOSS.VGGLoss().to(opt.device)
            self.parLoss = CrossEntropyLoss2d()#torch.nn.BCELoss()
            self.lpipsloss=lpips.LPIPS(net='alex').to(opt.device)
            self.triloss=LOSS.triplet_loss(batch_size=1)

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                               lr=opt.lr, betas=(0.9, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.9, 0.999))
            self.optimizer_SD = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_SD.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_SD)

        # load the pre-trained model and schedulers
        self.setup(opt)


    def set_input(self, input):
        # move to GPU and change data types
        self.input = input
        input_P1, input_BP1, input_SPL1, input_FB1 ,input_BP1_cor = input['P1'], input['BP1'], input['SPL1'], input['FB1'], input['BP1_cor']
        input_P2, input_BP2, input_SPL2, input_FB2, label_P2, input_BP2_cor = input['P2'], input['BP2'], input['SPL2'], input['FB2'], input['label_P2'], input['BP2_cor']
        input_P3, input_SPL3, input_FB3 = input['P3'], input['SPL3'], input['FB3']

        if len(self.gpu_ids) > 0:
            self.input_P1 = input_P1.cuda()
            self.input_BP1 = input_BP1.cuda()
            self.input_BP1_cor = input_BP1_cor.cuda()
            self.input_SPL1 = input_SPL1.cuda()
            self.input_FB1 = input_FB1.cuda()
            self.input_P2 = input_P2.cuda()
            self.input_BP2 = input_BP2.cuda()  
            self.input_BP2_cor = input_BP2_cor.cuda()  
            self.input_SPL2 = input_SPL2.cuda()
            self.input_FB2 = input_FB2.cuda()
            self.label_P2 = label_P2.cuda()   
            self.input_P3 = input_P3.cuda()
            self.input_SPL3 = input_SPL3.cuda()
            self.input_FB3 = input_FB3.cuda()


        self.image_paths=[]
        self.image_paths_p2=[]
        for i in range(self.input_P1.size(0)):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_' + input['P2_path'][i])
            self.image_paths_p2.append(os.path.join(input['P2_path'][i]))

        self.face_transform=transforms.Compose([
            transforms.Resize((160,160))
        ])
        self.facenet=facenet.Facenet(pretrained=True)
        device=torch.device('cuda:0')
        self.facenet.to(device)
        self.x=torch.zeros([6,3,3,160,160])
        self.x=self.x.cuda()


    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid - 1
        flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid).view(b, c, -1)
        return input_sample

    def test(self):
        """Forward function used in test time"""
        
        self.img_gen, self.coarse_gen, self.loss_reg, self.parsav, self.gen_poses, self.loss_kp, self.same_style, self.diff_style = self.net_G(self.input_P1, self.input_P2, self.input_P3, self.input_BP1, self.input_BP2, self.input_BP1_cor, self.input_BP2_cor, self.input_SPL1, self.input_SPL2, self.input_SPL3)
        # self.img_gen, self.coarse_gen, self.loss_reg, self.parsav, self.same_style, self.diff_style = self.net_G(self.input_P1, self.input_P2, self.input_P3, self.input_BP1, self.input_BP2, self.input_SPL1, self.input_SPL2, self.input_SPL3)
        ## test flow ##

        self.save_results(self.img_gen, data_name='vis')
        # if self.opt.save_input or self.opt.phase == 'test':
        #     self.save_results(self.input_P1, data_name='ref')
        #     self.save_results(self.input_P2, data_name='gt')
        #     result = torch.cat([self.input_P1, img_gen, self.input_P2], 3)
        #     self.save_results(result, data_name='all')
                       
                

    def forward(self):
        """Run forward processing to get the inputs"""
        self.img_gen, self.coarse_gen, self.loss_reg, self.parsav, self.gen_poses, self.loss_kp, self.same_style, self.diff_style = self.net_G(self.input_P1, self.input_P2, self.input_P3, self.input_BP1, self.input_BP2, self.input_BP1_cor, self.input_BP2_cor, self.input_SPL1, self.input_SPL2, self.input_SPL3)
        self.gen_face=get_face_crop(self.img_gen, self.input_FB2)
        self.gt_face = get_face_crop(self.input_P2, self.input_FB2)
        self.P1_face = get_face_crop(self.input_P1, self.input_FB1)
        self.P3_face = get_face_crop(self.input_P3, self.input_FB3)
        #below faceloss
        # if self.gen_face.shape[1]!=0 and self.gen_face.shape[1]!=0:
        x1=self.face_transform(self.gen_face)
        x2=self.face_transform(self.P1_face)
        x3=self.face_transform(self.P3_face)
        self.x[:,0,:,:,:]=x1
        self.x[:,1,:,:,:]=x2
        self.x[:,2,:,:,:]=x3
        self.face_out=torch.zeros(((self.img_gen.shape[0],3,128)))
        for i in range(self.img_gen.shape[0]):
            self.face_out[i,:,:]=self.facenet(self.x[i])
        # x2=self.facenet(x2)

    def backward_D_basic(self, netD, real, fake,fake_coarse):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        # print(D_real.shape)(6,1,16,16)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        #fake_coarse
        D_fake_coarse = netD(fake_coarse.detach())
        D_fake_coarse_loss = self.GANloss(D_fake_coarse, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss + D_fake_coarse_loss) * 0.6
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = LOSS.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.input_P2, self.img_gen,self.coarse_gen)

    def backward_SD_basic(self, net_SD, same, diff):
        base_function._unfreeze(net_SD)
        SD_same=net_SD(same)
        SD_diff=net_SD(diff)
        SD_same_loss = self.S_GANLoss(SD_same, True)
        SD_diff_loss = self.S_GANLoss(SD_diff, False)
        SD_loss=(SD_same_loss+SD_diff_loss)*0.3
        SD_loss.backward(retain_graph=True)
        return SD_loss

    def backward_SD(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_SD)
        self.loss_style_img = self.backward_SD_basic(self.net_SD, self.same_style, self.diff_style)


    def backward_G(self):
        """Calculate training loss for the generator"""
        # Calculate regularzation loss to make transformed feature and target image feature in the same latent space
        self.loss_reg_gen = self.loss_reg * self.opt.lambda_regularization

        #Calculate keypoints loss
        self.loss_kp=self.loss_kp*self.opt.lambda_kp

        #calculate constraint loss
        # self.loss_constraint=self.loss_constraint*self.opt.lambda_constraint

        # Calculate l1 loss 
        # loss_app_final_gen = self.L1loss(self.img_gen, self.input_P2)
        loss_app_final_gen = self.L1loss(self.img_gen, self.input_P2)
        loss_app_final_gen = loss_app_final_gen * self.opt.lambda_rec
        #calculate coarse_gen l1 loss
        loss_app_coarse_gen = self.L1loss(self.coarse_gen, self.input_P2)
        loss_app_coarse_gen = loss_app_coarse_gen * self.opt.lambda_crs_rec
        # self.loss_app_gen=loss_app_final_gen
        self.loss_app_gen=loss_app_final_gen+loss_app_coarse_gen

        # parsing loss
        label_P2 = self.label_P2.squeeze(1).long()
        #print(self.input_SPL2.min(), self.input_SPL2.max(), self.parsav.min(), self.parsav.max())
        self.loss_par = self.parLoss(self.parsav,label_P2)# * 20. 
        self.loss_par1 = self.L1loss(self.parsav, self.input_SPL2)  * 100

        # Calculate GAN loss
        base_function._freeze(self.net_D)
        D_fake = self.net_D(self.img_gen)
        self.loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        #Style GAN loss
        base_function._freeze(self.net_SD)
        same = self.net_SD(self.same_style)
        self.loss_ad_style = self.S_GANLoss(same,True) * self.opt.lambda_style_g

        # Calculate coarse gan loss
        D_coarse_fake = self.net_D(self.coarse_gen)
        self.loss_ad_coarse_gen = self.GANloss(D_coarse_fake, True, False) * self.opt.lambda_crs_g

        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(self.img_gen, self.input_P2) 
        self.loss_style_gen = loss_style_gen*self.opt.lambda_style
        self.loss_content_gen = loss_content_gen*self.opt.lambda_content

        # Calculate lpips loss
        loss_lpips=self.lpipsloss.forward(self.input_P2,self.img_gen)*self.opt.lambda_lpips
        self.loss_lpips=sum(loss_lpips[:,0,0,0])/self.opt.batchSize

        # Calculate face loss
        triplet=0
        for i in range(6):
            triplet+=self.triloss.forward(self.face_out[i])
        triplet/=6
        self.loss_face = (self.L1loss(self.gt_face, self.gen_face)) * self.opt.lambda_face 

        # self.loss_SD=self.loss_SD


        total_loss = 0

        for name in self.loss_names:
            if name != 'dis_img_gen':
                #print(getattr(self, "loss_" + name))
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()


    def optimize_parameters(self):
        """update network weights"""
        self.forward()

        self.optimizer_SD.zero_grad()
        self.backward_SD()
        self.optimizer_SD.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, inputs, targets):
        return self.nll_loss(self.softmax(inputs), targets)
