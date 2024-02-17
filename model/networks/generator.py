import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.networks.base_network import BaseNetwork
from model.networks.progress_generator import CBModel
from model.networks.base_function import *
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
from model.networks.discriminator import *

from util.utils import feature_normalize


class PoseGenerator(BaseNetwork):
    def __init__(self, opt, image_nc=3, structure_nc=18, output_nc=3, ngf=64, norm='instance',
                activation='LeakyReLU', use_spect=True, use_coord=False):
        super(PoseGenerator, self).__init__()

        self.use_coordconv = True
        self.match_kernel = 3

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        
        self.coarse_generator=CBModel(opt, input_nc=[3,(opt.keypoint_nc+opt.par_nc)*2],output_nc=output_nc,ngf=ngf,n_blocks=3)

        self.style_encoder = StyleEncoder(opt.image_nc, ngf)

        self.imgenc = VggEncoder()
        self.getMatrix = GetMatrix(ngf*4, 1)
        
        self.phi = nn.Conv2d(in_channels=ngf*4+3, out_channels=ngf*4, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=ngf*4+3, out_channels=ngf*4, kernel_size=1, stride=1, padding=0)

        #self.coarse_enc = RefineAtt(8+18, ngf)

        self.dec = BasicDecoder(3)

        #extract feature block
        self.isb = ISB(ngf*4, 256)
        self.res = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
                
        self.res1 = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
        
        net_SD=Style_Discriminator()
        self.net_SD=net_SD

        self.loss_SD = torch.nn.MSELoss()


    def addcoords(self, x):
        bs, _, h, w = x.shape
        xx_ones = torch.ones([bs, h, 1], dtype=x.dtype, device=x.device)
        xx_range = torch.arange(w, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(1)

        yy_ones = torch.ones([bs, 1, w], dtype=x.dtype, device=x.device)
        yy_range = torch.arange(h, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(1)

        xx_channel = xx_channel.float() / (w - 1)
        yy_channel = yy_channel.float() / (h - 1)
        xx_channel = 2 * xx_channel - 1
        yy_channel = 2 * yy_channel - 1

        rr_channel = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
        
        concat = torch.cat((x, xx_channel, yy_channel, rr_channel), dim=1)

        return concat

    def computecorrespondence(self, fea1, fea2, temperature=0.01,
                detach_flag=False,
                WTA_scale_weight=1,
                alpha=1):
        ## normalize the feature
        ## borrow from https://github.com/microsoft/CoCosNet
        batch_size = fea2.shape[0] 
        channel_size = fea2.shape[1]
        theta = self.theta(fea1)
        if self.match_kernel == 1:
            theta = theta.view(batch_size, channel_size, -1)  # 2*256*(feature_height*feature_width)
        else:
            theta = F.unfold(theta, kernel_size=self.match_kernel, padding=int(self.match_kernel // 2))

        dim_mean = 1
        theta = theta - theta.mean(dim=dim_mean, keepdim=True)
        theta_norm = torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon
        theta = torch.div(theta, theta_norm)
        theta_permute = theta.permute(0,2,1)

        phi = self.phi(fea2)
        if self.match_kernel == 1:
            phi = phi.view(batch_size, channel_size, -1)  # 2*256*(feature_height*feature_width)
        else:
            phi = F.unfold(phi, kernel_size=self.match_kernel, padding=int(self.match_kernel // 2))
        phi = phi - phi.mean(dim=dim_mean, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon
        phi = torch.div(phi, phi_norm)

        f = torch.matmul(theta_permute, phi)
        if WTA_scale_weight == 1:
            f_WTA = f
        else:
            f_WTA = WTA_scale.apply(f, WTA_scale_weight)
        f_WTA = f_WTA / temperature
        #print(f.shape)
        
        att = F.softmax(f_WTA.permute(0,2,1), dim=-1)
        return att
        



    def forward(self, img1, img2, img3, pose1, pose2, pose1_cor, pose2_cor, par1, par2, par3):
        #extract style
        codes_vector, exist_vector, img1code = self.style_encoder(img1, par1)
        #print(codes_vector.shape)(1,9,256)(area)
        #print(exist_vector.shape)(1,8)(label)
        #print(img1code.shape)(1,256,4,4)

        #coarse gen by coarse_generator
        coarse_gen, coarse_pcode, par2, gen_poses, kp_loss = self.coarse_generator([img1, pose1, pose2, pose1_cor, pose2_cor, par1])
        #print(coarse_gen.shape)(1,3,256,256)

        # instance transfer
        for _ in range(1):
            """share weights to normalize features use efb prograssively"""
            parcode = self.isb(coarse_pcode, par2, codes_vector, exist_vector)
            parcode = self.res(parcode)
        #print(parcode.shape)(1,256,64,64)
        
        #here is VGGenccoder
        img2code = self.imgenc(img2)
        #print(img2code.shape)(1,256,64,64)
        img2code1 = feature_normalize(img2code)
        loss_reg = F.mse_loss(img2code, parcode)

        # if True:
        #     img1code = self.imgenc(img1)
            
        #     parcode1 = feature_normalize(parcode)
        #     img1code1 = feature_normalize(img1code)
            
        #     if self.use_coordconv:
        #         parcode1 = self.addcoords(parcode1)
        #         img1code1 = self.addcoords(img1code1)
                #print(parcode1.shape)(1,259,64,64)
                #print(img1code1.shape)(1,259,64,64)

            # gamma, beta = self.getMatrix(img1code)
            # batch_size, channel_size, h,w = gamma.shape
            # att = self.computecorrespondence(parcode1, img1code1)
            # #print(att.shape)(1,4096,4096)
            # #here are the scale and bias
            # gamma = gamma.view(batch_size, 1, -1)
            # beta = beta.view(batch_size, 1, -1)
            # #print(gamma.shape)(1,1,4096)
            # #print(beta.shape)(1,1,4096)
            # imgamma = torch.bmm(gamma, att)
            # imbeta = torch.bmm(beta, att)
            # #print(imgamma.shape)(1,1,4096)
            #
            # imgamma = imgamma.view(batch_size,1,h,w).contiguous()
            # imbeta = imbeta.view(batch_size,1,h,w).contiguous()
            #
            # parcode = parcode*(1+imgamma)+imbeta
            # parcode = self.res1(parcode)

        final_gen = self.dec(parcode)
        #print(final_gen.shape)(1,3,256,256)
        #print(par2.shape)(1,8,256,256)

        codes_vector2, exist_vector2, img2code = self.style_encoder(final_gen, par2)
        codes_vector3, exist_vector3, img3code = self.style_encoder(img3, par3)

        same_style_code=torch.cat([codes_vector,codes_vector2],1)
        diff_style_code=torch.cat([codes_vector,codes_vector3],1)
        # SD_same=self.net_SD(same_style_code)
        # SD_diff=self.net_SD(diff_style_code)
        # # print(SD_same)
        # same=torch.ones_like(SD_same)
        # diff=torch.zeros_like(SD_diff)
        # SD_same_loss = self.loss_SD(SD_same, same)
        # # print(SD_same_loss)
        # SD_diff_loss = self.loss_SD(SD_diff, diff)
        # loss_SD=(SD_same_loss+SD_diff_loss)

        return final_gen, coarse_gen, loss_reg, par2, gen_poses, kp_loss, same_style_code, diff_style_code







