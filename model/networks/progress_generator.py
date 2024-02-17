import torch.nn as nn
import torch
import functools
import numpy as np
import torch.nn.functional as F
from model.networks.base_function import CrissCrossAttention
from model.networks.base_function import ResTAttention
from model.networks.base_function import RefineAtt
from model.networks.base_function import ParsingBlock
from model.networks.keypoints_generator import KPModel
# import lpips
from skimage.metrics import structural_similarity as ssim



class CoarseBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False):
        super(CoarseBlock, self).__init__()
        self.conv_block_stream1 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.conv_block_stream2 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.conv_block = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias, dilation=2), norm_layer(dim))
        
        # self.conv_block_stream2 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=True, cated_stream2=cated_stream2)
        # self.cca = CrissCrossAttention(256)
        self.ResT = ResTAttention(H=64,W=64,dim=256)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False, cal_att=False):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)


        conv_block += [nn.Conv2d(dim, dim//2, kernel_size=3, padding=1, bias=use_bias, dilation=2),
                           norm_layer(dim//2),
                           nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        # conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias, dilation=2), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x1, x2):
        # x1_out = self.conv_block_stream1(torch.cat((x1,x2), 1))
        # x1_out = self.conv_block_stream1(self.ResT(x1))
        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        x=torch.cat((x1_out,x2_out), 1)
        x=self.conv_block(torch.cat((x1_out,x2_out), 1))
        # x2_out = self.cca(x2)
        # att = F.sigmoid(x2_out)
        # x1_out = x1_out * att
        out = x1 + x # residual connection

        return out


class CBModel(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3, gpu_ids=[], padding_type='reflect', n_downsampling=2):
        super(CBModel, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]
        self.output_nc = output_nc
        self.n_blocks=n_blocks
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        #parsing+bone+facebox
        self.refine_att = RefineAtt(opt.keypoint_nc + opt.par_nc, ngf)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # down_sample
        model_stream1_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc_s1, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        model_stream2_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc_s2, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model_stream1_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult * 2),
                            nn.ReLU(True)]
            model_stream2_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult * 2),
                            nn.ReLU(True)]
        

        # att_block in place of res_block
        mult = 2**n_downsampling
        cated_stream2 = [True for i in range(n_blocks)]
        cated_stream2[0] = False

        #使用相同模块不同参数
        # CBs = nn.ModuleList()
        # for i in range(n_blocks):
        #     CBs.append(CoarseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, cated_stream2=cated_stream2[i]))
        
        # 使用相同模块相同参数
        CB1=CoarseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, cated_stream2=cated_stream2[i])
        CB2=CoarseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, cated_stream2=cated_stream2[i])
        CB3=CoarseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, cated_stream2=cated_stream2[i])

        KG=KPModel(opt.keypoint_nc, opt.batchSize)

        # up_sample
        model_stream1_up = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_stream1_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                            norm_layer(int(ngf * mult / 2)),
                            nn.ReLU(True)]

        model_stream1_up += [nn.ReflectionPad2d(3)]
        model_stream1_up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_stream1_up += [nn.Tanh()]


        # self.model = nn.Sequential(*model)
        self.PB =ParsingBlock(opt.par_nc+opt.keypoint_nc*2)
        self.stream1_down = nn.Sequential(*model_stream1_down)
        self.stream2_down = nn.Sequential(*model_stream2_down)
        # self.CB = CB
        self.CB1 = CB1
        self.CB2 = CB2
        self.CB3 = CB3
        self.stream1_up = nn.Sequential(*model_stream1_up)
        # KG_pretrained=torch.load('./pretrained_model/k_generator15.pth')
        # KG.load_state_dict(KG_pretrained)
        self.KG = KG

        # self.lpipsloss=lpips.LPIPS(net='alex').to(opt.device)
        # self.L1loss = torch.nn.L1Loss()

    def forward(self, input):
        # here x should be a tuple
        img1, pose1, pose2, pose1_cor, pose2_cor, par1 = input
        B,C,H,W=pose1.shape

        # step_vis=torch.zeros([3,6,18,256,256]) #用来查看中间步骤
        gen_poses=torch.zeros([2,B,C,H,W]).cuda() #用来生成姿态

        #keypoints generation step, here x.shape=(BS,18,2)
        self.KG.keypoint_input(input_cor=pose1_cor,data_agumentation=False)
        gen_pose1_cor, _, miss_mask1=self.KG()
        self.KG.keypoint_input(input_cor=pose2_cor,data_agumentation=False)
        gen_pose2_cor, _, miss_mask2=self.KG()

        self.KG.keypoint_input(input_cor=pose1_cor,data_agumentation=True)
        gen_pose1_cor_t, kp_loss1, _2=self.KG()
        self.KG.keypoint_input(input_cor=pose2_cor,data_agumentation=True)     
        gen_pose2_cor_t, kp_loss2, _2=self.KG()
        kp_loss=kp_loss1+kp_loss2

        gen_pose1_cor[miss_mask1]=pose1_cor[miss_mask1].float()
        gen_pose2_cor[miss_mask2]=pose2_cor[miss_mask2].float()

        #绘制生成好的pose keypoints,gen_poses第一个维度为pose1生成结果，第二个维度为pose2生成结果
        for j in range(B):
            for i in range(18):
                x=int(gen_pose1_cor[0,i,0])
                y=int(gen_pose1_cor[0,i,1])
                if x>=0 and x<=255 and y>=0 and y<=255:
                    gen_poses[0, j, i, x, y]=1
                x=int(gen_pose2_cor[0,i,0])
                y=int(gen_pose2_cor[0,i,1])
                if x>=0 and x<=255 and y>=0 and y<=255:
                    gen_poses[1, j, i, x, y]=1

        step=(gen_pose2_cor-gen_pose1_cor)//self.n_blocks

        par2, mask = self.PB(torch.cat((par1, pose1, pose2), 1))
        # print(par2.shape)(1,8,256,256)
        
        step_pose=gen_poses[0,:,:,:,:]
        origin_pose=gen_poses[0,:,:,:,:]
        non_pose=torch.zeros_like(pose1)
        step_par=par1
        origin_par=par1

        step_pose_cor=gen_pose1_cor

        # out = torch.cuda.FloatTensor(self.n_blocks,B,3,H,W).fill_(0)
        x1=img1
        x3=torch.cat((par2,gen_poses[1,:,:,:,:]), 1)
        # down_sample
        x1 = self.stream1_down(x1)

        # loss_constraint=0
        for idx in range(self.n_blocks):
            #change next step pose keypoints
            if idx==self.n_blocks-1:
                origin_pose=step_pose
                step_pose=gen_poses[1,:,:,:,:]
                origin_par=step_par
                step_par=par2
            else:
                step_pose_cor=step_pose_cor+step
                origin_pose=step_pose
                origin_par=step_par
                step_pose=non_pose
                for i in range(B):
                    for j in range(C):
                        if int(step_pose_cor[i,j,0])<0 or int(step_pose_cor[i,j,0])>255 or int(step_pose_cor[i,j,1])<0 or int(step_pose_cor[i,j,1])>255:
                            continue
                        step_pose[i,j,int(step_pose_cor[i,j,0]),int(step_pose_cor[i,j,1])]=1
                # step_vis[idx,:,:,:,:]=step_pose #用来查看中间步骤
                step_par, mask=self.PB(torch.cat((origin_par, origin_pose, step_pose), 1))

            x2=torch.cat((origin_pose, origin_par, step_pose, step_par), 1)
            x2 = self.stream2_down(x2)
            if idx==0:
                x1 = self.CB1(x1, x2)
            elif idx==1:
                x1 = self.CB2(x1, x2)
            elif idx==2:
                x1 = self.CB3(x1, x2)
            # out[idx,:] = self.stream1_up(x1)

        #计算限制损失
        # for i in range(self.n_blocks-1):
        #     # loss_lpips=self.lpipsloss.forward(out[i,:], out[i+1,:])
        #     # loss_lpips=sum(loss_lpips) * (2.0/(self.n_blocks-1))
        #     for j in range(B):
        #         x=np.array(torch.permute(out[i,j,:],(1,2,0)))
        #         y=np.array(torch.permute(out[i+1,j,:],(1,2,0)))
        #         print(ssim(x,y,multichannel=True))
        #         loss_tem = ssim(x,y,multichannel=True) * (10.0/(self.n_blocks-1))
        #         loss_constraint+=torch.mean(loss_tem)
        # loss_constraint=loss_constraint/B
        # print(loss_constraint)

        # up_sample to generate coarse result
        out = self.stream1_up(x1)
        # refine coarse code
        x1 = self.refine_att(x3,x1)
        # return out[self.n_blocks-1,:], x1, par2, gen_poses, kp_loss, loss_constraint
        return out, x1, par2, gen_poses, kp_loss
