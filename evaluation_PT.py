import numpy as np
import argparse
import math
# from data.image_folder import make_dataset
import os
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from numpy import trace
from numpy import cov
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from FID import calculate_fid_given_paths
from util.utils import get_face_crop
import torch



parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
parser.add_argument('--test_path', type = str, default='/home/user/mh/STEP/dataset/pair_and_bone/fasion-resize-pairs-test.csv',
                    help='path to save the test dataset path')
parser.add_argument('--num_test', type=int, default=1000,
                    help='how many images to load for each test')
# parser.add_argument('--sample_numbers',type=int,default=50, help='how many sample images for testing')
args = parser.parse_args()
data = open(args.test_path, 'r')

lpips_loss=lpips.LPIPS(net='alex')

box_root="./dataset/fashion_face_bboxes/"

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


if __name__ == "__main__":


    L1=0
    PSNR=0
    SSIM=0

    LPIPS=0
    FID=0
    whichmodel='CASD'

    miss=0

    for idx, i in enumerate(data):
        if idx == 0:
            continue
        paths=i.split(",")
        path_temp=paths[1].split("\n")
        path_gt = str(path_temp[0])
        # path_gen=path_gt.replace('.jpg','_vis.jpg')

        box_name=path_gt.replace(".jpg",".txt")
        box_name=box_name.replace('/','_')
        path_boxes=os.path.join(box_root + box_name)
        path_gt = os.path.join('/home/user/mh/STEP/dataset/' + path_gt)

        if(whichmodel!='mine' and whichmodel!='PATN' and whichmodel!='PT'):
            #test other author eval_result
            i = i.replace('_front', 'front')
            i = i.replace('_full', 'full')
            i = i.replace('_back', 'back')
            i = i.replace('_additional', 'additional')
            i = i.replace('_side', 'side')
            i = i.replace('_flat', 'flat')
            i = i.replace('id_0', 'id0')
        if(whichmodel=='PT'):
            i = i.replace('fashion/','')
            i = i.replace('/', '_')
        else:
            i = i.replace('/', '')


        path_gen_temp=i.split(',')
        if(whichmodel=='mine'):
            path_gen0 = path_gen_temp[0].replace('.jpg', '')
            x = path_gen_temp[1].replace('.jpg', '.jpg_vis.jpg')
        if(whichmodel=='PATN'):
            path_gen0 = path_gen_temp[0].replace('.jpg', '')
            x = path_gen_temp[1].replace('.jpg', '_vis.jpg')
        if(whichmodel=='PISE'):
            path_gen0 = path_gen_temp[0].replace('.jpg', '')
            x = path_gen_temp[1].replace('.jpg', '_vis.jpg')
        if(whichmodel=='ADGAN' or whichmodel=='PINet' or whichmodel=='PoNA'):
            path_gen0 = path_gen_temp[0]
            x = path_gen_temp[1].replace('.jpg', '.jpg_vis.jpg')
        if(whichmodel=='PT'):
            path_gen0 = path_gen_temp[0]
            x = path_gen_temp[1].replace('.jpg', '.jpg.png')
        if(whichmodel=='PIDM'or whichmodel == 'CASD' or whichmodel == 'NTED'):
            path_gen0 = path_gen_temp[0].replace('.jpg', '')
            x = path_gen_temp[1].replace('.jpg', '_vis.png')
        if(whichmodel=='UPGPT' ):
            path_gen0 = path_gen_temp[0].replace('.jpg', '')
            x = path_gen_temp[1].replace('.jpg', '.jpg')
        #如果跑dior，注释下面两行
        path_gen1_temp = x.split("\n")
        path_gen1=path_gen1_temp[0]
#fashionMENJackets_Vestsid0000401402_4full_2_fashionMENJackets_Vestsid0000401402_6flat_vis.png
        #mine
        if(whichmodel=='mine'):
            path_gen = os.path.join('/home/user/mh/STEP/eval_results/' + path_gen0+'_'+path_gen1)
        #PISE
        if(whichmodel=='PISE'):
            path_gen = os.path.join('/home/user/mh/eval_results_author/latest/' + path_gen0+'_'+path_gen1)
        #ADGAN
        if (whichmodel == 'ADGAN'):
            path_gen = os.path.join('/home/user/mh/other_result/ADGAN/' + path_gen0 + '___' + path_gen1)
        #PATN
        if (whichmodel == 'PINet'):
            path_gen = os.path.join('/home/user/mh/other_result/PINet/' + path_gen0 + '___' + path_gen1)
        # PINet 
        if (whichmodel == 'PATN'):
            path_gen = os.path.join('/home/user/mh/other_result/PATN/' + path_gen0 + '___' + path_gen1)
        #PoNA
        if (whichmodel == 'PoNA'):
            path_gen = os.path.join('/home/user/mh/other_result/PoNA/PoNA/' + path_gen0 + '___' + path_gen1)
        # DiOr
        if (whichmodel == 'DiOr'):
            path_gen = os.path.join('/home/user/mh/other_result/DiOr/' + 'generated_'+str(idx)+'.jpg')
        # PT
        if (whichmodel == 'PT'):
            path_gen = os.path.join('/home/user/mh/other_result/PT2_imgs/deepf/' + path_gen0 + 'to' + path_gen1)
        # PIDM
        if (whichmodel == 'PIDM'):
            path_gen = os.path.join('/home/user/mh/other_result/PIDM/' + path_gen0 + '_2_' + path_gen1)
        if (whichmodel == 'CASD' or whichmodel == 'NTED'):
            path_gen = os.path.join('/home/user/mh/other_result/CandN/' + path_gen0 + '_2_' + path_gen1)
        # UPGPT
        if (whichmodel == 'UPGPT'):
            path_gen = os.path.join('/home/user/mh/other_result/UPGPT/pose_transfer/samples/' + path_gen0 + '___' + path_gen1)

        if not os.path.exists(path_gen):
            miss+=1
            continue
#fashionMENJackets_Vestsid0000065304_2side_2_fashionMENJackets_Vestsid0000065304_3back_vis



        gt_image = Image.open(path_gt).convert('RGB')
        gen_image = Image.open(path_gen).convert('RGB')


        if whichmodel=='PIDM':
            t=transforms.Resize([256,176])
            gen_image=t(gen_image)
        if whichmodel == 'CASD' or whichmodel == 'NTED':
            t=transforms.Resize([256,1760])
            gen_image=t(gen_image)

        gt_numpy = np.array(gt_image)
        gen_numpy = np.array(gen_image)
        gt_numpy=gt_numpy/255
        gen_numpy=gen_numpy/255

        gt_numpy = gt_numpy[:, 40:216, :]
        if(whichmodel=='ADGAN'):
            gen_numpy = gen_numpy[:, 704:880, :]
        elif(whichmodel=='DiOr'):
            gen_numpy=gen_numpy[:,352:528,:]
        elif(whichmodel=='mine' or whichmodel=='PISE' or whichmodel=='PATN'):
            gen_numpy = gen_numpy[:, 40:216, :]
        elif(whichmodel == 'CASD' ):
            gen_numpy = gen_numpy[:, 1232:1408, :]
        elif(whichmodel == 'NTED' ):
            gen_numpy = gen_numpy[:, 1408:1584, :]

        #l1_temp, PSNR_temp, TV_temp = compute_errors(gt_numpy, gen_numpy)
        PSNR_temp=psnr(gen_numpy,gt_numpy)
        SSIM_temp = ssim(gt_numpy, gen_numpy,multichannel=True)
        img0 = lpips.im2tensor(gt_numpy*255)
        img1 = lpips.im2tensor(gen_numpy*255)
        LPIPS_temp=float(lpips_loss.forward(img0,img1))
        gt_f=gt_numpy[np.newaxis,:]
        gen_f = gen_numpy[np.newaxis, :]
        FID_temp=calculate_fid_given_paths(gen_f,gt_f,batch_size=1,cuda=True,dims=2048)/10

        #L1+=l1_temp
        PSNR+=PSNR_temp
        SSIM += SSIM_temp
        LPIPS+=LPIPS_temp
        FID += FID_temp
        #TV+=TV_temp
        if idx % 100 == 0:
            print("now:")
            print(PSNR_temp,SSIM_temp,LPIPS_temp,FID_temp)

    #L1=L1/idx
    PSNR=PSNR/(idx-miss)
    SSIM=SSIM/(idx-miss)
    LPIPS=LPIPS/(idx-miss)
    FID = FID / (idx-miss)
    #TV=TV/idx

    print('{:>10},{:>10},{:>10},{:>10}'.format('PSNR', 'SSIM','LPIPS','FID'))
    print('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(PSNR, SSIM,LPIPS,FID))


