import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import imageio
import cv2
import numpy as np
import random
from skimage.filters import difference_of_gaussians
import  platform

class calOrientationGabor(nn.Module):
    def __init__(self,channel_in=1,channel_out=1,stride=1):
        super(calOrientationGabor,self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.Tensor = torch.cuda.FloatTensor

        self.numKernels =180
        self.clamp_confidence_low=0.0
        self.clamp_confidence_high=0.2



    def filter(self,image,label,threshold,variance_data,orient_data,max_resp_data,sigma_x,sigma_y,Lambda,kernel_size):
        resArray = []
        for iOrient in range(self.numKernels):
            theta = nn.Parameter(torch.ones(self.channel_out) * (math.pi * iOrient / self.numKernels),
                                 requires_grad=False).cuda()
            GaborKernel = self.gabor_fn(kernel_size, self.channel_in, self.channel_out, theta,sigma_x,sigma_y,Lambda)
            # save_image(GaborKernel,r'test/{}.png'.format(iOrient))
            response = F.conv2d(image, GaborKernel, padding=kernel_size//2)
            resArray.append(response.clone())

        resTensor = resArray[0]
        H, W = resTensor.size()[2:4]
        orient = torch.zeros(1, 1, H, W).cuda()
        for iOrient in range(1, self.numKernels):
            resTensor = torch.cat([resTensor, resArray[iOrient]], dim=1)
            orient = torch.cat([orient, torch.ones(1, 1, H, W).cuda() * math.pi * iOrient / self.numKernels], dim=1)

        # argmax the response

        resTensor = torch.abs(resTensor)
        max_resp = torch.max(resTensor, dim=1, keepdim=True)[0]
        maxResTensor = torch.argmax(resTensor, dim=1, keepdim=True).float()
        best_orientTensor = maxResTensor * math.pi / self.numKernels

        orient_diff = torch.minimum(torch.abs(best_orientTensor - orient),
                           torch.minimum(torch.abs(best_orientTensor - orient - math.pi),
                                      torch.abs(best_orientTensor - orient + math.pi)))



        # #### compute confidence similar with
        # resTensor = torch.abs(resTensor)
        # res_norm = resTensor/torch.maximum(torch.sum(resTensor,dim=1,keepdim=True),1e-5*torch.ones_like(torch.sum(resTensor,dim=1,keepdim=True)))
        #
        # variance = orient_diff*orient_diff*res_norm*3
        # variance = torch.sum(variance,dim=1,keepdim=True)
        # variance = 1/(variance**2)
        #
        #
        # orient_data =best_orientTensor
        # variance_data = variance
        # confidenceTensor = variance_data


        #### compute confidence

        resp_diff = resTensor - max_resp
        variance = torch.sum(orient_diff * resp_diff * resp_diff, dim=1, keepdim=True)
        variance = variance ** (1 / 2)

        orient_data = torch.where(variance > variance_data, best_orientTensor, orient_data)
        max_resp_data = torch.where(variance > variance_data, max_resp, max_resp_data)
        variance_data = torch.where(variance > variance_data, variance, variance_data)

        max_all_resp = torch.max(max_resp_data)

        max_all_var = torch.max(variance_data)
        max_resp_data /= max_all_resp
        variance_data /= max_all_var

        confidenceTensor = (variance_data - self.clamp_confidence_low) / (
                        self.clamp_confidence_high - self.clamp_confidence_low)

        confidenceTensor = confidenceTensor.clamp(0, 1)

        return confidenceTensor,variance_data,orient_data



    def forward(self, image,label,iter=1,threshold=0.0):
        # filter the image with different orientations
        H,W=image.size()[2:4]
        variance_data=torch.ones(1,1,H,W).cuda()*0
        orient_data=torch.ones(1,1,H,W).cuda()*0
        max_resp_data=torch.ones(1,1,H,W).cuda()*0

        for i in range(iter):
            confidenceTensor,variance_data,orient_data = self.filter(image,label,threshold,variance_data,orient_data,max_resp_data,sigma_x=1.8,sigma_y=2.4,Lambda=4,kernel_size=17)
            image = confidenceTensor

        confidenceTensor[confidenceTensor < threshold] = 0
        best_orientTensor=orient_data
        orientTwoChannel = torch.cat([torch.sin(best_orientTensor), torch.cos( best_orientTensor)],dim=1)

        return orientTwoChannel,best_orientTensor,confidenceTensor

    def gabor_fn(self,kernel_size, channel_in, channel_out,theta,sigma_x,sigma_y,Lambda,phase=0.):
        sigma_x = nn.Parameter(torch.ones(channel_out) * sigma_x, requires_grad=False).cuda()
        sigma_y = nn.Parameter(torch.ones(channel_out) * sigma_y, requires_grad=False).cuda()
        Lambda = nn.Parameter(torch.ones(channel_out) * Lambda, requires_grad=False).cuda()
        psi = nn.Parameter(torch.ones(channel_out) * phase, requires_grad=False).cuda()

        # Bounding box
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax
        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).cuda().float()-0.5
        # y_0=y_0
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
        x_0 = torch.arange(xmin, xmax + 1).cuda().float()-0.5
        # x_0=x_0
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1,
                                   ksize).float()  # [channel_out, channelin, kernel, kernel]

        # Rotation
        # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

        # [channel_out, channel_in, kernel, kernel]
        gb = torch.exp(
            -.5 * (x_theta ** 2 / sigma_x.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2)) \
             * torch.cos(2 * math.pi* x_theta / Lambda.view(-1, 1, 1, 1)  + psi.view(-1, 1, 1, 1))

        return gb


def normalize_tensor(x):
    norm=torch.norm(x,p=2,dim=1)
    return x/torch.maximum(norm,1e-8)

def normalize(x):
    norm=np.linalg.norm(x,axis=-1)[...,None]
    # print(np.maximum(norm,1e-8).shape[:])
    return   x/np.maximum(norm,1e-8)

def convert_numpy(tensor):
    tensor=torch.squeeze(tensor,0)
    tensor=tensor.permute(1,2,0)
    tensor=tensor.data.cpu().numpy()
    return tensor


def calculate_orientation(image_dir,label_dir,save_root,filename=None,iter=1,threshold=0.0):

    ori_path = os.path.join(save_root,'Ori')
    conf_path = os.path.join(save_root,'conf')
    best_ori_path = os.path.join(save_root,'best_ori')
    if not os.path.exists(conf_path):
        os.makedirs(conf_path)
    if not os.path.exists(ori_path):
        os.makedirs(ori_path)
    if not os.path.exists(best_ori_path):
        os.makedirs(best_ori_path)

    ori_path = os.path.join(ori_path, filename)
    conf_path = os.path.join(conf_path, filename)
    best_ori_path = os.path.join(best_ori_path,filename)

    ##define gabor filter
    gabor = calOrientationGabor()
    gabor = gabor.cuda()

    transform1 = transforms.Compose([
        transforms.ToTensor(),

    ])

    ### difference of Gaussian
    image = np.array(Image.open(image_dir).convert('L'))
    label = np.array(Image.open(label_dir))/255.0
    image = difference_of_gaussians(image, 0.4, 10)
    # cv2. imwrite(os.path.join(save_root,'Ori','dg.png'),image*255*10)

    label = Image.open(label_dir)
    image = transform1(image).type(torch.float)
    label = transform1(label)
    image = torch.unsqueeze(image, 0)

    label = torch.unsqueeze(label, 0)
    label = label[:, 0:1, ...]
    gray=image
    gray = gray.cuda()
    label = label.cuda()
    label[label>=0]=1

    ori,best_ori, confidence = gabor(gray, label,iter,threshold=threshold)
    # save_image(best_ori / math.pi * 180 / 255,best_ori_path)
    cv2.imwrite(best_ori_path,best_ori[0].cpu().numpy().transpose(1,2,0) / math.pi * 180,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    save_image(confidence, conf_path)
    label=convert_numpy(label)
    confidence=convert_numpy(confidence)
    ori = convert_numpy(ori)


    # confidence[confidence>0]=1

    ori=(ori+1)/2
    confidence_mask = confidence.copy()
    confidence_mask[confidence_mask<threshold]=0
    confidence_mask[confidence_mask>threshold]=1
    # cv2.imwrite(save_root+'/Ori/conf.png',confidence_mask*255)
    H, W = ori.shape[:2]
    cv2.imwrite(ori_path,np.concatenate([np.ones((H, W, 1)), ori], axis=2)[..., ::-1] * 255,[int(cv2.IMWRITE_JPEG_QUALITY), 100])






def batch_generate(root,image_folder):

    files = os.listdir(os.path.join(root,image_folder))
    for file in files:
        image_dir = os.path.join(root,image_folder,file)
        label_dir = os.path.join(root,'hair_mask',file)
        calculate_orientation(image_dir, label_dir, save_root=root, filename=file, iter=1, threshold=0.0, )


if __name__ == '__main__':


    # root = 'E:\wukeyu\hair\data\mvshair\wky07-12\Real_data\wig07-12'
    # filename = 'IMG_3004.JPG'
    # image_dir = os.path.join(root,'capture_images',filename)
    # label_dir = os.path.join(root,'mask',filename)
    # calculate_orientation(image_dir,label_dir,save_root=root,filename=filename,iter=1,threshold=0.,)

    if platform.system()=='Windows':
        root = 'E:\wukeyu\hair\data\mvshair\wky07-27\Real_data1\person0'
    else:
        root = '/datasets/wky/data/mvs_hair/wky07-27/Real_data/wig3'
    batch_generate(root,'capture_images')



















