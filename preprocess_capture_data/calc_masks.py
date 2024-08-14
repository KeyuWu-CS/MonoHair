import cv2 as cv
import os 
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import argparse
import faulthandler
faulthandler.enable()


# sys.path.append(os.path.join(sys.path[0], '..'))

# calc silh masks
from submodules.MODNet.src.models.modnet import MODNet
# from hair_segment_matting.MODNet.src.models.modnet import MODNet
from tqdm import tqdm

# calc hair masks
sys.path.append('submodules/CDGNet')
from networks.CDGNet import Res_Deeplab
# from hair_segment_matting.CDGNet.networks.CDGNet import Res_Deeplab
import os
from copy import deepcopy


def postprocess_mask(tensor):
    image = np.array(tensor) * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    return image.astype(np.uint8)

def obtain_modnet_mask(im: torch.tensor, modnet: nn.Module,
                       ref_size = 512,):
    transes = [ transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]
    im_transform = transforms.Compose( transes)
    im = im_transform(im)
    im = im[None, :, :, :]

    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
    
    _, _, matte = modnet(im, True)
    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return matte[None]


def valid(model, valloader, input_size, image_size, num_samples, gpus):
    model.eval()

    parsing_preds = np.zeros((num_samples, image_size[0], image_size[1]),
                             dtype=np.uint8)

    hpreds_lst = []
    wpreds_lst = []

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    eval_scale=[0.66, 0.80, 1.0]
    # eval_scale=[1.0]
    flipped_idx = (15, 14, 17, 16, 19, 18)
    with torch.no_grad():
        for index, image in enumerate(valloader):
            # num_images = image.size(0)
            # print( image.size() )
            # image = image.squeeze()
            if index % 10 == 0:
                print('%d  processd' % (index * 1))
            #====================================================================================            
            mul_outputs = []
            for scale in eval_scale:

                interp_img = torch.nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
                scaled_img = interp_img( image )   
                # print( scaled_img.size() )

                outputs = model( scaled_img.cuda() )

                prediction = outputs[0][-1]
                #==========================================================
                hPreds = outputs[2][0]
                wPreds = outputs[2][1]
                hpreds_lst.append( hPreds[0].data.cpu().numpy() )
                wpreds_lst.append( wPreds[0].data.cpu().numpy() )
                #==========================================================
                single_output = prediction[0]
                flipped_output = prediction[1]
                flipped_output[14:20,:,:]=flipped_output[flipped_idx,:,:]
                single_output += flipped_output.flip(dims=[-1])
                single_output *=0.5
                # print( single_output.size() )
                single_output = interp( single_output.unsqueeze(0) )                 
                mul_outputs.append( single_output[0] )
            fused_prediction = torch.stack( mul_outputs )
            fused_prediction = fused_prediction.mean(0)
            fused_prediction = F.interpolate(fused_prediction[None], size=image_size, mode='bicubic')[0]
            fused_prediction = fused_prediction.permute(1, 2, 0)  # HWC
            fused_prediction = torch.argmax(fused_prediction, dim=2)
            fused_prediction = fused_prediction.data.cpu().numpy()
            parsing_preds[idx, :, :] = np.asarray(fused_prediction, dtype=np.uint8)
            #==================================================================================== 
            idx += 1

    parsing_preds = parsing_preds[:num_samples, :, :]
    return parsing_preds, hpreds_lst, wpreds_lst


    
def calculate_mask(args):
    print("Start calculating masks!")

    os.makedirs(os.path.join(args.scene_path, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(args.scene_path, 'mask_hair'), exist_ok=True)
    os.makedirs(os.path.join(args.scene_path, 'crop_images'), exist_ok=True)
    os.makedirs(os.path.join(args.scene_path, 'hair_mask'), exist_ok=True)

    images = sorted(os.listdir(os.path.join(args.scene_path, 'capture_images')))
    n_images = len(sorted(os.listdir(os.path.join(args.scene_path, 'capture_images'))))
    suffix = images[0][-4:]
    tens_list = []
    for i in range(n_images):
        tens_list.append(T.ToTensor()(Image.open(os.path.join(args.scene_path, 'capture_images', images[i])).convert('RGB')))


#     load MODNET model for silhouette masks
    modnet = nn.DataParallel(MODNet(backbone_pretrained=False))
    modnet.load_state_dict(torch.load(args.MODNET_ckpt))
    device = torch.device('cuda')
    modnet.eval().to(device)
    
    # Create silh masks
    silh_list = []
    for i in tqdm(range(len(tens_list))):
        silh_mask = obtain_modnet_mask(tens_list[i], modnet, 512)
        silh_list.append(silh_mask)
        cv2.imwrite(os.path.join(args.scene_path, 'mask', images[i]), postprocess_mask(silh_mask)[0].astype(np.uint8))
    
    print("Start calculating hair masks!")
#     load CDGNet for hair masks
    model = Res_Deeplab(num_classes=20)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Load existing state_dict and state_dict_old
    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(args.CDGNET_ckpt, map_location='cpu')

    # Create a temp dict to update state_dict while avoiding OrderedDict RuntimeError
    updated_state_dict = {}

    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            # Remove 'module.' from 'key'
            updated_state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            updated_state_dict[key] = deepcopy(state_dict_old[key])

    # Update state_dict after the loop
    state_dict.update(updated_state_dict)

    # Load and set the model's state_dict
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    basenames = sorted([s.split('.')[0] for s in os.listdir(os.path.join(args.scene_path, 'capture_images'))])
    input_size = (1024, 1024)
    # input_size = (1120, 1992)

    raw_images = []
    images = []
    masks = []
    min_w_list = []
    for basename in basenames:
        mask = np.asarray(Image.open(os.path.join(args.scene_path, 'mask', basename + suffix)))
        index = np.nonzero(mask)[1]
        min_w = int(max(np.mean(index) - mask.shape[0]//2 ,0))
        if min_w + mask.shape[0]>= mask.shape[1]:
            min_w -= min_w+mask.shape[0] -mask.shape[1]
        min_w_list.append(min_w)
        img = Image.open(os.path.join(args.scene_path, 'capture_images', basename +suffix)).convert('RGB')
        # raw_images.append(np.asarray(img)[...,:3])
        raw_images.append(np.asarray(img))

        ###crop
        # img_crop = np.asarray(img)
        # img_crop = img_crop[:, min_w:min_w + mask.shape[0]]
        # cv2.imwrite(os.path.join(args.scene_path, 'crop_images', basename + '.JPG'),img_crop[...,::-1])
        # img = Image.open(os.path.join(args.scene_path, 'crop_images', basename + '.JPG'))

        img = transform(img.resize(input_size))[None]
        img = torch.cat([img, torch.flip(img, dims=[-1])], dim=0)
        images.append(img)
        masks.append(mask)

    image_size = (mask.shape[1], mask.shape[0])
    # image_size = (img_crop.shape[1], img_crop.shape[0])
    raw_size = (mask.shape[1], mask.shape[0])
    parsing_preds, hpredLst, wpredLst = valid(model, images, input_size, image_size, len(images), gpus=1)

    for i in range(len(images)):
        hair_mask = np.zeros(raw_size[::-1])
        hair_mask = np.asarray(Image.fromarray((parsing_preds[i] == 2)).resize(image_size, Image.BICUBIC))
        # hair_mask_crop = np.asarray(Image.fromarray((parsing_preds[i] == 2)).resize(image_size, Image.BICUBIC))
        # min_w = min_w_list[i]
        # hair_mask[:,min_w:min_w+raw_size[1]] = hair_mask_crop
        hair_mask = hair_mask * masks[i]
        hair_mask = hair_mask.astype(np.uint8)
        source_img = raw_images[i]

        Image.fromarray(hair_mask).save(os.path.join(args.scene_path, 'hair_mask', basenames[i] + suffix))
        hair_mask = hair_mask[...,None].repeat(3,-1)

        mask_hair = hair_mask*0.5+source_img*0.5
        Image.fromarray(mask_hair.astype(np.uint8)).save(os.path.join(args.scene_path, 'mask_hair', basenames[i] +suffix))

    print('Results saved in folder: ', os.path.join(args.scene_path, 'hair_mask'))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--scene_path', default=r'/datasets/wky/data/mvs_hair/wky07-27/Real_data/wig3', type=str)
    parser.add_argument('--MODNET_ckpt', default='../hair_segment_matting/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt', type=str)
    parser.add_argument('--CDGNET_ckpt', default='../hair_segment_matting/CDGNet/snapshots/LIP_epoch_149.pth', type=str)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    calculate_mask(args)