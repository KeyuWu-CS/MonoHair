from PIL import Image
import numpy as np
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from skimage.filters import difference_of_gaussians
import math
import os
import tqdm
import cv2
import argparse

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def generate_gabor_filters(sigma_x, sigma_y, freq, num_filters):
    thetas = np.linspace(0, math.pi * (num_filters - 1) / num_filters, num_filters)
    kernels = []
    for theta in thetas:
        kernel = np.real(gabor_kernel(freq, theta=math.pi - theta, sigma_x=sigma_x, sigma_y=sigma_y))
        kernels.append(kernel)
    return kernels


def calc_orients(img, kernels):
    gray_img = rgb2gray(img)
    filtered_image = difference_of_gaussians(gray_img, 0.4, 10)
    gabor_filtered_images = [ndi.convolve(filtered_image, kernels[i], mode='wrap') for i in range(len(kernels))]
    F_orients = np.abs(np.stack(gabor_filtered_images)) # abs because we only measure angle in [0, pi]
    return F_orients


def calc_confidences(F_orients, orientation_map,args):
    orients_bins = np.linspace(0, math.pi * (args.num_filters - 1) / args.num_filters, args.num_filters)
    orients_bins = orients_bins[:, None, None]
    
    orientation_map = orientation_map[None]
    
    dists = np.minimum(np.abs(orientation_map - orients_bins), 
                       np.minimum(np.abs(orientation_map - orients_bins - math.pi),
                                  np.abs(orientation_map - orients_bins + math.pi)))

    F_orients_norm = F_orients / F_orients.sum(axis=0, keepdims=True)
    
    V_F = (dists**2 * F_orients_norm).sum(0)
    
    return V_F

def main(args):

    os.makedirs(args.orient_dir, exist_ok=True)
    os.makedirs(args.conf_dir, exist_ok=True)
    
    kernels = generate_gabor_filters(args.sigma_x, args.sigma_y, args.freq, args.num_filters)
    
    img_list = sorted(os.listdir(args.img_path))[:1]
    for img_name in tqdm.tqdm(img_list):
        basename = img_name.split('.')[0]
        img = np.array(Image.open(os.path.join(args.img_path, img_name)))
        mask = np.array(Image.open(os.path.join(args.mask_path, img_name)))

        mask = mask/np.max(mask)
        # img = img*mask[...,None]
        F_orients = calc_orients(img, kernels)
        orientation_map = F_orients.argmax(0)

        # orientation_map = np.where(orientation_map<=90,90 - orientation_map, orientation_map)
        # orientation_map = np.where(orientation_map>90,270 - orientation_map, orientation_map)
        # orientation_map_rad = orientation_map.astype('float16') / args.num_filters * math.pi
        orientation_map_rad = orientation_map / args.num_filters * math.pi

        # indices_cm2 = np.stack([np.cos(orientation_map_rad-0.5*np.pi) * 0.5 + 0.5,
        #                         -np.sin(orientation_map_rad-0.5*np.pi) * 0.5 + 0.5, np.zeros_like(orientation_map_rad)], axis=2)
        indices_cm2 = np.stack([np.cos(orientation_map_rad ) * 0.5 + 0.5,
                                np.sin(orientation_map_rad ) * 0.5 + 0.5,
                                np.zeros_like(orientation_map_rad)], axis=2)
        indices_cm2 =indices_cm2.astype(np.float32)*255
        cv2.imwrite(f'{args.orient_dir}/{basename}_ori.png',indices_cm2)

        indices_cm2 = cv2.cvtColor(indices_cm2, cv2.COLOR_RGB2BGR)
        confidence_map = calc_confidences(F_orients, orientation_map_rad,args)
        cv2.imwrite(f'{args.orient_dir}/{basename}1.png',indices_cm2)
        # confidence_map = 1-confidence_map
        confidence_map = 1/confidence_map**2
        # confidence_map[confidence_map<0.5]=0
        cv2.imwrite(f'{args.orient_dir}/{basename}_conf.png',confidence_map*255/10)


        cv2.imwrite(f'{args.orient_dir}/{basename}.png', orientation_map.astype('uint8'))
        np.save(f'{args.conf_dir}/{basename}.npy', confidence_map.astype('float16'))


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    root = 'E:\wukeyu\hair\data\mvshair\wky07-27\Real_data\wig1'
    parser.add_argument('--img_path', default=os.path.join(root,'capture_images'), type=str)
    parser.add_argument('--orient_dir', default= os.path.join(root,'orientation_maps'), type=str)
    parser.add_argument('--conf_dir', default= os.path.join(root,'confidence_maps'), type=str)
    parser.add_argument('--mask_path', default=os.path.join(root,'hair_mask'), type=str)
    parser.add_argument('--sigma_x', default=1.8, type=float)
    parser.add_argument('--sigma_y', default=2.4, type=float)
    parser.add_argument('--freq', default=0.23, type=float)
    parser.add_argument('--num_filters', default=180, type=int)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)