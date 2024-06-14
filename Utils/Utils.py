import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
from PIL import Image
import scipy.io
import scipy
from scipy.spatial.transform import Rotation as R
import trimesh
import open3d as o3d
import open3d.core as o3c
import time
import math
import copy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from trimesh.visual import texture,TextureVisuals
from tqdm import tqdm
import struct




def load_strand(file, return_strands=False):
    with open(file, mode='rb')as f:
        num_strand = f.read(4)

        (num_strand,) = struct.unpack('I', num_strand)
        point_count = f.read(4)

        (point_count,) = struct.unpack('I', point_count)

        # print("num_strand:",num_strand)
        segments = f.read(2 * num_strand)
        segments = struct.unpack('H' * num_strand, segments)
        segments = list(segments)

        num_points = sum(segments)

        points = f.read(4 * num_points * 3)
        points = struct.unpack('f' * num_points * 3, points)


    f.close()
    points = list(points)
    # points=[[points[i*3+0],points[i*3+1],points[i*3+2]] for i in range(len(points)//3)]
    points = np.array(points)
    points = np.reshape(points, (-1, 3))
    # points*=1000

    if return_strands:
        beg = 0
        strands = []
        oris = []
        for seg in segments:
            end = beg + seg
            strand = points[beg:end]
            strands.append(strand)
            dir = np.concatenate([strand[1:] - strand[:-1], strand[-1:] - strand[-2:-1]], 0)
            dir = dir/np.linalg.norm(dir,2,-1,keepdims=True)
            oris.append(dir)
            beg += seg
        return segments, points, strands, oris
    else:
        return segments, points

def load_bust(path):
    bust = trimesh.load(path)
    vertices = np.array(bust.vertices)
    faces = np.array(bust.faces)
    normals = np.array(bust.vertex_normals)
    return vertices, faces,normals


def getProjPoints(points, view, projection, tsfm=None):
    '''
    project 3D points on camera plane, return projected 2D locations (range [-1, 1])
    each view's contribution for each point, weighted by distance
    :param points: [4, N], N = num of points
    :param view: [V, 4, 4], V = num of views
    :param projection: [4, 4], projection matrix
    :param tsfm: [4, 4], model transformation
    :return: projected 2D points, [V, N, 1, 2]; each view's weight for each point, [V, 1, N]
    '''
    # [V, 4, N], world -> view -> projection
    model_points = torch.matmul(tsfm, points) if tsfm is not None else points
    view_points = torch.matmul(view, model_points)
    proj_points = torch.matmul(projection, view_points)
    # divide coordinates by homogeneous coefficient to perform perspective projection
    xy_coord = (proj_points[:, :2, :] / proj_points[:, 3:4, :]).transpose(1, 2).unsqueeze(2)
    # flip y-axis because y positive in OpenGL screen space is opposite to y positive in tensor/img coordinate
    xy_coord[..., 1] *= -1

    return xy_coord


def check_pts_in_views(coords, sample_range='all', view_ids=None):
    '''
    check if a point is in all visible views
    :param coords: [V, N, 1, 2]
    :return:
    '''
    ck_coords = torch.less(coords.abs(), 1.0)
    if sample_range == 'any':
        # in bound of any view
        in_view = torch.zeros((coords.shape[1], ), dtype=torch.bool).to(ck_coords.device)
        if view_ids is not None:
            for view_id in view_ids:
                ck_coord = ck_coords[view_id]
                in_view = torch.logical_or(in_view, torch.logical_and(ck_coord[:, 0, 0], ck_coord[:, 0, 1]))
        else:
            for ck_coord in ck_coords:
                in_view = torch.logical_or(in_view, torch.logical_and(ck_coord[:, 0, 0], ck_coord[:, 0, 1]))

        return in_view
    else:
        # in bound of all views
        in_view = torch.ones((coords.shape[1],), dtype=torch.bool).to(ck_coords.device)
        for ck_coord in ck_coords:
            in_view = torch.logical_and(in_view, torch.logical_and(ck_coord[:, 0, 0], ck_coord[:, 0, 1]))

        return in_view


def direct_valid_in_mask(coords, pts_view, masks, depths, view_ids=()):
    '''
    a trick to complement thin strands in frontal lower parts, which might be missed by network due to occlusion
    :param coords:
    :param pts_view:
    :param masks:
    :param depths:
    :param view_ids:
    :return:
    '''
    # [V, N]
    mask_feat_list = F.grid_sample(masks, coords, align_corners=False, padding_mode='border').squeeze()
    # [V, N]
    depth_feat_list = F.grid_sample(depths, coords, align_corners=False, padding_mode='zeros').squeeze()
    # to compensate inaccurate depth
    depth_eps = 0.02
    # [V, N]
    z_feat_list = -pts_view[:, 2]

    direct_valid = torch.ones([mask_feat_list.shape[1], ], dtype=torch.bool).to(mask_feat_list.device)
    # currently, only optimize front and back views
    # for side views, our bust model cannot handle accurate occlusion
    for i in view_ids:
        mask_feat = mask_feat_list[i]
        in_mask = torch.not_equal(mask_feat, 0)
        in_lower_parts = torch.greater(coords[i, :, 0, 1], 0.3)

        depth_feat = depth_feat_list[i]
        z_feat = z_feat_list[i]
        in_front_parts = torch.less_equal(z_feat, depth_feat + depth_eps)
        # points out of the image should not be excluded
        in_img = torch.greater(depth_feat, 0)

        # mask_valid = torch.logical_and(in_mask, front)
        # mask_valid = torch.logical_and(mask_valid, in_img)
        view_valid = torch.logical_and(in_img, in_mask)
        part_valid = torch.logical_and(in_front_parts, in_lower_parts)
        cur_valid = torch.logical_and(view_valid, in_lower_parts)

        direct_valid = torch.logical_and(direct_valid, cur_valid)

    return direct_valid


def check_pts_in_mask(coords, pts_view, masks, depths, view_ids=()):
    '''
    check if a point should be visible, used at inference time to exclude empty regions
    exclude points in front of bust depth and not in hair mask
    :param coords: [V, N, 1, 2]
    :param pts_view: [V, 4, N]
    :param masks: [V, 1, H, W]
    :param depths: [V, 1, H, W]
    :return:
    '''
    # [V, N]
    mask_feat_list = F.grid_sample(masks, coords, align_corners=False, padding_mode='border').squeeze()
    # [V, N]
    depth_feat_list = F.grid_sample(depths, coords, align_corners=False, padding_mode='zeros').squeeze()
    # to compensate inaccurate depth
    depth_eps = 0.02
    # [V, N]
    z_feat_list = -pts_view[:, 2]

    exclude = torch.zeros([mask_feat_list.shape[1], ], dtype=torch.bool).to(mask_feat_list.device)
    # currently, only optimize front and back views
    # for side views, our bust model cannot handle accurate occlusion
    for i in view_ids:
        mask_feat = mask_feat_list[i]
        out_mask = torch.eq(mask_feat, 0)

        depth_feat = depth_feat_list[i]
        z_feat = z_feat_list[i]
        front = torch.less_equal(z_feat, depth_feat + depth_eps)
        # points out of the image should not be excluded
        in_img = torch.greater(depth_feat, 0)

        empty = torch.logical_and(out_mask, front)
        empty = torch.logical_and(empty, in_img)

        exclude = torch.logical_or(exclude, empty)

    return torch.logical_not(exclude)


def resize_and_rename_imgs(args):
    '''
    resize image to a size suitable for network
    :param args:
    :return:
    '''

    img_folder = os.path.join(args.root_folder, args.img_folder)
    views_list = os.listdir(img_folder)
    for view in views_list:
        view_path = os.path.join(img_folder, view)
        origin_img = cv2.imread(os.path.join(view_path, args.origin_fname), cv2.IMREAD_COLOR)
        if args.img_scale == 1.0:
            cv2.imwrite(os.path.join(view_path, args.img_fname), origin_img)
        else:
            img_resize = cv2.resize(origin_img, dsize=None, fx=args.img_scale, fy=args.img_scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(view_path, args.img_fname), img_resize)

def get_pi_orient(args):
    '''
    tranform un-directional orientation to directional, double-pi range to pi range
    generate 2 maps, one with all direction pointing down, the other pointing up
    :param args:
    :return:
    '''
    img_folder = os.path.join(args.root_folder, args.img_folder)
    views_list = os.listdir(img_folder)
    for view in views_list:
        view_path = os.path.join(img_folder, view)
        dense_map = np.array(Image.open(os.path.join(view_path, args.dense_fname)))[..., :2].astype('float32') / 255.0 * 2. - 1.
        mask = np.greater(np.array(Image.open(os.path.join(view_path, args.mask_fname)).convert('L')),
                          args.mask_thresh)[..., np.newaxis]
        theta = np.arctan2(dense_map[..., 1:], dense_map[..., :1]) / 2.

        theta_down = np.where(theta < 0., theta, theta - np.pi)
        down_map = np.concatenate([np.cos(theta_down) * 0.5 + 0.5, np.sin(theta_down) * 0.5 + 0.5, np.zeros_like(theta_down)], axis=-1)
        down_map = cv2.cvtColor(down_map, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(view_path, args.down_fname), down_map * mask * 255.0)

        theta_up = np.where(theta > 0., theta, theta + np.pi)
        up_map = np.concatenate(
            [np.cos(theta_up) * 0.5 + 0.5, np.sin(theta_up) * 0.5 + 0.5, np.zeros_like(theta_up)], axis=-1)
        up_map = cv2.cvtColor(up_map, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(view_path, args.up_fname), up_map * mask * 255.0)


def correct_orient(args):
    '''
    invert direction in some regions to get correct orientation
    :param args:
    :return:
    '''
    img_folder = os.path.join(args.root_folder, args.img_folder)
    views_list = os.listdir(img_folder)
    for view in views_list[:1]:
        view_path = os.path.join(img_folder, view)
        pi_map = np.array(Image.open(os.path.join(view_path, args.dense_pi_fname)))[..., :2].astype(
            'float32') / 255.0 * 2. - 1.
        inv_map = np.greater(np.array(Image.open(os.path.join(view_path, args.inv_fname)).convert('L')),
                          args.mask_thresh)[..., np.newaxis]
        inv_sign = np.where(inv_map, -1., 1.).astype('float32')
        dir_map = pi_map * inv_sign
        dir_map = np.concatenate([dir_map * 0.5 + 0.5, np.zeros_like(inv_sign)], axis=-1)
        dir_map = cv2.cvtColor(dir_map, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(view_path, args.dense_dir_fname), dir_map * 255.0)

def gen_dir_orient(args):
    img_folder = os.path.join(args.root_folder, args.img_folder)
    views_list = os.listdir(img_folder)
    for view in views_list:
        view_path = os.path.join(img_folder, view)
        mask = np.array(Image.open(os.path.join(view_path, args.mask_fname)).convert('L'))[..., np.newaxis] > 100
        orient_dpi = np.array(Image.open(os.path.join(view_path, args.dense_fname)).convert('RGB'))[..., :2].astype('float32') / 255.0 * 2. - 1.
        theta = np.arctan2(orient_dpi[..., 1:], orient_dpi[..., :1]) / 2.
        orient_pi = np.concatenate([np.cos(theta), np.sin(theta)], axis=-1)
        gt_dir = np.array(Image.open(os.path.join(view_path, 'orient2d_dir.png')).convert('RGB'))[..., :2].astype('float32') / 255.0 * 2. - 1.
        dot = np.sum(orient_pi * gt_dir, axis=-1, keepdims=True)

        blank = np.zeros(mask.shape, dtype='float32')

        orient_dir = ((orient_pi * np.sign(dot)) * 0.5 + 0.5) * mask
        orient_dir = (np.concatenate([orient_dir, blank], axis=-1) * 255.0).astype('uint8')
        cv2.cvtColor(orient_dir, cv2.COLOR_RGB2BGR, dst=orient_dir)
        cv2.imwrite(os.path.join(view_path, args.dense_dir_fname), orient_dir)

def getViewOrients(orients, view):
    '''
    transform orientation to camera view coordinate, only rotation
    :param orients: [3, N], N = num of points
    :param view: [V, 4, 4], V = num of views
    :return: [N, 3, V], orients in camera view coordinates
    '''
    # only use rotation transform
    view = view[:, :3, :3]
    # [V, 3, N] -> [N, 3, V]
    view_orients = torch.matmul(view, orients).transpose(0, 2)
    return view_orients


def getRefViewOrients(orients, view):
    '''
    transform orientation to reference camera view coordinate, only rotation
    :param orients: [3, N]
    :param view: [4, 4]
    :return: [3, N], orientation in reference camera view
    '''
    view_rot = view[:3, :3]
    # [3, N]
    view_orients = torch.matmul(view_rot, orients)
    return view_orients


def getRelativeTrans(views):
    '''
    get relative transformations from other frames to the 1st frames
    input views: V_1, V_2, ..., V_n
    output trans: I, V_1 * V_2 ^ {-1}, ..., V_1 * V_n ^ {-1}
    :param views: [V, 4, 4], view matrices
    :return: [V, 4, 4], relative transformations
    '''
    # set 1st frame as reference frame
    view_ref = views[0]
    views_inv = torch.inverse(views)
    return torch.matmul(view_ref, views_inv)


class OccMetric():
    '''
    evaluate occupancy
    '''

    def __init__(self):
        self.tp_cnt = 0
        self.fp_cnt = 0
        self.fn_cnt = 0

    def clear(self):
        self.tp_cnt = 0
        self.fp_cnt = 0
        self.fn_cnt = 0

    def addBatchEval(self, input, target):
        '''
        evaluate occupancy prediction
        :param input: [N,], boolean
        :param target: [N,], boolean
        '''

        self.tp_cnt += torch.sum(torch.logical_and(input, target)).cpu().item()
        self.fp_cnt += torch.sum(torch.logical_and(input, torch.logical_not(target))).cpu().item()
        self.fn_cnt += torch.sum(torch.logical_and(torch.logical_not(input), target)).cpu().item()

    def getPrecisionAndRecall(self):
        return self.tp_cnt / max((self.tp_cnt + self.fp_cnt), 1), self.tp_cnt / max((self.tp_cnt + self.fn_cnt), 1)


def save_std_img(case_folder, imgs, fnames, unify_name='undistort.png'):
    '''
    generally save images under standard hierarchy
    imgs are read from any customized dataset
    :param case_folder:
    :param imgs:
    :param fnames:
    :param unify_name:
    :return:
    '''
    img_ufolder = os.path.join(case_folder, 'imgs')
    if not os.path.exists(img_ufolder):
        os.mkdir(img_ufolder)

    for index, fname in enumerate(fnames):
        img_folder = os.path.join(img_ufolder, fname)
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
        cv2.imwrite(os.path.join(img_folder, unify_name), imgs[index])


def norm_img_res(img, crop_w, crop_h, target_res=(1200, 800), scale=1, bg_color=0):
    '''
    normalize image resolution to (800, 1200), so that the pretrained model of DenoiseNet can perform well
    DenoiseNet's perfomance depends on the image resolution since it is trained using 800x1200 synthetic images
    This normalization is performed to avoid the nuisances caused by different image resolution
    :param img:
    :param crop_x: crop or padding along x axis
    :param crop_y: crop or padding along y axis
    :param target_res:
    :param scale:
    :return:
    '''
    bg = np.full((target_res[0], target_res[1], img.shape[-1]), fill_value=bg_color, dtype='uint8')
    img_cp = img.copy() if scale == 1 else cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img_cp.shape[:2]
    th, tw = target_res
    if crop_w:
        cp_w = (w - tw) // 2
        img_cp = img_cp[:, cp_w:cp_w + tw]
    if crop_h:
        cp_h = (h - th) // 2
        img_cp = img_cp[cp_h:cp_h + th]

    nh, nw = img_cp.shape[:2]
    pad_w = (tw - nw) // 2
    pad_h = (th - nh) // 2
    bg[pad_h:pad_h + nh, pad_w:pad_w+nw] = img_cp

    return bg

def batch_norm_img_res(case_folder, src_fname, tgt_fname, crop_w, crop_h, target_res=(1200, 800), scale=1):
    view_ufolder = os.path.join(case_folder, 'imgs')
    view_fname_list = os.listdir(view_ufolder)

    for view_fname in view_fname_list:
        src_img = cv2.imread(os.path.join(view_ufolder, view_fname, src_fname))
        tgt_img = norm_img_res(src_img, crop_w, crop_h, target_res, scale)
        cv2.imwrite(os.path.join(view_ufolder, view_fname, tgt_fname), tgt_img)

def recover_img_res(src_folder, tgt_folder, crop_w, crop_h, target_res=(1024, 1024), scale=1):

    img_fname_list = os.listdir(src_folder)
    for img_fname in img_fname_list:
        img = cv2.imread(os.path.join(src_folder, img_fname), cv2.IMREAD_UNCHANGED)
        resize = norm_img_res(img, crop_w, crop_h, target_res, scale, bg_color=0)
        cv2.imwrite(os.path.join(tgt_folder, img_fname), resize)

def gather_imgs(case_folder, src_fname_list, tgt_sub_list):
    view_ufolder = os.path.join(case_folder, 'imgs')
    view_list = os.listdir(view_ufolder)
    for index, src_fname in enumerate(src_fname_list):
        tgt_sub = tgt_sub_list[index]
        tgt_folder = os.path.join(case_folder, tgt_sub)
        if not os.path.exists(tgt_folder):
            os.mkdir(tgt_folder)
        for view in view_list:
            img = cv2.imread(os.path.join(view_ufolder, view, src_fname))
            cv2.imwrite(os.path.join(tgt_folder, '{}.png'.format(view)), img)

def norm_cam_intrin(intrin_mat, src_res, tgt_res=(1200, 800), scale=1, slct_vids=None):
    slct_intrin = intrin_mat if slct_vids is None else intrin_mat[slct_vids]
    fx, fy, cx, cy = slct_intrin[:, 0, 0], slct_intrin[:, 1, 1], slct_intrin[:, 0, 2], slct_intrin[:, 1, 2]
    nfx = fx * scale
    nfy = fy * scale
    h, w = src_res
    th, tw = tgt_res
    ncx = (cx - w / 2) * scale + tw / 2
    ncy = (cy - h / 2) * scale + th / 2

    ndc_fx = 2. * nfx / tw
    ndc_fy = 2. * nfy / th
    ndc_cx = 1 - 2. * ncx / tw
    ndc_cy = 1 - 2. * (th - ncy) / th

    return np.stack([nfx, nfy, ncx, ncy], axis=1), np.stack([ndc_fx, ndc_fy, ndc_cx, ndc_cy], axis=1)

def min_line_dist(rays_o, rays_d):
    A_i = np.eye(3).astype('float32') - rays_d * np.transpose(rays_d, [0, 2, 1])
    b_i = np.transpose(A_i, [0, 2, 1]) @ A_i @ rays_o
    pt_mindist = np.squeeze(np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
    return pt_mindist

def normalize(x):
    return x / np.linalg.norm(x)

def spherify_cam_poses(poses, front_vid, norm_rad=0.8, slct_vids=None):
    '''

    :param poses: c2w
    :param front_vid: set its location as world coord's +z, and its +y as world coord's +y
    :param slct_vids:
    :return:
    '''
    # to OpenGL coordinate style
    front_pose = poses[front_vid]
    slct_poses = poses if slct_vids is None else poses[slct_vids]

    ray_d = slct_poses[:, :3, 2:3]
    ray_o = slct_poses[:, :3, 3:4]

    center = min_line_dist(ray_o, ray_d)
    up = normalize((-1) * front_pose[:3, 1])
    front = normalize(front_pose[:3, 3] - center)
    left = normalize(np.cross(up, front))

    world = np.stack([left, up, front, center], axis=1)
    homo_base = np.array([[0, 0, 0, 1]], dtype='float32')
    world = np.concatenate([world, homo_base], axis=0)
    poses_recenter = np.linalg.inv(world) @ slct_poses

    radius = np.mean(np.linalg.norm(poses_recenter[:, :3, 3], axis=-1))
    rad_scale = norm_rad / radius
    poses_recenter[:, :3, 3] *= rad_scale

    recover_tsfm = world.copy()
    recover_tsfm[:3, :3] /= rad_scale

    poses_recenter[:, :3, 1:3] *= -1

    return poses_recenter, recover_tsfm

def save_cam_params(case_fname, poses, intrins, ndc_proj, recover_tsfm):
    save_folder = os.path.join('.\\camera\\calib_data', case_fname)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    poses_data = np.concatenate([np.float32([len(poses)]), poses.reshape(-1)]).astype('float32')
    ndc_data = np.concatenate([np.float32([len(ndc_proj)]), ndc_proj.reshape(-1)]).astype('float32')

    intrin_data = np.zeros((len(intrins), 13))
    intrin_data[:, 4:8] = intrins
    intrin_data = intrin_data.astype('float32')

    poses_data.tofile(os.path.join(save_folder, 'poses_recenter.dat'))
    intrin_data.tofile(os.path.join(save_folder, 'intrin.dat'))
    ndc_data.tofile(os.path.join(save_folder, 'ndc_proj.dat'))
    recover_tsfm.tofile(os.path.join(save_folder, 'recover_tsfm.dat'))


def cam_to_nerf_fmt(case_folder, img_folder, poses, intrins, res, scale=1):
    '''
    for quick verification in instant ngp
    :param poses:
    :param intrins:
    :param res:
    :return:
    '''
    import json
    class NpEncoder(json.JSONEncoder):
        '''
        json file format does not support np.float32 type
        use this class as a converter from np.* to python native types
        '''

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    h, w = res
    cam_params_dict = {
        'aabb_scale': 1,
        'scale': scale,
        'offset': [0.5, 0.5, 0.5],
        'frames': []
    }
    img_fname_list = sorted(os.listdir(os.path.join(case_folder, img_folder)))
    for index, img_fname in enumerate(img_fname_list):
        intrin = intrins[index]
        fx, fy, cx, cy = intrin[0][0], intrin[1][1], intrin[0][2], intrin[1][2]
        img_info = {
            'file_path': os.path.join(img_folder, img_fname),
            # 'sharpness': sharpness(os.path.join(case_path, imgFolder, img_fname)),
            'transform_matrix': poses[index],
            'camera_angle_x': np.arctan(w / 2.0 / fx) * 2,
            'camera_angle_y': np.arctan(h / 2.0 / fy) * 2,
            'fl_x': fx,
            'fl_y': fy,
            'k1': 0.0,
            'k2': 0.0,
            'p1': 0.0,
            'p2': 0.0,
            'cx': cx,
            'cy': cy,
            'w': w,
            'h': h
        }
        cam_params_dict['frames'].append(img_info)

    with open(os.path.join(case_folder, 'transforms.json'), 'w') as save_file:
        json.dump(cam_params_dict, save_file, cls=NpEncoder, indent=4)
        # save_file.write(json.dumps(cam_params_dict, cls=NpEncoder, indent=4))

# def integrate_fix_view_data(root_folder, dst_folder, inte_list):
#     '''
#     save images as a union binary file, to speed up file reading during training
#     origin file: [model_tsfm.dat, dense_orient.png, raw_conf.png, mask.png]
#     :return:
#     '''
#     import shutil
#     from PIL import Image
#     item_list = os.listdir(root_folder)
#     item_list.sort()
#
#     if not os.path.exists(dst_folder):
#         os.mkdir(dst_folder)
#     check=True
#     for item in item_list:
#         print('==> item {}'.format(item))
#         if item=="502":
#             check=False
#         # if check:
#         #    continue
#
#         cases_folder = os.path.join(root_folder, item, 'cases')
#         case_list = os.listdir(cases_folder)
#
#         case_list.sort()
#         for case in case_list:
#             print('=> case {}'.format(case))
#             dst_case_folder = os.path.join(dst_folder, item + '_' + case)
#             if not os.path.exists(dst_case_folder):
#                 os.mkdir(dst_case_folder)
#
#             # 1. model transform
#             if 'model_tsfm' in inte_list:
#                 shutil.copyfile(os.path.join(cases_folder, case, 'model_tsfm_complete.dat'), os.path.join(dst_case_folder, 'model_tsfm.dat'))
#
#             views_folder = os.path.join(cases_folder, case, 'views')
#             views_list = os.listdir(views_folder)
#             dir_list = []
#             dir_list1 = []
#             orient_list = []
#             conf_list = []
#             mask_list = []
#             depth_list = []
#             for view in views_list:
#                 view_path = os.path.join(views_folder, view)
#                 if 'dir' in inte_list:
#                     # dir = np.array(Image.open(os.path.join(view_path, 'orient2d_dir.png')).convert('RGB'))[..., :2]
#                     # dir_list.append(dir)
#                     dir = np.array(Image.open(os.path.join(view_path, 'dir.png')).convert('RGB'))[..., :2]
#                     dir1 = np.array(Image.open(os.path.join(view_path, 'orient2d_dir.png')).convert('RGB'))[..., :2]
#                     dir_list.append(dir)
#                     dir_list1.append(dir1)
#                 # orient = np.array(Image.open(os.path.join(view_path, 'dense.png')).convert('RGB'))[..., :2]
#                 # orient_list.append(orient)
#                 if 'conf' in inte_list:
#                     conf = np.array(Image.open(os.path.join(view_path, 'raw_conf.png')).convert('L'))
#                     conf_list.append(conf)
#                 if 'mask' in inte_list:
#                     mask = np.array(Image.open(os.path.join(view_path, 'mask.png')).convert('L'))
#                     mask_list.append(mask)
#                 if 'depth' in inte_list:
#                     depth = np.array(Image.open(os.path.join(view_path, 'bust_depth.png')).convert('L'))
#                     depth_list.append(depth)
#
#             # 2.1 directional orientation
#             if 'dir' in inte_list:
#                 dir_union = np.stack(dir_list, axis=0).transpose((0, 3, 1, 2))
#                 dir_union.tofile(os.path.join(dst_case_folder, 'dir_union.dat'),)
#                 dir_union1 = np.stack(dir_list1, axis=0).transpose((0, 3, 1, 2))
#                 dir_union1.tofile(os.path.join(dst_case_folder, 'orient2d_dir_union.dat'), )
#             # # 2.2 un-directional orientation
#             # orient_union = np.stack(orient_list, axis=0).transpose((0, 3, 1, 2))
#             # orient_union.tofile(os.path.join(dst_case_folder, 'orient_union.dat'))
#             # 3. confidence
#             if 'conf' in inte_list:
#                 conf_union = np.stack(conf_list, axis=0)
#                 conf_union.tofile(os.path.join(dst_case_folder, 'conf_union.dat'))
#             # 4. hair mask
#             if 'mask' in inte_list:
#                 mask_union = np.stack(mask_list, axis=0)
#                 mask_union.tofile(os.path.join(dst_case_folder, 'mask_union.dat'))
#             # 5. bust depth
#             if 'depth' in inte_list:
#                 depth_union = np.stack(depth_list, axis=0)
#                 depth_union.tofile(os.path.join(dst_case_folder, 'depth_union.dat'))

def integrate_fix_view_data(root_folder, dst_folder, inte_list):
    '''
    save images as a union binary file, to speed up file reading during training
    origin file: [model_tsfm.dat, dense_orient.png, raw_conf.png, mask.png]
    :return:
    '''
    import shutil
    from PIL import Image
    item_list = os.listdir(root_folder)
    item_list.sort()

    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    for item in item_list:
        print('==> item {}'.format(item))

        cases_folder = os.path.join(root_folder, item, 'cases')
        case_list = os.listdir(cases_folder)

        case_list.sort()
        for case in case_list:
            print('=> case {}'.format(case))
            dst_case_folder = os.path.join(dst_folder, item + '_' + case)
            if not os.path.exists(dst_case_folder):
                os.mkdir(dst_case_folder)

            # 1. model transform
            if 'model_tsfm' in inte_list:
                shutil.copyfile(os.path.join(cases_folder, case, 'model_tsfm_complete.dat'), os.path.join(dst_case_folder, 'model_tsfm.dat'))

            views_folder = os.path.join(cases_folder, case, 'views')
            views_list = os.listdir(views_folder)
            views_list.sort()
            # print(views_list)
            dir_list = []
            orient_list = []
            conf_list = []
            mask_list = []
            depth_list = []
            bust_hair_depth_list = []
            for view in views_list:
                view_path = os.path.join(views_folder, view)
                if 'dir' in inte_list:
                    # dir = np.array(Image.open(os.path.join(view_path, 'orient2d_dir.png')).convert('RGB'))[..., :2]
                    # dir_list.append(dir)
                    orient = np.array(Image.open(os.path.join(view_path, 'dense.png')).convert('RGB'))[..., :2]
                    orient_list.append(orient)
                if 'conf' in inte_list:
                    conf = np.array(Image.open(os.path.join(view_path, 'raw_conf.png')).convert('L'))
                    conf_list.append(conf)
                if 'mask' in inte_list:
                    mask = np.array(Image.open(os.path.join(view_path, 'mask.png')).convert('L'))
                    mask_list.append(mask)
                if 'depth' in inte_list:
                    depth = np.array(Image.open(os.path.join(view_path, 'bust_depth.png')).convert('L'))
                    depth_list.append(depth)
                    bust_hair_depth = np.array(Image.open(os.path.join(view_path, 'bust_hair_depth.png')).convert('L'))
                    bust_hair_depth_list.append(bust_hair_depth)

            # 2.1 directional orientation
            if 'dir' in inte_list:
                # dir_union = np.stack(dir_list, axis=0).transpose((0, 3, 1, 2))
                # dir_union.tofile(os.path.join(dst_case_folder, 'dir_union.dat'))
            # 2.2 un-directional orientation
                orient_union = np.stack(orient_list, axis=0).transpose((0, 3, 1, 2))
                orient_union.tofile(os.path.join(dst_case_folder, 'orient_union.dat'))
            # 3. confidence
            if 'conf' in inte_list:
                conf_union = np.stack(conf_list, axis=0)
                conf_union.tofile(os.path.join(dst_case_folder, 'conf_union.dat'))
            # 4. hair mask
            if 'mask' in inte_list:
                mask_union = np.stack(mask_list, axis=0)
                mask_union.tofile(os.path.join(dst_case_folder, 'mask_union.dat'))
            # 5. bust depth
            if 'depth' in inte_list:
                depth_union = np.stack(depth_list, axis=0)
                depth_union.tofile(os.path.join(dst_case_folder, 'depth_union.dat'))
                bust_hair_depth_union = np.stack(bust_hair_depth_list, axis=0)
                bust_hair_depth_union.tofile(os.path.join(dst_case_folder, 'bust_hair_depth_union.dat'))



def normalize_rot(poses):
    '''
    normalize rotation matrices in case the original poses are not strictly orthonormal
    :param poses:
    :return:
    '''
    from scipy.spatial.transform import Rotation as R
    norm_poses = poses.copy()
    for pose in norm_poses:
        rot = pose[:3, :3]
        norm_rot = R.from_matrix(rot).as_matrix()
        pose[:3, :3] = norm_rot
    return norm_poses

def recover_tsfm_str(recover_path, str_path, save_path):
    recover_tsfm = np.fromfile(recover_path, dtype='float32').reshape(4, 4)
    str_data = np.fromfile(str_path, dtype='float32')
    origin_str_data = str_data.copy()

    s_cnt = int(str_data[0])
    vertices_data = str_data[s_cnt + 2:].reshape(-1, 3)
    homo_base = np.ones((vertices_data.shape[0], 1), dtype='float32')
    vertices = np.concatenate([vertices_data, homo_base], axis=1).transpose((1, 0))

    recover_vertices = recover_tsfm @ vertices
    origin_str_data[s_cnt + 2:] = recover_vertices[:3].transpose((1, 0)).reshape(-1)
    origin_str_data.tofile(save_path)

def reproject_pts(vertices_data, poses_w2c, intrins, resolution):
    '''

    :param vertices_data: [N, 3]
    :param poses_w2c:
    :param intrins:
    :param resolution:
    :return:
    '''
    homo_base = np.ones((vertices_data.shape[0], 1), dtype='float32')
    vertices = np.concatenate([vertices_data, homo_base], axis=1).transpose((1, 0))

    h, w = resolution
    import matplotlib.pyplot as plt
    for oid in range(len(poses_w2c)):
        print('===> {}'.format(oid))
        blank = np.zeros((h, w), dtype='uint8')
        origin_pose_w2c = poses_w2c[oid]
        cam_intrin = intrins[oid]

        view_vert = (origin_pose_w2c @ vertices)[:3]

        proj_vert = cam_intrin @ view_vert
        proj_vert[:2] = proj_vert[:2] / proj_vert[2:]
        proj_vert = proj_vert[:2].transpose((1, 0)).astype('int32')
        print('proj', proj_vert.shape)
        valid_vert_w = np.logical_and(np.greater(proj_vert[:, 0], 0), np.less(proj_vert[:, 0], w))
        valid_vert_h = np.logical_and(np.greater(proj_vert[:, 1], 0), np.less(proj_vert[:, 1], h))
        valid_vert = np.logical_and(valid_vert_w, valid_vert_h)
        proj_vert = proj_vert[valid_vert]
        print('proj valid', proj_vert.shape)

        blank[proj_vert[:, 1], proj_vert[:, 0]] = 255
        plt.figure(figsize=(8, 8))
        plt.title(str(oid))
        plt.imshow(blank)
        plt.show()

def recover_verify_str(str_path, poses_w2c, intrins, resolution):
    str_data = np.fromfile(os.path.join(str_path), dtype='float32')

    s_cnt = int(str_data[0])
    vertices_data = str_data[s_cnt + 2:].reshape(-1, 3)

    reproject_pts(vertices_data, poses_w2c, intrins, resolution)

def recover_verify_bust(bust_path, poses_w2c, intrins, resolution):
    import trimesh
    bust_mesh = trimesh.load(bust_path)
    vertices_data = np.array(bust_mesh.vertices)
    reproject_pts(vertices_data, poses_w2c, intrins, resolution)


def save_ori_mat(positive_points,orientation,voxel_min,voxel_size,grid_resolution,path,model_tsfm):
    print('model_tsfm:',model_tsfm)
    grid_resolution = grid_resolution.astype(np.int32)
    occ = np.zeros(grid_resolution)
    ori = np.zeros((*(grid_resolution),3))

    up_index = orientation[:,1]>0
    orientation[up_index]*=-1


    positive_points = positive_points.transpose(1,0)
    positive_points = model_tsfm[:3,:3] @ positive_points +model_tsfm[:3,3:4]
    positive_points = positive_points.transpose(1,0)
    positive_points[:, 1:] *= -1

    indexs = (positive_points-voxel_min)/voxel_size

    indexs = np.round(indexs)
    indexs = indexs.astype(np.int32)
    x,y,z = np.split(indexs,3,-1)

    x = np.clip(x,0,grid_resolution[0]-1)
    y = np.clip(y,0,grid_resolution[1]-1)
    z = np.clip(z,0,grid_resolution[2]-1)
    # z = (grid_resolution[2]-z-1).astype(np.int32)
    # y = (grid_resolution[1]-y-1).astype(np.int32)
    x=np.squeeze(x)
    y=np.squeeze(y)
    z=np.squeeze(z)
    occ[x,y,z] = 1
    ori[x,y,z] = orientation
    # ori[...,1:]*=-1
    ori = ori.transpose((0,1,3,2))
    ori = np.reshape(ori,[grid_resolution[0],grid_resolution[1],grid_resolution[2]*3])
    ori = np.transpose(ori,(1,0,2))
    occ = np.transpose(occ,(1,0,2))

    scipy.io.savemat(os.path.join(path, 'Ori3D.mat'), {'Ori': ori})
    scipy.io.savemat(os.path.join(path, 'Occ3D.mat'), {'Occ': occ})




def out_message(loss):
    message="{:.4f}".format(loss.all)
    for key,vaule in loss.items():
        message+=",{}={:.5f}".format(key,vaule)
    return message


def matrix_to_quat(matrix):
    r = R.from_matrix(matrix)
    quat = r.as_quat()

    return quat


def eularToMatrix_np(theta,type='yzx'):
    pi=3.1415926
    pi = math.pi
    c1,c2,c3=np.cos(theta*pi)
    s1,s2,s3=np.sin(theta*pi)
    c1=c1[None]
    c2=c2[None]
    c3=c3[None]
    s1=s1[None]
    s2=s2[None]
    s3=s3[None]

    ## xyz
    if type=='xyz':
        v1=np.concatenate([c2*c3,-c2*s3,s2],axis=0)
        v2=np.concatenate([c1*s3+c3*s1*s2,c1*c3-s1*s2*s3,-c2*s1],axis=0)
        v3=np.concatenate([s1*s3-c1*c3*s2,c3*s1+c1*s2*s3,c1*c2],axis=0)
        matrix = np.concatenate([v1[None], v2[None], v3[None]], axis=0)

    ## yzx
    elif type=='yzx':
        v1=np.concatenate([c1*c2, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3],0)
        v2=np.concatenate([s2,c2*c3,-c2*s3],0)
        v3=np.concatenate([-c2*s1,c1*s3+c3*s1*s2, c1*c3-s1*s2*s3],0)
        matrix=np.concatenate([v1[None],v2[None],v3[None]],axis=0)
    elif type =='xzy':
        v1 = np.concatenate([c2*c3, -s2, c2*s3],0)
        v2 = np.concatenate([s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1],0)
        v3 = np.concatenate([c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3])
        matrix = np.concatenate([v1[None], v2[None], v3[None]], axis=0)

    return matrix

def generate_headtrans_from_tsfm(tsfm_semantic_path,save_path):

    model_tsfm = np.fromfile(tsfm_semantic_path,dtype=np.float32)


    # r = R.from_euler('xzy', [model_tsfm[3], -model_tsfm[5], -model_tsfm[4]], degrees=True)   #have some bug
    # matrix = r.as_matrix()
    # print(matrix[:3,:3]*model_tsfm[6])
    # quat = r.as_quat()
    # trans_and_scale = np.array([model_tsfm[0],-model_tsfm[1],-model_tsfm[2],model_tsfm[6]])
    # tsfm = np.concatenate([quat[None],trans_and_scale[None]],0)
    # np.savetxt(save_path,tsfm)


    matrix = eularToMatrix_np(np.array([model_tsfm[3], -model_tsfm[5], -model_tsfm[4]])/180,'xzy')
    r = R.from_matrix(matrix)
    quat = r.as_quat()
    trans_and_scale = np.array([model_tsfm[0], -model_tsfm[1], -model_tsfm[2], model_tsfm[6]])
    tsfm = np.concatenate([quat[None], trans_and_scale[None]], 0)
    np.savetxt(save_path, tsfm)
    # matrix = np.eye(4)
    # matrix[:3,:3] = eularToMatrix_np(np.array([model_tsfm[3], model_tsfm[5], model_tsfm[4]])/180,'xzy')
    # matrix[:3,:3]*=model_tsfm[6]
    # matrix[:3,3] = np.array([model_tsfm[0],model_tsfm[1],model_tsfm[2]])
    # print(matrix)
    # return matrix


def preprocess_bust(path,type='DECA'):

    mesh = trimesh.load(path+'/bust.obj')


    v = np.array(mesh.vertices)

    if type == 'DECA':
        kpt3d = np.loadtxt(path + '/kpt3d.txt')
        v = v + np.array([0, 1.75, 0.04])
        kpt3d = kpt3d + np.array([0, 1.75, 0.04])
    elif type =='PIXIE':
        kpt3d = np.load(path+'/keypoints.npy')

        kpt3d = kpt3d[0, -90:-22]
        kpt3d = np.concatenate([kpt3d[-17:], kpt3d[:-17]], 0)
        v[:, 1:] *= -1
        v = v + np.array([0, 0.77, -0.07])
        kpt3d[:,1:] *= -1
        kpt3d = kpt3d + np.array([0, 0.77, -0.07])
    elif type == 'PyMAF-X':
        kpt3d = np.load(path + '/keypoints.npy')
        kpt3d = np.concatenate([kpt3d[-17:], kpt3d[:-17]], 0)
        v[:, 1:] *= -1
        v += np.array([0.015, 0.72, -0.05])
        kpt3d[:, 1:] *= -1
        kpt3d += np.array([0.015, 0.72, -0.05])
    mesh = trimesh.Trimesh(vertices=v, faces=mesh.faces)
    trimesh.exchange.export.export_mesh(mesh,os.path.join(path,'bust_long.obj'), include_texture=False)
    np.savetxt(path+'/kpt3d_world.txt',kpt3d)

    mesh =  trimesh.Trimesh(vertices=kpt3d)
    trimesh.exchange.export.export_mesh(mesh, os.path.join(path, 'kpt3d_world.obj'), include_texture=False)

def blend_images(root):
    files = os.listdir(root+'/render_images')
    os.makedirs(os.path.join(root,'blend_images'),exist_ok=True)
    for file in files:
        capture_image = cv2.imread(os.path.join(root, 'capture_images',file))
        render_image = cv2.imread(os.path.join(root, 'render_images',file))
        mask = cv2.imread(os.path.join(root, 'hair_mask',file))
        blend_image = np.where(mask>50,capture_image, render_image*0.7+capture_image*0.3)
        cv2.imwrite(os.path.join(root,'blend_images', file),blend_image)


def export_pointcouldwithcolor(root,save_path, bust_to_origin,mul=1000,threshold=0.001):
    refine = True
    patch = '10-09-patch9-04'
    if refine:
        patch += '-refine'

    select_points = np.load(os.path.join(root, 'select_p.npy'))
    select_ori = np.load(os.path.join(root,  'select_o.npy'))
    min_loss = np.load(os.path.join(root, 'min_loss.npy'))

    if refine:
        filter_unvisible_points = np.load(os.path.join(root,  'filter_unvisible.npy'))
        filter_unvisible_ori = np.load(os.path.join(root,'filter_unvisible_ori.npy'))
        up_index = filter_unvisible_ori[:, 1] > 0
        filter_unvisible_ori[up_index] *= -1


    index = np.where(min_loss < 0.05)[0]
    index1 = np.zeros_like(min_loss)
    index1[index] = 1
    index1 = index1.astype(np.bool_)
    select_ori = select_ori[index1]
    select_points = select_points[index1]
    print('num select', select_ori.shape[:])

    reverse_index = select_ori[:, 1] > 0
    select_ori[reverse_index] *= -1
    # print(np.max(select_ori[:,1]))


    select_points = np.concatenate([select_points, filter_unvisible_points], 0)
    select_ori = np.concatenate([select_ori, filter_unvisible_ori], 0)
    points=select_points
    normals = select_ori





    # index = np.where(min_loss < threshold)[0]
    # select_ori = select_ori[index]
    # select_points = select_points[index]
    #
    # # points = select_points.astype(np.float32)
    # # normals = select_ori.astype(np.float32)
    #
    # coarse_data = np.load(root + '/raw.npy')
    # coarse_points = coarse_data[:,:3]
    # coarse_ori = coarse_data[:,3:6]
    # points =  np.concatenate([coarse_points,select_points], 0).astype(np.float32)
    # normals = np.concatenate([ coarse_ori,select_ori], 0).astype(np.float32


    points-=bust_to_origin
    points*=mul

    colors = np.zeros_like(normals)
    colors[:,0]=1

    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3c.Tensor(points, dtype)
    pcd.point.colors = o3c.Tensor(colors, dtype)
    pcd.point.normals = o3c.Tensor(normals, dtype)
    o3d.t.io.write_point_cloud(save_path, pcd)
    points/=mul
    points+=bust_to_origin
    return points, normals

def sample_scalp(scalp_path,sample_num,save_path,mul=1000,tsfm_path=None):
    if tsfm_path is not None:
        scalp_tsfm_path = os.path.join(scalp_path[:-4]+'_tsfm.obj')
        transform_bust(scalp_path,tsfm_path,scalp_tsfm_path)
        scalp_path = scalp_tsfm_path
    mesh = o3d.io.read_triangle_mesh(scalp_path)  # 加载mesh
    mesh.compute_vertex_normals()
    pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=sample_num,use_triangle_normal=False)  # 采样点云
    points = np.asarray(pcd.points)*mul
    normals = np.asanyarray(pcd.normals)
    # from visualization import vis_point_colud,draw_scene
    # pc = vis_point_colud(points)
    # draw_scene([pc,mesh])

    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3c.Tensor(points, dtype)
    pcd.point.normals = o3c.Tensor(normals, dtype)
    o3d.t.io.write_point_cloud(save_path, pcd)


def transform_bust(mesh_path,tsfm_path,save_path):
    head_mesh = trimesh.load(mesh_path,force='mesh')

    v = np.array(head_mesh.vertices)
    model_tsfm = np.fromfile(os.path.join(tsfm_path),
                             dtype=np.float32).reshape(4, 4).T
    bust_to_origin = np.array([0.006, -1.644, 0.010])
    v += bust_to_origin

    normals = np.asarray(head_mesh.vertex_normals)
    normals = (model_tsfm[:3, :3] @ normals.T).T
    Root = v.transpose(1, 0)
    Root = model_tsfm[:3, :3] @ Root + model_tsfm[:3, 3:4]
    Root = Root.transpose(1, 0)
    Root -= bust_to_origin
    mesh = trimesh.Trimesh(vertices=Root, faces=head_mesh.faces,vertex_normals=normals,visual=head_mesh.visual)
    trimesh.exchange.export.export_mesh(mesh, save_path,
                                        include_texture=True)
    # mesh.export(save_path)
def export_CTHair_data(root,file,threshold=0.001):
    bust_to_origin =  np.array([0.006, -1.644, 0.010])
    pc_root = os.path.join(root,file)
    save_path = pc_root+'/meta_bob_oriens_wd_2.4.ply'
    points, ori = export_pointcouldwithcolor(pc_root,save_path,bust_to_origin,mul=1000,threshold=threshold)
    scalp_path = os.path.join(root,'DECA_PIXIE/scalp.obj')
    tsfm_path = os.path.join(root,'model_tsfm.dat')
    scalp_save_path = os.path.join(root,file,'scalp_meta_bob_pts.ply')
    sample_scalp(scalp_path, 30000, scalp_save_path,1000,tsfm_path)
    return  points, ori


def generate_scalp_from_head(head_path,texture_path,scalp_save_path,mul=1000):
    head_mesh = trimesh.load(head_path)
    head_mesh.vertices = np.asarray(head_mesh.vertices)*mul
    uv_coords = head_mesh.visual.uv  # num_vertices X 2
    head_texture = cv2.imread(texture_path)
    head_tex_width, head_tex_height, _ = head_texture.shape
    # for each face determiner whether it is scalp
    num_faces = head_mesh.faces.shape[0]
    face_uv_coords = uv_coords[head_mesh.faces] * [head_tex_height, head_tex_width]
    face_uv_coords = np.around(face_uv_coords).astype(np.uint16)
    face_uv_coords = np.clip(face_uv_coords, [0, 1], [head_tex_width - 1, head_tex_height])
    face_uv_colors = head_texture[head_tex_height - face_uv_coords[:, :, 1], face_uv_coords[:, :, 0], :]
    face_avg_colors = np.sum(face_uv_colors, axis=1, keepdims=False)

    scalp_faces_mask = face_avg_colors[:, 0] > 255 * 0.3
    scalp_faces_idx = np.where(face_avg_colors[:, 0] > 255 * 0.3)[0]
    scalp_mesh = copy.deepcopy(head_mesh)
    scalp_mesh.update_faces(scalp_faces_mask)
    scalp_mesh.remove_unreferenced_vertices()

    scalp_mesh.export(scalp_save_path)

    return head_mesh, scalp_mesh, scalp_faces_idx



def smnooth_strand(strand, lap_constraint=2.0, pos_constraint=1.0, fix_tips=False):
    num_pts = strand.shape[0]
    num_value = num_pts * 3 - 2 + num_pts
    smoothed_strand = np.copy(strand)

    # construct laplacian sparse matrix
    i, j, v = np.zeros(num_value, dtype=np.int16), np.zeros(num_value, dtype=np.int16), np.zeros(num_value)

    i[0], i[1], i[2 + (num_pts - 2) * 3], i[2 + (num_pts - 2) * 3 + 1] = 0, 0, num_pts - 1, num_pts - 1
    i[2 : num_pts * 3 - 4] = np.repeat(np.arange(1, num_pts - 1), 3)
    i[num_pts * 3 - 2:] = np.arange(num_pts) + num_pts

    j[0], j[1], j[2 + (num_pts - 2) * 3], j[2 + (num_pts - 2) * 3 + 1] = 0, 1, num_pts - 2, num_pts - 1
    j[2 : num_pts * 3 - 4] = np.repeat(np.arange(1, num_pts - 1), 3) \
                           + np.repeat(np.array([-1, 0, 1], dtype=np.int16), num_pts - 2).reshape(num_pts - 2, 3, order='F').ravel()
    j[num_pts * 3 - 2:] = np.arange(num_pts)

    v[0], v[1], v[2 + (num_pts - 2) * 3], v[2 + (num_pts - 2) * 3 + 1] = 1, -1, -1, 1
    v[2 : num_pts * 3 - 4] = np.repeat(np.array([-1, 2, -1], dtype=np.int16), num_pts - 2).reshape(num_pts - 2, 3, order='F').ravel()
    v = v * lap_constraint
    v[num_pts * 3 - 2:] = pos_constraint

    A = coo_matrix((v, (i, j)), shape=(num_pts * 2, num_pts))
    At = A.transpose()
    AtA = At.dot(A)

    # solving
    for j_axis in range(3):
        b = np.zeros(num_pts * 2)
        b[num_pts:] = smoothed_strand[:, j_axis] * pos_constraint
        Atb = At.dot(b)

        x = spsolve(AtA, Atb)
        smoothed_strand[:, j_axis] = x[:num_pts]

    if fix_tips:
        strand[1:-1] = smoothed_strand[1:-1]
    else:
        strand = smoothed_strand

    return strand


def smooth_strands(strands, lap_constraint=2.0, pos_constraint=1.0, fix_tips=False):
    loop = tqdm(range(len(strands)))
    loop.set_description("Smoothing strands")
    for i_strand in loop:

        strands[i_strand] = smnooth_strand(strands[i_strand], lap_constraint, pos_constraint, fix_tips)

    return strands

def compute_similar(A,B):
    return np.sum(A*B,axis=-1)/(np.maximum(np.linalg.norm(A,2,axis=-1)*np.linalg.norm(B,2,axis=-1),1e-4))


def check_strand(strand,scalp_mean,scalp_tree,scalp_min,scalp_max):
    num = strand.shape[0]
    beg_pos = strand[0]
    end_pos = strand[-1]
    # mean_x1 = np.mean(strand[:num//2,0])
    # mean_x2 = np.mean(strand[num//2:,0])
    index1 = strand[:,0]>scalp_mean[0]
    index2 = strand[:,0]<scalp_mean[0]
    nei_distance,_ = scalp_tree.query(beg_pos,1)
    # print(beg_pos)
    # print(end_pos)
    # print(scalp_mean)

    check =False
    if (beg_pos[0]< scalp_min[0]/6 and end_pos[0]>scalp_max[0])/6  or (beg_pos[0]> scalp_max[0]/6 and end_pos[0]<scalp_min[0]/6):
        check = True


    # if np.sum(index1)>20 and np.sum(index2)>20 and beg_pos[1]<scalp_mean[1] and beg_pos[1]<scalp_mean[1] and num>50 and np.abs(beg_pos[0]-end_pos[0])>0.06 and abs(beg_pos[0])>0.03 and abs(end_pos[0])>0.03 and nei_distance>0.1:
    # if np.sum(index1)>20 and np.sum(index2)>20 and beg_pos[1]<scalp_mean[1] and beg_pos[1]<scalp_mean[1] and num>50 and np.abs(beg_pos[0]-end_pos[0])>0.06 and check:
    if check:
        ori = np.abs(strand[1:-1] - strand[:-2]) + np.abs(strand[2:] - strand[1:-1])
        index = np.argmin(ori[:,1])
        return False,index
    else:
        return True,None


    # if mean_x1*mean_x2<0 and np.abs(mean_x1-mean_x2)>


def check_strand1(strand,scalp_min):
    ori = np.concatenate([strand[1:] -strand[:-1],strand[-1:]-strand[-2:-1]],0)
    up_index = ori[:,1]>-0.0001
    up_index = np.logical_and(up_index,strand[:,1]>scalp_min[1])
    if np.sum(up_index)==0:
        return True, None
    else:
        index = np.argmax(strand[:,1],-1)
        return False, index


def save_hair_strands(path,strands):
    segments = [strands[i].shape[0] for i in range(len(strands))]
    hair_count=len(segments)
    point_count=sum(segments)
    points = np.concatenate(strands,0)
    # points = voxel_to_points(points)
    with open(path, 'wb')as f:
        f.write(struct.pack('I', hair_count))
        f.write(struct.pack('I', point_count))
        for num_every_strand in segments:
            f.write(struct.pack('H', num_every_strand ))

        for vec in points:
            f.write(struct.pack('f', vec[0]))
            f.write(struct.pack('f', vec[1]))
            f.write(struct.pack('f', vec[2]))
    f.close()


def generate_flame(smplx_mesh,target_path,flame_template_path,flame_idx_path):
    index = np.load(flame_idx_path)
    v = np.array(smplx_mesh.vertices)
    flame_v = v[index]


    with open(flame_template_path,'r')as f:
        with open(target_path,'w') as flame_f:
            lines = f.readlines()
            count = 0
            for line in lines:
                content = line.split(' ')
                if content[0] =='v':
                    flame_f.writelines("v {} {} {}\n".format(flame_v[count][0], flame_v[count][1], flame_v[count][2]))
                    count+=1
                else:
                    flame_f.writelines(line)
            flame_f.close()
        f.close()

def replace_scalp(template_mesh, source_mesh,scalp_uv_path,save_path):

    uv = source_mesh.visual.uv
    source_v = source_mesh.vertices
    img = cv2.imread(scalp_uv_path)
    img = img[..., 0]
    uv *= np.array([img.shape[:2]]) - 1
    uv = np.round(uv).astype(np.int32)
    mask = img[img.shape[1] - uv[:, 1], uv[:, 0]]
    index = np.argwhere(mask > 200)[:, 0]
    source_v[index] = template_mesh.vertices[index]

    source_mesh.vertices = source_v
    trimesh.exchange.export.export_mesh(source_mesh, save_path)


def generate_flame_scalp(target_path,source_mesh,flame_template_path,scalp_mask_path):
    flame_template = trimesh.load_mesh(flame_template_path)
    faces = source_mesh.faces
    source_v = np.array(source_mesh.vertices)
    uv = flame_template.visual.uv
    img = cv2.imread(scalp_mask_path)
    img = img[...,0]

    img[img>0]=255
    # cv2.imshow('1',img)
    # cv2.waitKey(0)
    uv *= np.array([img.shape[:2]])-1
    uv = np.round(uv).astype(np.int32)
    mask = []

    for uv_ in uv:
        if np.sum(img[img.shape[1]-uv_[1]-3:img.shape[1]-uv_[1]+3, uv_[0]-3:uv_[0]+3])>0:
            mask.append(np.array([255]))
        else:
            mask.append(np.array([0]))

    mask = np.concatenate(mask,0)
    index = np.argwhere(mask>200)[:,0]

    scalp_v = source_v[index]


    original_index = index.copy()
    scalp_faces = []
    scalp_face_idx = []
    for i,f in enumerate(faces):
        check = False
        v1 = f[0]
        v2 = f[1]
        v3 = f[2]
        if v1 in index and v2 in index and v3 in index:
            index_v1 = np.argwhere(index==v1)
            index_v2 = np.argwhere(index==v2)
            index_v3 = np.argwhere(index==v3)

            scalp_face = np.concatenate([index_v1,index_v2,index_v3],1)
            scalp_faces.append(scalp_face)
            scalp_face_idx.append(i)


    scalp_faces = np.concatenate(scalp_faces,0)
    scalp_face_idx = np.asarray(scalp_face_idx)
    scalp = trimesh.Trimesh(vertices=scalp_v,faces=scalp_faces)
    trimesh.exchange.export.export_mesh(scalp,target_path,include_texture=False)
    # np.save(scalp_face_idx_path,scalp_face_idx)

def generate_bust(source_mesh, template_mesh, scalp_mask_path, flame_template_path, flame_idx_path ,save_root):
    source_v = source_mesh.vertices
    template_v = template_mesh.vertices

    index = np.load(flame_idx_path)
    vertices = []
    vertices_uv_repeat = []
    map = np.ones_like(index)*-1


    with open(flame_template_path,'r')as f:
        lines = f.readlines()
        for line in lines:
            content = line.split(' ')
            if content[0] == 'v':
                v = np.array([float(content[1]), float(content[2]), float(content[3])])[None, ...]

                vertices.append(v)
            if content[0] =='vt':
                vertices_uv_repeat.append(np.array([float(content[1]), float(content[2])])[None, ...])

            if content[0] =='f':
                for m in content[1:]:
                    v_index,uv_index = m.split('/')
                    map[int(v_index)-1] = int(uv_index)-1
    assert np.min(map)>=0
    vertices_uv_repeat = np.concatenate(vertices_uv_repeat,0)
    vertices_uv = vertices_uv_repeat[map]

    img = cv2.imread(scalp_mask_path)
    img = img[..., 0]

    img[img > 0] = 255
    uv = vertices_uv
    uv *= np.array([img.shape[:2]]) - 1
    uv = np.round(uv).astype(np.int32)
    mask = []

    for uv_ in uv:
        if np.sum(img[img.shape[1] - uv_[1] - 3:img.shape[1] - uv_[1] + 3, uv_[0] - 3:uv_[0] + 3]) > 0:
            mask.append(np.array([255]))
        else:
            mask.append(np.array([0]))

    mask = np.concatenate(mask, 0)
    scalp_index = np.argwhere(mask > 200)[:, 0]

    scalp_v = template_v[index][scalp_index]
    source_v[index[scalp_index]] = scalp_v
    source_mesh.vertices = source_v
    trimesh.exchange.export.export_mesh(source_mesh, os.path.join(save_root,'bust_long.obj'))

    generate_flame(source_mesh,os.path.join(save_root,'flame_bust.obj'),flame_template_path,flame_idx_path)

    # scalp_v = source_mesh.vertices[index][scalp_index]
    faces = source_mesh.faces
    scalp_faces = []
    scalp_face_idx = []

    index = index[scalp_index]
    for i, f in enumerate(faces):

        v1 = f[0]
        v2 = f[1]
        v3 = f[2]
        if v1 in index and v2 in index and v3 in index:
            index_v1 = np.argwhere(index == v1)
            index_v2 = np.argwhere(index == v2)
            index_v3 = np.argwhere(index == v3)

            scalp_face = np.concatenate([index_v1, index_v2, index_v3], 1)
            scalp_faces.append(scalp_face)
            scalp_face_idx.append(i)

    scalp_faces = np.concatenate(scalp_faces, 0)
    scalp = trimesh.Trimesh(vertices=scalp_v, faces=scalp_faces)
    trimesh.exchange.export.export_mesh(scalp, os.path.join(save_root,'scalp.obj'))


if __name__ == '__main__':
    pass








