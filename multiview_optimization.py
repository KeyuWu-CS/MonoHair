import os
import sys
from log import log
import options
from dataprocess import DataProcessor
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from skimage.transform import resize
from skimage.io import imread
import pickle
from submodules.DELTA.lib.model.smplx import SMPLX
from submodules.DELTA.lib.utils import rotation_converter
from pathlib import Path
from tqdm import tqdm
from submodules.DELTA.lib.utils import util, lossfunc
from submodules.DELTA.lib.render.mesh_helper import render_shape
from submodules.DELTA.lib.utils.deca_render import SRenderY
from submodules.DELTA.lib.model.FLAME import FLAMETex
import json
from Utils.Camera_utils import Camera
from itertools import cycle
import trimesh
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    SoftSilhouetteShader,
)
import torch.nn.functional as F

def load_cam(path):
    with open(path, 'r')as f:
        cam = json.load(f)
    f.close()
    cam = cam['cam_list']
    camera = {}
    for c in cam:
        camera[c['file']] = Camera(c['ndc_prj'], np.linalg.inv(np.array(c['pose'])), c['file'])
    return camera


def eularToMatrix(theta, device, type='yzx'):  # Y X Z 内旋    theta(z,x,y)
    pi = 3.1415926
    c1, c2, c3 = torch.cos(theta * pi)
    s1, s2, s3 = torch.sin(theta * pi)
    c1 = c1[None]
    c2 = c2[None]
    c3 = c3[None]
    s1 = s1[None]
    s2 = s2[None]
    s3 = s3[None]

    ## xyz
    if type == 'xyz':
        v1 = torch.cat([c2 * c3, -c2 * s3, s2], dim=0)
        v2 = torch.cat([c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1], dim=0)
        v3 = torch.cat([s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2], dim=0)
        matrix = torch.cat([v1[None], v2[None], v3[None]], dim=0)

    ## yzx
    elif type == 'yzx':
        v1 = torch.cat([c1 * c2, s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3], 0)
        v2 = torch.cat([s2, c2 * c3, -c2 * s3], 0)
        v3 = torch.cat([-c2 * s1, c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3], 0)
        matrix = torch.cat([v1[None], v2[None], v3[None]], 0)

    elif type == 'xzy':
        v1 = torch.cat([c2 * c3, -s2, c2 * s3], 0)
        v2 = torch.cat([s1 * s3 + c1 * c3 * s2, c1 * c2, c1 * s2 * s3 - c3 * s1], 0)
        v3 = torch.cat([c3 * s1 * s2 - c1 * s3, c2 * s1, c1 * c3 + s1 * s2 * s3], 0)
        matrix = torch.cat([v1[None], v2[None], v3[None]], 0)

    return matrix.to(device)

class NerfDataset(torch.utils.data.Dataset):
    """Synthetic_agora Dataset"""

    def __init__(self, cfg, given_imagepath_list=None):
        super().__init__()
        subject = cfg.subject
        self.dataset_path = os.path.join(cfg.path, subject)
        self.subject_id = subject
        root_dir = os.path.join(self.dataset_path, 'cache')
        os.makedirs(root_dir, exist_ok=True)
        self.pose_cache_path = os.path.join(root_dir, 'pose.pt')
        self.cam_cache_path = os.path.join(root_dir, 'cam.pt')
        self.exp_cache_path = os.path.join(root_dir, 'exp.pt')
        self.beta_cache_path = os.path.join(root_dir, 'beta.pt')
        self.tex_cache_path = os.path.join(root_dir, 'tex.pt')
        self.light_cache_path = os.path.join(root_dir, 'light.pt')



        imagepath_list = given_imagepath_list

        self.data = imagepath_list
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"

        self.image_size = cfg.data.image_size
        # self.white_bg = cfg.white_bg
        # self.load_lmk = cfg.load_lmk
        # self.load_normal = cfg.load_normal
        self.load_fits = cfg.load_fits
        self.mode = 'train'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ## load smplx
        imagepath = self.data[index]
        image = imread(imagepath) / 255.
        imagename = imagepath.split('/')[-1].split('.')[0]
        alpha_image = image[:, :, -1:]

        image = image[:, :, :3]



        # if self.white_bg:
        #     image = image[..., :3] * alpha_image + (1. - alpha_image)
        # else:
        #     image = image[..., :3] * alpha_image

        image = image[..., :3] * alpha_image
        # ## add alpha channel
        image = np.concatenate([image, alpha_image[:, :, :1]], axis=-1)
        # image = resize(image, [self.image_size, self.image_size])

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = image[3:]
        image = image[:3]

        mask = (mask > 0.5).float() # for hair matting, remove this

        name = self.subject_id
        data = {
            'idx': index,
            'frame_id': imagename,
            'name': name,
            'imagepath': imagepath,
            'image': image,
            # 'cam': cam
            'mask': mask,
        }

        # --- load keypoints

        if os.path.exists(os.path.join(self.dataset_path, 'landmark2d', f'{imagename}.txt')):
            lmk = np.loadtxt(os.path.join(self.dataset_path, 'landmark2d', f'{imagename}.txt'))
            # normalize lmk

            # lmk = torch.from_numpy(lmk).float() / torch.tensor(self.image_size)
            lmk = torch.from_numpy(lmk).float()
            lmk = np.concatenate([lmk, np.ones([lmk.shape[0], 1])], axis=-1)
            data['lmk'] = lmk
            ## iris
            iris = np.loadtxt(os.path.join(self.dataset_path, 'iris', f'{imagename}.txt'))
            # normalize lmk
            iris = torch.from_numpy(iris).float()
            # iris[:, :2] = iris[:, :2] / torch.tensor(self.image_size)
            iris[:, :2] = iris[:, :2]
            # iris[:, :2] = iris[:, :2] * 2. - 1
            data['iris'] = iris


        # --- masks from hair matting and segmentation
        ''' for face parsing from https://github.com/zllrunning/face-parsing.PyTorch/issues/12
        [0 'backgruond' 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
        # 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        '''
        parsing_file = os.path.join(self.dataset_path, 'face_parsing', f'{imagename}.png')
        if os.path.exists(os.path.join(parsing_file)):
            semantic = imread(parsing_file)
            labels = np.unique(semantic)

            if 'b0_0' in self.subject_id:
                mask_np = (mask.squeeze().numpy() * 255).astype(np.uint8)
                skin_cloth_region = np.ones_like(mask_np).astype(np.float32)
                skin_cloth_region[semantic == 17] = 0
                skin_cloth_region[mask_np < 100] = 0
                face_region = np.zeros_like(semantic)
                face_labels = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
                for label in face_labels:
                    face_region[semantic == label] = 255

                skin_cloth_region = resize(skin_cloth_region, [self.image_size[0], self.image_size[1]])
                face_region = resize(face_region, [self.image_size[0], self.image_size[1]])
                skin_cloth_region = torch.from_numpy(skin_cloth_region).float()[None, ...]
                face_region = torch.from_numpy(face_region).float()[None, ...]
                data['hair_mask'] = mask * (1 - skin_cloth_region)
                data['skin_mask'] = skin_cloth_region
                data['face_mask'] = face_region
                # cv2.imwrite('mask.png', mask_np)
                # cv2.imwrite('mask_hair.png', (data['hair_mask'][0].numpy()*255).astype(np.uint8))
                # cv2.imwrite('mask_nonhair.png', (data['skin_mask'][0].numpy()*255).astype(np.uint8))
                # cv2.imwrite('mask_face.png', (data['face_mask'][0].numpy()*255).astype(np.uint8))
                # exit()
            else:
                skin_cloth_region = np.zeros_like(semantic)
                face_region = np.zeros_like(semantic)
                # fix semantic labels, if there's background inside the body, then make it as skin
                mask_np = mask.squeeze().numpy().astype(np.uint8) * 255

                semantic[(semantic + mask_np) == 255] = 1
                for label in labels[:-1]:
                    # last label is hair/hat
                    if label == 0 or label == 17 or label == 18:
                        continue
                    skin_cloth_region[semantic == label] = 255
                    # if label in [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14]:
                    if label in [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]:
                        face_region[semantic == label] = 255
                skin_cloth_region = resize(skin_cloth_region, [self.image_size[0], self.image_size[1]])
                face_region = resize(face_region, [self.image_size[0], self.image_size[1]])
                skin_cloth_region = torch.from_numpy(skin_cloth_region).float()[None, ...]
                face_region = torch.from_numpy(face_region).float()[None, ...]
                data['hair_mask'] = mask * (1 - skin_cloth_region)
                data['skin_mask'] = skin_cloth_region
                data['face_mask'] = face_region
            ### face and skin
            if self.mode == 'val':
                face_neck_region = np.ones_like(semantic) * 255
                face_neck_region[semantic == 0] = 0
                face_neck_region[semantic == 15] = 0
                face_neck_region[semantic == 16] = 0
                face_neck_region[semantic == 18] = 0
                face_neck_region = resize(face_neck_region, [self.image_size[0], self.image_size[1]])
                face_neck_region = torch.from_numpy(face_neck_region).float()[None, ...]
                data['face_neck_mask'] = face_neck_region


        return data


class PoseModel(nn.Module):

    def __init__(self,
                 dataset,
                 model_cfg,
                 optimize_cam=False,
                 use_perspective=False,
                 use_appearance=False,
                 appearance_dim=0,
                 use_deformation=False,
                 use_light=False,
                 n_lights=3,
                 deformation_dim=0):
        super(PoseModel, self).__init__()
        self.device = 'cuda:0'
        self.subject_id = dataset.subject_id

        ## initialize
        # assume: global pose zero (body facing front), change head pose
        # optimize: cam, beta, head/neck pose, exp

        init_exp = torch.zeros([1, model_cfg.n_exp]).float().to(self.device)
        init_light = torch.zeros([1, 9, 3]).float().to(self.device)

        init_full_pose = torch.zeros([55, 3]).float().to(self.device) + 0.00001
        init_full_pose[0, 0] = np.pi
        ## init shoulder
        init_full_pose[16,2] = -np.pi*60/180
        init_full_pose[17,2] = np.pi*60/180

        init_full_pose = rotation_converter.batch_euler2axis(init_full_pose)[None, ...]
        self.init_full_pose = init_full_pose
        self.register_parameter('pose', torch.nn.Parameter(init_full_pose))
        self.register_parameter('exp', torch.nn.Parameter(init_exp))
        # self.register_parameter(f'{name}_light', torch.nn.Parameter(init_light))
        # self.register_parameter('scale',torch.nn.Parameter(torch.tensor([1.]).float().to(self.device)))
        # self.register_parameter('global_trans',)

        for imagepath in dataset.data:
            imagename = Path(imagepath).stem
            # frame_id = int(imagename.split('_f')[-1])
            frame_id = int(imagename)
            name = self.subject_id

        #     # init cam
        #     # self.register_parameter(f'{name}_cam_{frame_id}', torch.nn.Parameter(init_cam))
        #     # init full pose
        #     self.register_parameter(f'{name}_pose_{frame_id}', torch.nn.Parameter(init_full_pose))
            self.register_parameter(f'{name}_light_{frame_id}', torch.nn.Parameter(init_light))
            # self.register_parameter(f'{name}_exp_{frame_id}', torch.nn.Parameter(init_exp))

    def forward(self, batch, extra_fix_idx=None):
        # return poses of given frame_ids
        name = self.subject_id
        frame_ids = batch['frame_id']
        names = [name] * len(batch['frame_id'])
        batch_size = len(frame_ids)
        # batch_pose = torch.cat([getattr(self, f'{names[i]}_pose_{frame_ids[i]}') for i in range(batch_size)])
        batch_pose = torch.cat([self.pose])

        batch_pose = rotation_converter.batch_axis2matrix(batch_pose.reshape(-1, 3)).reshape(batch_size, 55, 3, 3)
        batch['init_full_pose'] = rotation_converter.batch_axis2matrix(self.init_full_pose.clone().reshape(
            -1, 3)).reshape(batch_size, 55, 3, 3)
        batch['full_pose'] = batch_pose
        # do not optimize body pose
        # global: 0, neck: 12, head: 15, leftarm: 16, rightarm: 17, jaw: 22, lefteye: 23, righteye: 24
        fix_idx = list(range(1, 12)) + [13, 14, 18, 19, 20, 21] + list(range(25, 55))
        if extra_fix_idx is not None:
            fix_idx += extra_fix_idx
        batch['full_pose'][:, fix_idx] = batch[
            'init_full_pose'][:,
                              fix_idx]  #torch.eye(3).to(batch_pose.device)[None,None,...].expand(batch_size, len(fix_idx), -1, -1)
        # batch['init_cam'] = batch['cam'].clone()
        # batch['cam'] = torch.cat([getattr(self, f'{names[i]}_cam_{frame_ids[i]}') for i in range(batch_size)])
        # batch['exp'] = torch.cat([getattr(self, f'{names[i]}_exp_{frame_ids[i]}') for i in range(batch_size)])
        batch['exp'] = torch.cat([self.exp])
        # batch['scale'] = self.scale

        batch['light'] = torch.cat([getattr(self, f'{names[i]}_light_{frame_ids[i]}') for i in range(batch_size)])
        return batch

def projection( vertices, proj, pose):
        '''

        :param v: N*3
        :param prj_mat: 4*4
        :param poses: 4*4
        :return:
        '''
        # self.proj = self.proj.to(vertices.device)
        # self.pose = self.pose.to(vertices.device)
        vertices = vertices.permute(1, 0)
        vertices = torch.cat([vertices, torch.ones((1, vertices.size(1)), device=vertices.device)])
        camera_v = torch.matmul(pose, vertices)
        z = camera_v[2:3, :]
        uv = torch.matmul(proj, camera_v)
        uv[:2] /= z
        uv = uv.transpose(1, 0)
        return uv[:, :2], z[0]



class SMPLX_optimizer(torch.nn.Module):

    def __init__(self, args, dataset=None, device='cuda:0',image_size=[1920,1080],light_type='SH'):
        super(SMPLX_optimizer, self).__init__()
        self.cfg = args
        self.model_cfg = args.smplx
        self.device = device
        self.image_size = image_size
        self.light_type =light_type

        # smplx_cfg.model.n_shape = 100
        # smplx_cfg.model.n_exp = 50
        self.dataset = dataset
        self._setup_model()
        self._setup_renderer()
        self._setup_loss_weight()
        self.configure_optimizers()
        self.cam = load_cam(args.camera_path)
        # loss
        # self.id_loss = lossfunc.VGGFace2Loss(pretrained_model=model_cfg.fr_model_path)


    def _setup_renderer(self):
        ## setup raterizer
        uv_size = 1024
        topology_path = self.model_cfg.topology_path
        # cache data for smplx texture
        self.smplx_texture = imread(self.model_cfg.smplx_tex_path) / 255.
        self.cached_data = np.load(self.model_cfg.flame2smplx_cached_path, allow_pickle=True, encoding='latin1').item()

        self.render = SRenderY(self.image_size,
                               obj_filename=topology_path,
                               uv_size=uv_size,
                               rasterizer_type='pytorch3d').to(self.device)
        mask = imread(self.model_cfg.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.flame_face_eye_mask = F.interpolate(mask, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)

        # face region mask in flame texture map
        mask = imread(self.model_cfg.face_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.flame_face_mask = F.interpolate(mask, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)

        ########### silhouette rendering
        ## camera
        R = torch.eye(3).unsqueeze(0)
        T = torch.zeros([1, 3])
        batch_size = 1
        self.cameras = pytorch3d.renderer.cameras.FoVOrthographicCameras(R=R.expand(batch_size, -1, -1),
                                                                         T=T.expand(batch_size, -1),
                                                                         znear=0.0).to(self.device)

        blend_params = BlendParams(sigma=1e-7, gamma=1e-4)
        raster_settings = RasterizationSettings(image_size=self.image_size,
                                                blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                                                faces_per_pixel=50,
                                                bin_size=0)
        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.cameras,
                                                                          raster_settings=raster_settings),
                                                shader=SoftSilhouetteShader(blend_params=blend_params))

    def _setup_model(self):
        ## pose model
        self.posemodel = PoseModel(model_cfg=self.model_cfg,dataset=self.dataset,
                                   optimize_cam=True,
                                   use_perspective=False,
                                   use_appearance=False,
                                   use_deformation=False,
                                   use_light=True,
                                   n_lights=9).to(self.device)

        ## smplx model
        self.smplx = SMPLX(self.model_cfg).to(self.device)
        self.flametex = FLAMETex(self.model_cfg).to(self.device)
        self.verts = self.smplx.v_template
        self.faces = self.smplx.faces_tensor
        ## iris index
        self.idx_iris = [9503, 10049]  # right, left
        ## part index
        ##--- load vertex mask
        with open(self.model_cfg.mano_ids_path, 'rb') as f:
            hand_idx = pickle.load(f)
        flame_idx = np.load(self.model_cfg.flame_ids_path)
        with open(self.model_cfg.flame_vertex_masks_path, 'rb') as f:
            flame_vertex_mask = pickle.load(f, encoding='latin1')
        # verts = torch.nn.Parameter(self.v_template, requires_grad=True)
        exclude_idx = []
        exclude_idx += list(hand_idx['left_hand'])
        exclude_idx += list(hand_idx['right_hand'])
        exclude_idx += list(flame_vertex_mask['face'])
        exclude_idx += list(flame_vertex_mask['left_eyeball'])
        exclude_idx += list(flame_vertex_mask['right_eyeball'])
        exclude_idx += list(flame_vertex_mask['left_ear'])
        exclude_idx += list(flame_vertex_mask['right_ear'])
        all_idx = range(self.smplx.v_template.shape[1])
        face_idx = list(flame_vertex_mask['face'])
        body_idx = [i for i in all_idx if i not in face_idx]
        self.part_idx_dict = {
            'face': flame_vertex_mask['face'],
            'hand': list(hand_idx['left_hand']) + list(hand_idx['right_hand']),
            'exclude': exclude_idx,
            'body': body_idx
        }

    def configure_optimizers(self):
        ###--- optimizer for training nerf model.
        # whether to use apperace code
        # nerf_params = list(self.model.mlp_coarse.parameters()) + list(self.model.mlp_fine.parameters())
        init_beta = torch.zeros([1, self.model_cfg.n_shape]).float().to(self.device)
        self.register_parameter('beta', torch.nn.Parameter(init_beta))
        init_tex = torch.zeros([1, self.model_cfg.n_tex]).float().to(self.device)
        self.register_parameter('tex', torch.nn.Parameter(init_tex))
        self.register_parameter('model_scale', torch.nn.Parameter(torch.tensor([1.]).float().to(self.device)))
        self.register_parameter('model_trans', torch.nn.Parameter(torch.tensor([0.,0.,0.]).float().to(self.device)))
        self.register_parameter('model_rotate', torch.nn.Parameter(torch.tensor([0.,0.,0.]).float().to(self.device)))

        parameters = [{'params': [self.beta, self.tex], 'lr': 1e-3}]
        parameters.append({'params':self.model_scale,'lr':1e-3})
        parameters.append({'params':self.model_trans,'lr':1e-2})
        parameters.append({'params':self.model_rotate,'lr':5e-2})

        # parameters.append(
        #     {'params': self.posemodel.parameters(), 'lr': 1e-3})
        # for name, param in self.posemodel.named_parameters():
        #     print(name)
        # cam_parameters = [param for name, param in self.posemodel.named_parameters() if 'cam' in name]
        pose_parameters = [param for name, param in self.posemodel.named_parameters() if 'cam' not in name]
        parameters.append({'params': pose_parameters, 'lr': 1e-3})
        # parameters.append({'params': cam_parameters, 'lr': 1e-2})


        self.optimizer = torch.optim.Adam(params=parameters)


    def combine_tsfm(self):
        matrix =  torch.eye(4).to(self.cfg.device)
        matrix[:3, :3] = eularToMatrix(self.model_rotate[[0, 2, 1]] / 180., self.cfg.device, 'xzy')
        matrix[:3, 3] = self.model_trans
        matrix[:3, :3] *= self.model_scale
        self.model_tsfm = torch.cat([self.model_trans, self.model_rotate, self.model_scale], 0)
        self.matrix = matrix
        return matrix

    def forward_model(self, batch, returnMask=False, returnRendering=False, returnNormal=False):
        ''' forward SMPLX model
        Args:
            batch: dict, batch data
                'beta': [B, n_shape(200)], shape parameters
                'exp': [B, n_exp(100)], expression parameters
                'full_pose': [B, n_pose(55), 3, 3], pose parameters, in Rotatation matrix format
                'cam': [B, 3], camera parameters, [scale, tx, ty], use orthographic projection
            returnMask: bool, whether to return mask
            returnRendering: bool, whether to return rendering
            returnNormal: bool, whether to return normal
        Returns:
            opdict: dict, output dict

        '''
        opdict = {}
        verts, landmarks, joints = self.smplx(shape_params=batch['beta'],
                                              expression_params=batch['exp'],
                                              full_pose=batch['full_pose'])
        verts_ori,_,_ = self.smplx(full_pose=batch['full_pose'])

        verts_ori[:, :, 1:] *= -1
        verts_ori[:, :, 1] += 0.7
        verts_ori += torch.tensor([0.006, -1.644, 0.010],dtype=torch.float32,device=verts_ori.device)

        verts[:,:,1:] *= -1
        verts[:,:,1] += 0.7
        verts += torch.tensor([0.006, -1.644, 0.010],dtype=torch.float32,device=verts_ori.device)
        landmarks[:, :, 1:] *= -1
        landmarks[:, :, 1] += 0.7
        landmarks[:, :, ] += torch.tensor([0.006, -1.644, 0.010],dtype=torch.float32,device=verts_ori.device)


        # verts*=1.2
        # landmarks*=1.2
        verts = verts[0]  #### N,3
        landmarks = landmarks[0] #### 68*3
        verts_ori = verts_ori[0]
        opdict['verts_template'] = verts.clone()
        opdict['verts_template_ori'] = verts_ori.clone()



        matrix = self.combine_tsfm()
        #### matrix is c2w, we need w2c
        # matrix[:3,:3] /= self.model_scale
        # matrix[:3,:3] = matrix[:3,:3].transpose(1,0)
        # matrix[:3,:3] *= self.model_scale
        # matrix[:3,3] *= -1
        # opdict['matrix'] = matrix

        verts = verts.permute(1,0)
        landmarks = landmarks.permute(1,0)
        verts_ori = verts_ori.permute(1,0)

        verts = torch.matmul(matrix[:3,:3],verts) + matrix[:3,3:4]
        landmarks = torch.matmul(matrix[:3,:3],landmarks) + matrix[:3,3:4]
        verts_ori = torch.matmul(matrix[:3,:3],verts_ori) + matrix[:3,3:4]

        verts =verts.permute(1,0)
        landmarks = landmarks.permute(1,0)
        verts_ori = verts_ori.permute(1,0)

        # verts *= batch['scale']
        # landmarks *= batch['scale']
        # mesh = trimesh.Trimesh(vertices=verts[0].detach().cpu().numpy(),faces=faces.detach().cpu().numpy())
        # trimesh.exchange.export.export_mesh(mesh,'smplx1.obj',include_texture=False)

        cam = self.cam[batch['frame_id'][0]]

        uv,z =cam.projection(verts)
        uv[:, 0:1] = uv[:, 0:1] * -1
        # cam.render_img(verts[0],self.cfg.data.image_size,self.cfg.device,'test.png')
        trans_verts = torch.cat([uv,-z[:,None]],1)[None,...]


        pred_lmk,_ = cam.projection(landmarks)
        pred_lmk = cam.uv2pixel(pred_lmk,self.cfg.data.image_size,pred_lmk.device)[None,...]


        # convert smpl-x landmarks to flame landmarks (right order)
        pred_lmk = torch.cat([pred_lmk[:, -17:], pred_lmk[:, :-17]], dim=1)
        pred_lmk = pred_lmk[:,:,[1,0]]

        verts = verts[None,...]
        opdict['verts'] = verts
        opdict['trans_verts'] = trans_verts
        opdict['pred_lmk'] = pred_lmk
        opdict['verts_ori'] = verts_ori

        # render mask for silhouette loss
        batch_size = verts.shape[0]
        faces = self.faces.unsqueeze(0).expand(batch_size, -1, -1)
        if returnMask:
            trans_verts_mask = trans_verts.clone()
            trans_verts_mask[:, :, :2] = -trans_verts_mask[:, :, :2]
            trans_verts_mask[:, :, -1] = -trans_verts_mask[:, :, -1] + 50
            mesh = Meshes(
                verts=trans_verts_mask,
                faces=faces
            )
            mesh_mask = self.silhouette_renderer(meshes_world=mesh).permute(0, 3, 1, 2)[:, 3:]
            opdict['mesh_mask'] = mesh_mask
        # # render image for image loss
        if returnRendering:
            # calculate normal for shading
            normal_verts = trans_verts.clone()

            normal_verts[..., 0] = -normal_verts[..., 0]

            normals = util.vertex_normals(normal_verts, faces)
            trans_verts[..., -1] = trans_verts[..., -1] + 50
            albedo = self.flametex(batch['tex'])
            rendering_out = self.render(verts,
                                        trans_verts,
                                        albedo,
                                        lights=batch['light'],
                                        light_type=self.light_type,
                                        given_normal=normals)
            opdict['image'] = rendering_out['images']
            opdict['albedo_image'] = rendering_out['albedo_images']
            opdict['shading_image'] = rendering_out['shading_images']

        return opdict

    def _setup_loss_weight(self):
        loss_cfg = self.cfg.loss
        loss_cfg.lmk = 0.3
        loss_cfg.eyed = 0
        # loss_cfg.lipd = 0.5
        # # mask
        loss_cfg.inside_mask = 1.
        loss_cfg.mesh_mask = 1.
        # # image
        loss_cfg.image = 2.
        loss_cfg.albedo = 2.

        self.loss_cfg = loss_cfg

    def optimize(self, dataloader, iters, vis_step=100, vispath=None,
                 data_type='else',
                 lmk_only=True,
                 use_mask=False,
                 use_rendering=False,
                 use_normal=False):
        ''' optimize the pose and shape parameters of the model, using lmk loss only
        # global: 0, neck: 12, head: 15, leftarm: 16, rightarm: 17, jaw: 22, lefteye: 23, righteye: 24
        '''
        iter_dataloader = cycle(dataloader)
        os.makedirs(vispath, exist_ok=True)
        for i in tqdm(range(iters)):
            batch = next(iter_dataloader)

            # first stage, only optimize global and neck pose
            if data_type == 'fix_shoulder':
                extra_fix_idx = [15, 16, 17, 23, 24, 22]
            elif data_type == 'fix_neck':
                extra_fix_idx = [12, 15, 16, 17, 23, 24, 22]
            else:
                extra_fix_idx = []
            batch['image'] =batch['image'].to(self.cfg.device).type(torch.float32)
            image = batch['image']
            batch['mask'] = batch['mask'].to(self.cfg.device)
            batch['hair_mask'] =batch['hair_mask'].to(self.cfg.device)
            batch['face_mask'] =batch['face_mask'].to(self.cfg.device)
            batch_size = image.shape[0]
            batch['beta'] = self.beta.expand(batch_size, -1)
            batch['tex'] = self.tex.expand(batch_size, -1)
            batch = self.posemodel(batch, extra_fix_idx=extra_fix_idx)
            opdict = self.forward_model(batch, returnMask=use_mask, returnRendering=use_rendering,
                                        returnNormal=use_normal)

            losses = {}
            if 'iris' in batch.keys() and self.cfg.optimize.use_iris:
                pred_iris = opdict['trans_verts'][:, self.idx_iris, :2]
                pred_lmk = torch.cat([opdict['pred_lmk'], pred_iris], dim=1)
                gt_lmk = torch.cat([batch['lmk'][:, :], batch['iris'][:, :]], dim=1)

            else:
                pred_lmk = opdict['pred_lmk']

                gt_lmk = batch['lmk']
            gt_lmk = gt_lmk.to(pred_lmk.device).type(torch.float32)
            # lmk loss, use confidence, if confidence is 0, then ignore this points (e.g. especially for iris points)

            weight = torch.ones_like(pred_lmk)[0,:,0]
            weight[0:16]*=5
            weight[31:35]*=5

            losses['lmk'] = lossfunc.batch_kp_2d_l1_loss(gt_lmk, pred_lmk,weight) * self.loss_cfg.lmk
            # eye distance loss from DECA
            if self.cfg.loss.eyed > 0.:
                losses['eyed'] = lossfunc.eyed_loss(pred_lmk[:, :68, :2], gt_lmk[:, :68, :2]) * self.cfg.loss.eyed

            if use_mask:
                losses['mesh_inside_mask'] = (torch.relu(
                    opdict['mesh_mask'] - batch['mask'])).abs().mean() * self.loss_cfg.inside_mask
                mesh_mask = opdict['mesh_mask']
                non_skin_mask = 1 - mesh_mask.detach()
                hair_only = non_skin_mask * batch['hair_mask']
                mesh_mask = mesh_mask + hair_only
                opdict['mesh_mask'] = mesh_mask
                losses['mesh_mask'] = lossfunc.huber(opdict['mesh_mask'], batch['mask']) * self.loss_cfg.mesh_mask
            if use_rendering:
                losses['image'] = (batch['face_mask'] * (
                            batch['image'] - opdict['image'])).abs().mean() * self.loss_cfg.image
                losses['image_albedo'] = (batch['face_mask'] *
                                          (batch['image'] - opdict['albedo_image'])).abs().mean() * self.loss_cfg.albedo

            losses['reg_shape'] = (torch.sum(batch['beta'] ** 2) / 2).mean() * 1e-5
            losses['reg_exp'] = (torch.sum(batch['exp'] ** 2) / 2).mean() * 1e-5
            losses['reg_tex'] = (torch.sum(batch['tex'] ** 2) / 2).mean() * 5e-5
            # regs in init shoulder

            losses['reg_scale'] = torch.sum(self.model_scale) *self.loss_cfg.scale_weight

            shoulder_pose = batch['full_pose'][:, 16:18]
            shoulder_pose_axis = rotation_converter.batch_matrix2axis(shoulder_pose.reshape(-1, 3, 3)).reshape(
                batch_size, -1, 3)
            losses['reg_shoulder'] = (shoulder_pose_axis - self.posemodel.init_full_pose[:, 16:18,
                                                           :].detach()).mean().abs() * 1000

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss

            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()

            if vis_step < 1000 and i % vis_step == 0:
                print('model_tsfm:',self.model_tsfm)
                loss_info = f"Iter: {i}/{iters}: "
                for k, v in losses.items():
                    loss_info = loss_info + f'{k}: {v:.6f}, '
                print(loss_info)
                visdict = {
                    'inputs': image,
                    'lmk_gt': util.tensor_vis_landmarks(image, gt_lmk, isScale=False),
                    'lmk_pred': util.tensor_vis_landmarks(image, pred_lmk, isScale=False)
                }
                # render shape
                faces = self.smplx.faces_tensor

                shape_image = render_shape(vertices=opdict['trans_verts'].detach(),
                                           faces=faces.expand(batch_size, -1, -1),
                                           image_size=self.cfg.data.image_size,
                                           background=image)
                visdict['shape'] = shape_image
                for key in opdict.keys():
                    if 'image' in key:
                        visdict[key] = opdict[key]
                util.visualize_grid(visdict, os.path.join(vispath, f'{i:06}.png'), return_gird=False)
            if (i+1)%1000==0:
                mesh = trimesh.Trimesh(
                    vertices=opdict['verts'][0].detach().cpu().numpy() - np.array([0.006, -1.644, 0.010]),
                    faces=faces.detach().cpu().numpy())
                trimesh.exchange.export.export_mesh(mesh, os.path.join(vispath, f'{i:06}.obj'), include_texture=False)
                mesh = trimesh.Trimesh(
                    vertices=opdict['verts_ori'].detach().cpu().numpy() - np.array([0.006, -1.644, 0.010]),
                    faces=faces.detach().cpu().numpy())
                trimesh.exchange.export.export_mesh(mesh, os.path.join(vispath, f'{i:06}_ori.obj'), include_texture=False)

        visdict = {
            'inputs': image,
            'lmk_gt': util.tensor_vis_landmarks(image, gt_lmk, isScale=False),
            'lmk_pred': util.tensor_vis_landmarks(image, pred_lmk, isScale=False)
        }
        # render shape
        faces = self.smplx.faces_tensor
        shape_image = render_shape(vertices=opdict['trans_verts'].detach(),
                                   faces=faces.expand(batch_size, -1, -1),
                                   image_size=self.cfg.data.image_size,
                                   background=image)
        visdict['shape'] = shape_image
        for key in opdict.keys():
            if 'image' in key:
                visdict[key] = opdict[key]
        util.visualize_grid(visdict, os.path.join(vispath, f'{i:06}.png'), return_gird=False)

        self.save_tsfm()


        mesh = trimesh.Trimesh(vertices=opdict['verts_template'].detach().cpu().numpy()- np.array([0.006, -1.644, 0.010]),faces=faces.detach().cpu().numpy())
        trimesh.exchange.export.export_mesh(mesh,os.path.join(vispath, 'final_template.obj'),include_texture=False)     #### template smplx with new shape params
        mesh = trimesh.Trimesh(
            vertices=opdict['verts_template_ori'].detach().cpu().numpy() - np.array([0.006, -1.644, 0.010]),
            faces=faces.detach().cpu().numpy())
        trimesh.exchange.export.export_mesh(mesh, os.path.join(vispath, 'final_template_ori.obj'), include_texture=False)  #### template smplx without shape optimization.



    def save_tsfm(self):
        self.combine_tsfm()
        print(self.matrix)
        self.matrix = self.matrix.cpu().detach().numpy().astype(np.float32).T

        print(self.model_tsfm)

        self.matrix.tofile(self.save_path + '/model_tsfm.dat')

        self.model_tsfm = self.model_tsfm.cpu().detach().numpy().astype(np.float32)
        self.model_tsfm.tofile(self.save_path + '/model_tsfm_semantic.dat')

    def run(self,
            savepath=None,
            optimize_iter=3000,
            batch_size=1,
            vis_step=100,
            data_type='fix_shoulder',):
        '''
        Args:
            iter_list: list, number of iterations for each stage
            savepath: str, path to save the results
            args: dict, additional arguments
        '''
        self.save_path = savepath
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=min(batch_size, 4),
                                                 pin_memory=True,
                                                 drop_last=False)

        self.optimize(dataloader,optimize_iter,vis_step=vis_step,vispath=os.path.join(savepath,'vis'),
                      data_type=data_type,use_mask=self.cfg.optimize.use_mask,use_rendering=self.cfg.optimize.use_rendering)




    def save(self, savepath, dataloader):
        subject = self.posemodel.subject_id
        for batch in tqdm(dataloader):
            util.move_dict_to_device(batch, self.device)
            batch = self.posemodel(batch)
            image = batch['image']
            batch_size = image.shape[0]
            for i in range(batch_size):
                frame_id = batch['frame_id'][i]
                pixie_param = {
                    'shape': self.beta,
                    'full_pose': batch['full_pose'][i],
                    'light': batch['light'][i],
                    'cam': batch['cam'][i],
                    'exp': batch['exp'][i],
                    'tex': self.tex
                }
                util.save_params(os.path.join(savepath, f'{subject}_f{frame_id:06}_param.pkl'), pixie_param)

        #     shape_image = render_shape(vertices=trans_verts.detach(),
        #                             faces=self.smplx.faces_tensor.expand(1, -1, -1),
        #                             image_size=image.shape[-1],
        #                             background=image)
        #     visdict = {
        #     'inputs': image,
        #     'landmarks2d_gt': util.tensor_vis_landmarks(image, gt_lmk, isScale=True),
        #     'landmarks2d': util.tensor_vis_landmarks(image, predicted_landmarks, isScale=True),
        #     'shape_images': shape_image
        # }
        # grid_image = util.visualize_grid(visdict, savepath=None, return_gird=True)
        # return pixie_param, grid_image


def get_config():
    log.process(os.getpid())
    opt_cmd = options.parse_arguments(sys.argv[1:])
    args = options.set(opt_cmd=opt_cmd)
    # args.output_path = os.path.join(args.data.root, args.data.case,args.output_root,args.name)
    args.output_path = os.path.join(args.output_root, args.name)
    os.makedirs(args.output_path, exist_ok=True)
    options.save_options_file(args)

    return args



if __name__ == '__main__':
    args = get_config()
    dataprocess = DataProcessor(args)
    dataprocess.run(args.subject_path,vis=True,ignore_existing=args.ignore_existing)
    current_irispath_list =  glob(os.path.join(args.path, args.subject, 'iris', '*.txt'))
    current_lmkpath_list =  glob(os.path.join(args.path, args.subject, 'landmark2d', '*.txt'))


    imagepath_list = []
    for path in current_irispath_list:
        name = path.split('/')[-1][:-4]
        if os.path.exists(os.path.join(args.path, args.subject, 'landmark2d', name+ '.txt')):
            imagepath_list.append(os.path.join(args.path, args.subject, 'matting', name + '.png'))


    dataset = NerfDataset(args, given_imagepath_list = imagepath_list)
    optimizer = SMPLX_optimizer(args, dataset, args.device,args.data.image_size)
    optimizer.run(os.path.join(args.path,args.subject,'optimize'), args.optimize.iter, args.batch_size, vis_step=args.optimize.vis_step, data_type=args.optimize.data_type)
