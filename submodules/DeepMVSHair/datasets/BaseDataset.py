import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as T
import json
from scipy.spatial.transform import Rotation as R
from torchvision.utils import save_image


class BaseDataset(Dataset):

    def __init__(self, root_folder, calib_folder, device, data_type='union', num_views=4, slct_vids=None, use_dir=True, use_hair_depth=True, cat_depth=True,case=None):
        self.root_folder = root_folder
        self.items_fname_list = os.listdir(self.root_folder)
        if case is not None:
            self.items_fname_list = [case]

        self.device = device
        self.data_type = data_type
        self.calib_folder = calib_folder
        self.num_views = num_views
        self.slct_vids = np.array(slct_vids, dtype=np.int64) if slct_vids is not None else None
        self.use_dir = use_dir
        self.use_hair_depth = use_hair_depth
        self.cat_depth = cat_depth
        print('cat_depth:',self.cat_depth)
        print('use_dir:',use_dir)

        self.img_size = (1280, 720)

        self.cam_fnm = 'cam_params.json'

        self.read_cam_params()


        if data_type == 'union':
            # currently used for training and testing
            self.model_tsfm_fname = 'model_tsfm.dat'
            self.mask_fname = 'mask_union.dat'
            self.conf_fname = 'conf_union.dat'
            self.orient_fname = 'orient_union.dat'
            self.dir_fname = 'dir_union.dat'
            self.depth_fname = 'depth_union.dat'
            self.bust_hair_depth_fname = 'bust_hair_depth_union.dat'

        elif data_type == 'synthetic':
            self.model_tsfm_fname = 'model_tsfm_complete.dat'
            self.mask_fname = 'mask.png'
            self.conf_fname = 'raw_conf.png'
            self.orient_fname = 'dense.png'
            self.dir_fname = 'dir.png'
            self.depth_fname = 'bust_depth.png'

        elif data_type == 'real':
            self.model_tsfm_fname = 'model_tsfm.dat'
            # self.mask_fname = 'mask.png'
            # self.mask_fname1 = 'mask1.png'
            # # self.conf_fname = 'raw_conf.png'
            # self.conf_fname = 'mask.png'

            ### synthetic
            self.mask_fname = 'mask.png'
            # self.mask_fname1 = 'dense1.png'
            # self.conf_fname = 'raw_conf.png'
            self.conf_fname = 'undirectional_map.png'


            self.orient_fname = 'undirectional_map.png'
            self.dir_fname = 'dense.png'
            self.depth_fname = 'bust_depth.png'
            # self.bust_hair_depth_fname = 'bust_hair_depth.png'
            self.bust_hair_depth_fname = 'hair_depth.png'

        else:
            raise RuntimeError('data type {} is not supported'.format(self.data_type))


    def readIntrinAndPose(self):
        '''
        not used currently, use read_cam_params to read json file instead
        :return:
        '''
        # ndc projection
        ndc_data = torch.tensor(np.fromfile(os.path.join(self.calib_folder, 'ndc_proj.dat'), dtype='float32')[1:].reshape(self.num_views, 4))
        self.ndc_proj = torch.zeros((self.num_views, 4, 4), dtype=torch.float32)
        self.ndc_proj[:, 0, 0] = ndc_data[:, 0]
        self.ndc_proj[:, 1, 1] = ndc_data[:, 1]
        self.ndc_proj[:, 0, 2] = ndc_data[:, 2]
        self.ndc_proj[:, 1, 2] = ndc_data[:, 3]
        self.ndc_proj[:, 3, 2] = -1.
        self.ndc_proj = self.ndc_proj.to(self.device)
        if self.slct_vids is not None:
            self.ndc_proj = self.ndc_proj[self.slct_vids]

        # camera poses, world-to-camera
        cam_poses_c2w = np.fromfile(os.path.join(self.calib_folder, 'poses_recenter.dat'), dtype='float32')[1:].reshape(self.num_views, 4, 4)
        self.cam_poses_w2c = torch.inverse(torch.tensor(cam_poses_c2w)).to(self.device)
        if self.slct_vids is not None:
            self.cam_poses_w2c = self.cam_poses_w2c[self.slct_vids]


    def read_cam_params(self):
        with open(os.path.join(self.calib_folder, self.cam_fnm), 'r') as f:
            json_obj = json.load(f)

            # ndc projection
            ndc_data = torch.tensor([cam_item['ndc_prj'] for cam_item in json_obj['cam_list']], dtype=torch.float32)
            self.ndc_proj = torch.zeros((self.num_views, 4, 4), dtype=torch.float32)
            self.ndc_proj[:, 0, 0] = ndc_data[:, 0]
            self.ndc_proj[:, 1, 1] = ndc_data[:, 1]
            self.ndc_proj[:, 0, 2] = ndc_data[:, 2]
            self.ndc_proj[:, 1, 2] = ndc_data[:, 3]
            self.ndc_proj[:, 3, 2] = -1.
            self.ndc_proj = self.ndc_proj.to(self.device)
            if self.slct_vids is not None:
                self.ndc_proj = self.ndc_proj[self.slct_vids]

            # camera poses, world-to-camera
            cam_poses_c2w = torch.tensor([cam_item['pose'] for cam_item in json_obj['cam_list']], dtype=torch.float32)
            self.cam_poses_w2c = torch.inverse(cam_poses_c2w).to(self.device)
            if self.slct_vids is not None:
                self.cam_poses_w2c = self.cam_poses_w2c[self.slct_vids]

    def getViewIdOfFname(self, fname):
        '''

        :param fname: file name, eg. '12.txt'
        :return: file id, eg. 12
        '''
        return int(fname[:fname.index('.')])

    def readImageFromFile(self, fname, format):
        # img = cv2.imread(fname, cv2.IMREAD_COLOR)
        img = np.array(Image.open(fname).convert(format))
        return img

    def readMatrixFromFile(self, fname):
        '''
        read 4x4 matrix saved by opengl
        :param fname:
        :return:
        '''
        data = np.fromfile(fname, dtype='float32').reshape(4, 4)
        # transpose the matrix, to convert it from column-major to row-major
        return np.array(data).T

    def read_union_data(self, idx):
        '''
        read synthetic data, where imgs from different views are integrated for fast reading
        :param idx:
        :return:
        '''

        # index = [2,9,7,10,3,11,12,0,8,6,4,14,15,1,5,13]
        case_path = os.path.join(self.root_folder, self.items_fname_list[idx])

        # 1. model transform, generated by OpenGL, needs to be transposed first
        model_tsfm = np.fromfile(os.path.join(case_path, self.model_tsfm_fname), dtype='float32').reshape(4, 4).T
        model_tsfm = torch.tensor(model_tsfm)

        # 2. orientation
        orient_union = np.fromfile(os.path.join(case_path, self.dir_fname if self.use_dir else self.orient_fname), dtype='uint8').reshape(
            self.num_views, 2, self.img_size[0], self.img_size[1])
        if self.slct_vids is not None:
            orient_union = orient_union[self.slct_vids]
        orient_union = torch.tensor(orient_union).float() / 255.0
        # orient_union = orient_union[index]

        # 3. confidence
        conf_union = np.fromfile(os.path.join(case_path, self.conf_fname), dtype='uint8').reshape(
            self.num_views, 1, self.img_size[0], self.img_size[1])
        if self.slct_vids is not None:
            conf_union = conf_union[self.slct_vids]
        conf_union = torch.tensor(conf_union).float() / 255.0
        # conf_union = conf_union[index]
        # 4. mask
        mask_union = np.fromfile(os.path.join(case_path, self.mask_fname), dtype='uint8').reshape(
            self.num_views, 1, self.img_size[0], self.img_size[1])
        if self.slct_vids is not None:
            mask_union = mask_union[self.slct_vids]
        mask_union = torch.tensor(mask_union).float() / 255.0
        # mask_union = mask_union[index]
        # 5. depth
        depth_union = np.fromfile(os.path.join(case_path, self.depth_fname), dtype='uint8').reshape(
            self.num_views, 1, self.img_size[0], self.img_size[1])
        if self.slct_vids is not None:
            depth_union = depth_union[self.slct_vids]
        depth_union = torch.tensor(depth_union).float() / 255.0 * 2.0

        # 6. bust_hair_depth
        bust_hair_depth_union = np.fromfile(os.path.join(case_path, self.bust_hair_depth_fname), dtype='uint8').reshape(
            self.num_views, 1, self.img_size[0], self.img_size[1])
        if self.slct_vids is not None:
            bust_hair_depth_union = bust_hair_depth_union[self.slct_vids]
        bust_hair_depth_union = torch.tensor(bust_hair_depth_union).float() / 255.0 * 2.0

        if self.use_hair_depth:
            if self.cat_depth:
                bust_hair_depth_union = torch.where(mask_union>0.5, bust_hair_depth_union, torch.ones_like(bust_hair_depth_union)*2.0)
                depth_union = torch.cat([bust_hair_depth_union,depth_union],1)
            else:
                depth_union = bust_hair_depth_union


        # depth_union = depth_union[index]
        item = {}
        item['item_id'], item['case_id'] = self.items_fname_list[idx].split('_')
        # [4, 4]
        item['model_tsfm'] = model_tsfm.to(self.device)
        # [V, 1, H, W]
        item['masks'] = mask_union.to(self.device)
        # [V, 2, H, W]
        item['orient_map'] = (orient_union.to(self.device) * 2. - 1.) * item['masks']
        # [V, 1, H, W]
        item['conf_map'] = conf_union.to(self.device)
        # [V, 1, H, W]
        item['depth_map'] = depth_union.to(self.device)

        return item

    def read_synthetic_data(self, idx):
        '''
        read real data captured by ARkit, images are saved in separated folders
        :param idx:
        :return:
        '''
        cases_folder = os.path.join(self.root_folder, self.items_fname_list[idx], 'cases')
        cases_list = os.listdir(cases_folder)

        case_id = np.random.randint(0, len(cases_list), 1)[0]
        case_path = os.path.join(cases_folder, cases_list[case_id])

        # 1. model transform, generated by OpenGL, needs to be transposed first
        model_tsfm = np.fromfile(os.path.join(case_path, self.model_tsfm_fname), dtype='float32').reshape(4, 4).T
        model_tsfm = torch.tensor(model_tsfm)

        orient_list = []
        mask_list = []
        mask_list1 = []
        conf_list = []

        depth_list = []

        views_folder = os.path.join(case_path, 'views')
        views_list = os.listdir(views_folder)

        for vid in self.slct_vids:

            view = views_list[vid]

            view_path = os.path.join(views_folder, view)

            orient_map = T.to_tensor(self.readImageFromFile(os.path.join(view_path, self.dir_fname if self.use_dir else self.orient_fname), 'RGB')[..., :2])
            orient_list.append(orient_map)

            mask_map = T.to_tensor(self.readImageFromFile(os.path.join(view_path, self.mask_fname), 'L'))
            mask_list.append(mask_map)
            mask_map1 = T.to_tensor(self.readImageFromFile(os.path.join(view_path, self.mask_fname1), 'L'))
            mask_list1.append(mask_map1)

            conf_map = T.to_tensor(self.readImageFromFile(os.path.join(view_path, self.conf_fname), 'L'))
            conf_list.append(conf_map)

            depth_map = T.to_tensor(self.readImageFromFile(os.path.join(view_path, self.depth_fname), 'L'))
            depth_list.append(depth_map)

        mask_union = torch.stack(mask_list, dim=0)
        mask_union1 = torch.stack(mask_list1, dim=0)
        conf_union = torch.stack(conf_list, dim=0)
        orient_union = torch.stack(orient_list, dim=0)
        depth_union = torch.stack(depth_list, dim=0) * 2.0

        item = {}
        item['item_id'] = self.items_fname_list[idx]
        item['case_id'] = cases_list[case_id]
        # [4, 4]
        item['model_tsfm'] = model_tsfm.to(self.device)
        # [V, 1, H, W]
        item['masks'] = mask_union.to(self.device)
        # [V, 2, H, W]
        item['orient_map'] = (orient_union.to(self.device) * 2. - 1.) * item['masks']
        # item['orient_map'] = (orient_union.to(self.device) * 2. - 1.)
        # [V, 1, H, W]
        item['conf_map'] = conf_union.to(self.device)

        # [V, 1, H, W]
        item['depth_map'] = depth_union.to(self.device)

        return item

    def read_real_data(self, idx):

        item_folder = os.path.join(self.root_folder, self.items_fname_list[idx])

        model_tsfm = np.fromfile(os.path.join(item_folder, self.model_tsfm_fname), dtype='float32').reshape(4, 4).T
        model_tsfm = torch.tensor(model_tsfm)
        model_tsfm_semantic_path = os.path.join(item_folder, 'model_tsfm_semantic.dat')

        img_folder = os.path.join(item_folder, 'imgs')

        orient_list = []
        mask_list = []
        mask_list1 = []
        conf_list = []
        depth_list = []
        bust_hair_depth_list = []

        views_list = os.listdir(img_folder)

        views_list.sort()
        if self.slct_vids is None:
            self.slct_vids = [i for i in range(len(views_list))]


        for vid in self.slct_vids:

            view = views_list[vid]
            view_path = os.path.join(img_folder, view)

            orient_map = T.to_tensor(
                self.readImageFromFile(os.path.join(view_path, self.dir_fname if self.use_dir else self.orient_fname), 'RGB')[..., :2])
            orient_list.append(orient_map)

            mask_map = T.to_tensor(self.readImageFromFile(os.path.join(view_path, self.mask_fname), 'L'))
            mask_list.append(mask_map)
            # mask_map1 = T.to_tensor(self.readImageFromFile(os.path.join(view_path, self.mask_fname1), 'L'))
            # mask_list1.append(mask_map1)

            conf_map = T.to_tensor(self.readImageFromFile(os.path.join(view_path, self.conf_fname), 'L'))
            conf_list.append(conf_map)

            depth_map = T.to_tensor(self.readImageFromFile(os.path.join(view_path, self.depth_fname), 'L'))
            depth_list.append(depth_map)

            bust_hair_depth_map = T.to_tensor(self.readImageFromFile(os.path.join(view_path, self.bust_hair_depth_fname), 'L'))
            bust_hair_depth_list.append(bust_hair_depth_map)

        mask_union = torch.stack(mask_list, dim=0)
        # mask_union1 = torch.stack(mask_list1, dim=0)
        conf_union = torch.stack(conf_list, dim=0)
        orient_union = torch.stack(orient_list, dim=0)
        depth_union = torch.stack(depth_list, dim=0) * 2.0
        bust_hair_depth_union = torch.stack(bust_hair_depth_list,dim=0) * 2.0
        mask_union[mask_union<0.5]=0
        mask_union[mask_union>=0.5]=1

        if self.use_hair_depth:
            if self.cat_depth:
                # bust_hair_depth_union = torch.where(mask_union > 0.5, bust_hair_depth_union,
                #                                     torch.ones_like(bust_hair_depth_union) * 2.0)
                depth_union = torch.cat([bust_hair_depth_union, depth_union], 1)
            else:
                # depth_union = torch.where(mask_union1 > 0.8, bust_hair_depth_union,
                #                                     torch.ones_like(bust_hair_depth_union) * 2.0)
                depth_union = bust_hair_depth_union

        item = {}
        item['item_id'] = self.items_fname_list[idx]
        # [4, 4]
        item['model_tsfm'] = model_tsfm.to(self.device)
        # [V, 1, H, W]
        item['masks'] = mask_union.to(self.device)
        # [V, 2, H, W]
        # item['orient_map'] = (orient_union.to(self.device) * 2. - 1.) * item['masks']
        item['orient_map'] = (orient_union.to(self.device) * 2. - 1.)
        # [V, 1, H, W]
        item['conf_map'] = conf_union.to(self.device)
        # [V, 1, H, W]
        item['depth_map'] = depth_union.to(self.device)
        item['model_tsfm_semantic_path'] = model_tsfm_semantic_path

        return item

    def __getitem__(self, idx):
        '''

        :param idx:
        :return:
        '''
        if self.data_type == 'union':
            return self.read_union_data(idx)
        elif self.data_type == 'synthetic':
            return self.read_synthetic_data(idx)
        elif self.data_type == 'real':
            return self.read_real_data(idx)
        else:
            raise RuntimeError('data type {} is not supported'.format(self.data_type))

    def __len__(self):
        return len(self.items_fname_list)


if __name__ == "__main__":
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    bd = BaseDataset('D:\\Hair_Dataset\\syf_cmp_process', '..\\camera\\calib_data\\syf_cmp', device, num_views=8)




