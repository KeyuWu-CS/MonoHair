import torch
import numpy as np
import os
import torch.nn.functional as F
from datasets.BaseDataset import BaseDataset
from util import getProjPoints


class OriDataset(BaseDataset):

    def __init__(self, root_folder, ori_samples_folder, calib_folder, device, data_type='union', num_views=4, slct_vids=None, num_pt_required=None, use_dir=True,use_hair_depth=True, cat_depth=True):
        super().__init__(root_folder, calib_folder, device, data_type, num_views=num_views, slct_vids=slct_vids, use_dir=use_dir, use_hair_depth=use_hair_depth,cat_depth=cat_depth)

        self.ori_samples_folder = ori_samples_folder
        self.points_fname = 'ori3d_{}.dat'
        self.num_pt_required = num_pt_required
        self.translate_to_origin = np.array([0.006, -1.644, 0.010])
        self.bbox_min = np.array([-0.24, -0.2, -0.35]) - self.translate_to_origin
        self.bbox_max = np.array([0.24, 0.4, 0.25]) - self.translate_to_origin

    def readOriSamplesFromFile(self, fname):
        data = np.fromfile(fname, dtype='float32').reshape(-1, 6)

        coord = data[:, :3].T
        homogeneous = np.ones((1, coord.shape[1]), dtype='float32')
        coord = np.concatenate([coord, homogeneous], axis=0)

        orient = data[:, 3:].T
        return coord, orient

    def __getitem__(self, idx):
        '''

        :param idx:
        :return:
        '''
        item = super().__getitem__(idx)
        points_path = os.path.join(self.ori_samples_folder, self.points_fname.format(item['item_id']))
        points, orients = self.readOriSamplesFromFile(points_path)


        # print('id {}\nnum pts {}'.format(idx, points.shape))
        #
        # # only sample in the bounding box
        # in_bbox = (points[0] > self.bbox_min[0]) & (points[0] < self.bbox_max[0]) & \
        #           (points[1] > self.bbox_min[1]) & (points[1] < self.bbox_max[1]) & \
        #           (points[2] > self.bbox_min[2]) & (points[2] < self.bbox_max[2])
        #
        # points = points[:, in_bbox]
        # orients = orients[:, in_bbox]
        # print('in bbox {}'.format(points.shape))

        rand_perm = torch.randperm(points.shape[1]).to(self.device)

        # [4, N]
        item['points'] = torch.tensor(points).to(self.device)[:, rand_perm]
        # [3, N]
        orients = torch.tensor(orients).to(self.device)[:, rand_perm]
        # get only required number of points
        if self.num_pt_required is not None:
            item['points'] = item['points'][:, :self.num_pt_required]
            orients = orients[:, :self.num_pt_required]

        orients = F.normalize(item['model_tsfm'][:3, :3] @ orients, dim=0)

        xy_coords = getProjPoints(item['points'], self.cam_poses_w2c, self.ndc_proj, item['model_tsfm'])

        # [V, N, 1, 2]
        item['xy_coords'] = xy_coords
        orients = orients.transpose(0, 1)
        # [N, 3]
        item['gt_targets'] = orients

        # [4, N]
        item['pts_world'] = item['model_tsfm'] @ item['points']
        # [V, 4, N]
        item['pts_view'] = self.cam_poses_w2c @ item['pts_world']

        # [N, 1, 3]
        item['pts_world'] = item['pts_world'][:3].transpose(0, 1).unsqueeze(1)
        # [N, V, 3]
        item['pts_view'] = item['pts_view'][:, :3, :].permute(2, 0, 1)

        return item


if __name__ == "__main__":
    pass
