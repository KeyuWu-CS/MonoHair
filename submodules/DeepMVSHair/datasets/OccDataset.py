import torch
import numpy as np
import os
from datasets.BaseDataset import BaseDataset
from util import getProjPoints


class OccDataset(BaseDataset):

    def __init__(self, root_folder, occ_samples_folder, calib_folder, device, data_type='union', num_views=4, slct_vids=None, num_pt_required=None, use_dir=True,use_hair_depth=True,cat_depth=True):
        super().__init__(root_folder, calib_folder, device, data_type, num_views=num_views, slct_vids=slct_vids, use_dir=use_dir,use_hair_depth=use_hair_depth,cat_depth=cat_depth)

        self.occ_samples_folder = occ_samples_folder
        self.occ_fname = 'occ_samples_{}.dat'

        self.num_pt_required = num_pt_required
        # to match orignially sampled points
        self.translate_to_origin = np.array([0.006, -1.644, 0.010])
        self.bbox_min = np.array([-0.24, -0.2, -0.35]) - self.translate_to_origin
        self.bbox_max = np.array([0.24, 0.4, 0.25]) - self.translate_to_origin

    def readOccSamplesFromFile(self, fname):
        data = np.fromfile(fname, dtype='float32').reshape(-1, 4)
        coord = data[:, :3].T
        homogeneous = np.ones((1, coord.shape[1]), dtype='float32')
        coord = np.concatenate([coord, homogeneous], axis=0)

        label = data[:, 3]
        return coord, label

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        points_path = os.path.join(self.occ_samples_folder, self.occ_fname.format(item['item_id']))
        points, labels = self.readOccSamplesFromFile(points_path)

        # print('id {}\nnum pts {}'.format(idx, points.shape))
        #
        # # only sample in the bounding box
        # in_bbox = (points[0] > self.bbox_min[0]) & (points[0] < self.bbox_max[0]) & \
        #             (points[1] > self.bbox_min[1]) & (points[1] < self.bbox_max[1]) & \
        #             (points[2] > self.bbox_min[2]) & (points[2] < self.bbox_max[2])
        #
        # points = points[:, in_bbox]
        # labels = labels[in_bbox]
        # print('in bbox {}'.format(points.shape))

        # randomly sample positive/negative points
        rand_perm = torch.randperm(len(labels)).to(self.device)

        # [4, N]
        item['points'] = torch.tensor(points).to(self.device)[:, rand_perm]
        if self.num_pt_required is not None:
            item['points'] = item['points'][:, :self.num_pt_required]

        xy_coords = getProjPoints(item['points'], self.cam_poses_w2c, self.ndc_proj, item['model_tsfm'])

        # [V, N, 1, 2]
        item['xy_coords'] = xy_coords
        # [N, ]
        item['gt_targets'] = torch.tensor(labels).to(self.device)[rand_perm].long()
        if self.num_pt_required is not None:
            item['gt_targets'] = item['gt_targets'][:self.num_pt_required]

        # [4, N]
        item['pts_world'] = item['model_tsfm'] @ item['points']
        # [V, 4, N]
        item['pts_view'] = self.cam_poses_w2c @ item['pts_world']

        # [N, 1, 3]
        item['pts_world'] = item['pts_world'][:3].transpose(0, 1).unsqueeze(1)
        # [N, V, 3]
        item['pts_view'] = item['pts_view'][:, :3, :].permute(2, 0, 1)

        return item

