from datasets.BaseDataset import BaseDataset
from sampleOcc import sampleGridCorner, sampleGridCenter
import numpy as np
import os
import torch


class EvalDataset(BaseDataset):

    def __init__(self, root_folder, calib_folder, device, data_type='synthetic', num_views=4, slct_vids=None, sample_style='corner', with_occ=False,use_hair_depth=True,cat_depth=True,use_colmap_points=False,use_dir = True,case=None):
        super().__init__(root_folder, calib_folder, device, data_type=data_type, num_views=num_views, slct_vids=slct_vids, use_hair_depth=use_hair_depth, cat_depth=cat_depth,use_dir=use_dir,case=case)

        # bbox_min = (-0.24, -0.15, -0.35)
        # bbox_max = (0.24, 0.4, 0.25)
        bbox_min = (-0.32, -0.32, -0.24)
        bbox_max = (0.32, 0.32, 0.24)
        self.bust_to_origin = np.array([0.006, -1.644, 0.010])
        self.vsize = 0.005

        self.samples = sampleGridCorner(vsize=self.vsize/2, bbox_min=bbox_min, bbox_max=bbox_max).to(self.device) if sample_style=='corner' else sampleGridCenter(vsize=self.vsize, bbox_min=bbox_min, bbox_max=bbox_max).to(self.device)
        self.grid_resolution = np.array([(bbox_max[0] - bbox_min[0]) / self.vsize, (bbox_max[1] - bbox_min[1]) / self.vsize, (bbox_max[2] - bbox_min[2]) / self.vsize], dtype='float32')
        self.voxel_min = np.array(bbox_min, dtype='float32')

        self.with_occ = with_occ
        self.fv_samples_fname = os.path.join('cmp_fv', 'voxels_fv.dat')
        self.use_colmap_points = use_colmap_points
        self.case =None
        self.sample_style = sample_style


    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        fv_samples_path = os.path.join(self.root_folder, self.items_fname_list[idx], self.fv_samples_fname)
        if self.with_occ and os.path.exists(fv_samples_path):
            fv_samples = torch.tensor(np.fromfile(fv_samples_path, dtype='float32').reshape(-1, 3)).to(self.device)
            homogeneous = torch.ones(fv_samples.shape[0], 1).to(self.device).float()
            item['fv_samples'] = torch.cat([fv_samples, homogeneous], dim=1).transpose(0, 1)
        if self.use_colmap_points and item['item_id']==self.case:
            colmap_points_path = os.path.join(self.root_folder,self.items_fname_list[idx],'ours/colmap_points.obj')
            self.colmap_points = load_colmap_points(colmap_points_path,self.voxel_min, self.bust_to_origin,self.vsize/8,[1024,1024,768] ,True,1)
            self.colmap_points = np.concatenate([self.colmap_points,np.ones((self.colmap_points.shape[0],1))],1)
            bbox_min = (np.min(self.colmap_points[:,0]),np.min(self.colmap_points[:,1]),np.min(self.colmap_points[:,2]))
            bbox_max = (np.max(self.colmap_points[:,0]),np.max(self.colmap_points[:,1]),np.max(self.colmap_points[:,2]))
            sample= sampleGridCorner(vsize=self.vsize/4, bbox_min=bbox_min, bbox_max=bbox_max).to(
                self.device) if self.sample_style == 'corner' else sampleGridCenter(vsize=self.vsize, bbox_min=bbox_min,
                                                                                    bbox_max=bbox_max).to(self.device)
            item['num_sample'] =sample.size(1)

            item['colmap_points'] = torch.from_numpy(self.colmap_points).type(torch.float).to(self.device).permute(1,0)
            # item['colmap_points'] = torch.cat([self.samples,item['colmap_points']],dim=1)
            item['colmap_points'] = torch.cat([sample,item['colmap_points']],dim=1)
        else:
            sample = self.samples
            item['num_sample'] = sample.size(1)
            item['colmap_points'] =sample


        return item

    def set_case(self,case):
        self.case =case

