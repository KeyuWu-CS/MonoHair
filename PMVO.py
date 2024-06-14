from scipy.spatial import KDTree
import options
from log import log
from Utils.PMVO_utils import *
import sys
import torch.nn as nn
import torch
import os
from tqdm import trange
from Utils.Camera_utils import load_cam,parsing_camera


class PMVO(nn.Module):
    def __init__(self, camera,depths, Ori, Conf, masks, device='cuda:0', image_size=[1120, 1992],patch_size=5,visible_threshold=1,conf_threshold=0.4):
        super().__init__()
        self.camera_dict = camera
        self.visible_threshold = visible_threshold

        self.camera = []
        self.camera_key = []
        for k,v in camera.items():

            depths[k] = torch.from_numpy(np.array(depths[k])).to(device).type(torch.float)
            Ori[k] = torch.from_numpy(Ori[k]).to(device).type(torch.float)
            Conf[k] = torch.from_numpy(Conf[k]).to(device).type(torch.float)
            masks[k] = torch.from_numpy(masks[k]).to(device).type(torch.float)
            self.camera_key.append(k)
            self.camera.append(v)

        self.depths_dict = depths
        self.Ori_dict = Ori
        self.Conf_dict = Conf
        self.mask_dict = masks
        self.device = device
        self.image_size = image_size
        self.patch_size = patch_size
        self.conf_threshold =conf_threshold

    def forward(self, points):
        points = torch.from_numpy(points).to(self.device).type(torch.float)  ### N,3
        self.Compute_Visible_and_Ori(points)
        base_view_indexs,base_view_confs = self.Find_max_conf_from_visible_view()

        min_loss = None
        high_conf_Index = None
        best_sample_points = torch.zeros_like(points)
        best_surface_points = torch.zeros_like(points)

        weight = self.compute_weight(self.visible, self.Conf, self.mask)
        for i in range(0,20,2):
            base_view_index = base_view_indexs[i]
            sample_next_points,surface_points = self.sample_next_3d_pos(points, base_view_index)  ### N * num_sample * 3
            Prj_Ori_2D = self.compute_reproject_ori(surface_points, sample_next_points)
            # Prj_Ori_2D = self.compute_reproject_ori(points, sample_next_points)

            loss, index,hcIndex = self.compute_prj_loss(Prj_Ori_2D, self.Ori, weight)
            if min_loss is None:
                min_loss = loss
                min_index = torch.arange(0,index.size(0),1).type(torch.long)
                high_conf_Index = hcIndex
            else:
                # min_index = torch.lt(loss, min_loss)
                # min_loss = torch.minimum(min_loss,loss)
                min_index = torch.logical_and(torch.lt(loss, min_loss),torch.gt(base_view_confs[i],0))
                min_loss[min_index] = loss[min_index]
                high_conf_Index[min_index] = hcIndex[min_index]

            # N = torch.arange(0,min_index.size(0),1).type(torch.long)
            best_sample_points[min_index] = sample_next_points[min_index, index[min_index],:]
            best_surface_points[min_index] = surface_points[min_index]
            # best_sample_points = sample_next_points[N, min_index,:]
        # line_ori = best_sample_points - points  ### N * 3
        line_ori = best_sample_points - best_surface_points  ### N * 3
        line_ori = line_ori/torch.linalg.norm(line_ori,2,dim=-1,keepdim=True)
        # index = torch.lt(min_loss,1)
        select_points = points
        select_ori = line_ori
        return select_points, select_ori, min_loss, high_conf_Index


    def refine(self,points,ori):
        self.Compute_Visible_and_Ori(points)
        filter_index = self.filter_head_points(points,self.visible_threshold)
        # filter_index = self.filter_head_points(points,0.1)

        next_points = points + ori * 0.005/4
        next_points = torch.unsqueeze(next_points,dim=1)
        Prj_Ori_2D = self.compute_reproject_ori(points, next_points)

        loss, _,hcIndex = self.compute_prj_loss(Prj_Ori_2D, self.Ori, weight=None)
        # filter_index = torch.logical_and(filter_index,~hcIndex )
        loss[filter_index]=-1
        return loss


    def filter_head_points(self,points,visible_threshold):

        points_numpy = points.clone().cpu().numpy()
        nei_dist,nei_index = bust_tree.query(points_numpy,k=1)

        # filter_head_index = nei_dist<0.005
        # filter_head_index = torch.from_numpy(filter_head_index).to(points.device)  ### to filter some points very cloest to head

        nei_scalp_dist,_ = scalp_tree.query(points_numpy,k=1)
        head_top_index = nei_scalp_dist<0.04    #### sometimes the points above the head (cloest to human head) may be filter due to incorrect hair segmentation
        head_top_index = np.logical_and(head_top_index,points_numpy[:,2]<scalp_max[2]-0.01)
        head_top_index = torch.from_numpy(head_top_index).to(points.device)


        indexs = []
        unvisibles = []
        mask = []
        count = 0
        for view,camera in self.camera_dict.items():
            count+=1
            uv, z, unvisible_index = self.project_points(points,camera, self.image_size)
            m = self.get_mask(uv, view)

            d = self.get_depth(uv, view)

            unvisible = torch.where(z * 255 - d >= visible_threshold,torch.ones_like(d), torch.zeros_like(d))
            # unvisible[unvisible_index] = 1

            m[m>0.2]=1
            index = (1-unvisible) * m

            indexs.append(index[None])
            unvisibles.append(unvisible[None])
            mask.append(m[None])
        indexs =torch.cat(indexs,0)

        unvisibles = torch.cat(unvisibles,0)
        visibles = 1 - unvisibles

        filter_index = torch.lt(torch.sum(visibles, dim=0) - torch.sum(indexs, dim=0),
                                torch.sum(visibles, dim=0) * 1 / 2)
        filter_index = ~filter_index



        filter_index = torch.logical_and(filter_index,~head_top_index)
        # filter_index = torch.logical_or(filter_index, filter_head_index) ### to filter some points very cloest to head

        return filter_index






    def compute_prj_loss(self,Prj_Ori_2D, Ori, weight):
        '''

        :param Prj_Ori_2D: view * N * num_sample * 2
        :param Ori: view * N * 2
        :param Conf: view * N
        :return:
        '''
        num_sample = Prj_Ori_2D.size(2)

        min_loss = None
        highConfPatch = torch.gt(torch.max(self.Conf_patch,-1)[0],self.conf_threshold)
        highConfPatch = torch.repeat_interleave(highConfPatch[...,None],num_sample,dim=2)
        for i in range(self.Conf_patch.size(-1)):
            Ori = self.Ori_patch[...,i,:]
            Ori = torch.unsqueeze(Ori, dim=2)
            Ori = torch.repeat_interleave(Ori, num_sample, dim=2)  ### view * N * num_sample * 2
            Conf = self.Conf_patch[...,i]
            Conf = torch.unsqueeze(Conf,dim=2)
            Conf = torch.repeat_interleave(Conf, num_sample, dim=2)
            similar = torch.maximum(torch.cosine_similarity(Ori, Prj_Ori_2D, dim=-1),torch.cosine_similarity(-Ori, Prj_Ori_2D, dim=-1))
            # similar =torch.cosine_similarity(Ori, Prj_Ori_2D, dim=-1)
            if min_loss is None:
                min_loss = 1-similar
                best_Conf = Conf
            else:
                index = torch.lt(1-similar,min_loss)
                index1 = torch.logical_and(index,Conf>self.conf_threshold)
                min_loss = torch.where(torch.logical_and(index1,highConfPatch),1-similar,min_loss)
                min_loss = torch.where(torch.logical_and(index,~highConfPatch),1-similar,min_loss)
                best_Conf = torch.where(torch.logical_and(index1,highConfPatch),Conf,best_Conf)
                best_Conf = torch.where(torch.logical_and(index,~highConfPatch),Conf,best_Conf)

                #### original
                # best_Conf = torch.where(torch.logical_and(1 - similar < min_loss,Conf>self.conf_threshold), Conf, best_Conf)
                # min_loss = torch.where(torch.logical_and(1-similar<min_loss,Conf>self.conf_threshold),1-similar,min_loss)

                #min_loss = torch.minimum(min_loss, 1 - similar)  ### view * N * num_sample
        visible = torch.unsqueeze(self.visible,dim=2)
        visible = torch.repeat_interleave(visible,num_sample,2)
        mask = torch.unsqueeze(self.mask,dim=2)
        mask = torch.repeat_interleave(mask,num_sample,dim=2)
        # print(min_loss[:,:100])
        weight = self.compute_weight(visible, best_Conf,mask)
        min_loss = min_loss * weight
        weight1 = weight>0

        positive_index = torch.gt(torch.sum(weight,dim=0)/torch.sum(weight1,dim=0),self.conf_threshold)   ## N * num_sample
        low_conf_index = torch.lt(torch.sum(positive_index,dim=-1),5) ### N

        min_loss = torch.sum(min_loss,dim=0)/torch.sum(weight,dim=0)
        min_loss_copy = min_loss.clone()
        min_loss = torch.where(positive_index,min_loss,torch.ones_like(min_loss))
        min_loss[low_conf_index] = min_loss_copy[low_conf_index]
        min_loss, min_index = torch.min(min_loss,dim=-1)
        high_conf_index = positive_index[torch.arange(positive_index.size(0)).type(torch.long),min_index]


        return  min_loss, min_index,high_conf_index

    def compute_weight(self,visible, Conf, mask):
        weight = torch.where(visible==-1 , torch.zeros_like(visible), torch.ones_like(visible))
        weight*=Conf
        weight = torch.where(mask>0, weight, weight)
        return weight



    def compute_reproject_ori(self, points, sample_next_points):
        '''
        :param points: N * 3
        :param sample_next_points: N*  numsample * 3
        :return: view * N * num_sample * 2
        '''

        num_sample = sample_next_points.size(1)
        sample_next_points = torch.reshape(sample_next_points,(-1,3))
        Line_Ori_2D = []
        for i in range(len(self.camera)):
            uv_sample, _ = self.camera[i].projection(sample_next_points)
            uv_sample = self.camera[i].uv2pixel(uv_sample,self.image_size,points.device) ### (N*numsample) * 2
            uv_sample = torch.reshape(uv_sample, (-1, num_sample, 2))   ### N * numsample * 2
            uv, _ = self.camera[i].projection(points)
            uv = self.camera[i].uv2pixel(uv, self.image_size, points.device)   ### N * 2
            uv = torch.unsqueeze(uv, dim=1)
            uv = torch.repeat_interleave(uv, num_sample, dim=1) ### N * numsample * 2
            prj_ori = uv_sample - uv
            Line_Ori_2D.append(prj_ori[None])
        Line_Ori_2D = torch.cat(Line_Ori_2D,0) ### view * N * num_sample * 2

        return Line_Ori_2D

    def compute_points_prj_ori(self, points, next_points):
        '''
        :param points: N * 3
        :param sample_next_points: N* 3
        :return: view * N * 2
        '''

        Line_Ori_2D = []
        for i in range(len(self.camera)):
            uv_next, _ = self.camera[i].projection(next_points)
            uv_next = self.camera[i].uv2pixel(uv_next,self.image_size,points.device) ### (N*numsample) * 2
            uv, _ = self.camera[i].projection(points)
            uv = self.camera[i].uv2pixel(uv, self.image_size, points.device)   ### N * 2
            prj_ori = uv_next - uv
            Line_Ori_2D.append(prj_ori[None])
        Line_Ori_2D = torch.cat(Line_Ori_2D,0) ### view * N * 2

        return Line_Ori_2D


    def sample_next_3d_pos(self, points, base_view_index ,num_sample = 90):
        '''
        ### We use a alternative sample method to calculate the correct direction for each
        3D points. We can find the next 3D point for each point to obtain the growth direction, and the next point
        must on the ray reproject by next 2D pixel. We can only find the a suitable depth, then find this point.
        Also can use a optimization algorithm, but it will cost lots of time and have similar results.

        :param points: N,3
        :param base_view_index: N   the best init view for each point
        :return: sample_points: N * num_sample * 3
        '''
        sample1 = torch.arange(-0.005,-0.001,0.004/(num_sample/4),device = points.device)
        sample2 = torch.arange(-0.001,0.001,0.002/(num_sample/2),device = points.device)
        sample3 = torch.arange(0.001,0.005,0.004/(num_sample/4),device = points.device)
        sample = torch.cat([sample1,sample2,sample3],0)
        sample = sample[:num_sample]
        sample_points = torch.zeros_like(points) ### N,3
        sample_points = torch.unsqueeze(sample_points,dim=1)
        sample_points = torch.repeat_interleave(sample_points,num_sample,dim=1) ### N*180*3

        surface_points = points.clone()

        for i in range(len(self.camera)):

            index = torch.eq(base_view_index,i)  ### M
            if torch.sum(index) ==0:
                continue
            uv, z = self.camera[i].projection(points[index])

            uv_normalized = uv.clone()

            uv[:, 0:1] = - uv[:, 0:1]
            uv[:, :2] = (uv[:, :2] + 1) / 2
            uv[:, :2] *= torch.tensor(self.image_size[::-1], device=points.device, dtype=torch.float)


            ### trace next point
            next_pos_2d = uv + self.Ori[i][index][:,[1,0]]*2       ### M*2

            next_pos_2d /= torch.tensor(self.image_size[::-1], device=points.device, dtype=torch.float)
            next_pos_2d[:,:2] = next_pos_2d * 2 -1
            next_pos_2d[:,0:1] = - next_pos_2d[:,0:1]

            uv = torch.round(uv).type(torch.long)
            uv = uv[:,[1,0]]
            uv[:,0] = torch.clamp(uv[:,0],0,self.image_size[0]-1)
            uv[:,1] = torch.clamp(uv[:,1],0,self.image_size[1]-1)
            depth = self.get_depth(uv.type(torch.long), self.camera_key[i])
            unvisible_index = torch.gt((-z / 2 * 255) - depth, 0.1)
            # next_z = z
            depth = depth/255*2*-1   #### depth = -1* z/2 * 255



            next_z = z.clone()
            # next_z[unvisible_index] = depth[unvisible_index]
            next_z = torch.unsqueeze(next_z,dim=1)   ### M*1
            next_z = torch.repeat_interleave(next_z, num_sample, dim=1)   ### M*180
            sample_next_z = next_z + sample   ### M*180
            sample_next_z = torch.flatten(sample_next_z)
            next_pos_2d = torch.unsqueeze(next_pos_2d,dim=1) ###M*1*2
            next_pos_2d = torch.repeat_interleave(next_pos_2d,num_sample,dim=1) ### M*180*2
            next_pos_2d = torch.reshape(next_pos_2d,(-1,2))

            sample_world_v = self.camera[i].reprojection(next_pos_2d, sample_next_z,to_world=True)[:,:3]
            sample_world_v = torch.reshape(sample_world_v,(-1,num_sample,3))    #### M*180*3

            surface_p = self.camera[i].reprojection(uv_normalized, depth,to_world=True)[:,:3]

            sample_points[index] = sample_world_v
            surface_points[index][unvisible_index] = surface_p[unvisible_index]
            surface_points[index][~unvisible_index] = points[index][~unvisible_index]
        return sample_points,surface_points



    def Find_max_conf_from_visible_view(self):
        Conf = torch.where(self.visible<1,self.Conf*(torch.maximum(self.visible,torch.zeros_like(self.visible))),self.Conf)
        base_view_conf, base_view_index = torch.topk(Conf,20,dim=0,largest=True)

        return base_view_index,base_view_conf


    def Compute_Visible_and_Ori(self,points):
        visible = []
        Ori = []
        Conf = []
        mask = []
        Ori_patch = []
        Conf_patch = []
        for view, c in self.camera_dict.items():
            uv, z, unvisible_index = self.project_points(points,c, self.image_size)
            render_depth = self.get_depth(uv, view)
            o2d = self.get_ori(uv,view)
            c = self.get_conf(uv, view)
            o_patch = self.get_ori_patch(uv, view,self.patch_size)
            c_patch = self.get_c_patch(uv, view, self.patch_size)
            m = self.get_mask(uv,view)
            vb = self.compute_visible(render_depth, z*255.)
            vb[unvisible_index] = -1
            visible.append(vb[None])
            Ori.append(o2d[None])
            Conf.append(c[None])
            mask.append(m[None])
            Ori_patch.append(o_patch[None])
            Conf_patch.append(c_patch[None])
        self.visible = torch.cat(visible, 0) ### num_view,N
        self.Ori = torch.cat(Ori,0)    ### num_view,N,2
        self.Conf = torch.cat(Conf,0)   ### num_view,N
        self.Conf = torch.clamp(self.Conf, 1e-6, 1)
        self.mask = torch.cat(mask,0)
        self.Ori_patch = torch.cat(Ori_patch,0)       ### num_view,N,patch_size,2
        self.Conf_patch = torch.cat(Conf_patch,0)     ### num_view,N,patch_size
        self.Conf_patch = torch.clamp(self.Conf_patch,1e-6,1)

    def project_points(self, points,camera, image_size):
        uv,z = camera.projection(points)
        uv[:, 0:1] = - uv[:, 0:1]
        uv[:, :2] = (uv[:, :2] + 1) / 2
        uv[:, :2] *= torch.tensor(image_size[::-1], device=points.device, dtype=torch.float)
        uv = torch.round(uv).type(torch.long)
        unvisible_index1 = torch.gt(uv[:,0],image_size[1] - 1)
        unvisible_index2 = torch.lt(uv[:,0],0)
        unvisible_index3 = torch.gt(uv[:,1],image_size[0] - 1)
        unvisible_index4 = torch.lt(uv[:,1],0)

        uv[:, 0] = torch.clamp(uv[:, 0], 0, image_size[1] - 1)
        uv[:, 1] = torch.clamp(uv[:, 1], 0, image_size[0] - 1)

        z = -z/2
        unvisible_index = torch.logical_or(unvisible_index1,unvisible_index2)
        unvisible_index = torch.logical_or(unvisible_index,unvisible_index3)
        unvisible_index = torch.logical_or(unvisible_index,unvisible_index4)

        return uv[:,[1,0]],z, unvisible_index




    def filter_points(self,points):
        indexs = []

        unvisibles = []
        unvisibles1 = []
        mask = []
        count = 0
        for view,camera in self.camera_dict.items():
            count+=1
            uv, z,unvisible_index = self.project_points(points,camera, self.image_size)

            m = self.get_mask(uv, view)
            d = self.get_depth(uv, view)
            c = self.get_c_patch(uv, view,self.patch_size)
            # m[unvisible_index] = 0

            c = torch.max(c, dim=-1)[0]

            c[unvisible_index] = 0
            # bg = torch.where(m==0,torch.ones_like(m), torch.zeros_like(m))   ###  not hair mask
            unvisible = torch.where(z * 255 - d > 0.1,torch.ones_like(d), torch.zeros_like(d))
            unvisible[unvisible_index] = 1
            unvisible1 = torch.where(z * 255 - d > self.visible_threshold,torch.ones_like(d), torch.zeros_like(d))
            unvisible1[unvisible_index] = 1
            low_c = torch.where(c < self.conf_threshold, torch.ones_like(m), torch.zeros_like(m))   ### low confidence
            m[m>0.2]=1
            # index = (1-unvisible) * low_c*(1-m)    ### visible with low conf
            index = (1-unvisible) * low_c    ### visible with low conf
            indexs.append(index[None])
            unvisibles.append(unvisible[None])
            unvisibles1.append(unvisible1[None])
            mask.append(m[None])
        indexs =torch.cat(indexs,0)
        masks = torch.cat(mask,0)
        unvisibles = torch.cat(unvisibles,0)
        unvisibles1 = torch.cat(unvisibles1,0)
        visibles = 1 - unvisibles
        visibles1 = 1 - unvisibles1


        low_conf_indexs = torch.gt(torch.sum(indexs,0), 4)  ### visible point with high conf less than 2

        hair_index = torch.lt(torch.sum(visibles,dim=0)- torch.sum(visibles * masks,dim=0), torch.sum(visibles,dim=0)*1/2)  #### more than 1/2 visible views has mask=1
        hair_index1 = torch.lt(torch.sum(visibles1,dim=0)- torch.sum(visibles1 * masks,dim=0), torch.sum(visibles1,dim=0)*1/2)

        surface_index = torch.gt(torch.sum(visibles,dim=0),1)

        filter_index = torch.gt(torch.sum(visibles1,dim=0),1)
        filter_index = torch.logical_and(filter_index,~surface_index)  ### points with depth between 0.1 - threshold


        surface_index =  torch.logical_and(surface_index,torch.logical_and(~low_conf_indexs,hair_index))
        filter_index = torch.logical_and(filter_index,torch.logical_and(~low_conf_indexs,hair_index1))

        surface_points =points[surface_index]

        # return surface_index,points[surface_index],filter_index
        return surface_index,surface_points,filter_index

    def compute_unvisible_points(self,points):

        unvisibles = []
        count = 0
        for view, camera in self.camera_dict.items():
            count += 1
            uv, z, unvisible_index = self.project_points(points, camera, self.image_size)
            d = self.get_depth(uv, view)

            unvisible = torch.where(z * 255 - d > 0.9, torch.ones_like(d), torch.zeros_like(d))
            unvisible[unvisible_index] = 1
            unvisibles.append(unvisible[None])

        unvisibles = torch.cat(unvisibles, 0)
        visibles = 1 - unvisibles

        visible_indexs = torch.gt(torch.sum(visibles, dim=0), 2)  #### at least 4 visible view


        return ~visible_indexs

    def get_depth(self,uv,view):
        depth = self.depths_dict[view]

        return depth[uv[:,0],uv[:,1]][:,0]

    def get_ori(self,uv,view):
        Ori_2D = self.Ori_dict[view]
        return Ori_2D[uv[:,0],uv[:,1]]

    def get_ori_patch(self, uv ,view,size=1):
        Ori_2D = self.Ori_dict[view]
        ori_patch = []
        for i in range(-(size//2),size//2+1):
            for j in range(-(size//2), size//2+1):
                h = torch.clamp(uv[:,0]+i,0,Ori_2D.size(0)-1)
                w = torch.clamp(uv[:,1]+j,0,Ori_2D.size(1)-1)
                # o = Ori_2D[uv[:,0]+i,uv[:,1]+j]
                o = Ori_2D[h,w]
                ori_patch.append(o[...,None,:])
        ori_patch = torch.cat(ori_patch,-2)
        return ori_patch

    def get_c_patch(self, uv, view, size=1):
        Conf = self.Conf_dict[view]
        Conf_patch = []
        for i in range(-(size // 2), size // 2 + 1):
            for j in range(-(size // 2), size // 2 + 1):
                h = torch.clamp(uv[:, 0] + i, 0, Conf.size(0)-1)
                w = torch.clamp(uv[:, 1] + j, 0, Conf.size(1)-1)
                c = Conf[h,w]
                # c = Conf[uv[:, 0] + i, uv[:, 1] + j]
                Conf_patch.append(c[..., None])
        Conf_patch = torch.cat(Conf_patch, -1)
        return Conf_patch

    def get_conf(self,uv, view):
        Conf = self.Conf_dict[view]
        return Conf[uv[:,0],uv[:,1]]

    def get_mask(self,uv, view):
        mask = self.mask_dict[view]
        return mask[uv[:,0],uv[:,1]][:,0]

    def compute_visible(self, depth, z):
        visible = torch.zeros_like(depth)
        visible = torch.where(z - depth < 0.1, 1 - (z-depth)/0.1, torch.ones_like(visible) * -1)
        visible = torch.clamp(visible,-1,1)
        return visible





def filter_negative_points(points,pmvo,args,step=30):
    if points.shape[0] % step !=0:
        step = step+1
    num_sub_p = points.shape[0] // 30
    surface_indexs = []
    surface_points = []
    filter_indexs = []
    for i in range(step):
        sub_points = points[i * num_sub_p:(i + 1) * num_sub_p]
        surface_index,surface_p,filter_index = pmvo.filter_points(
            torch.from_numpy(sub_points).to(args.device).type(torch.float))
        surface_points.append(surface_p)
        filter_indexs.append(filter_index)
        surface_indexs.append(surface_index)
    surface_points = torch.cat(surface_points, 0)
    filter_indexs = torch.cat(filter_indexs,0)
    surface_indexs = torch.cat(surface_indexs,0)
    surface_points = surface_points.cpu().numpy()
    filter_indexs = filter_indexs.cpu().numpy()
    surface_indexs = surface_indexs.cpu().numpy()
    print('surface_num:', surface_points.shape[:])
    print('num filter_unvisible:',np.sum(filter_indexs))
    return surface_indexs,surface_points,filter_indexs







def optimize(points, pmvo, args):
    num_sub_p = 5000
    step = Num_points // num_sub_p + 1
    select_points = []
    select_ori = []
    min_loss = []
    high_conf_index =[]
    for i in trange(step):
        sub_points = points[i * num_sub_p:(i + 1) * num_sub_p]
        select_p, select_o, loss, hcIndex = pmvo.forward(sub_points)
        select_points.append(select_p)
        select_ori.append(select_o)
        min_loss.append(loss)
        high_conf_index.append(hcIndex)
    select_points = torch.cat(select_points, 0)
    select_ori = torch.cat(select_ori, 0)
    min_loss = torch.cat(min_loss, 0)
    high_conf_index = torch.cat(high_conf_index,0)

    select_points = select_points.cpu().numpy()
    select_ori = select_ori.cpu().numpy()
    min_loss = min_loss.cpu().numpy()
    high_conf_index = high_conf_index.cpu().numpy()

    os.makedirs(args.save_root, exist_ok=True)

    suffix = ''
    np.save(args.save_root + '/{}select_p.npy'.format(suffix), select_points)
    np.save(args.save_root + '/{}select_o.npy'.format(suffix), select_ori)
    np.save(args.save_root + '/{}min_loss.npy'.format(suffix), min_loss)
    np.save(args.save_root + '/{}high_conf_index.npy'.format(suffix), high_conf_index)






def refine(points,ori,loss,pmvo,filter_unvisible_points,args,infer_inner=True,threshold=0.001,genrate_ori_only = False):
    if not genrate_ori_only :
        print('filter nosiy points...')
        points_tree = KDTree(data=points)
        sub_num = 5000
        step = points.shape[0]//sub_num+1
        for i in trange(step):
            sub_points = points[i*sub_num:min((i+1)*sub_num,points.shape[0])]
            sub_ori = ori[i*sub_num:min((i+1)*sub_num,points.shape[0])]  ### sub_num,3
            sub_loss = loss[i*sub_num:min((i+1)*sub_num,points.shape[0])] ### sub_num
            _, index = points_tree.query(sub_points,100)

            Neighbor_ori = ori[index]

            # vis_p = vis_point_colud(Neighbor_points[0],(Neighbor_ori[0]+1)*0.5)
            # draw_scene([vis_p,vis_bust])

            # Neighbor_points = points[index]
            # Neighbor_points = torch.from_numpy(Neighbor_points).to(device)
            sub_points = torch.from_numpy(sub_points).to(device)
            sub_loss = torch.from_numpy(sub_loss).to(device)
            sub_ori = torch.from_numpy(sub_ori).to(device)
            Neighbor_ori = torch.from_numpy(Neighbor_ori).to(device)

            center_ori = compute_points_similarity(Neighbor_ori)


            update_loss = pmvo.refine(sub_points,center_ori)
            # index = torch.lt(update_loss,sub_loss)
            similar1 = torch.cosine_similarity(center_ori,sub_ori,dim=-1)
            similar2 = torch.cosine_similarity(center_ori,-sub_ori,dim=-1)
            similar = torch.maximum(similar1,similar2)
            index = torch.lt(similar,0.95)
            # sub_loss[index] = update_loss[index]
            sub_ori[index] = center_ori[index]
            sub_loss = update_loss
            # sub_ori = center_ori
            sub_loss[sub_loss==-1] = 0.5
            ori[i * sub_num:min((i + 1) * sub_num, points.shape[0])] = sub_ori.cpu().numpy()
            loss[i * sub_num:min((i + 1) * sub_num, points.shape[0])] = sub_loss.cpu().numpy()



        os.makedirs(args.output_path+'/refine', exist_ok=True)
        np.save(args.output_path + '/refine/select_p.npy', points)
        np.save(args.output_path + '/refine/select_o.npy', ori)
        np.save(args.output_path + '/refine/min_loss.npy', loss)




    points =  np.load(args.output_path + '/refine/select_p.npy')
    ori =  np.load(args.output_path + '/refine/select_o.npy')
    min_loss = np.load(args.output_path + '/refine/min_loss.npy')
    index = np.where(min_loss < threshold)[0]
    select_ori = ori[index]
    select_points = points[index]

    points_tree = KDTree(data=select_points)

    #### compute ori of unvisible points
    sub_num = 5000
    step = filter_unvisible_points.shape[0] // sub_num + 1
    filter_unvisible_ori = []
    select_filter_unvisible_points = []
    print('compute points orientation near the surface... ')
    for i in trange(step):

        sub_points = filter_unvisible_points[i * sub_num:min((i + 1) * sub_num, filter_unvisible_points.shape[0])]
        _, index = points_tree.query(sub_points, 100)
        sub_points = torch.from_numpy(sub_points).type(torch.float).to(device)
        pmvo.Compute_Visible_and_Ori(sub_points)
        filter_index = pmvo.filter_head_points(sub_points,args.PMVO.visible_threshold)

        Neighbor_ori = select_ori[index]
        Neighbor_ori = torch.from_numpy(Neighbor_ori).to(device)
        center_ori = compute_points_similarity(Neighbor_ori)
        filter_unvisible_ori.append(center_ori[~filter_index])
        select_filter_unvisible_points.append(sub_points[~filter_index])
    filter_unvisible_ori = torch.cat(filter_unvisible_ori,0)
    filter_unvisible_ori = filter_unvisible_ori.cpu().numpy()
    select_filter_unvisible_points = torch.cat(select_filter_unvisible_points,0)
    select_filter_unvisible_points = select_filter_unvisible_points.cpu().numpy()
    np.save(args.output_path + '/refine/filter_unvisible.npy', select_filter_unvisible_points)
    np.save(args.output_path + '/refine/filter_unvisible_ori.npy', filter_unvisible_ori)


    select_ori = np.concatenate([select_ori,filter_unvisible_ori],0)
    select_points = np.concatenate([select_points,select_filter_unvisible_points],0)




    grid_resolution = np.array([256, 256, 192]).astype(np.int32)
    occ = np.zeros(grid_resolution)
    ori = np.zeros((*(grid_resolution), 3))

    voxel_min = np.array([-0.32, -0.32, -0.24])
    voxel_size = 0.005 / 2

    up_index = select_ori[:, 1] > 0
    select_ori[up_index] *= -1
    total_x, total_y, total_z = p2v(select_points, voxel_min, voxel_size, grid_resolution)


    ori_dict = {}
    for x,y,z,dir in zip(total_x,total_y,total_z,select_ori):
        # occ[x, y, z] +=1
        # ori[x, y, z] += dir
        key = str(x)+'_'+str(y)+'_'+str(z)
        if ori_dict.__contains__(key):
            ori_dict[key].append(dir[None])
        else:
            ori_dict[key] = []
            ori_dict[key].append(dir[None])
    for key,value in ori_dict.items():
        key = key.split('_')
        x = int(key[0])
        y = int(key[1])
        z = int(key[2])
        occ[x,y,z] = 1
        value = np.concatenate(value,0)
        value = torch.from_numpy(value).type(torch.float).to(device)
        avg_ori = compute_points_similarity(value[None])[0]
        ori[x,y,z] = avg_ori.cpu().numpy()


    # ori = ori/np.maximum(occ[...,None],1e-6)
    # ori = ori/np.maximum(np.linalg.norm(ori,2,-1,keepdims=True),1e-6)
    # occ[occ>0] = 1

    if infer_inner:
        coarse_data = np.load(args.data.root + '/ours/raw.npy')
        coarse_points = coarse_data[:, :3]
        coarse_ori = coarse_data[:, 3:6]
        points = coarse_points.astype(np.float32)
        coarse_ori = coarse_ori.astype(np.float32)
        up_index = coarse_ori[:, 1] > 0
        coarse_ori[up_index] *= -1
        points = torch.from_numpy(points).to(args.device)
        unvisible_index = pmvo.compute_unvisible_points(points)
        points = points.cpu().numpy()
        unvisible_index = unvisible_index.cpu().numpy()
        un_visible_points = points[unvisible_index]
        unvisible_ori = coarse_ori[unvisible_index]
        x, y, z = p2v(un_visible_points.copy(), voxel_min, voxel_size, grid_resolution)
        occ[x, y, z] = 1
        ori[x, y, z] = unvisible_ori
        np.save(os.path.join(args.save_path,'coarse.npy'),un_visible_points)
        np.save(os.path.join(args.save_path,'coarse_ori.npy'),unvisible_ori)

    ori = ori.transpose((0, 1, 3, 2))
    ori = np.reshape(ori, [grid_resolution[0], grid_resolution[1], grid_resolution[2] * 3])
    ori = np.transpose(ori, (1, 0, 2))
    occ = np.transpose(occ, (1, 0, 2))

    path = args.save_path
    # if infer_inner:
    #     scipy.io.savemat(os.path.join(path, 'Ori3D_fused.mat'), {'Ori': ori})
    #     scipy.io.savemat(os.path.join(path, 'Occ3D_fused.mat'), {'Occ': occ})
    # else:
    scipy.io.savemat(os.path.join(path, 'Ori3D.mat'), {'Ori': ori})
    scipy.io.savemat(os.path.join(path, 'Occ3D.mat'), {'Occ': occ})


def config_parser():
    log.process(os.getpid())
    opt_cmd = options.parse_arguments(sys.argv[1:])
    args = options.set(opt_cmd=opt_cmd)
    args.output_path = os.path.join(args.data.root, args.data.case, args.output_root, args.name)
    os.makedirs(args.output_path, exist_ok=True)
    options.save_options_file(args)
    args.data.root = os.path.join(args.data.root, args.data.case)


    ### path
    args.bbox_min = np.array(args.bbox_min)
    args.bust_to_origin = np.array(args.bust_to_origin)

    args.data.strands_path = os.path.join(args.data.root, args.data.strands_path)
    args.data.bust_path = os.path.join(args.data.root, args.data.bust_path)


    args.image_camera_path = os.path.join(args.data.root, args.image_camera_path)
    args.data.raw_points_path = os.path.join(args.data.root, args.data.raw_points_path)
    args.data.depth_path = os.path.join(args.data.root, args.data.depth_path)
    args.data.Ori2D_path = os.path.join(args.data.root, args.data.Ori2D_path)
    args.data.Conf_path = os.path.join(args.data.root, args.data.Conf_path)
    args.data.mask_path = os.path.join(args.data.root, args.data.mask_path)
    # args.render_images_path = os.path.join(args.root, args.render_images_path)
    args.save_root = os.path.join(args.output_path,'optimize')
    # args.save_path = os.path.join(args.output_path,args.save_path)
    if args.PMVO.infer_inner and not args.PMVO.optimize:
        args.save_path = os.path.join(args.output_path, 'full')
    else:
        args.save_path = os.path.join(args.output_path, 'refine')
    os.makedirs(args.save_path,exist_ok=True)

    return args




if __name__ == '__main__':
    print('Run PMVO...')
    args = config_parser()

    device = args.device
    ### load bust
    vertices, faces,normals = load_bust(args.data.bust_path)
    vertices += args.bust_to_origin

    bust_tree = KDTree(data=vertices)
    scalp = o3d.io.read_triangle_mesh(os.path.join(args.data.root,'ours/scalp_tsfm.obj'))
    scalp_vertices = np.asarray(scalp.vertices)
    scalp_vertices +=args.bust_to_origin
    scalp_tree = KDTree(data=scalp_vertices)
    scalp_mean = np.mean(scalp_vertices,axis=0)
    scalp_max = np.max(scalp_vertices,axis=0)


    ### load camera
    camera = load_cam(args.image_camera_path)
    image_path = os.path.join(args.data.root, 'capture_images')
    camera = parsing_camera(camera,image_path)
    print('num of view:',len(camera))

    ### load depth
    depths = load_depth(camera, args.data.depth_path)

    ### Load Conf,Ori,mask
    Ori, Conf = Load_Ori_And_Conf(camera, args.data.Ori2D_path, args.data.Conf_path)
    masks = load_mask(camera,args.data.mask_path)

    ### initalize PMVO
    pmvo = PMVO(camera,depths, Ori, Conf,masks,device=device, image_size=args.data.image_size, patch_size=args.PMVO.patch_size,visible_threshold = args.PMVO.visible_threshold,conf_threshold=args.PMVO.conf_threshold)



    if args.PMVO.optimize:
        print('load raw mesh...')
        #### load colmap points
        colmap_points = load_colmap_points(args.data.raw_points_path, args.bbox_min, args.bust_to_origin, 0.005 / 4,
                                           [512, 512, 384], True, args.PMVO.num_sample_per_grid)
        ### filter negative points
        points = colmap_points
        raw_points = points.copy()
        print('total points:', points.shape[0])
        print('filter low conf points...')
        if args.PMVO.filter_point:
            surface_index, surface_points, filter_index = filter_negative_points(points, pmvo, args)
            points = surface_points
            # index = np.where(np.logical_and(surface_indexs, sample_occ.astype(np.bool_)), True, False)
            # index = ~filter_index
            os.makedirs(args.save_root, exist_ok=True)
            np.save(os.path.join(args.save_root, 'surface.npy'), raw_points[surface_index])
            np.save(os.path.join(args.save_root, 'filter_unvisible.npy'), raw_points[filter_index])

            del surface_points, surface_index, raw_points, filter_index, colmap_points

        Num_points = points.shape[0]
        print('process points:', Num_points)
        optimize(points, pmvo,args)
        select_points = np.load(args.save_root + '/select_p.npy')
        select_ori = np.load(args.save_root + '/select_o.npy')
        min_loss = np.load(args.save_root + '/min_loss.npy')
        filter_unvisible_points = np.load(args.save_root + '/filter_unvisible.npy')


        refine(select_points, select_ori, min_loss, pmvo, filter_unvisible_points, args, infer_inner=False,
               threshold=args.PMVO.threshold, genrate_ori_only=False)

    else:
        select_points = np.load(args.save_root + '/select_p.npy')
        select_ori = np.load(args.save_root + '/select_o.npy')
        min_loss = np.load(args.save_root + '/min_loss.npy')
        filter_unvisible_points = np.load(args.save_root + '/filter_unvisible.npy')

        refine(select_points,select_ori,min_loss,pmvo,filter_unvisible_points,args,infer_inner=args.PMVO.infer_inner,threshold=args.PMVO.threshold, genrate_ori_only=True)

















