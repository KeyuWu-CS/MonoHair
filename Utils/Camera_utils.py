import os
import cv2
import json
import torch
import numpy as np




class Camera():
    def __init__(self, proj, pose, id, to_tensor=True):
        self.proj = self.get_projection_matrix(*proj)
        self.pose = pose
        self.id = id
        if to_tensor:
            self.proj = torch.from_numpy(self.proj).type(torch.float)
            self.pose = torch.from_numpy(self.pose).type(torch.float)

    def get_projection_matrix(self, fx, fy, cx, cy):
        zfar = 100
        znear = 0.1
        mat = np.array([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, (-zfar - znear) / (zfar - znear), -2. * zfar * znear / (zfar - znear)],
            [0, 0, -1, 0]
        ]
        )
        # mat = np.array([
        #     [fx, 0, cx, 0],
        #     [0, fy, cy, 0],
        #     [0, 0, 1,0]
        # ]
        # )

        return mat

    def projection(self, vertices,debug=False):
        '''

        :param v: N*3
        :param prj_mat: 4*4
        :param poses: 4*4
        :return:
        '''
        self.proj = self.proj.to(vertices.device)
        self.pose = self.pose.to(vertices.device)
        vertices = vertices.permute(1, 0)
        vertices = torch.cat([vertices, torch.ones((1, vertices.size(1)), device=vertices.device)])
        camera_v = torch.matmul(self.pose, vertices)
        z = camera_v[2:3, :]
        uv = torch.matmul(self.proj, camera_v)
        uv[:2] /= z
        uv = uv.transpose(1, 0)



        return uv[:, :2], z[0]

    def uv2pixel(self,uv, image_size,device):
        '''
        :param uv: [-1,1]
        :param image_size:
        :param device:
        :return: [0,image_size]
        '''
        uv[:, 0:1] = uv[:, 0:1] * -1
        uv[:, :2] = (uv[:, :2] + 1) / 2
        uv[:, :2] *= torch.tensor(image_size[::-1], device=device, dtype=torch.float)
        uv = torch.flip(uv,dims=[1])
        return uv

    def pixel2uv(self,uv,image_size,device):
        uv = uv[:,[1,0]]
        uv /= torch.tensor(image_size[::-1], device=device, dtype=torch.float) ### 0,1
        uv[:, :2] = uv * 2 - 1   #### -1,1
        uv[:, 0:1] = - uv[:, 0:1]
        return uv


    def reprojection(self, uv, z, to_world = False):
        zfar = 100
        znear = 0.1
        m = (-zfar - znear) / (zfar - znear)
        n = -2. * zfar * znear / (zfar - znear)
        Homogeneous = torch.ones_like(uv)
        Homogeneous[:,0] *= z*m+n

        Homogeneous[:,1] *= -z
        Homogeneous_uv = torch.cat([uv * z[:,None], Homogeneous], 1)

        Homogeneous_uv = Homogeneous_uv.permute(1, 0)
        # camera_v = torch.matmul(torch.linalg.inv(self.proj), Homogeneous_uv)
        camera_v = Homogeneous_uv
        camera_v[0] = (uv[:,0] - self.proj[0,2])/self.proj[0,0] * z
        camera_v[1] = (uv[:,1] - self.proj[1,2])/self.proj[1,1] * z
        camera_v[2] = z
        camera_v[3] = 1

        if to_world:
            # print('pose:',self.pose)
            # print('pose_inv:',torch.linalg.inv(self.pose))
            # world_v = torch.matmul(torch.linalg.inv(self.pose), camera_v)
            world_v = torch.matmul(torch.linalg.inv(self.pose[:3,:3]),camera_v[:3]-self.pose[:3,3:4])
            world_v = world_v.permute(1,0)
            return world_v
        else:
            camera_v = camera_v.permute(1,0)
            return camera_v

    def camera2world(self,points):
        points = torch.cat([points,torch.ones((points.size(0),1),device = points.device)],1)
        points = points.permute(1,0)
        world_v = torch.matmul(torch.linalg.inv(self.pose),points)
        world_v = world_v.permute(1,0)
        return world_v




    def render_img(self, vertices, image_size, device,save_path, color = None):
        img = torch.ones((image_size[0], image_size[1], 3), device=device)
        uv, z = self.projection(vertices)
        uv[:, 0:1] = - uv[:, 0:1]
        uv[:, :2] = (uv[:, :2] + 1) / 2
        uv[:, :2] *= torch.tensor(image_size[::-1], device=device, dtype=torch.float)
        uv = torch.round(uv).type(torch.long)
        uv[:, 0] = torch.clamp(uv[:, 0], 0, image_size[1] - 1)
        uv[:, 1] = torch.clamp(uv[:, 1], 0, image_size[0] - 1)
        if color is not None:
            img[uv[:, 1], uv[:, 0]] *= color
        else:
            img[uv[:, 1], uv[:, 0]] *= -z / 2
        img = img.cpu().numpy()
        cv2.imwrite(save_path, img * 255)





def load_cam(path):
    with open(path, 'r')as f:
        cam = json.load(f)
    f.close()
    cam = cam['cam_list']
    return cam

def parsing_camera(cam,image_path=None):
    step=1
    if image_path is not None:
        files = os.listdir(image_path)
        if len(files)>500:
            step=4
        elif len(files)>300:
            step = 2

    camera = {}
    for c in cam[::step]:
        if image_path is None:
            camera[c['file']] = Camera(c['ndc_prj'], np.linalg.inv(np.array(c['pose'])), c['file'])
        elif c['file']+'.png' in files or c['file']+'.JPG' in files or c['file']+'.jpg':
            camera[c['file']] = Camera(c['ndc_prj'], np.linalg.inv(np.array(c['pose'])), c['file'])
    return camera