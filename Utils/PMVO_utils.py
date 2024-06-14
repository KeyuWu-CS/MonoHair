import os
import cv2
import torch
import numpy as np
import trimesh
import struct
import scipy
import scipy.io
from scipy.interpolate import splprep, splev
import torch.nn.functional as F
import math
import open3d as o3d
from scipy.interpolate import CubicHermiteSpline


def save_mesh(vertices,faces=None,path='test.obj'):
    mesh = trimesh.Trimesh(vertices=vertices, )
    trimesh.exchange.export.export_mesh(mesh, path)



def read_hair_data(file_path):
    '''
    read strands from a binary file, following our defined format
    :param filePath:
    :return: a list of Nx3 numpy arrays, each array represents the ordered vertices of a strand
    '''
    raw_data = np.fromfile(file_path, dtype='float32')
    # number of strands
    s_cnt = int(raw_data[0])
    # each strand's begin index in 'vertices_data'
    s_begin_indices = raw_data[1:s_cnt + 1]
    # number of total vertices (sum up all strands)
    total_vcnt = int(raw_data[s_cnt + 1])
    # (N * 3)
    vertices_data = raw_data[s_cnt + 2:]
    # a list of strands
    strands = []
    for i in range(s_cnt):
        begin_index = int(s_begin_indices[i])
        end_index = total_vcnt if i == (s_cnt - 1) else int(s_begin_indices[i + 1])
        strands.append(vertices_data[(begin_index * 3):(end_index * 3)].reshape(-1, 3))

    return strands.copy(),vertices_data.copy()


def load_strand(file):
    with open(file, mode='rb')as f:
        num_strand = f.read(4)
        (num_strand,) = struct.unpack('I', num_strand)
        point_count = f.read(4)
        (point_count,) = struct.unpack('I', point_count)

        segments = f.read(2 * num_strand)
        segments = struct.unpack('H' * num_strand, segments)
        segments = list(segments)
        num_points = sum(segments)

        points = f.read(4 * num_points * 3)
        points = struct.unpack('f' * num_points * 3, points)
    f.close()
    points = list(points)
    points = np.array(points)
    points = np.reshape(points, (-1, 3))

    return segments, points


def write_strand(points, path, segments):
    hair_count = len(segments)
    point_count = sum(segments)
    with open(path, 'wb')as f:
        f.write(struct.pack('I', hair_count))
        f.write(struct.pack('I', point_count))
        for num_every_strand in segments:
            f.write(struct.pack('H', num_every_strand))

        for vec in points:
            f.write(struct.pack('f', vec[0]))
            f.write(struct.pack('f', vec[1]))
            f.write(struct.pack('f', vec[2]))

    f.close()


def get_ground_truth_3D_occ(d, flip=False):
    occ = scipy.io.loadmat(d, verify_compressed_data_integrity=False)['Occ'].astype(np.float32)
    occ = np.transpose(occ, [2, 0, 1])
    occ = np.expand_dims(occ, -1)  # D * H * W * 1

    if flip:
        occ = occ[:, :, ::-1, :]

    occ = np.ascontiguousarray(occ)
    return occ


def get_ground_truth_3D_ori(d, flip=False, growInv=False):
    transfer = False

    ori = scipy.io.loadmat(d, verify_compressed_data_integrity=False)['Ori'].astype(np.float32)

    ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, -1]) ###128,128,3,96
    ori = ori.transpose([0, 1, 3, 2]).transpose(2, 0, 1, 3)  #128,128,96,3    96,128,128,3

    if flip:
        ori = ori[:, :, ::-1, :] * np.array([-1.0, 1.0, 1.0])

    ori = np.ascontiguousarray(ori)
    if transfer:
        return ori * np.array([1, -1, -1])  # scaled
    else:
        return ori


def B_spline_interpolate(X, num):
    # print(X.shape[:])
    tck, u = splprep([X[:, 0], X[:, 1], X[:, 2]], s=0., k=3)
    U = np.linspace(0, 1, num)
    new_points = splev(U, tck)
    return new_points


def interpolation(segments, points):
    new_points = []

    new_segments = []
    begin = 0
    for strand, num in enumerate(segments):
        X = np.zeros((num, 3))
        for i in range(num):
            vec = points[begin + i]
            X[i] = vec.copy()
        #### num 代表原strand的点的数量，*20 扩大20倍 再每隔4个点采样一个点， 新strand点的数量是原strand的5倍。
        #### 若要生成等距的训练数据控制新strand的数量与原strand数量相等即可，  即num*4
        begin = begin + (num)
        if num > 5:
            # inter_points = B_spline_interpolate(X,num*4)
            inter_points = B_spline_interpolate(X, 100)
            inter_points = [[inter_points[0][i], inter_points[1][i], inter_points[2][i]] for i in
                            range(0, len(inter_points[0]))]

            new_points.append(inter_points)
            new_segments.append(len(inter_points))
    return new_segments, new_points


def Interpolatehair(segments, points, path):
    # segments, points = load_strand(path+'/hair.hair', False)

    new_segments, new_points = interpolation(segments, points)
    new_points = np.array(new_points)
    new_points = np.reshape(new_points, (-1, 3))
    save_path = os.path.join(path, 'uniform_hair.hair')
    write_strand(new_points.copy(), save_path, new_segments)

    return new_segments, new_points


def load_and_interpolate_strands(root):
    strands_path = os.path.join(root, 'result.hair')

    segments, points = load_strand(strands_path)
    mesh = trimesh.Trimesh(vertices=points, )
    trimesh.exchange.export.export_mesh(mesh, root + '/strands.obj',
                                        include_texture=False)
    segments, points = Interpolatehair(segments, points, root)

    return segments, points






def load_bust(path):
    bust = trimesh.load(path)
    vertices = np.array(bust.vertices)
    faces = np.array(bust.faces)
    normals = np.array(bust.vertex_normals)
    return vertices, faces,normals


def DenseSampleFromTriangle(vertices, faces, mul=1):
    points = [vertices]
    # points=[]

    for f in faces:
        v1 = vertices[f[0]]
        v2 = vertices[f[1]]
        v3 = vertices[f[2]]
        edge12 = np.sqrt(np.sum((v2 - v1) ** 2))
        edge13 = np.sqrt(np.sum((v3 - v1) ** 2))
        edge23 = np.sqrt(np.sum((v3 - v2) ** 2))
        # print(v1.shape[:])

        NumSample = int(max(edge12, edge23, edge13) / 0.001 * mul)

        # if definN is not None:
        #     NumSample=definN
        uI = np.linspace(0, 1., NumSample)
        vI = np.linspace(0, 1., NumSample)
        # print(vI.shape[:])
        # print('NumSample:',NumSample)
        for u in uI:
            uvI = u + vI
            uvI = uvI[uvI < 1.]
            uvI = uvI[:, None]
            wI = 1. - uvI
            SamplePoints = v1 * u + v2 * (uvI - u) + wI * v3
            points.append(SamplePoints)
    points = np.concatenate(points, 0)

    return points


def SamplePointsAroundVolume(occ, kernel=3, close_volume=True, erosion = False,return_numpy=False,):

    if close_volume:
        enlarge_occ = F.max_pool3d(occ, kernel, 1, kernel//2)
        close_occ = F.avg_pool3d(enlarge_occ,kernel,1,kernel//2)
        close_occ[close_occ<1] = 0
        occ = close_occ

    enlarge_occ = F.max_pool3d(occ, kernel, 1, kernel//2)
    sample_occ = enlarge_occ - occ

    if erosion:
        erosion_occ = F.avg_pool3d(occ,3,1,1)
        erosion_occ[erosion_occ<1]=0
        erosion_occ = occ - erosion_occ
        sample_occ = sample_occ + erosion_occ
    indices = torch.nonzero(sample_occ,as_tuple=False)[:,1:]   ### zyx
    indices = torch.flip(indices,dims=[1])  ### xyz
    samples = randSampleFromGrid(indices,8)
    if return_numpy:
        samples = samples.cpu().numpy()
    return samples



def randSampleFromGrid(indices, sample_per_grid):
    '''
    :param indices: tensors, [N, 3]
    :param sample_per_grid
    :return: [N * sample_per_grid, 3]
    '''

    base = torch.cat([indices]*sample_per_grid, dim=0) if sample_per_grid>0 else base
    random_offset = torch.rand(base.shape,device= base.device)
    samples = base +random_offset
    return samples


def Load_Ori_And_Conf(camera, Ori_path, Conf_path):
    Ori = {}
    Conf = {}
    suffix = '.JPG'

    for view,_ in camera.items():
        if not os.path.exists(os.path.join(Ori_path, view + '.JPG')):
            suffix = '.png'
        if not os.path.exists(os.path.join(Ori_path, view + '.png')):
            suffix = '.jpg'
        o = cv2.imread(os.path.join(Ori_path, view + suffix), cv2.IMREAD_GRAYSCALE)
        o = (180-o)/180 * math.pi
        # o = np.stack([(np.cos(o) + 1) * 0.5, (-np.sin(o) + 1) * 0.5, np.zeros_like(o)], -1)
        # cv2.imwrite('test.png', o[...,::-1] * 255.)
        # o = np.stack([np.cos(o), -np.sin(o)], -1) #### used to visualize
        o = np.stack([np.sin(o), np.cos(o)], -1)

        c = cv2.imread(os.path.join(Conf_path, view + suffix), cv2.IMREAD_GRAYSCALE) / 255.
        Ori[view] = o
        Conf[view] = c

    return Ori, Conf

def load_depth(camera,path,type='npy'):
    # files = os.listdir(path)
    # files.sort()
    depths = {}
    # for file in files:
    #     if type not in file:
    #         continue
    #     if type == 'img':
    #         depth = cv2.imread(os.path.join(path,file))
    #     else:
    #         depth = np.load(os.path.join(path,file[:-4]+'.npy')).astype(np.float32)
    #     depths[file[:-4]] = depth
    for view, _ in camera.items():

        depth =  np.load(os.path.join(path,view+'.npy')).astype(np.float32)
        depths[view] = depth

    return depths

def load_mask(camera,path):
    files = os.listdir(path)
    suffix = files[0][-4:]
    masks = {}
    for view, _ in camera.items():
        mask = cv2.imread(os.path.join(path, view+suffix))
        mask[mask < 50] = 0
        masks[view] = mask / 255.

    # files.sort()
    # masks = {}
    # for file in files:
    #
    #     mask = cv2.imread(os.path.join(path, file))
    #     mask[mask<50] = 0
    #     masks[file[:-4]] = mask/255.
    return masks


def SamplePointsAroundmesh(colmap_points,bbox_min,vsize,num_per_grid = 32,grid_resolution = [512,512,384]):
    occ = np.zeros(grid_resolution)
    colmap_points[:,1:]*=-1
    indexs = (colmap_points - bbox_min)/vsize
    indexs = np.round(indexs)
    indexs = indexs.astype(np.int32)
    x, y, z = np.split(indexs, 3, -1)

    x = np.clip(x, 0, grid_resolution[0] - 1)
    y = np.clip(y, 0, grid_resolution[1] - 1)
    z = np.clip(z, 0, grid_resolution[2] - 1)

    x = np.squeeze(x)
    y = np.squeeze(y)
    z = np.squeeze(z)
    occ[x, y, z] = 1
    x,y,z = np.nonzero(occ)
    indices = np.concatenate([x[:,None],y[:,None],z[:,None]],1)
    base = np.concatenate([indices]*num_per_grid,0)
    random_offset = np.random.random(base.shape[:])*1
    sample = base + random_offset
    sample = sample*vsize + bbox_min
    sample[:,1:]*=-1
    return sample

def load_colmap_points(path, bbox_min ,bust_to_origin,vsize=0.005,grid_resolution=[128,128,96],sample=True,num_per_grid=8):
    mesh = o3d.io.read_triangle_mesh(path)  # 加载mesh
    # mesh = trimesh.load(path) # 加载mesh
    num_p = np.asarray(mesh.vertices).shape[0]
    print('num_p:',num_p)
    pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=num_p*5,
                                                            use_triangle_normal=False)  # 采样点云
    colmap_points = np.asarray(pcd.points)

    # colmap_points = np.asarray(mesh.vertices)
    # mesh = trimesh.load(path)
    # colmap_points = np.array(mesh.vertices)
    # colmap_faces = np.array(mesh.faces)


    colmap_points += bust_to_origin
    if sample:
        sample_points = SamplePointsAroundmesh(colmap_points.copy(), bbox_min, vsize, num_per_grid=num_per_grid,grid_resolution=grid_resolution)
        print('num sample:', sample_points.shape[:])
        return sample_points
    else:
        return colmap_points



def compute_points_similarity(ori):
    '''

    :param ori: N,K,3
    :return:
    '''
    N,K,_ = ori.size()
    ori_copy = ori.clone()
    ori = torch.unsqueeze(ori,dim=2)
    ori = torch.repeat_interleave(ori,K,dim=2) ### N,K,K,3
    similar1 = torch.cosine_similarity(ori,ori.permute(0,2,1,3),dim=-1) ### N,K,K
    similar2 = torch.cosine_similarity(-ori,ori.permute(0,2,1,3),dim=-1) ### N,K,K
    similar = torch.maximum(similar1,similar2)
    similar = torch.mean(similar,dim=-1)
    max_index = torch.argmax(similar,dim=-1) # N
    B = torch.arange(N).type(torch.long)
    return ori_copy[B,max_index]



def p2v(points,voxel_min,voxel_size,grid_resolution):
    points[:, 1:] *= -1

    indexs = (points - voxel_min) / voxel_size

    indexs = np.round(indexs)

    indexs = indexs.astype(np.int32)
    x, y, z = np.split(indexs, 3, -1)

    x = np.clip(x, 0, grid_resolution[0] - 1)
    y = np.clip(y, 0, grid_resolution[1] - 1)
    z = np.clip(z, 0, grid_resolution[2] - 1)
    # z = (grid_resolution[2]-z-1).astype(np.int32)
    # y = (grid_resolution[1]-y-1).astype(np.int32)
    x = np.squeeze(x)
    y = np.squeeze(y)
    z = np.squeeze(z)
    return x,y,z


def voxel_to_points(voxels):
    voxel_size = 0.005 / 2
    # voxel_min = np.array([-0.32, -0.32, -0.24])
    voxel_min = torch.tensor([-0.32, -0.32, -0.24],dtype=torch.float,device=voxels.device)
    points = voxels * voxel_size+voxel_min
    points[...,1:]*=-1
    return points

def points_to_voxel(points):
    voxel_min = torch.tensor([-0.32, -0.32, -0.24],dtype=torch.float,device=points.device)
    voxel_size = 0.005 / 2
    points[...,1:] *= -1
    indexs = (points - voxel_min)/voxel_size
    # indexs = torch.round(indexs)
    return indexs


def clear_scalp_ori(points,normals,ori,occ):
    voxel_size = 0.005 / 2

    for i, (point, normal) in enumerate(zip(points, normals)):
        if i % 500 == 0:
            print(i)
        point_copy = point.clone()
        normal_copy = normal.clone()

        idx = points_to_voxel(point_copy.clone()).type(torch.long)
        conf = occ[0, idx[..., 2], idx[..., 1], idx[..., 0]]
        if conf!=0:
            count = 0
            while count<3:
                count+=1
                point_copy = point_copy + normal_copy * voxel_size
                idx = points_to_voxel(point_copy.clone()).type(torch.long)
                if occ[0, idx[..., 2], idx[..., 1], idx[..., 0]]==0:
                    break
                else:
                    occ[0, idx[..., 2], idx[..., 1], idx[..., 0]]=0
                    ori[:, idx[..., 2], idx[..., 1], idx[..., 0]]=0
            count = 0
            point_copy = point.clone()
            while count<10:
                count+=1
                point_copy = point_copy - normal_copy * voxel_size
                idx = points_to_voxel(point_copy.clone()).type(torch.long)
                if occ[0, idx[..., 2], idx[..., 1], idx[..., 0]] == 0:
                    break
                else:
                    occ[0, idx[..., 2], idx[..., 1], idx[..., 0]] = 0
                    ori[:, idx[..., 2], idx[..., 1], idx[..., 0]] = 0
            point_copy = point.clone()
            idx = points_to_voxel(point_copy.clone()).type(torch.long)
            occ[0, idx[..., 2], idx[..., 1], idx[..., 0]] = 0
            ori[:, idx[..., 2], idx[..., 1], idx[..., 0]] = 0


    return ori, occ



def diffusion_scalp(points, normals,ori,occ):
    idxs = points_to_voxel(points.clone()).type(torch.long)
    conf = occ[0,idxs[:,2],idxs[...,1],idxs[...,0]]
    print(torch.sum(conf))

    # ori,occ = clear_scalp_ori(points,normals,ori,occ)


    voxel_size = 0.005/2

    threshold = 0.5
    total_sample = []
    total_normal = []
    count1=0

    idxs = points_to_voxel(points.clone()).type(torch.long)
    conf = occ[0,idxs[:,2],idxs[...,1],idxs[...,0]]
    print(torch.sum(conf))
    diffusion_ori = torch.zeros_like(ori)
    diffusion_occ = torch.zeros_like(occ)

    trace_step = 10
    for i,(point,normal) in enumerate(zip(points,normals)):


        if i%500==0:
            print(i)
        point_copy = point.clone()
        normal_copy = normal.clone()
        step = 0
        normal_bias = torch.zeros_like(normal_copy)
        point_set = []
        normal_set = []
        fail_count = 0
        while True:
            if fail_count>8:
                break
            point_set.append(point_copy)
            idx = points_to_voxel(point_copy.clone()).type(torch.long)

            conf = occ[0,idx[...,2],idx[...,1],idx[...,0]]
            if conf==0 and step<trace_step:
                normal_copy = 0.8*normal_copy + 0.2 * normal_bias
                normal_copy = normal_copy/torch.linalg.norm(normal_copy,2,dim=-1)
                normal_set.append(normal_copy)
                point_copy = point_copy + normal_copy * voxel_size
                step+=1
            else:
                if step==0:
                    break
                if step>=trace_step:
                    count1+=1
                    break
                grow_dir = ori[:,idx[...,2],idx[...,1],idx[...,0]]
                if torch.cosine_similarity(grow_dir,normal_copy,dim=-1)>threshold:
                    normal_set.append(grow_dir)
                    break
                elif torch.cosine_similarity(-grow_dir,normal_copy,dim=-1)>threshold:
                    normal_set.append(-grow_dir)
                    break
                else:
                    point_copy = point.clone()
                    if torch.cosine_similarity(grow_dir,normal_copy,dim=-1)<0:
                        normal_bias = -grow_dir
                    else:
                        normal_bias = grow_dir
                    step = 0
                    point_set = []
                    normal_set = []
                    fail_count+=1


        if step!=0 and step!=trace_step:
            point_set = torch.row_stack(point_set)
            normal_set = torch.row_stack(normal_set)
            point_set = point_set.cpu().numpy()
            normal_set = normal_set.cpu().numpy()

            spline_evaluator = CubicHermiteSpline(np.linspace(0, 1, num=2), [point_set[0],point_set[-1]], [normal_set[0]*voxel_size*step,normal_set[-1]*voxel_size*step])
            u = np.linspace(0, 1, num=len(point_set), endpoint=True)
            sample = spline_evaluator(u)
            sample_normal = np.concatenate([sample[1:] - sample[:-1],sample[-1:]-sample[-2:-1]],0)
            total_sample.append(sample)
            total_normal.append(sample_normal)
    total_normal = np.concatenate(total_normal,0)
    total_sample = np.concatenate(total_sample,0)
    np.save('total_normal.npy',total_normal)
    np.save('total_sample.npy',total_sample)
    # vis_line = vis_normals(total_sample,total_normal)
    # vis_pc = vis_point_colud(points.cpu().numpy())
    # draw_scene([vis_scalp,vis_line])
    total_sample = torch.from_numpy(total_sample).to(ori.device)
    total_normal = torch.from_numpy(total_normal).to(ori.device)
    total_normal = total_normal/torch.linalg.norm(total_normal,2,-1,keepdim=True)
    idxs = points_to_voxel(total_sample).type(torch.long)

    for i,(idx,n) in enumerate(zip(idxs,total_normal)):
        if i == 0:
            print('idx:',idx)
        diffusion_ori[:,idx[2],idx[1],idx[0]] += n
        diffusion_occ[:,idx[2],idx[1],idx[0]] +=1
    diffusion_ori = diffusion_ori/torch.maximum(diffusion_occ,torch.ones_like(diffusion_occ)*1e-6)
    diffusion_occ[diffusion_occ>0]=1

    # diffusion_occ = diffusion_occ[None]
    # diffusion_ori = diffusion_ori[None]
    # enlarge_occ = F.max_pool3d(diffusion_occ, 7, 1, 7 // 2)
    # enlarge_ori = F.avg_pool3d(diffusion_ori, 7, 1, 7 // 2)
    # close_occ = F.avg_pool3d(enlarge_occ, 7, 1, 7 // 2)
    # enlarge_ori = enlarge_ori/torch.maximum(close_occ,torch.ones_like(close_occ)*1e-6)
    #
    # close_occ[close_occ < 1] = 0
    # diffusion_occ = close_occ
    # diffusion_ori = enlarge_ori*diffusion_occ
    # diffusion_ori = diffusion_ori[0]
    # diffusion_occ = diffusion_occ[0]



    #ori = ori*(1-diffusion_occ) + diffusion_ori
    #occ = occ*(1-diffusion_occ) + diffusion_occ
    #ori = ori/torch.linalg.norm(ori,2,0,keepdim=True)

    ori = ori+ (1-occ)*diffusion_ori
    # print(ori.size())
    occ = occ + (1-occ)*diffusion_occ
    return ori,occ


def compute_strands_confidence(strand, occ,ori):
    ss_tensor = torch.from_numpy(strand).to(occ.device)
    strand_ori = torch.cat([ss_tensor[1:]-ss_tensor[:-1],ss_tensor[-1:]-ss_tensor[-2:-1]],0)
    strand_ori[:, 1:] *= -1
    idx = torch.round(ss_tensor).type(torch.long)

    idx[:,0] = torch.clamp(idx[:,0],0,255)
    idx[:,1] = torch.clamp(idx[:,1],0,255)
    idx[:,2] = torch.clamp(idx[:,2],0,192)

    ss_occ = occ[0, idx[:, 2], idx[:, 1], idx[:, 0]]
    ss_ori = ori[:, idx[:, 2], idx[:, 1], idx[:, 0]]

    similar1 = torch.cosine_similarity(ss_ori.permute(1, 0), strand_ori, dim=-1)
    similar2 = torch.cosine_similarity(-ss_ori.permute(1, 0), strand_ori, dim=-1)
    similar = torch.maximum(similar2, similar1)
    similar = torch.sum(similar) / torch.sum(ss_occ)

    out_ratio = 1 - (torch.sum(ss_occ) / ss_occ.size(0))
    confidence = torch.sum(ss_occ) / ss_occ.size(0)
    return confidence,similar

def random_move_strands(original_strand,occ,ori,threshold=0.4, index=-1):
    count = 0

    strand = original_strand.copy()
    while True:
        out_ratio = 0
        ss = strand.copy()[:index]
        ss_tensor = torch.from_numpy(ss).to(occ.device)
        strand_ori = torch.cat([ss_tensor[1:]-ss_tensor[:-1],ss_tensor[-1:]-ss_tensor[-2:-1]],0)
        # strand_ori[:,1:]*=-1
        # idx = points_to_voxel(ss_tensor)
        idx = torch.round(ss_tensor).type(torch.long)
        if torch.max(idx[:, 2]) >= 192 or torch.max(idx[:, 1] >= 256) or torch.max(idx[:, 0] >= 256):
            check = False
            break

        ss_occ = occ[0, idx[:, 2], idx[:, 1], idx[:, 0]]
        ss_ori = ori[:, idx[:, 2], idx[:, 1], idx[:, 0]]

        similar1 = torch.cosine_similarity(ss_ori.permute(1,0),strand_ori,dim=-1)
        similar2 = torch.cosine_similarity(-ss_ori.permute(1,0),strand_ori,dim=-1)
        similar = torch.maximum(similar2,similar1)
        similar = torch.sum(similar)/torch.sum(ss_occ)

        out_ratio = 1-(torch.sum(ss_occ) / ss_occ.size(0))
        # print('out:',out_ratio)
        # print('index:',index)
        if torch.sum(ss_occ) / ss_occ.size(0) > threshold and similar>0.3:
        # if torch.sum(ss_occ) / ss_occ.size(0) > threshold and similar>0.:
            check =True
            break
        # strand = original_strand.copy()
        # strand += np.random.random((3)) * 0.5
        count += 1
        if count >= 1:
            check = False
            break
    if check:
        return strand,check,out_ratio
    else:
        return original_strand,check,out_ratio



def save_hair_strands(path,strands,bust_to_origin,translate=True):
    segments = [strands[i].shape[0] for i in range(len(strands))]
    hair_count=len(segments)
    point_count=sum(segments)
    points = np.concatenate(strands,0)
    # points = voxel_to_points(points)
    if translate:
        points -= bust_to_origin
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