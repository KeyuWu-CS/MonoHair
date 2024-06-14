from Utils.PMVO_utils import *
import platform
import torch
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import CubicHermiteSpline
from Utils.Utils import smooth_strands,compute_similar,save_hair_strands,load_strand
from tqdm import trange,tqdm
import options
from log import log
import sys

def cubic_interp(p1,p2,n1,n2, num_interp):
    spline_evaluator = CubicHermiteSpline(np.linspace(0, 1, num=2), [p1, p2],
                                          [n1* num_interp, n2 * num_interp])
    u = np.linspace(0, 1, num= num_interp+1, endpoint=True)
    sample = spline_evaluator(u)
    sample_normal = np.concatenate([sample[1:] - sample[:-1], sample[-1:] - sample[-2:-1]], 0)
    return sample, sample_normal


def points_to_voxel(points):
    voxel_min = torch.tensor([-0.32, -0.32, -0.24],dtype=torch.float,device=points.device)
    voxel_size = 0.005 / 2
    points[...,1:] *= -1
    indexs = (points - voxel_min)/voxel_size
    # indexs = torch.round(indexs)
    return indexs

def voxel_to_points(voxels):
    voxel_size = 0.005 / 2
    # voxel_min = np.array([-0.32, -0.32, -0.24])
    voxel_min = torch.tensor([-0.32, -0.32, -0.24],dtype=torch.float,device=voxels.device)
    points = voxels * voxel_size+voxel_min
    points[...,1:]*=-1
    return points



class HairGrowing():
    def __init__(self,occ_path,ori_path,device='cuda:0', image_size=[1120, 1992]):
        super().__init__()
        self.device = device
        self.image_size = image_size
        occ = get_ground_truth_3D_occ(occ_path)
        ori = get_ground_truth_3D_ori(ori_path)  # Z,H,W,C  ---> ZYX
        occ = torch.from_numpy(occ).to(device)
        ori = torch.from_numpy(ori).to(device)
        occ = occ.permute(3, 0, 1, 2)
        ori = ori.permute(3, 0, 1, 2)
        ori = ori.type(torch.float32)

        self.occ = occ
        self.ori = ori
        self.ori[1:, ...] *= -1



    def trace(self,seedPos,flag,thrDot,W,H,Z):


        seedPos+=torch.tensor([0.5,0.5,0.5],dtype=torch.float,device=seedPos.device)
        seedPos += torch.rand_like(seedPos)*0.5
        seedPos_copy = seedPos.clone()

        seedPos_index = seedPos.type(torch.long)
        seedPos_index[0] = torch.clamp(seedPos_index[0], 0, W - 1)
        seedPos_index[1] = torch.clamp(seedPos_index[1], 0, H - 1)
        seedPos_index[2] = torch.clamp(seedPos_index[2], 0, Z - 1)
        if flag[seedPos_index[2], seedPos_index[1], seedPos_index[0]] >= 3:
            return False
        Tan = self.ori[:, seedPos_index[2], seedPos_index[1], seedPos_index[0]]
        strand = []
        Tans = []
        strand.append(seedPos[None])
        Tans.append(Tan[None])
        count = 0
        while True:
            if self.occ[0, seedPos_index[2], seedPos_index[1], seedPos_index[0]] == 0:
                break
            nextPos = seedPos + Tan
            nextPos_index = nextPos.type(torch.long)
            nextPos_index[0] = torch.clamp(nextPos_index[0], 0, W - 1)
            nextPos_index[1] = torch.clamp(nextPos_index[1], 0, H - 1)
            nextPos_index[2] = torch.clamp(nextPos_index[2], 0, Z - 1)
            nextTan = self.ori[:, nextPos_index[2], nextPos_index[1], nextPos_index[0]]

            if torch.dot(nextTan, Tan) < thrDot:
                break
                # if torch.dot(-nextTan, Tan) < thrDot:
                #     break
                # else:
                #     nextTan = - nextTan
            seedPos = nextPos
            Tan = nextTan

            strand.append(seedPos[None])
            Tans.append(nextTan[None])
            seedPos_index = seedPos.type(torch.long)
            seedPos_index[0] = torch.clamp(seedPos_index[0], 0, W - 1)
            seedPos_index[1] = torch.clamp(seedPos_index[1], 0, H - 1)
            seedPos_index[2] = torch.clamp(seedPos_index[2], 0, Z - 1)
            count += 1
            if count >= 256:
                break

        seedPos = seedPos_copy
        seedPos_index = seedPos.type(torch.long)
        seedPos_index[0] = torch.clamp(seedPos_index[0], 0, W - 1)
        seedPos_index[1] = torch.clamp(seedPos_index[1], 0, H - 1)
        seedPos_index[2] = torch.clamp(seedPos_index[2], 0, Z - 1)

        Tan = self.ori[:, seedPos_index[2], seedPos_index[1], seedPos_index[0]]

        count = 0
        while True:
            if self.occ[0, seedPos_index[2], seedPos_index[1], seedPos_index[0]] == 0:
                break

            nextPos = seedPos - Tan
            nextPos_index = nextPos.type(torch.long)
            nextPos_index[0] = torch.clamp(nextPos_index[0], 0, W - 1)
            nextPos_index[1] = torch.clamp(nextPos_index[1], 0, H - 1)
            nextPos_index[2] = torch.clamp(nextPos_index[2], 0, Z - 1)
            nextTan = self.ori[:, nextPos_index[2], nextPos_index[1], nextPos_index[0]]

            if torch.dot(nextTan, Tan) < thrDot:
                break
                # if torch.dot(-nextTan, Tan) < thrDot:
                #     break
                # else:
                #     nextTan = - nextTan
            seedPos = nextPos
            Tan = nextTan
            strand.insert(0, seedPos[None])
            Tans.insert(0, nextTan[None])
            seedPos_index = seedPos.type(torch.long)
            seedPos_index[0] = torch.clamp(seedPos_index[0], 0, W - 1)
            seedPos_index[1] = torch.clamp(seedPos_index[1], 0, H - 1)
            seedPos_index[2] = torch.clamp(seedPos_index[2], 0, Z - 1)
            count += 1
            if count >= 256:
                break
        if len(strand)>=5:
            strand = torch.cat(strand, 0)
            Tans = torch.cat(Tans, 0)
            return strand
        else:
            return False




    def traceFromScalp(self,seedPos,seedNormal, thrDot, W, H, Z,pointsTree):



        seedPos_index = seedPos.type(torch.long)
        seedPos_index[0] = torch.clamp(seedPos_index[0], 0, W - 1)
        seedPos_index[1] = torch.clamp(seedPos_index[1], 0, H - 1)
        seedPos_index[2] = torch.clamp(seedPos_index[2], 0, Z - 1)
        d = torch.tensor([0,1.0,0.],device= seedNormal.device)

        Normal = seedNormal + d* min(torch.dot(seedNormal,d)+1, 1)
        Normal = Normal/torch.linalg.norm(Normal,2,-1)


        Tan = Normal.clone()
        strand = []
        Tans = []
        strand.append(seedPos[None])
        Tans.append(Tan[None])
        count = 0
        Grow_Inner = True
        while True:
            if self.occ[0, seedPos_index[2], seedPos_index[1], seedPos_index[0]] == 0 and not Grow_Inner:
                break
            nextPos = seedPos + Tan
            nextPos_index = nextPos.type(torch.long)
            nextPos_index[0] = torch.clamp(nextPos_index[0], 0, W - 1)
            nextPos_index[1] = torch.clamp(nextPos_index[1], 0, H - 1)
            nextPos_index[2] = torch.clamp(nextPos_index[2], 0, Z - 1)
            nextTan = self.ori[:, nextPos_index[2], nextPos_index[1], nextPos_index[0]]
            if torch.linalg.norm(nextTan,2) <0.1 and Grow_Inner:
                # if torch.linalg.norm(nextTan,2) >0.1:
                # nextTan = Normal.clone()
                if torch.dot(Tan,seedNormal)<0.85:
                    nextTan = Tan
                else:
                    nextTan = Tan + d * min(torch.dot(seedNormal, d) + 1, 1)
                    nextTan = nextTan / torch.linalg.norm(nextTan, 2, -1)

            else:
                if torch.dot(nextTan, Tan) < thrDot and not Grow_Inner:
                    # break
                    if torch.dot(-nextTan, Tan) < thrDot and not Grow_Inner:
                        break
                    else:
                        nextTan = - nextTan
                if torch.dot(nextTan, Tan) <0 and Grow_Inner:
                    nextTan = - nextTan

                Grow_Inner = False
            seedPos = nextPos
            Tan = nextTan

            strand.append(seedPos[None])
            Tans.append(nextTan[None])
            seedPos_index = seedPos.type(torch.long)
            seedPos_index[0] = torch.clamp(seedPos_index[0], 0, W - 1)
            seedPos_index[1] = torch.clamp(seedPos_index[1], 0, H - 1)
            seedPos_index[2] = torch.clamp(seedPos_index[2], 0, Z - 1)
            count += 1
            if count >= 256:
                break
            if count>=25 and Grow_Inner:
                break

        strand = torch.cat(strand, 0)
        if Grow_Inner:
            return None
        else:
            return strand


    def GenerateGuideStrandFromScalp(self,scalp_points,scalp_normals, pointsTree,thrDot=0.8):
        print('generate from scalp')
        Z, H, W = self.occ.size()[1:]
        print('voxel size:',Z,H,W)
        postive_indexs = torch.nonzero(self.occ[0], as_tuple=False)
        postive_indexs = torch.flip(postive_indexs, dims=[1])
        postive_indexs = postive_indexs.type(torch.float)

        strands = []
        flag = torch.zeros_like(self.occ)[0]
        for i in trange(scalp_points.size(0)):
            seedPos = scalp_points[i]
            seedNormal = scalp_normals[i]
            strand = self.traceFromScalp(seedPos,seedNormal,thrDot,W,H,Z,pointsTree)
            if strand is not None:
                strands.append(strand)
                strand_index = strand.type(torch.long)
                strand_index[:, 0] = torch.clamp(strand_index[:, 0], 0, W - 1)
                strand_index[:, 1] = torch.clamp(strand_index[:, 1], 0, H - 1)
                strand_index[:, 2] = torch.clamp(strand_index[:, 2], 0, Z - 1)
                flag[strand_index[:, 2], strand_index[:, 1], strand_index[:, 0]] = 1

        print('num guide:',len(strands))
        num_root =len(strands)
        for _ in range(2):
            for i in trange(postive_indexs.size(0)):
                seedPos = postive_indexs[i]
                strand = self.trace(seedPos, flag, thrDot, W, H, Z)
                if strand is not False:
                    strands.append(strand)
                    strand_index = strand.type(torch.long)
                    strand_index[:, 0] = torch.clamp(strand_index[:, 0], 0, W - 1)
                    strand_index[:, 1] = torch.clamp(strand_index[:, 1], 0, H - 1)
                    strand_index[:, 2] = torch.clamp(strand_index[:, 2], 0, Z - 1)
                    flag[strand_index[:, 2], strand_index[:, 1], strand_index[:, 0]] += 1


        self.strands = strands
        print('done...')
        return strands,num_root



    def randomlyGenerateSegments(self,thrDot=0.8):
        print('generate segments...')
        Z,H,W = self.occ.size()[1:]

        postive_indexs = torch.nonzero(self.occ[0],as_tuple=False)
        postive_indexs = torch.flip(postive_indexs,dims=[1])
        postive_indexs = postive_indexs.type(torch.float)
        strands = []
        strandsTan = []


        flag = torch.zeros_like(self.occ)[0]

        step = 3
        for iter in range(step):
            for i in trange(postive_indexs.size(0)):
                seedPos = postive_indexs[i]
                strand = self.trace(seedPos,flag,thrDot,W,H,Z)
                if strand is not False:
                    strands.append(strand)
                    strand_index = strand.type(torch.long)
                    strand_index[:, 0] = torch.clamp(strand_index[:, 0], 0, W - 1)
                    strand_index[:, 1] = torch.clamp(strand_index[:, 1], 0, H - 1)
                    strand_index[:, 2] = torch.clamp(strand_index[:, 2], 0, Z - 1)
                    flag[strand_index[:, 2], strand_index[:, 1], strand_index[:, 0]] += 1


        self.strands = strands
        self.strandsTan = strandsTan
        print('done...')
        return strands



    def connect_segments(self,strands_connect_info,strands,i):

        type_dict = {
            'tip':'root',
            'root':'tip'
        }

        long_strand = []
        long_strand.append(strands[i])
        connect_info = strands_connect_info[i]

        root_connect_info = connect_info['root']
        tip_connect_info = connect_info['tip']
        connect_list = [i]

        def connect(connected_strand,best_strand,type,along_with_root=True):
            connect_list.append(best_strand)
            strand = strands[best_strand]
            if type =='root':
                if along_with_root:
                    connected_strand = self.connect_strands(connected_strand,strand[::-1],False)
                else:
                    connected_strand = self.connect_strands(connected_strand, strand, True)
            else:
                if along_with_root:
                    connected_strand = self.connect_strands(connected_strand, strand, False)
                else:
                    connected_strand = self.connect_strands(connected_strand, strand[::-1], True)
            connect_info = strands_connect_info[best_strand][type_dict[type]]
            if connect_info is not None:
                if connect_info[0] not in connect_list:
                    connected_strand = connect(connected_strand, connect_info[0], connect_info[1], along_with_root=along_with_root)


            return connected_strand


        if root_connect_info is not None:
            long_strand = connect(long_strand,root_connect_info[0],root_connect_info[1],along_with_root=True)
        if tip_connect_info is not None:
            long_strand = connect(long_strand,tip_connect_info[0],tip_connect_info[1],along_with_root=False)
        # print(long_strand)
        long_strand = np.concatenate(long_strand,0)
        return long_strand


    def connect_strands(self,strand1,strand2,push_back,cubic_sample=False,add_mid=True,need_weight=False):
        num_connect = strand2.shape[0]


        if push_back:
            base_strand = strand1[-1]
            connect_strand = []
            seedPos = base_strand[-1]


            if add_mid:
                mid_point = seedPos*0.5 + strand2[0]*0.5
                connect_strand.append(mid_point[None])
                seedPos = mid_point
            for i in range(num_connect-1):
                nextPos = seedPos + (strand2[i+1] - strand2[i])
                # weight = np.sin(0.5*np.pi * min(1, i / (0.7*num_connect)))
                # if num_connect - i <=4:
                #     weight = 0.9
                # else:
                #     weight=0
                weight = 0
                nextPos = nextPos * (1-weight) + strand2[i+1]*weight
                connect_strand.append(nextPos[None])
                seedPos = nextPos

            # dir = strand2[1:] - strand2[:-1]
            # dir = np.cumsum(dir,axis=0)
            # connect_strand.append(seedPos+dir)



            connect_strand = np.concatenate(connect_strand, 0)
            strand1.append(connect_strand)

        else:

            base_strand = strand1[0]
            connect_strand = []
            seedPos = base_strand[0]

            if add_mid:
                mid_point = seedPos * 0.5 + strand2[-1] * 0.5
                connect_strand.append(mid_point[None])
                seedPos = mid_point
            for i in range(num_connect-1):
                nextPos = seedPos + (strand2[-2-i]-strand2[-1-i])
                # weight = np.sin(0.5 * np.pi * min(1, i / (0.7 * num_connect)))
                if need_weight:
                    weight = np.sin(0.5 * np.pi * min(1, i / (1.5 * num_connect)))
                else:
                    weight=0
                # if num_connect - i <= 4:
                #     weight = 0.9
                # else:
                #     weight = 0

                nextPos = nextPos * (1 - weight) + strand2[-2-i] * weight
                connect_strand.append(nextPos[None])
                seedPos = nextPos

            # temp_ss = strand2[::-1]
            # dir = temp_ss[1:] - temp_ss[:-1]
            # dir = np.cumsum(dir, axis=0)
            # connect_strand.append(seedPos + dir)

            connect_strand = np.concatenate(connect_strand,0)
            strand1.insert(0,connect_strand[::-1])

        return strand1



    def query(self,point,tree,k,dist,i):
        nei_distance, nei_strands_index = tree.query(point, k=k,
                                                           distance_upper_bound=dist)  ### root connect with root
        prune_index = nei_distance < 9999
        nei_distance = nei_distance[prune_index]

        nei_strands_index = nei_strands_index[prune_index]

        delet_self_index = nei_strands_index == i
        nei_distance = nei_distance[~delet_self_index]
        nei_strands_index = nei_strands_index[~delet_self_index]
        return nei_distance,nei_strands_index


    def find_connect_info(self,strands,connect_threshold=0.005,connect_dot_threshold=0.7,occ=None):
        print('connect segments...')
        new_strands = []
        strands_root = []
        strands_tip = []
        strands_root_ori = []
        strands_tip_ori = []
        strands_tree = []

        for strand in strands:
            strands_root.append(strand[:1])
            strands_tip.append(strand[-1:])
            strands_root_ori.append(strand[1:2]-strand[:1])
            strands_tip_ori.append(strand[-1:]-strand[-2:-1])
            strands_tree.append(KDTree(data=strand))

        strands_root_ori = np.concatenate(strands_root_ori,0)
        strands_tip_ori = np.concatenate(strands_tip_ori,0)
        strands_root = np.concatenate(strands_root,0)
        strands_tip = np.concatenate(strands_tip,0)
        roots_tree = KDTree(data=strands_root)
        tips_tree = KDTree(data=strands_tip)

        strands_connect_info = []



        # for strand in strands:
        for i in trange(len(strands)):
            strand = strands[i]
            root_Ori = strand[1:2] - strand[:1]
            tip_Ori = strand[-1:] - strand[-2:-1]
            connect_info = {}

            ##### find connection for root
            nei_distance, nei_strands_index = self.query(strand[0],roots_tree ,k=50,dist=connect_threshold,i=i)
            # best_strand = self.find_best_connect_strands(root_Ori,strands_root_ori[nei_strands_index],strands_root[i:i+1],strands_root[nei_strands_index],
            #                                              strands_tip[i:i+1],strands_tip[nei_strands_index],nei_distance,nei_strands_index,type='root2root',threshold=connect_dot_threshold)
            best_strand = self.find_best_connect_strands(root_Ori, strands_root_ori[nei_strands_index],strand,strands_tree,
                                                         nei_distance, nei_strands_index, type='root2root',
                                                         threshold=connect_dot_threshold,strands =strands)


            if best_strand is None:
                nei_distance, nei_strands_index = self.query(strand[0], tips_tree, k=50, dist=connect_threshold,i=i)
                # best_strand = self.find_best_connect_strands(root_Ori, strands_tip_ori[nei_strands_index],strands_root[i:i+1],strands_root[nei_strands_index],
                #                                              strands_tip[i:i+1], strands_tip[nei_strands_index],nei_distance,nei_strands_index, type='root2tip',threshold=connect_dot_threshold)
                best_strand = self.find_best_connect_strands(root_Ori, strands_tip_ori[nei_strands_index],strand,strands_tree,
                                                             nei_distance,nei_strands_index, type='root2tip',threshold=connect_dot_threshold,strands =strands)
                if best_strand is None:
                    connect_info['root'] = None
                else:
                    connect_info['root'] = [best_strand, 'tip']
            else:
                connect_info['root'] = [best_strand,'root']


            nei_distance, nei_strands_index = self.query(strand[-1], roots_tree, k=50, dist=connect_threshold,i=i)
            # best_strand = self.find_best_connect_strands(tip_Ori, strands_root_ori[nei_strands_index],strands_root[i:i+1],strands_root[nei_strands_index],
            #                                              strands_tip[i:i+1], strands_tip[nei_strands_index],nei_distance,nei_strands_index, type='tip2root',threshold=connect_dot_threshold)
            best_strand = self.find_best_connect_strands(tip_Ori, strands_root_ori[nei_strands_index],strand, strands_tree,
                                                        nei_distance,nei_strands_index, type='tip2root',threshold=connect_dot_threshold,strands =strands)
            if best_strand is None:

                nei_distance, nei_strands_index = self.query(strand[-1], tips_tree, k=50, dist=connect_threshold,i=i)

                # best_strand = self.find_best_connect_strands(tip_Ori, strands_tip_ori[nei_strands_index],strands_root[i:i+1],strands_root[nei_strands_index],
                #                                              strands_tip[i:i+1], strands_tip[nei_strands_index],nei_distance,nei_strands_index, type='tip2tip',threshold=connect_dot_threshold)
                best_strand = self.find_best_connect_strands(tip_Ori, strands_tip_ori[nei_strands_index],strand, strands_tree,
                                                             nei_distance,nei_strands_index, type='tip2tip',threshold=connect_dot_threshold,strands =strands)
                if best_strand is None:
                    connect_info['tip'] = None
                else:
                    connect_info['tip'] = [best_strand,'tip']
            else:
                connect_info['tip'] = [best_strand,'root']
            strands_connect_info.append(connect_info)

        fail = 0
        for i in trange(len(strands_connect_info)):
            strand = self.connect_segments(strands_connect_info,strands,i)
            ss = strand.copy()
            count=0

            while True:
                ss_tensor = torch.from_numpy(ss).to(occ.device)
                idx = points_to_voxel(ss_tensor)
                idx = torch.round(idx).type(torch.long)
                if torch.max(idx[:, 2]) >= 192 or torch.max(idx[:, 1] >= 256) or torch.max(idx[:, 0] >= 256):
                    check = False
                    break
                ss_occ = occ[0, idx[:, 2], idx[:, 1], idx[:, 0]]
                if torch.sum(ss_occ) / ss_occ.size(0) >0.8:
                    strand = ss
                    check = True
                    break
                else:
                    ss = strand.copy()
                    ss += np.random.random((3)) * 0.005
                count+=1
                if count>=50:
                    check = False
                    break
            if not check:
                fail+=1
            new_strands.append(strand)


        print('fail:',fail)
        print('done...')
        return new_strands



    def find_best_connect_strands(self, root_ori, nei_strands, strand, strands_tree, nei_distance, nei_strands_index, type='root2root', threshold=0.7,strands=None):
        root_ori = root_ori.repeat(nei_strands.shape[0], 0)
        Ori_similar = np.sum(root_ori * nei_strands, -1) / (
                    np.linalg.norm(root_ori, axis=-1) * np.linalg.norm(nei_strands, axis=-1))




        if type == 'root2root' or type == 'tip2tip':
            index = Ori_similar < -threshold

        if type == 'root2tip' or type == 'tip2root':
            index = Ori_similar > threshold
        if np.sum(index) == 0:
            return None

        dist_index = np.ones_like(nei_strands_index).astype(np.bool_)
        for i,nei_strands_I in enumerate(nei_strands_index):
            if strand.shape[0] + strands[nei_strands_I].shape[0]>=80:
                dist_index[i] = False

            dist,_ = strands_tree[nei_strands_I].query(strand,1)
            if strand.shape[0]<6:
                dist_index[i] = np.sum(dist<0.005)<4
            else:
                dist_index[i] = np.sum(dist<0.01) <= 6
            strand_lenght = np.linalg.norm(strand[0]- strand[-1],2)
            if dist[0]<strand_lenght*2/3  and dist[-1]<strand_lenght*2/3 and len(strand)>20:
                dist_index[i] = False

        index = np.logical_and(index, dist_index)

        if np.sum(index) == 0:
            return None

        loss = nei_distance[index] * (1 - np.abs(Ori_similar[index]))
        best_strand = nei_strands_index[index][np.argmin(loss)]
        return best_strand





    def _connect_to_scalp(self,strand, root_tree, scalp_points,scalp_normals):
        begPos = strand[0]
        begTan = strand[1] - strand[0]
        _, nei_root_index = root_tree.query(begPos,k=1)
        nei_root = scalp_points[nei_root_index]
        nei_root_normal = scalp_normals[nei_root_index]
        num = np.linalg.norm(begPos-nei_root,2)
        sample_points = cubic_interp(nei_root,begPos,nei_root_normal,begTan,int(num))[0]
        sample_points = np.array(sample_points)
        strand = np.concatenate([sample_points,strand],0)
        return strand


    def connect_to_scalp(self,strands,num_root):


        root_flag =np.zeros((len(strands),))
        root_flag[:num_root] =1
        out_ratio = np.zeros_like(root_flag)

        print('num of strands:',len(strands))
        print('num of good strands:',np.sum(root_flag))
        flag = True
        iter = 0
        root_flag = root_flag.astype(np.bool_)
        out_root_flag = np.zeros_like(root_flag).astype(np.bool_)
        print('connect poor strands to good strands...')
        thr_dist = 0.5
        thr_dot = 0.9
        max_thr_dist = 2.0
        max_dot_dist = 0.6
        if args.PMVO.infer_inner:
            thr_dist = 0.5
            thr_dot = 0.9
            max_thr_dist = 2
            max_dot_dist = 0.6




        while flag:
            print('iter:',iter)
            print('num of good strands:', np.sum(root_flag))
            print('num of out strands:', np.sum(out_root_flag))
            print('current thr_dist:',thr_dist)
            print('current thr_dot:',thr_dot)

            num_good = np.sum(root_flag)
            strands_info = []
            core_strands = []
            strands_tree = []
            for i in range(len(strands)):
                strand = strands[i]
                if root_flag[i]:
                    core_strands.append(strand)
                    strands_info.extend([i]*strand.shape[0])
                strands_tree.append(KDTree(data=strand))
            strands_info = np.array(strands_info)




            core_strands = np.concatenate(core_strands,0)
            core_strands_tree = KDTree(data=core_strands)


            for i in tqdm(range(len(strands))):
                if root_flag[i] or out_root_flag[i]:
                    continue
                strand = strands[i]
                nei_index = core_strands_tree.query_ball_point(strand[0],thr_dist)
                nei_strands = strands_info[nei_index]
                nei_index_inv = core_strands_tree.query_ball_point(strand[-1], thr_dist * 2)
                neiI_strands_inv = strands_info[nei_index_inv]
                if len(neiI_strands_inv) != 0 and len(np.union1d(nei_index_inv, nei_index)) == 0:
                    continue
                if len(nei_index)!=0:
                    cloest_nei_strand = nei_strands[0]
                    nei_pos_dist,nei_pos_index = strands_tree[cloest_nei_strand].query(strand,1)


                    # _,nei_pos_index_end = strands_tree[cloest_nei_strand].query(strand[-1],1)
                    nei_pos_index_beg = nei_pos_index[0]
                    nei_pos_index_end = nei_pos_index[-1]
                    ss = strands[cloest_nei_strand]
                    if nei_pos_index_beg == ss.shape[0]-1:
                        tan1 = ss[nei_pos_index_beg] - ss[nei_pos_index_beg-1]
                    else:
                        tan1 =ss[nei_pos_index_beg+1] - ss[nei_pos_index_beg]
                    tan2 = strand[1]-strand[0]
                    if compute_similar(tan1,tan2)<0 and nei_pos_index_beg>nei_pos_index_end and np.mean(nei_pos_dist)<5:
                        strands[i] = strand[::-1]
                        strand = strands[i]


                connect = False
                min_loss = np.inf
                best_pos_index = None
                min_nei_point_index = None
                min_nei_strandI =None

                check = []
                count = 0
                for neiI in nei_strands:
                    if neiI in check:
                        continue
                    check.append(neiI)
                    count+=1
                    nei_strand = strands[neiI]
                    _,nei_point_index = strands_tree[neiI].query(strand[0],1)
                    nei_distance,_ = strands_tree[neiI].query(strand[:5],1)
                    if np.mean(nei_distance)<1:
                        continue

                    if len(strand)>60 and len(strand)+nei_point_index>150:
                        continue



                    Tan = strand[1]-strand[0]
                    if nei_point_index<=1:
                        continue
                    # beg_index = max(nei_point_index - 4,max(nei_point_index-int(thr_dist)-2,0))
                    # loss,pos_index = self.compute_strands_similar(strand[0],nei_strand[beg_index:nei_point_index+1],Tan,thr_dist,thr_dot)
                    if nei_point_index==0:
                        nei_ori = nei_strand[nei_point_index+1]-nei_strand[nei_point_index]
                    else:
                        nei_ori = nei_strand[nei_point_index] - nei_strand[nei_point_index-1]
                    loss,pos_index = self.compute_strands_similar(strand[0],nei_strand[nei_point_index:nei_point_index+1],Tan,thr_dist,thr_dot,nei_ori)
                    loss+= out_ratio[neiI]
                    if loss<min_loss:
                        min_loss = loss
                        min_nei_strandI = neiI
                        best_pos_index = pos_index
                        min_nei_point_index = nei_point_index + best_pos_index
                        connect = True
                    if count>=30:
                        break



                if not connect:
                    continue
                else:
                    if min_nei_point_index<=1 or best_pos_index is None:
                        continue
                    else:

                        ss = strands[min_nei_strandI]
                        min_index = 0


                        mid_point = strand[min_index]*0.95 + ss[min_nei_point_index]*0.05   ### usually0.8 0.2
                        #mid_point = strand[min_index]*0.8 + ss[min_nei_point_index]*0.2   ### usually0.8 0.2
                        # mid_point = strand[min_index]*0.9 + ss[min_nei_point_index]*0.1   ### usually0.8 0.2
                        min_point = mid_point[None]

                        connect_strand = self.connect_strands([min_point,strand[min_index:]],ss[:min_nei_point_index + 1], push_back=False,cubic_sample=False,add_mid=False)


                    connect_strand = np.concatenate(connect_strand,0)
                    connect_strand,in_check,out_r = random_move_strands(connect_strand, self.occ, self.ori,args.HairGenerate.out_ratio,index = min_nei_point_index+1)
                    out_ratio[i] = out_r
                    if in_check:
                        strands[i] = connect_strand
                        root_flag[i] = True
                    else:
                        strands[i] = connect_strand
                        out_root_flag[i] = True
            if np.sum(root_flag)-num_good>(len(strands)-num_root)//500:
                flag = True
            else:
                if thr_dist == max_thr_dist and thr_dot==max_dot_dist:
                    flag=False
                else:
                    thr_dist = min(thr_dist+0.25,max_thr_dist)
                    thr_dot = max(thr_dot-0.075,max_dot_dist)
                    flag = True
            iter+=1

        print('done...')


        print('connect to scalp...')

        new_strands = []
        for i in trange(len(strands)):
            strand = strands[i]
            if root_flag[i] or out_root_flag[i]:
                new_strands.append(strand)

        return new_strands



    def compute_strands_similar(self,strand1,strand2,Tan,thr_dist,thr_dot,ori):

        # ori = np.concatenate([strand2[1:]-strand2[:-1],strand2[-1:]-strand2[-2:-1]],0)

        min_loss = np.inf
        min_index = None
        for i in range(strand2.shape[0]):
            dist = np.linalg.norm(strand2[i] - strand1)
            # similar = compute_similar(strand1[i+1]-strand1[i],Tan)
            similar_connect = compute_similar(strand1-strand2[i], Tan)
            # similar = compute_similar(ori[i], Tan)
            similar = compute_similar(ori, Tan)
            # if similar > thr_dot and dist < thr_dist + strand2.shape[0]-i-1 and similar_connect>0.5:
            # loss = (1 - similar_connect) * dist
            # if similar > thr_dot and dist < thr_dist + strand2.shape[0]-i-1 and similar_connect>0.5:
            if similar > thr_dot and dist < thr_dist + strand2.shape[0]-i-1:
                loss = (1 - similar_connect)+0.1*thr_dist
                if loss < min_loss:
                    min_loss = loss
                    min_index = i
        if min_index is not None:
            min_index = min_index - strand2.shape[0] + 1


        return min_loss,min_index



    def VoxelToWorld(self,strands,bust_to_origin=None):
        new_strands = []
        for ss in strands:
            ss = voxel_to_points(ss)
            ss = ss.cpu().numpy()
            if bust_to_origin is not None:
                ss -= bust_to_origin
            new_strands.append(ss)
        return new_strands

    def WorldToVoxel(self,strands,bust_to_origin=None):
        new_strands = []
        for ss in strands:
            if bust_to_origin is not None:
                ss += bust_to_origin
            ss = torch.from_numpy(ss).type(torch.float).to(args.device)
            ss = points_to_voxel(ss)

            new_strands.append(ss.cpu().numpy())
        return new_strands

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
    args.data.scalp_path = os.path.join(args.data.root, args.data.scalp_path)


    suffix = ''
    if args.scalp_diffusion:
        suffix = '_diffusion'


    args.image_camera_path = os.path.join(args.data.root, args.image_camera_path)
    if args.PMVO.infer_inner:
        args.save_path = os.path.join(args.output_path, 'full')
    else:
        args.save_path = os.path.join(args.output_path, 'refine')

    args.data.Occ3D_path = os.path.join(args.save_path, 'Occ3D{}.mat'.format(suffix))
    args.data.Ori3D_path = os.path.join(args.save_path, 'Ori3D{}.mat'.format(suffix))

    return args


if __name__ == '__main__':
    args = config_parser()


    mesh = o3d.io.read_triangle_mesh(args.data.scalp_path)
    scalp = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=60000,
                                                              use_triangle_normal=False)  # 采样点云
    scalp_points = np.asarray(scalp.points)
    scalp_normals = np.asarray(scalp.normals)


    scalp_points += args.bust_to_origin
    scalp_points = torch.from_numpy(scalp_points).to(args.device)
    scalp_normals = torch.from_numpy(scalp_normals).to(args.device)

    scalp_normals = scalp_normals / torch.linalg.norm(scalp_normals, 2, -1, keepdims=True)

    scalp_points = points_to_voxel(scalp_points)
    scalp_normals[:, 1:] *= -1

    scalp_normals = scalp_normals.type(torch.float32)
    scalp_points = scalp_points.type(torch.float32)

    treepoints = scalp_points.cpu().numpy()
    pointsTree = KDTree(data=treepoints)

    HairGrowSolver = HairGrowing(args.data.Occ3D_path, args.data.Ori3D_path, device=args.device,
                                 image_size=args.data.image_size)

    # HairGrowSolver.GenerateGuideStrandFromScalp(occ,ori,scalp_points,scalp_normals,args.HairGenerate.grow_threshold)

    if args.HairGenerate.generate_segments:
        strands,num_root = HairGrowSolver.GenerateGuideStrandFromScalp(scalp_points, scalp_normals, pointsTree,
                                                               args.HairGenerate.grow_threshold)

        strands = HairGrowSolver.VoxelToWorld(strands,args.bust_to_origin)

        save_path = os.path.join(args.save_path, 'scalp_segment.hair')
        save_hair_strands(save_path, strands)
        strands = smooth_strands(strands, 4.0, 2.0)
        save_path = os.path.join(args.save_path, 'scalp_segment_smooth.hair')
        save_hair_strands(save_path, strands)

        np.save(args.save_path + '/num_root.npy', np.array(num_root))
    else:
        # num_root = 51865
        num_root = np.load(args.save_path + '/num_root.npy')


    if args.HairGenerate.connect_segments:
        segment, points = load_strand(os.path.join(args.save_path, 'scalp_segment.hair'),return_strands=False)

        strands = []
        beg = 0
        for i, seg in enumerate(segment):
            end = beg + seg
            strand = points[beg:end]
            if i >= num_root:
                strand += args.bust_to_origin
            strands.append(strand)
            beg += seg
        # strands = smooth_strands(strands, 4.0, 2.0)

        HairGrowSolver.strands = strands
        connected_strands = HairGrowSolver.find_connect_info(strands[num_root:], args.HairGenerate.connect_threshold,
                                                             args.HairGenerate.connect_dot_threshold, HairGrowSolver.occ)
        # new_strands = []
        new_strands = strands[:num_root]

        for i in trange(len(connected_strands)):

            ss = connected_strands[i] - args.bust_to_origin
            new_strands.append(ss)

        new_strands = smooth_strands(new_strands, 4.0, 2.0)
        save_path = os.path.join(args.save_path, 'strands.hair')
        save_hair_strands(save_path, new_strands)

    if args.HairGenerate.connect_scalp:
        segment, points, strands, oris = load_strand(os.path.join(args.save_path, 'strands.hair'),return_strands=True)
        strands = HairGrowSolver.WorldToVoxel(strands,args.bust_to_origin)


        scalp_points = scalp_points.cpu().numpy()
        scalp_normals = scalp_normals.cpu().numpy()
        root_tree = KDTree(data=scalp_points)

        connect_strands = HairGrowSolver.connect_to_scalp(strands,num_root)

        strands = []

        for ss in connect_strands:

            ss = torch.from_numpy(ss.copy())
            ss = voxel_to_points(ss)
            ss = ss.cpu().numpy()
            ss -= args.bust_to_origin
            strands.append(ss)

        strands = smooth_strands(strands, 4.0, 2.0)
        save_hair_strands(os.path.join(args.save_path, 'connected_strands.hair'), strands)







