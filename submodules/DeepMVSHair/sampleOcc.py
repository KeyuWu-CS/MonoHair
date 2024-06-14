import numpy as np
import pandas as pd
import torch
import os
from pyntcloud import PyntCloud
import torch.nn.functional as F

occ_folder = os.path.join('data', 'occ_data')
point_cloud_folder = os.path.join('data', 'point_cloud')
occ_sample_folder = os.path.join('data', 'occ_samples')
grid_size = [128, 128, 96]

bbox_min = torch.tensor([-0.370043, 1.22692, -0.259537]).cuda()
grid_step = torch.tensor([0.00565012, 0.00565012, 0.00565012]).cuda()

total_voxels = 128 * 128 * 96

def sampleGridCorner(vsize=0.005, bbox_min=(-0.3, -0.4, -0.25), bbox_max=(0.3, 0.4, 0.25)):
    '''
    sample around the origin of bust
    generate querying points on voxel corners
    :return:
    '''
    x = torch.arange(bbox_min[0], bbox_max[0] + vsize, vsize).cuda().float()
    y = torch.arange(bbox_min[1], bbox_max[1] + vsize, vsize).cuda().float()
    z = torch.arange(bbox_min[2], bbox_max[2] + vsize, vsize).cuda().float()

    xv, yv, zv = torch.meshgrid([x, y, z])

    coords = torch.stack([xv, yv, zv], dim=3).reshape(-1, 3)

    homogeneous = torch.ones(coords.shape[0], 1).cuda().float()
    samples = torch.cat([coords, homogeneous], dim=1).transpose(0, 1)

    return samples

def sampleGridCenter(vsize=0.005, bbox_min=(-0.3, -0.4, -0.25), bbox_max=(0.3, 0.4, 0.25)):
    '''
    sample around the origin of bust
    generate querying points on voxel centers
    :param vsize:
    :param bbox_min:
    :param bbox_max:
    :return:
    '''
    x = torch.arange(bbox_min[0], bbox_max[0], vsize).cuda().float()
    y = torch.arange(bbox_min[1], bbox_max[1], vsize).cuda().float()
    z = torch.arange(bbox_min[2], bbox_max[2], vsize).cuda().float()

    xv, yv, zv = torch.meshgrid([x, y, z])

    coords = torch.stack([xv, yv, zv], dim=3).reshape(-1, 3) + 0.5 * vsize

    homogeneous = torch.ones(coords.shape[0], 1).cuda().float()
    samples = torch.cat([coords, homogeneous], dim=1).transpose(0, 1)

    return samples


def denseSampleGrid():
    '''
    sample grid uniformly (without labels), only works on synthetic data where sampling range is known
    :return:
    '''
    x = torch.arange(grid_size[0]).cuda()
    y = torch.arange(grid_size[1]).cuda()
    z = torch.arange(grid_size[2]).cuda()

    xv, yv, zv = torch.meshgrid([x, y, z])
    coords = torch.cat([xv.reshape(*grid_size, 1), yv.reshape(*grid_size, 1), zv.reshape(*grid_size, 1)], dim=3).reshape(-1, 3)
    base = bbox_min + coords * grid_step
    # basis = torch.cat([base] * sample_per_grid, dim=0) if sample_per_grid > 1 else base
    #
    # random_offset = torch.rand(basis.shape).cuda() * grid_step
    # samples = basis + random_offset
    samples = base + grid_step * 0.5
    homogeneous = torch.ones(samples.shape[0], 1).cuda()
    samples = torch.cat([samples, homogeneous], dim=1).transpose(0, 1)

    return samples


def randSampleFromGrid(indices, sample_per_grid, label, color):
    '''
    sample synthetic GT data (with labels)
    :param indices: tensors, [N, 3]
    :param sample_per_grid:
    :param label:
    :param color:
    :return: [N * sample_per_grid, 3]
    '''

    base = bbox_min + indices * grid_step
    basis = torch.cat([base] * sample_per_grid, dim=0) if sample_per_grid > 1 else base
    random_offset = torch.rand(basis.shape).cuda() * grid_step
    samples = basis + random_offset
    samples = samples.cpu().numpy().astype('float32')

    labels = np.full(samples.shape[0], label).astype('float32')[:, np.newaxis]

    colors = np.zeros(samples.shape).astype('uint8')
    colors[:] = color

    return samples, labels, colors


def savePointCloud(fname, samples, colors):
    '''

    :param fname:
    :param samples: [N, 3]
    :param colors: [N, 3]
    :return:
    '''
    d = {
        'x': samples[:, 0],
        'y': samples[:, 1],
        'z': samples[:, 2],
        'red': colors[:, 0],
        'green': colors[:, 1],
        'blue': colors[:, 2]
    }

    cloud = PyntCloud(pd.DataFrame(data=d))
    cloud.to_file(fname)


def sampleOcc(data_id, kernel=5):
    fname = os.path.join(occ_folder, 'occ_' + str(data_id) + '.dat')

    positive_indices = np.fromfile(fname, dtype='int32').reshape(-1, 3)

    valid_indices = torch.tensor(positive_indices).cuda().long()
    # voxels containing hair strand segments
    positive_samples, labels1, colors1 = randSampleFromGrid(valid_indices, sample_per_grid=6, label=1, color=torch.tensor((255, 0, 0)))
    # savePointCloud(os.path.join(point_cloud_folder, 'samples_posi_' + str(data_id) + '.ply'), positive_samples, colors1)

    voxels = torch.zeros([1, 128, 128, 96]).cuda()
    voxels[0, valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = 1

    voxels_enlarge = F.max_pool3d(voxels, kernel, 1, kernel // 2)
    # empty voxels close to valid voxels
    voxels_empty_close = (1 - voxels) * voxels_enlarge
    # other empty voxels
    voxels_empty = 1 - voxels_enlarge

    negative_close_indices = torch.nonzero(voxels_empty_close, as_tuple=False)[:, 1:]
    negative_close_samples, labels2, colors2 = randSampleFromGrid(negative_close_indices, sample_per_grid=4, label=0, color=torch.tensor((0, 255, 0)))
    # savePointCloud(os.path.join(point_cloud_folder, 'samples_close_' + str(data_id) + '.ply'), negative_close_samples, colors2)

    negative_indices = torch.nonzero(voxels_empty, as_tuple=False)[:, 1:]
    # reduce negative samples in distant empty space
    select_indices = torch.arange(0, negative_indices.shape[0], step=10)
    negative_samples, labels3, colors3 = randSampleFromGrid(negative_indices[select_indices], sample_per_grid=1, label=0, color=torch.tensor((0, 0, 255)))
    # savePointCloud(os.path.join(point_cloud_folder, 'samples_nega_' + str(data_id) + '.ply'), negative_samples, colors3)

    samples_union = np.concatenate([positive_samples, negative_close_samples, negative_samples], axis=0)
    labels_union = np.concatenate([labels1, labels2, labels3], axis=0)

    data_union = np.concatenate([samples_union, labels_union], axis=1)

    data_union.tofile(os.path.join(occ_sample_folder, 'occ_samples_' + str(data_id) + '.dat'))

    print('\nsave to {}'.format(fname))
    print('positive samples: {}'.format(len(positive_samples)))
    print('total negative samples: {}'.format(len(negative_close_samples) + len(negative_samples)))
    print('posi / nega ratio: {}'.format(len(positive_samples) / (len(negative_close_samples) + len(negative_samples))))

    return len(positive_samples) / (len(negative_close_samples) + len(negative_samples))


def dat2ply(dat_fname, ply_fname):
    data = np.fromfile(dat_fname, dtype='float32').reshape(-1, 4)
    samples = data[:, :3]
    labels = data[:, 3].astype('int32')

    samples = samples[labels == 1]

    colors = np.zeros((samples.shape[0], 3), dtype='uint8')
    colors[:, 1] = 255

    savePointCloud(ply_fname, samples, colors)


def getId(fname, prefix):
    return int(fname[len(prefix):fname.index('.')])


def tensor2ply(tensor_fname, ply_fname):
    # [4, N] -> [N, 4]
    tensor_points = torch.load(tensor_fname)['points'].cpu().numpy()
    colors = np.zeros((tensor_points.shape[0], 3), dtype='uint8')
    colors[:, 0] = 255
    savePointCloud(ply_fname, tensor_points, colors)


def data2samples():
    fname_list = os.listdir(occ_folder)
    ratio_sum = 0
    for fname in fname_list:
        id = getId(fname, 'occ_')
        ratio = sampleOcc(id)
        ratio_sum+=ratio

    print('avg pn ratio: {}'.format(ratio_sum / len(fname_list)))


def old():
    import torch.nn.functional as F
    # convert predict points to ply file
    # tensor_folder = 'C:\\Users\\musinghead\\PycharmProjects\\HairMVSNet\\results\\0716'
    # ply_folder = 'C:\\Users\\musinghead\\PycharmProjects\\HairMVSNet\\results\\point_clouds'
    # fname_list = os.listdir(tensor_folder)
    # for fname in fname_list[:20]:
    #     id = getId(fname, '')
    #     path = os.path.join(tensor_folder, fname)
    #     tensor2ply(path, os.path.join(ply_folder, str(id) + '.ply'))

    # convert GT points to ply file
    tensor_folder = 'results\\1029'

    dst_folder = 'results\\1029_pc'
    # ply_folder = 'C:\\Users\\musinghead\\PycharmProjects\\HairMVSNet\\results\\point_clouds'
    # hair_data_folder = 'C:\\Users\\musinghead\\Dataset\\Hair'
    fname_list = os.listdir(tensor_folder)
    for index, fname in enumerate(fname_list):
        id = getId(fname, '')
        data_path = os.path.join(tensor_folder, fname)
        data = torch.load(data_path)
        points = data['points'].cpu().numpy()
        orients = data['orients'].cpu().numpy()

        cnt = points.shape[1]
        new_data = np.concatenate([points, orients], axis=0).transpose((1, 0)).reshape(-1)

        new_data_w_cnt = np.concatenate([[np.float32(cnt)], new_data])

        dst_path = os.path.join(dst_folder, str(id) + '.dat')
        new_data_w_cnt.tofile(dst_path)

        print('finish', index)
    #     path = os.path.join(hair_data_folder, str(id), 'occ_samples.dat')
    #     dat2ply(path, os.path.join(ply_folder, str(id) + '_gt.ply'))


if __name__ == '__main__':
    tensor2ply('results\\1029\\370.pth', 'results\\point_clouds\\370.ply')