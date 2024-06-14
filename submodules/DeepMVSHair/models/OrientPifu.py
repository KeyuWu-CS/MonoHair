import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Unet import Unet
from models.MLP import MLP
from models.BackBone import BackBone


class OrientPifu(nn.Module):

    def __init__(self):
        super(OrientPifu, self).__init__()
        self.features = BackBone(in_feat=3)
        self.query = MLP(input_feat=736)

    def forward(self, orient_map, xy_points, z_feat):
        '''

        :param orient_map: [B, 2, H, W]
        :param xy_points: [B, N, 1, 2], N = num of sample points
        :param z_feat: [B, 1, N]
        :return:
        '''
        sample_feat = self.features(orient_map, xy_points)
        prediction = self.query(sample_feat, z_feat)

        return prediction


if __name__ == '__main__':
    op = OrientPifu()
    img = torch.randn((1,2,1024,1024))
    points = torch.randn((1, 2000, 1, 2)) * 2 - 1
    z = torch.randn((1, 1, 2000))
    out = op(img, points, z)
    print('out', out.shape)
