import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Shallow import Shallow
from models.UnetSimple import UNetSimple, ShallowEncoder
from models.ViT import OccViT


# Positional encoding from nerf-pytorch
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class SoftL1Loss(nn.Module):
    def __init__(self, reduction=None):
        super(SoftL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, eps=0.0, lamb=0.0, thresh=0.5):
        ret = torch.abs(input - target) - eps
        ret = torch.clamp(ret, min=0.0)
        ret = ret * (1 + lamb * torch.sign(target - thresh) * torch.sign(thresh - input))
        if self.reduction is None or self.reduction == "mean":
            return torch.mean(ret)
        else:
            return torch.sum(ret)


class ConvModule1d(nn.Module):

    def __init__(self, in_feat, out_feat):
        super(ConvModule1d, self).__init__()
        self.conv = nn.Conv1d(in_feat, out_feat, 1)
        self.bn = nn.BatchNorm1d(out_feat)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)

        return F.relu(y)


class OrientDecoder(nn.Module):

    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.conv1 = ConvModule1d(in_feat, 128)
        self.conv2 = ConvModule1d(128, 128)
        self.conv3 = ConvModule1d(128, 128)
        self.conv4 = ConvModule1d(128, 128)
        self.conv5 = ConvModule1d(128, 64)
        self.conv6 = nn.Conv1d(64, out_feat, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = F.normalize(self.conv6(y), dim=1)
        return y


class OccDecoder(nn.Module):

    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.conv1 = ConvModule1d(in_feat, 128)
        self.conv2 = ConvModule1d(128, 128)
        self.conv3 = ConvModule1d(128, 128)
        self.conv4 = ConvModule1d(128, 64)
        self.conv5 = nn.Conv1d(64, out_feat, (1,))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        return y

class OccLinearDecoder(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_feat)
        )

    def forward(self, x):
        y = self.net(x)
        return x


class Occ_attn(nn.Module):
    '''
    view independent feature
    '''

    def __init__(self, in_feat=3, output_dim=2, vit_dim=128, vit_depth=2, vit_heads=8, num_views=4, pt_res=5,
                 with_gt=True, fuse_func='vit', backbone='unet', use_pos=True, use_pt=True):
        super().__init__()
        self.with_gt = with_gt
        self.fuse_func = fuse_func
        self.use_pos = use_pos
        self.use_pt = use_pt

        # self.backbone = Shallow(in_feat=in_feat, kernel=3)
        if backbone == 'unet':
            self.backbone = UNetSimple(in_feat=in_feat, ksize=5)
            print('backbone: {}'.format(backbone))
        elif backbone == 'shallow':
            self.backbone = ShallowEncoder(in_feat=in_feat, ksize=5)
            print('backbone: {}'.format(backbone))
        else:
            print('[error] invalid backbone option')

        print('==> occ attn network info ==>')
        print('img feat dim {}'.format(self.backbone.output_feat))

        self.pt_embed, self.pt_dim = get_embedder(pt_res, input_dims=3)
        print('pt dim {}'.format(self.pt_dim))

        self.occ_vit = OccViT(output_dim=output_dim, token_dim=vit_dim, feat_dim=self.backbone.output_feat, pt_dim=self.pt_dim,
                              depth=vit_depth, heads=vit_heads, mlp_dim=vit_dim, num_views=num_views, dim_head=vit_dim,
                              use_pos=use_pos, use_pt=use_pt, fuse_func=fuse_func)
        print('==> vit info ==>')
        print('vit dim {}'.format(vit_dim))
        print('vit depth {}'.format(vit_depth))
        print('vit heads {}'.format(vit_heads))

        # cross entropy loss
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, data):
        if self.with_gt:
            return self.forward_with_gt(data['imgs'], data['masks'], data['pts_world'], data['pts_view'], data['sample_coord'], data['gt_labels'])
        else:
            return self.forward_raw(data['imgs'], data['masks'], data['pts_world'], data['pts_view'], data['sample_coord'])

    def forward_with_gt(self, imgs, masks, pts_world, pts_view, sample_coord, gt_labels):
        '''

        :param imgs: [V, C_img, H, W]
        :param masks: [V, 1, H, W]
        :param sample_coord: [V, N, 1, 2], projected 2d coordinates of sample points
        :param cams: [V, 7], camera parameters, quaternion (4) + translation (3)
        :param pts: [N, 3],
        :param gt_labels: [N, ]
        :return: cross entropy loss
        '''

        # [V, C_f, N] -> [N, V, C_f]
        img_feat = self.backbone(imgs, masks, sample_coord).permute(2, 0, 1)
        vit_input_data = {
            'img_feat': img_feat,
            'pts_world_feat': self.pt_embed(pts_world),
            'pts_view_feat': self.pt_embed(pts_view)
        }

        # [N, 2]
        cls_result = self.occ_vit(vit_input_data)

        return self.cross_entropy(cls_result, gt_labels), cls_result

    def forward_raw(self, imgs, masks, sample_coord, pts_world, pts_view):
        '''

        :param imgs: [V, C_img, H, W]
        :param masks: [V, 1, H, W]
        :param sample_coord: [V, N, 1, 2], projected 2d coordinates of sample points
        :param cams: [V, 7], camera parameters, quaternion (4) + translation (3)
        :param pts: [N, 3],
        :return:
        '''

        # [V, C_f, N] -> [N, V, C_f]
        img_feat = self.backbone(imgs, masks, sample_coord).permute(2, 0, 1)
        vit_input_data = {
            'img_feat': img_feat
        }

        # [N, 2]
        cls_result = self.occ_vit(vit_input_data)

        return cls_result

    def get_feat(self, imgs):
        return self.backbone.get_feat(imgs)

    def forward_with_feat(self, feats, pts_world, pts_view, masks, sample_coord):
        # masks_feat = F.grid_sample(masks, sample_coord, align_corners=False).squeeze(dim=3)
        sample_feats = torch.cat([F.grid_sample(feat, sample_coord, align_corners=False).squeeze(dim=3) for feat in feats], dim=1)
        img_feat = sample_feats.permute(2, 0, 1)
        vit_input_data = {
            'img_feat': img_feat,
            'pts_world_feat': self.pt_embed(pts_world),
            'pts_view_feat': self.pt_embed(pts_view)
        }

        # [N, 2]
        cls_result = self.occ_vit(vit_input_data)

        return cls_result


class Ori_attn(nn.Module):
    '''
    view dependent feature
    '''

    def __init__(self, in_feat=3, output_dim=3, vit_dim=128, vit_depth=2, vit_heads=8, num_views=4, pt_res=5,
                 with_gt=True, fuse_func='vit', backbone='unet', use_pos=True, use_pt=True):
        super().__init__()
        self.with_gt = with_gt
        self.fuse_func = fuse_func
        self.use_pos = use_pos
        self.use_pt = use_pt

        # self.backbone = Shallow(in_feat=in_feat, kernel=3)
        if backbone == 'unet':
            self.backbone = UNetSimple(in_feat=in_feat, ksize=5)
            print('backbone: {}'.format(backbone))
        elif backbone == 'shallow':
            self.backbone = ShallowEncoder(in_feat=in_feat, ksize=5)
            print('backbone: {}'.format(backbone))
        else:
            print('[error] invalid backbone option')

        print('==> ori attn network info ==>')
        print('img feat dim {}'.format(self.backbone.output_feat))

        self.pt_embed, self.pt_dim = get_embedder(pt_res, input_dims=3)
        print('pt dim {}'.format(self.pt_dim))

        self.vit = OccViT(output_dim=output_dim, token_dim=vit_dim, feat_dim=self.backbone.output_feat, pt_dim=self.pt_dim,
                          depth=vit_depth, heads=vit_heads, mlp_dim=vit_dim, num_views=num_views, dim_head=vit_dim,
                          use_pos=use_pos, use_pt=use_pt, fuse_func=fuse_func)

        print('==> vit info ==>')
        print('vit dim {}'.format(vit_dim))
        print('vit depth {}'.format(vit_depth))
        print('vit heads {}'.format(vit_heads))

    def forward(self, data):
        if self.with_gt:
            return self.forward_with_gt(data['imgs'], data['masks'], data['pts_world'], data['pts_view'], data['sample_coord'], data['gt_targets'])
        else:
            return self.forward_raw(data['imgs'], data['masks'], data['sample_coord'])

    def forward_with_gt(self, imgs, masks, pts_world, pts_view, sample_coord, gt_orients):
        '''

        :param imgs: [V, C_img, H, W]
        :param masks: [V, 1, H, W]
        :param sample_coord: [V, N, 1, 2], projected 2d coordinates of sample points
        :param cams: [V, 7], camera parameters, quaternion (4) + translation (3)
        :param pts: [N, 3],
        :param gt_orients: [N, 3]
        :return: cross entropy loss
        '''

        # [V, C_f, N] -> [N, V, C_f]
        img_feat = self.backbone(imgs, masks, sample_coord).permute(2, 0, 1)

        vit_input_data = {
            'img_feat': img_feat,
            'pts_world_feat': self.pt_embed(pts_world),
            'pts_view_feat': self.pt_embed(pts_view)
        }

        # [N, 3]
        ori_result = F.normalize(self.vit(vit_input_data), dim=1)

        posi_loss = torch.mean(F.l1_loss(ori_result, gt_orients, reduction='none'), dim=1)
        nega_loss = torch.mean(F.l1_loss(ori_result, -gt_orients, reduction='none'), dim=1)

        smaller_loss = torch.min(posi_loss, nega_loss)
        return torch.mean(smaller_loss)
        # return F.l1_loss(ori_result, gt_orients)


    def forward_raw(self, imgs, masks, sample_coord):
        '''

        :param imgs: [V, C_img, H, W]
        :param masks: [V, 1, H, W]
        :param sample_coord: [V, N, 1, 2], projected 2d coordinates of sample points
        :return:
        '''

        # [V, C_f, N] -> [N, V, C_f]
        img_feat = self.backbone(imgs, masks, sample_coord).permute(2, 0, 1)

        vit_input_data = {
            'img_feat': img_feat
        }

        # [N, 3]
        ori_result = F.normalize(self.vit(vit_input_data), dim=1)

        return ori_result

    def get_feat(self, imgs):
        return self.backbone.get_feat(imgs)

    def forward_with_feat(self, feats, pts_world, pts_view, masks, sample_coord):
        # masks_feat = F.grid_sample(masks, sample_coord, align_corners=False).squeeze(dim=3)
        sample_feats = torch.cat([F.grid_sample(feat, sample_coord, align_corners=False).squeeze(dim=3) for feat in feats], dim=1)
        img_feat = sample_feats.permute(2, 0, 1)
        vit_input_data = {
            'img_feat': img_feat,
            'pts_world_feat': self.pt_embed(pts_world),
            'pts_view_feat': self.pt_embed(pts_view)
        }
        # [N, 3]
        ori_result = F.normalize(self.vit(vit_input_data), dim=1)

        return ori_result


class SRF(nn.Module):

    def __init__(self, train=True):
        super(SRF, self).__init__()
        self.is_train = train
        self.BackBone = Shallow(in_feat=3)
        self.featMatchConv1d = nn.Conv1d(371, 128, kernel_size=2, stride=2)
        self.featAggrConv1d_1 = nn.Conv1d(128, 128, kernel_size=4)
        self.featAggrConv1d_2 = nn.Conv1d(128, 128, kernel_size=4)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.decoder = OrientDecoder(371 + 128, 3)

    def getPairFeatures(self, feature):
        '''
        combine image features in pairs, order matters
        :param feature: [N, V, C_f], V = num of views, N = num of points
        :return: [N, C_f, 2 * V * (V - 1)], paired image features
        '''
        # V
        num_view = feature.shape[1]
        # [0, 1, ..., V - 1]
        index = torch.arange(num_view)
        # pair indices
        pairs = torch.combinations(index)
        # inverse pair indices
        pairs_inv = torch.flip(pairs, [1])
        # combine pair and inverse pair, then flatten it to a index list
        indices = torch.cat([pairs, pairs_inv], dim=0).reshape(-1).to(feature.device)
        # [N, 2 * V * (V - 1), C_f]
        pair_feat = torch.index_select(feature, 1, indices)
        # [N, C_f, 2 * V * (V - 1)]
        pair_feat = torch.transpose(pair_feat, 1, 2)

        return pair_feat

    def DotProductLoss(self, pred, target):
        '''
        dot product loss = 1 - V_pred \cdot V_target
        :param pred: [N, 3, V]
        :param target: [N, 3, V]
        :return: mean dot product loss
        '''
        dot_product = torch.sum(pred * target, dim = 1)
        dp_loss = dot_product * (-1) + 1
        mean_dp_loss = torch.mean(dp_loss)
        return mean_dp_loss


    def forward(self, img, sample_coord, gt_orients):
        '''

        :param img: [V, C, H, W]
        :param sample_coord: [V, N, 2]
        :param gt_orients: [N, 3, V]
        :return:
        '''

        # [V, C_f, N]
        feat = self.BackBone(img, sample_coord)
        # [V, N, C_f]
        feat_t = feat.transpose(1, 2)
        # [N, V, C_f]
        feat_t = feat_t.transpose(0, 1)
        # [N, C_f, 2 * S], S = V * (V - 1)
        pair_feat = self.getPairFeatures(feat_t)
        # [N, K, S]
        stereo_feat = self.featMatchConv1d(pair_feat)
        # [N, K, S - 3]
        aggr_feat = self.featAggrConv1d_1(stereo_feat)
        # [N, K, S - 6]
        aggr_feat = self.featAggrConv1d_2(aggr_feat)
        # [N, K, 1]
        global_encode = self.max_pool(aggr_feat)
        # [N, K, V]
        global_encode_expand = global_encode.expand(-1, -1, img.shape[0])
        # [N, C_f, V]
        view_encode = feat_t.transpose(1, 2)
        # [N, K + C_f, V]
        final_encode = torch.cat([global_encode_expand, view_encode], dim=1)
        # [N, 3, V], view-dependent orientation
        orient_prediction = self.decoder(final_encode)

        if self.is_train:
            # loss = self.DotProductLoss(orient_prediction, gt_orients)
            loss = F.mse_loss(orient_prediction, gt_orients)
            return loss
        else:
            return orient_prediction


