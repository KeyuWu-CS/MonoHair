import torch
from datasets import get_dataset
import numpy as np
from torch.utils.data import DataLoader
from models import get_model
import math
from tqdm import tqdm
from util import getProjPoints, check_pts_in_views,save_ori_mat,generate_headtrans_from_tsfm
import os
import torch.nn.functional as F
import configargparse
@torch.no_grad()
def deep_mvs_eval(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    eval_ds_ops = {
        'root_folder': args.root_folder,
        'calib_folder': args.calib_folder,
        'device': device,
        'data_type': 'real',
        'num_views': args.ds_num_views,
        'slct_vids': args.slct_vids,
        'sample_style': args.sample_style,
        'use_hair_depth': args.use_hair_depth,
        'cat_depth': args.cat_depth,
        'use_colmap_points':args.use_colmap_points,
        'use_dir':args.use_dir,
        'case':args.case
    }

    eval_dataset = get_dataset(args.mode, eval_ds_ops)
    eval_dataset = torch.utils.data.Subset(eval_dataset, [i for i in range(len(eval_dataset))])

    # save voxels information
    grid_resolution = eval_dataset.dataset.grid_resolution
    voxel_min = eval_dataset.dataset.voxel_min
    vsize = eval_dataset.dataset.vsize
    voxels_save_info = np.concatenate([grid_resolution, voxel_min, np.array([vsize], dtype='float32')], axis=0)
    print('voxels save info {} {}'.format(voxels_save_info, voxels_save_info.dtype))

    print('eval dataset {}'.format(len(eval_dataset)))
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=args.eval_bs,
                                 shuffle=False,
                                 num_workers=0)
    eval_dataset.dataset.set_case(args.case)

    in_feat = 3
    if args.cat_depth:
        in_feat = 4

    occ_ops = {
        'in_feat': in_feat,
        'output_dim': 2,
        'vit_dim': args.occ_vit_dim,
        'vit_depth': args.occ_vit_depth,
        'vit_heads': args.occ_vit_heads,
        'num_views': args.md_num_views,
        'pt_res': args.occ_pts_embed_res,
        'with_gt': False,
        'backbone': args.backbone,
        'fuse_func': args.fuse_func,
        'use_pos': args.use_pos,
        'use_pt': args.use_pt
    }

    occ_model = get_model('occ', occ_ops).to(device)
    occ_model.load_state_dict(torch.load(args.occ_model_path, map_location=device)['model_state_dict'], strict=True)

    ori_ops = {
        'in_feat': in_feat,
        'output_dim': 3,
        'vit_dim': args.ori_vit_dim,
        'vit_depth': args.ori_vit_depth,
        'vit_heads': args.ori_vit_heads,
        'num_views': args.md_num_views,
        'pt_res': args.ori_pts_embed_res,
        'with_gt': False,
        'backbone': args.backbone,
        'fuse_func': args.fuse_func,
        'use_pos': args.use_pos,
        'use_pt': args.use_pt
    }

    ori_model = get_model('ori', ori_ops).to(device)
    ori_model.load_state_dict(torch.load(args.ori_model_path, map_location=device)['model_state_dict'], strict=True)

    occ_model.eval()
    ori_model.eval()

    with torch.no_grad():
        cnt = 0
        for index, item in enumerate(eval_dataloader):


            print('==> item {} / {}'.format(cnt + 1, len(eval_dataloader)))

            item_id = item['item_id'][0]
            print('=> id {}'.format(item_id))
            if args.case is not None:
                if item_id!=args.case:
                    continue
            orient_map = item['orient_map'][0]
            conf_map = item['conf_map'][0]
            depth = item['depth_map'][0]
            # imgs = torch.cat([orient_map, conf_map, depth], dim=1)
            imgs = torch.cat([orient_map, depth], dim=1)
            masks = item['masks'][0]

            model_tsfm = item['model_tsfm'][0]
            item['colmap_points']= item['colmap_points'][0]
            # count the last batch (possibly incomplete), in order to test all sample points
            # batches_per_item = int(math.ceil(eval_dataset.dataset.samples.shape[1] / args.eval_pts_per_batch))

            batches_per_item = int(math.ceil(item['colmap_points'].shape[1] / args.eval_pts_per_batch))

            # only filter points out of masks
            raw_pts = []
            positive_points = []
            positive_ori = []
            orients = []
            occ = []

            occ_img_feats = occ_model.get_feat(imgs)
            ori_img_feats = ori_model.get_feat(imgs)

            print('colmap_points:',item['colmap_points'].shape[1])
            print('sample step:',item['num_sample']//args.eval_pts_per_batch)
            count = 1
            for i in tqdm(range(batches_per_item), ascii=True):
                low_bound = args.eval_pts_per_batch * i
                high_bound = args.eval_pts_per_batch * (i + 1)

                # points in normalized space, origin at bust center
                # ns_pts_batch = eval_dataset.dataset.samples[:, low_bound:high_bound].clone()
                ns_pts_batch = item['colmap_points'][:, low_bound:high_bound].clone()
                # [4, N]
                # pts_world_batch = model_tsfm @ ns_pts_batch
                pts_world_batch = ns_pts_batch


                # [V, 4, N]
                pts_view_batch = eval_dataset.dataset.cam_poses_w2c @ pts_world_batch
                # [V, N, 1, 2]
                xy_coords_batch = getProjPoints(pts_world_batch, eval_dataset.dataset.cam_poses_w2c,
                                                eval_dataset.dataset.ndc_proj)
                visible = check_pts_in_views(xy_coords_batch, sample_range="any")
                #visible = check_pts_in_views(xy_coords_batch, sample_range=args.sample_range)

                if visible.sum() == 0:
                    count+=1
                    print(True)
                    print(count)
                    continue

                # [V, N, 1, 2]
                # xy_coords_batch = xy_coords_batch[:, visible]
                xy_coords_batch = xy_coords_batch
                # [N, 1, 3]
                # pts_world_batch = pts_world_batch[:3].transpose(0, 1)[visible].unsqueeze(1)
                pts_world_batch = pts_world_batch[:3].transpose(0, 1).unsqueeze(1)
                # [N, V, 3]
                # pts_view_batch = pts_view_batch[:, :3, :].permute(2, 0, 1)[visible]
                pts_view_batch = pts_view_batch[:, :3, :].permute(2, 0, 1)
                # [4, N]
                # ns_pts_batch = ns_pts_batch[:, visible]

                # predict positive occupancy points
                pred_occ = occ_model.forward_with_feat(occ_img_feats, pts_world_batch, pts_view_batch, masks,
                                                       xy_coords_batch)
                bin_pred_occ = torch.gt(pred_occ[:, 1], pred_occ[:, 0])
                if i < item['num_sample'] // args.eval_pts_per_batch:
                    if torch.sum(bin_pred_occ)==0:
                        continue


                pred_ori = ori_model.forward_with_feat(ori_img_feats, pts_world_batch, pts_view_batch, masks,
                                                       xy_coords_batch)

                # extract positive samples
                ns_pts_valid = torch.linalg.inv(model_tsfm) @ ns_pts_batch   ###[4,N]
                ns_pts_valid = ns_pts_valid[:3, bin_pred_occ].transpose(0, 1)

                positive_points.append(ns_pts_valid)
                positive_local_ori = pred_ori[bin_pred_occ]
                positive_local_ori = torch.inverse(model_tsfm)[:3, :3] @ positive_local_ori.transpose(0, 1)
                positive_ori.append(positive_local_ori.transpose(0,1))

                if i< (item['num_sample']//args.eval_pts_per_batch)+1:
                    orients.append(pred_ori[bin_pred_occ])
                    occ.append(bin_pred_occ[bin_pred_occ])
                    raw_pts.append(ns_pts_batch[:3,bin_pred_occ].transpose(0, 1))
                else:
                    orients.append(pred_ori)
                    occ.append(bin_pred_occ)
                    raw_pts.append(ns_pts_batch[:3].transpose(0, 1))

            positive_points = torch.cat(positive_points, dim=0)
            positive_ori = F.normalize(torch.cat(positive_ori, dim=0), dim=1)
            orients = F.normalize(torch.cat(orients, dim=0), dim=1)
            raw_pts = torch.cat(raw_pts, dim=0)
            print("raw",raw_pts.size())
            occ = torch.cat(occ, 0)
            save_points = positive_points.cpu().numpy()
            save_orients = positive_ori.cpu().numpy()
            orients = orients.cpu().numpy()
            raw_pts = raw_pts.cpu().numpy()
            occ = occ.cpu().numpy()

            print('positive points cnt {}'.format(len(save_points)))

            # num_p = eval_dataset.dataset.samples.shape[1]
            # num_p = item['num_sample']
            # sample_pts = raw_pts[:num_p]
            # sample_ori = orients[:num_p]
            # sample_occ =occ[:num_p]
            # colmap_pts = raw_pts[num_p:]
            # colmap_ori = orients[num_p:]
            # colmap_occ = occ[num_p:]

            save_data = np.concatenate([save_points, save_orients], axis=1).reshape(-1)
            save_data_w_cnt = np.concatenate([voxels_save_info, [np.float32(len(save_points))], save_data], axis=0)

            save_folder_path = os.path.join(args.root_folder, item_id, args.save_folder)
            os.makedirs(save_folder_path,exist_ok=True)
            save_data_w_cnt.tofile(os.path.join(save_folder_path, args.save_vname))

            mat_save_root = os.path.join(save_folder_path, 'Voxel_hair')
            print(mat_save_root)
            os.makedirs(mat_save_root,exist_ok=True)
            save_ori_mat(save_points, save_orients, voxel_min, vsize/2, grid_resolution*2, mat_save_root,
                         model_tsfm.cpu().numpy())

            generate_headtrans_from_tsfm(item['model_tsfm_semantic_path'][0], os.path.join(mat_save_root, 'head.trans'))


            # print('num sample:',sample_occ.shape[0])
            # print(np.sum(sample_occ))

            # sample_data = np.concatenate([sample_pts[sample_occ],sample_ori[sample_occ],sample_occ[sample_occ,None].astype(np.float32)],1)

            # colmap_data = np.concatenate([colmap_pts,colmap_ori,colmap_occ[:,None].astype(np.float32)],1)
            # print('num colmap:', colmap_data.shape[0])
            # print('pos num sample:',sample_data.shape[0])
            # raw_data = np.concatenate([sample_data,colmap_data],0)


            raw_data = np.concatenate([raw_pts,orients,occ[:,None].astype(np.float32)],1)
            print('raw data:',raw_data.shape[:])
            print(os.path.join(save_folder_path,'raw.npy'))
            np.save(os.path.join(save_folder_path,'raw.npy'),raw_data)

            cnt += 1



def config_ds_real_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    parser.add_argument('--root_folder', type=str,
                        help='the folder contains items')
    parser.add_argument('--calib_folder', type=str,
                        help='the folder contains calibrated camera data')
    parser.add_argument('--data_type', type=str,
                        help='union / synthetic / real')

    parser.add_argument('--mode', type=str, default='eval',
                        help='evaluation mode')

    parser.add_argument('--sample_style', type=str, default='corner',
                        help='corner / center')
    parser.add_argument('--save_folder', type=str, default='ours',
                        help='saved file name of voxels')
    parser.add_argument('--save_vname', type=str, default='voxels.dat',
                        help='saved file name of voxels')
    parser.add_argument('--in_mask_vids', nargs='+', type=int, default=[],
                        help='indices of views used in mask check')
    parser.add_argument("--use_hair_depth", action='store_true',default=False,
                        help='if true, use hair depth')
    parser.add_argument("--cat_depth", action='store_true',default=False,
                        help='if true, separate hair depth and bust depth')
    parser.add_argument("--use_dir", action='store_true',default=False,
                        help='if true, separate hair depth and bust depth')

    parser.add_argument('--ds_num_views', type=int, default=4,
                        help='number of total views contained in data cases')
    parser.add_argument('--slct_vids', nargs='+', type=int, default=None,
                        help='select a subset of views')
    parser.add_argument('--md_num_views', type=int, default=4,
                        help='number of views input to model')
    parser.add_argument('--backbone', type=str, default='unet',
                        help='backbone feature extraction network')

    # occupancy model params
    # parser.add_argument('--occ_cam_embed_res', type=int, default=2,
    #                     help='embedding resolution of camera parameters')
    parser.add_argument('--occ_pts_embed_res', type=int, default=2,
                        help='embedding resolution of points coordinates')
    parser.add_argument('--occ_vit_dim', type=int, default=128,
                        help='internal vector dimensions of vision transformer')
    parser.add_argument('--occ_vit_depth', type=int, default=2,
                        help='depth of vision transformer layers')
    parser.add_argument('--occ_vit_heads', type=int, default=4,
                        help='number of heads in attention module of vision transformer')

    parser.add_argument('--fuse_func', type=str, default='vit',
                        help='avg / mlp / vit')
    parser.add_argument("--use_pos", action='store_true', default=False,
                        help='ablation study on vit, w.o. position embedding')
    parser.add_argument("--use_pt", action='store_true', default=False,
                        help='ablation study on vit, w.o. points embedding')
    # parser.add_argument('--num_views', type=int, default=8,
    #                     help='number of fixed views')
    parser.add_argument('--sample_range', type=str, default='all',
                        help='any / all')

    # orientation model params
    # parser.add_argument('--ori_cam_embed_res', type=int, default=2,
    #                     help='embedding resolution of camera parameters')
    parser.add_argument('--ori_pts_embed_res', type=int, default=2,
                        help='embedding resolution of points coordinates')
    parser.add_argument('--ori_vit_dim', type=int, default=128,
                        help='internal vector dimensions of vision transformer')
    parser.add_argument('--ori_vit_depth', type=int, default=2,
                        help='depth of vision transformer layers')
    parser.add_argument('--ori_vit_heads', type=int, default=4,
                        help='number of heads in attention module of vision transformer')

    # load trained model, for evaluation
    parser.add_argument('--occ_model_path', type=str, default=None,
                        help='path of trained occupancy model')
    parser.add_argument('--ori_model_path', type=str, default=None,
                        help='path of trained orientation model')

    # dataset
    parser.add_argument('--eval_bs', type=int, default=1,
                        help='batch size in eval dataset')
    parser.add_argument('--eval_pts_per_batch', type=int, default=2048,
                        help='points per batch in eval dataset')

    # result saving
    # parser.add_argument('--result_save_folder', type=str,
    #                     help='the folder to save point cloud')

    return parser

if __name__ == '__main__':
    pass