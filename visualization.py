from Utils.visual_utils import *
import os
import trimesh

if __name__ == '__main__':

    root = r'E:\wukeyu\hair\project\MonoHair_release\data\ksyusha1'






    bust = trimesh.load(os.path.join(root,'Bust/bust_long_tsfm.obj'))

    vertices = np.asarray(bust.vertices)
    vertices+= np.array([0.006, -1.644, 0.010])
    faces = np.asarray(bust.faces)
    vis_bust = vis_mesh(vertices, faces)

    root = root+r'\output\10-16'
    surface = np.load(root+'/optimize/surface.npy')
    filter = np.load(root+'/optimize/filter_unvisible.npy')
    surface_pcd = vis_point_colud(surface[:,:3])
    filter_pcd = vis_point_colud(filter[:,:3])
    draw_scene([vis_bust,surface_pcd])
    draw_scene([filter_pcd,surface_pcd])

    select_points = np.load(os.path.join(root, 'refine','select_p.npy'))
    select_ori = np.load(os.path.join(root, 'refine','select_o.npy'))
    min_loss = np.load(os.path.join(root, 'refine','min_loss.npy'))

    filter_unvisible_points = np.load(os.path.join(root + '/refine', 'filter_unvisible.npy'))
    filter_unvisible_ori = np.load(os.path.join(root + '/refine', 'filter_unvisible_ori.npy'))
    up_index = filter_unvisible_ori[:, 1] > 0
    filter_unvisible_ori[up_index] *= -1

    index = np.where(min_loss <= 0.01)[0]
    select_ori = select_ori[index]
    select_points = select_points[index]
    print('num select', select_ori.shape[:])
    reverse_index = select_ori[:, 1] > 0
    select_ori[reverse_index] *= -1
    select_ori_visual = np.abs(select_ori)
    # select_ori_visual = (select_ori+1)/2
    vis_points = vis_point_colud(select_points, select_ori_visual)

    vis_norm = vis_normals(select_points, select_ori * 0.004, select_ori_visual)
    draw_scene([vis_points, vis_norm, vis_bust])
    draw_scene([vis_points, vis_norm])

    # select_points = np.concatenate([select_points, filter_unvisible_points], 0)
    # select_ori = np.concatenate([select_ori, filter_unvisible_ori], 0)
    select_points = filter_unvisible_points
    select_ori = filter_unvisible_ori


    reverse_index = select_ori[:, 1] > 0
    select_ori[reverse_index] *= -1

    # select_ori_visual = (select_ori + 1) / 2
    select_ori_visual = np.abs(select_ori)

    vis_fuse = vis_point_colud(select_points, select_ori_visual)
    vis_fuse_norm = vis_normals(select_points, select_ori * 0.004, select_ori_visual)
    draw_scene([vis_fuse, vis_fuse_norm])