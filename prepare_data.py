import os
import platform
import cv2
from log import log
import options
import sys
from preprocess_capture_data.calc_masks import calculate_mask
from preprocess_capture_data.GaborFilter import batch_generate
from Utils.ingp_utils import generate_ngp_posefrom_cam_params,generate_mvs_pose_from_base_cam, convert_ngp_to_nerf,convert_mesh_to_mvs
import shutil
import trimesh
from Utils.Utils import transform_bust,generate_headtrans_from_tsfm,generate_bust



from Utils.Render_utils import render_bust_hair_depth


def get_config():


    log.process(os.getpid())

    opt_cmd = options.parse_arguments(sys.argv[1:])
    args = options.set(opt_cmd=opt_cmd)
    args.output_path = os.path.join(args.data.root, args.data.case,args.output_root,args.name)
    os.makedirs(args.output_path, exist_ok=True)
    options.save_options_file(args)
    args.data.root = os.path.join(args.data.root, args.data.case)
    args.segment.scene_path = args.data.root



    return args

if __name__ == '__main__':

    args = get_config()


    case = args.data.case
    camera_path = args.camera_path
    root = args.data.root
    os.makedirs(os.path.join(root,'ours'),exist_ok=True)

    #### 0. run colmap and colmap2nerf.py


    #### 1. drag imgs to instant-ngp.exe

    #### 2. add key frame using instant-ngp.exe(to do: using one of the front image as a key frame) generate "key_frame.json"



    ##### select about 150 images
    if args.prepare_data.select_images:
        raw_root = os.path.join(args.data.root,'colmap/images')
        files = os.listdir(raw_root)
        files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

        os.makedirs(args.data.root+'/capture_images',exist_ok=True)
        max_sharpless = 0
        for i,file in enumerate(files):
            frame = cv2.imread(os.path.join(raw_root, file))
            img2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
            if imageVar > max_sharpless:
                max_file = file
                max_sharpless = imageVar
            if (i + 1) % args.data.frame_interval == 0:  #### set frame_interval, let the num of images around 150 (100-200 is also ok)
                max_sharpless = 0
                shutil.copyfile(os.path.join(raw_root, max_file), os.path.join(root, 'capture_images', max_file))


    if args.prepare_data.process_camera:
        #### 3. generate 16 fixed camera pose      generate "base_cam.json"
        base_cam_save_path = os.path.join(root,'colmap','base_cam.json')
        generate_ngp_posefrom_cam_params(os.path.join(root,'colmap'),camera_path,base_cam_save_path)

        #### 4. generate pose for each capture images    generate "cam_params.json"
        select_files = []
        files = os.listdir(os.path.join(root,'capture_images'))
        for i, file in enumerate(files):
            select_files.append(file[:-4])
        data_folder = os.path.join(root, 'colmap')
        generate_mvs_pose_from_base_cam(data_folder, select_files,camera_path, image_size=args.data.image_size)
        shutil.copyfile(os.path.join(root,'colmap/cam_params.json'), os.path.join(root,'ours','cam_params.json'))


    if args.prepare_data.run_ngp:
        #### 5. render trainning images   generate "base_transform.json" "base.obj" and render imgs
        base_cam_path = os.path.join(root,'colmap/base_cam.json')
        save_path = os.path.join(root,'colmap/base_transform.json')
        convert_ngp_to_nerf(base_cam_path,save_path,image_size=[1920,1080])
        scene_path = os.path.join(root,'colmap')
        load_snapshot = os.path.join(root,'colmap/base.ingp')
        screenshot_transforms = os.path.join(root,'colmap/base_transform.json')
        screenshot_dir = os.path.join(root,'trainning_images/capture_images')
        os.makedirs(screenshot_dir,exist_ok=True)
        save_mesh_path = os.path.join(root,'colmap/base.obj')
        # cmd = 'python E:/wukeyu/Instant-NGP/instant-ngp-new/instant-ngp/scripts/run.py  --scene={}'.format(scene_path) + ' ' + \

        cmd = 'python submodules/instant-ngp/scripts/run.py  --scene={}'.format(scene_path) +' '+\
              '--load_snapshot={}'.format(load_snapshot)+ ' ' +\
              '--screenshot_transforms={}'.format(screenshot_transforms)+ ' '+ \
              '--screenshot_dir={}'.format(screenshot_dir)+ ' '+ \
              '--save_mesh={}'.format(save_mesh_path)+ ' '+\
              '--fov_axis 1'+' '+\
              '--marching_cubes_density_thresh {}'.format(args.ngp.marching_cubes_density_thresh)
        os.system(cmd)
        files = os.listdir(screenshot_dir)
        for file in files:
            os.makedirs(os.path.join(root,'imgs',file[:-4]),exist_ok=True)
            shutil.copyfile(os.path.join(screenshot_dir,file),os.path.join(root,'imgs',file[:-4],'origin.png'))

        ####  convert nerf mesh to mvs  generate "colmap_points.obj"
        colmap_points_root = os.path.join(root,'colmap')
        colmap_points_save_path = os.path.join(root,'ours/colmap_points.obj')
        convert_mesh_to_mvs(colmap_points_root,camera_path,colmap_points_save_path)



    if args.prepare_data.fit_bust:
        print('fiting ...')
        cmd = 'python multiview_optimization.py  --yaml=configs/Bust_fit/{} '.format(case)
        os.system(cmd)
        shutil.copyfile(os.path.join(args.data.root,'optimize','model_tsfm.dat'),os.path.join(args.data.root,'model_tsfm.dat'))
        shutil.copyfile(os.path.join(args.data.root,'optimize','model_tsfm_semantic.dat'),os.path.join(args.data.root,'model_tsfm_semantic.dat'))
        Bust_root = os.path.join(args.data.root,'Bust')
        os.makedirs(Bust_root,exist_ok=True)
        shutil.copyfile(os.path.join(args.data.root,'optimize/vis','final_template.obj'),os.path.join(Bust_root,'final_template.obj'))
        shutil.copyfile(os.path.join(args.data.root,'optimize/vis','final_template_ori.obj'),os.path.join(Bust_root,'final_template_ori.obj'))

        flame_template_path = 'assets/data/head_template.obj'
        smplx_source_mesh = trimesh.load(os.path.join(Bust_root,'final_template.obj'))
        smplx_template_mesh = trimesh.load(os.path.join(Bust_root, 'final_template_ori.obj'))
        generate_bust(smplx_source_mesh,smplx_template_mesh,'assets/data/scalp_mask.png',flame_template_path,'assets/data/SMPL-X__FLAME_vertex_ids.npy',Bust_root)

    if args.prepare_data.process_bust:
        #### 9. bust transform

        head_path = os.path.join(root,'Bust','bust_long.obj')
        head_mesh = trimesh.load(head_path)
        flame_template_path = 'assets/data/head_template.obj'
        scalp_texture_path = 'assets/data/scalp_mask.png'
        scalp_save_path = os.path.join(root,'Bust','scalp.obj')

        os.makedirs(os.path.join(root,'ours/Voxel_hair'),exist_ok=True)
        shutil.copyfile(os.path.join(root,'Bust','bust_long.obj'),os.path.join(root,'ours/Voxel_hair','bust_long.obj'))
        shutil.copyfile(os.path.join(root,'Bust','scalp.obj'),os.path.join(root,'ours/Voxel_hair','scalp.obj'))
        shutil.copyfile(os.path.join(root,'Bust','flame_bust.obj'),os.path.join(root,'ours/Voxel_hair','flame_bust.obj'))
        shutil.copyfile(os.path.join(root,'model_tsfm.dat'),os.path.join(root,'ours/Voxel_hair','model_tsfm.dat'))
        transform_bust(os.path.join(root,'ours/Voxel_hair','bust_long.obj'),os.path.join(root,'ours/Voxel_hair','model_tsfm.dat'),os.path.join(root,'ours/bust_long_tsfm.obj'))
        transform_bust(os.path.join(root,'ours/Voxel_hair','scalp.obj'),os.path.join(root,'ours/Voxel_hair','model_tsfm.dat'),os.path.join(root,'ours/scalp_tsfm.obj'))
        transform_bust(os.path.join(root,'ours/Voxel_hair','flame_bust.obj'),os.path.join(root,'ours/Voxel_hair','model_tsfm.dat'),os.path.join(root,'ours/flame_bust_tsfm.obj'))
        generate_headtrans_from_tsfm(os.path.join(root,'model_tsfm_semantic.dat'),os.path.join(root,'ours/Voxel_hair/head.trans'))

    if args.prepare_data.render_depth:
        #### 8. generate bust_hair_depth  generate "bust_hair_depth.png"
        save_root = os.path.join(root,'imgs')
        bust_path = os.path.join(root, 'ours/bust_long_tsfm.obj')
        if platform.system()=='Windows':
            Headless =False
        else:
            Headless = True
        render_bust_hair_depth(os.path.join(root,'ours/colmap_points.obj'), camera_path, save_root,bust_path=bust_path,Headless=Headless)
        save_root = os.path.join(root, 'render_depth')
        os.makedirs(save_root,exist_ok=True)
        capture_img_cam_path = os.path.join(root,'ours','cam_params.json')
        bust_path = os.path.join(root,'ours/bust_long_tsfm.obj')
        bust_path = None

        render_bust_hair_depth(os.path.join(root,'ours/colmap_points.obj'), capture_img_cam_path, save_root,image_size=args.data.image_size,capture_imgs=True,bust_path=bust_path,Headless=Headless)


    if args.prepare_data.process_imgs:
        ### 10. compute mask orientation and confidence for capture images
        segment_args = args.segment
        calculate_mask(segment_args)
        image_folder = 'capture_images'
        batch_generate(root, image_folder)