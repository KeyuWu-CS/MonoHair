import os
from Utils.Utils import save_hair_strands
from HairGrow import HairGrowing

from Utils.Render_utils import render_data
from Utils.Camera_utils import load_cam,parsing_camera
from Utils.Utils import load_strand,load_bust
import numpy as np
from log import log
import options
import sys
import torch
def get_config():


    log.process(os.getpid())

    opt_cmd = options.parse_arguments(sys.argv[1:])
    args = options.set(opt_cmd=opt_cmd)
    args.output_path = os.path.join(args.data.root, args.data.case,args.output_root,args.name)
    # os.makedirs(args.output_path, exist_ok=True)
    # options.save_options_file(args)
    args.data.root = os.path.join(args.data.root, args.data.case)
    args.segment.scene_path = args.data.root



    return args

if __name__ == '__main__':

    args = get_config()


    case = args.data.case
    camera_path = args.camera_path
    root = args.data.root
    os.makedirs(os.path.join(root,'ours'),exist_ok=True)

    if args.infer_inner.render_data:
        args.save_path = os.path.join(args.output_path, 'refine')
        args.data.Occ3D_path = os.path.join(args.save_path, 'Occ3D.mat')
        args.data.Ori3D_path = os.path.join(args.save_path, 'Ori3D.mat')

        HairGrowSolver = HairGrowing(args.data.Occ3D_path ,args.data.Ori3D_path ,args.device ,args.data.image_size)
        strands = HairGrowSolver.randomlyGenerateSegments(args.HairGenerate.grow_threshold)
        strands = HairGrowSolver.VoxelToWorld(strands.copy())

        args.bust_to_origin = np.array(args.bust_to_origin)
        save_hair_strands(os.path.join(args.output_path ,'refine', 'render_segments.hair') ,strands)




        # segments, points = load_strand(os.path.join(args.output_path ,'refine', 'render_segments.hair'))
        # strands = []
        # beg = 0
        # for seg in segments:
        #     end = beg + seg
        #     strand = points[beg:end]
        #
        #     strand += args.bust_to_origin
        #     strands.append(strand)
        #     beg += seg
        vertices, faces, _ = load_bust(os.path.join(args.data.root ,args.data.bust_path))
        vertices += np.array(args.bust_to_origin)

        camera = load_cam(args.camera_path)
        image_path = os.path.join(args.data.root, 'trainning_images/capture_images')
        camera = parsing_camera(camera, image_path)
        render_data(camera ,strands ,vertices ,faces ,[1280 ,720] ,os.path.join(args.data.root ,'imgs'))



    if args.infer_inner.run_mvs:
        print('infer inner...')

        sys.path.append('submodules/DeepMVSHair')
        from mvs_eval import config_ds_real_parser ,deep_mvs_eval
        ### 12. run deepMVSHair

        mvs_args = config_ds_real_parser()
        mvs_args.set_defaults(config='configs/dense_sample_real/vit_standard.txt')
        mvs_args.add_argument('--use_colmap_points' ,action='store_true' ,default=False)
        mvs_args.add_argument('--case' ,type=str ,default=args.data.case)
        mvs_args =mvs_args.parse_known_args()[0]
        deep_mvs_eval(mvs_args)
        torch.cuda.empty_cache()
        cmd = 'python PMVO.py --yaml=configs/reconstruct/{} --PMVO.infer_inner --PMVO.optimize='.format(case)  ### if out of memory, run this code independent
        os.system(cmd)

