import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import math
import os
import open3d as o3d

def nerf_to_ngp(xf, convert_quat=True):
	mat = np.copy(xf)
	mat = mat[:-1, :]
	mat[:, 1] *= -1  # flip axis
	mat[:, 2] *= -1
	mat[:, 3] *= 0.33  # scale
	mat[:, 3] += [0.5, 0.5, 0.5]  # offset

	mat = mat[[1, 2, 0], :]  # swap axis

	if convert_quat:
		rm = R.from_matrix(mat[:, :3])
		return rm.as_quat(), mat[:, 3]
	return mat[:3, :3], mat[:, 3]


# def ngp_to_nerf(cam_matrix):
# 	cam_matrix = cam_matrix[[2, 0, 1], :]  # flip axis (yzx->xyz)
# 	cam_matrix[:, 3] -= 0.5  # reversing offset
# 	cam_matrix[:, 3] /= 0.33  # reversing scale
# 	cam_matrix[:, 1] /= -1  # flipping y axis
# 	cam_matrix[:, 2] /= -1  # z flipping
#
# 	return cam_matrix


def ngp_to_nerf(R,T):
	mat = np.eye(4)
	mat[:3,:3]=R
	mat[:3,3]=T
	mat = mat[[2,0,1,3],:]
	mat[:3,3]-=[0.5, 0.5, 0.5]
	mat[:3,3]/=0.33
	mat[:,2]*=-1
	mat[:,1]*=-1
	return mat

def quat2mat(Rq):
	Rm = R.from_quat(Rq)
	rotation_matrix = Rm.as_matrix()
	return rotation_matrix


def mat2quat(mat):
	rm = R.from_matrix(mat)
	return rm.as_quat()


def load_transofrm_json(path):
	with open(path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	camera_angle_y = data["camera_angle_y"]
	fov = camera_angle_y * 180 / math.pi
	n_frames = len(data['frames'])
	xforms = {}
	for i in range(n_frames):
		file = data['frames'][i]['file_path'].split('/')[-1][:-4]
		xform = data['frames'][i]['transform_matrix']
		xforms[file] = xform
	xforms = dict(sorted(xforms.items()))

	return xforms, fov


def load_cam_params(path):
	w, h = (1080, 1920)
	scale = 2 / 3
	Rotation = []
	Translate = []
	fovs_x = []
	fovs_y = []
	intrin_op = []
	ndc_prj = []
	with open(path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	items = data['cam_list']
	for item in items:
		pose = np.array(item['pose'])
		Rotation.append(pose[:3, :3])
		Translate.append(pose[:3, 3])
		fovs_x.append((math.atan(w / (item['intrin_op'][0] / scale * 2)) * 2) * 180 / math.pi)

		fovs_y.append((math.atan(h / (item['intrin_op'][1] / scale * 2)) * 2) * 180 / math.pi)
		intrin_op.append(item['intrin_op'])
		ndc_prj.append(item['ndc_prj'])
	return Rotation, Translate, fovs_x, fovs_y


def convert_ngp_to_nerf(base_cam_path,save_path,image_size=[1080,1920]):
	Rotation, Trans, fovs = load_base_cam(base_cam_path,True)
	mat_all = []
	for q,t,fov in zip(Rotation,Trans,fovs):
		R = quat2mat(q)
		mat = ngp_to_nerf(R,t)
		mat_all.append(mat)
	save_transform_json(mat_all,fovs[0],save_path,image_size)


def save_transform_json(matrix, fov,save_path,image_size):
	camera_angle_x = fov * np.pi / 180
	out = {"camera_angle_x": camera_angle_x, "is_fisheye": False, "cx": image_size[1]//2, "cy":image_size[0]//2 , "w": image_size[1], "h": image_size[0]}
	frame = []
	for i, mat in enumerate(matrix):
		frame.append(

			{
				"file_path": "%03d" % i,
				"transform_matrix": mat.tolist()
			}
		)

	out["frame"] = frame
	with open(save_path, 'w')as outfile:
		json.dump(out, outfile, indent=2)



def load_base_cam(path,reture_fov=False):
	with open(path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	num_view = len(data['path'])
	Rotation = []
	Trans = []
	fovs = []
	for i in range(num_view):
		Rq = data['path'][i]['R']
		Rotation.append(Rq)
		Trans.append(np.array(data['path'][i]['T']))
		fovs.append(data['path'][i]['fov'])
	if reture_fov:
		return Rotation, Trans,fovs
	else:
		return Rotation, Trans


def save_base_cam_json(quat, trans, fovs, save_path):
	# def save_base_cam_json(xforms,fov):
	out = {"path": [], "time": 0.0}
	for q, t, fov in zip(quat, trans, fovs):
		# for _,xform in xforms.items():
		# 	q,t = nerf_to_ngp(np.array(xform))

		out['path'].append({
			"R": list(q),
			"T": list(t),
			"aperture_size": 0.0,
			"fov": fov,
			"glow_mode": 0,
			"glow_y_cutoff": 0.0,
			"scale": 0,
			"slice": 0.0
		}
		)
	# break

	with open(save_path, 'w')as outfile:
		json.dump(out, outfile, indent=2)


# def save_base_cam_json1(quat,trans,fov):
def save_base_cam_json1(xforms, fov):
	out = {"path": [], "time": 1.0}
	# for q, t in zip(quat,trans):
	for _, xform in xforms.items():
		q, t = nerf_to_ngp(np.array(xform))

		out['path'].append({
			"R": list(q),
			"T": list(t),
			"aperture_size": 0.0,
			"fov": fov,
			"glow_mode": 0,
			"glow_y_cutoff": 0.0,
			"scale": 0,
			"slice": 0.0
		}
		)
		break
	with open('E:/wukeyu/Instant-NGP/data1/hair_data_0604/base_cam.json', 'w')as outfile:
		json.dump(out, outfile, indent=2)


def cut_video(video_path):
	capture = cv2.VideoCapture(video_path)
	frames = []
	while True:
		ret, frame = capture.read()
		if not ret:
			break
		frames.append(frame)
	cv2.imwrite('test.png', frames[0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def generate_pose(pose_path, transforms_path, filename):
	Rotation, Trans = load_base_cam(pose_path)
	xforms, fov = load_transofrm_json(transforms_path)
	xf = xforms[filename]
	q, t = nerf_to_ngp(np.array(xf))
	R = quat2mat(q)
	R0inv = np.linalg.inv(quat2mat(Rotation[0]))
	R0 = quat2mat(Rotation[0])
	T0 = Trans[0].copy()
	Rc = np.dot(R, R0inv)
	Tc = t - np.dot(Rc, T0)

	for i in range(0, len(Rotation)):
		Rotation[i] = quat2mat(Rotation[i])

		# Ri0 = np.dot(R0,np.linalg.inv(Rotation[i]))
		# Ti0 = T0 - np.dot(Ri0,Trans[i])

		Rotation[i] = np.dot(Rc, Rotation[i])
		Trans[i] = Tc + np.dot(Rc, Trans[i])
		# Trans[i] -= [0.5, 0.5, 0.5]  # offset

		# Trans[i] *= 1/0.33  # scale
		# print(Rotation[i])
		# print(Trans[i])
		# Rotation[i] = np.dot(Rc,np.dot(Ri0,R0))

		# Rotation[i] = np.dot(Ri0,np.dot(Rc,R0))
		# Trans[i] = Trans[i] - np.dot(Ri0,T0)*(Tc+)

		# Rotation[i] = np.dot(R,Rotation[i]*R0)
		# Trans[i] = t+np.dot(Trans[i]-T0,R)
		Rotation[i] = mat2quat(Rotation[i])

	save_base_cam_json(Rotation, Trans, fov)


def mvs_to_ngp(mat):
	mat[:, 2] *= -1
	mat[:, 1] *= -1
	# mat[:,0]*=-1
	# mat = mat[[1,2,0]]
	return mat


def generate_ngp_posefrom_cam_params(data_folder,camera_path,save_path):
	# data_folder = 'E:\wukeyu\hair\data\mvshair\wky07-27\Real_data\wig3\colmap1'
	# save_path = 'E:\wukeyu\Instant-NGP\data1\wig1/base_cam.json'

	# Rotation, Translate, fovs_x, fovs_y = load_cam_params(
	# 	'E:\wukeyu\hair\DynamicHair\mvs_hair\HairMVSNet_clean\camera\calib_data\wky07-22/cam_params.json')
	Rotation, Translate, fovs_x, fovs_y = load_cam_params(camera_path)
	for i in range(len(Rotation)):
		Rotation[i] = mvs_to_ngp(Rotation[i])
		Translate[i] += [1, 1, 1]
		Translate[i] *= 2           #scale

	q, t = load_base_cam(data_folder + '/key_frame.json')
	q = q[0]
	t = t[0]
	R = quat2mat(q)
	# R = Rotation[0].copy()
	# t = Translate[0].copy()

	R0 = Rotation[0].copy()
	T0 = Translate[0].copy()
	Rinv = np.linalg.inv(R)

	### 0 to c
	Rc = np.dot(Rinv, R0)
	Tc = np.dot(Rinv, T0 - t)
	R_w2c = np.linalg.inv(R0)
	T_w2c = - np.dot(R_w2c, T0)

	print('distance:', np.linalg.norm(Translate[2] - Translate[0]))

	for i in range(0, len(Rotation)):
		R_pose = np.dot(np.linalg.inv(Rotation[i]), R0)
		T_pose = np.dot(np.linalg.inv(Rotation[i]), T0 - Translate[i])

		# R_temp = np.dot(Rc,R_w2c)
		R_temp = np.dot(R_pose, np.dot(Rc, R_w2c))
		# T_temp = Tc + np.dot(Rc,T_w2c)
		T_temp = np.dot(R_pose, Tc + np.dot(Rc, T_w2c)) + T_pose

		Rotation[i] = np.linalg.inv(R_temp)
		Translate[i] = - np.dot(np.linalg.inv(R_temp), T_temp)

		Translate[i] = Translate[i].tolist()
		Rotation[i] = mat2quat(Rotation[i])
		Rotation[i] = Rotation[i].tolist()

	print('distance:', np.linalg.norm(np.array(Translate[2]) - np.array(Translate[0])))

	save_base_cam_json(Rotation, Translate, fovs_y, save_path)
	video_path = os.path.join(data_folder, 'video')
	if not os.path.exists(video_path):
		os.makedirs(video_path)
	for i in range(len(Rotation)):
		save_base_cam_json(Rotation[i:i + 1], Translate[i:i + 1], fovs_y[i:i + 1],
						   os.path.join(video_path, "%03d.json" % i))


def generate_mvs_pose_from_base_cam(data_folder, select_files,camera_path, image_size):
	h, w = image_size

	xforms, fov = load_transofrm_json(data_folder + '/transforms.json')
	quat = []
	trans = []
	fovs = []
	file_name = []
	for file, xf in xforms.items():
		if file in select_files:
			q, t = nerf_to_ngp(np.array(xf), True)
			quat.append(q)
			trans.append(t)
			fovs.append(fov)
			file_name.append(file)

	Rotation, Translate, fovs_x, fovs_y = load_cam_params(camera_path)
	for i in range(len(Rotation)):
		Rotation[i] = mvs_to_ngp(Rotation[i])
		Translate[i] += [1, 1, 1]
		Translate[i] *= 2                    #scale

	mvs_c2w_R = Rotation[0]
	mvs_c2w_T = Translate[0]
	mvs_w2c_R = np.linalg.inv(mvs_c2w_R)
	mvs_w2c_T = -mvs_w2c_R @ mvs_c2w_T

	base_q, base_t = load_base_cam(data_folder + '/base_cam.json')
	base_q = base_q[0]
	base_t = base_t[0]

	intrin = h / 2 / math.tan(fov * math.pi / 180 / 2)

	pose = []
	intrin_op = []
	ndc_prj = []

	for q, t, fov in zip(quat, trans, fovs):
		mat = quat2mat(q)
		R_pose = np.linalg.inv(mat) @ quat2mat(base_q)
		T_pose = np.linalg.inv(mat) @ (base_t - t)

		w2c_R = np.dot(R_pose, mvs_w2c_R)
		w2c_T = np.dot(R_pose, mvs_w2c_T) + T_pose

		c2w_R = np.linalg.inv(w2c_R)
		c2w_T = - np.dot(np.linalg.inv(w2c_R), w2c_T)
		mat = np.eye(4)
		c2w_T /= 2                            #scale
		c2w_T -= np.array([1, 1, 1])
		c2w_R[:, 1:3] *= -1
		mat[:3, :3] = c2w_R
		mat[:3, 3] = c2w_T
		pose.append(mat)
		intrin_op.append(np.array([intrin, intrin, 0, 0]))
		ndc_prj.append(np.array([intrin * 2 / w, intrin * 2 / h, 0, 0]))
	save_path = data_folder + '/cam_params.json'
	save_camera_json(intrin_op, pose, ndc_prj, file_name, save_path)


def save_camera_json(intrins, poses, ndcs, file_name, save_path):
	class NpEncoder(json.JSONEncoder):
		'''
        json file format does not support np.float32 type
        use this class as a converter from np.* to python native types
        '''

		def default(self, obj):
			if isinstance(obj, np.integer):
				return int(obj)
			if isinstance(obj, np.floating):
				return float(obj)
			if isinstance(obj, np.ndarray):
				return obj.tolist()
			return json.JSONEncoder.default(self, obj)

	camera = {}
	cam_list = []
	# print(cam_list)
	for i, (intrin, pose, ndc, file) in enumerate(zip(intrins, poses, ndcs, file_name)):
		temp = {}
		temp["file"] = file
		temp["intrin"] = [0., 0., 0., 0.]
		temp["intrin_op"] = [intrin[0], intrin[1], intrin[2], intrin[3]]
		temp["dist"] = [0., 0., 0., 0., 0.]
		temp["pose"] = pose
		temp["ndc_prj"] = ndc
		cam_list.append(temp)

	camera["cam_list"] = cam_list
	with open(save_path, 'w') as save_file:
		json.dump(camera, save_file, cls=NpEncoder, indent=4)


def convert_mesh_to_mvs(root,camera_path, save_path):
	# cam_matrix = cam_matrix[[2, 0, 1], :]  # flip axis (yzx->xyz)
	# cam_matrix[:, 3] -= 0.5  # reversing offset
	# cam_matrix[:, 3] /= 0.33  # reversing scale
	# cam_matrix[:, 1] /= -1  # flipping y axis
	# cam_matrix[:, 2] /= -1  # z flipping

	#### nerf to ngp
	mesh_path = os.path.join(root,'base.obj')
	# mesh = trimesh.load(mesh_path)
	mesh = o3d.io.read_triangle_mesh(mesh_path)
	vertices = np.asarray(mesh.vertices)
	# vertices[:,1:] *= -1
	# vertices*=0.3275
	vertices*=0.33
	vertices += np.array([0.5, 0.5, 0.5])
	# vertices = vertices[:,[1,2,0]]


	Rotation, Translate, fovs_x, fovs_y = load_cam_params(camera_path)
	for i in range(len(Rotation)):
		Rotation[i] = mvs_to_ngp(Rotation[i])
		Translate[i] += [1, 1, 1]
		Translate[i] *= 2                   #scale

	q, t = load_base_cam(root + '/key_frame.json')
	q = q[0]
	t = t[0]
	R = quat2mat(q)

	R0 = Rotation[0].copy()
	T0 = Translate[0].copy()
	Rinv = np.linalg.inv(R)
	Tinv = -np.dot(Rinv,t)   ### w2c ngp
	vertices = vertices.transpose(1,0)
	vertices = Rinv @ vertices + Tinv[:,None]


	vertices = R0 @ vertices + T0[:,None]   ### c2w
	vertices = vertices.transpose(1,0)

	vertices/=2                           #scale
	vertices-= np.array([1, 1, 1])
	# vertices[:,1:]*= -1

	vertices-= np.array([0.006, -1.644, 0.010])
	# mesh = trimesh.Trimesh(vertices=vertices,faces=mesh.faces)
	# mesh = trimesh.Trimesh(vertices=vertices,faces=mesh.faces)
	# trimesh.exchange.export.export_mesh(mesh,save_path)
	new_mesh = o3d.geometry.TriangleMesh()
	new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
	new_mesh.triangles = mesh.triangles
	o3d.io.write_triangle_mesh(save_path,new_mesh)


# def convert_mesh_to_mvs(root,camera_path, save_path):
#
# 	#### nerf to ngp
# 	mesh_path = os.path.join(root,'base.obj')
# 	# mesh = trimesh.load(mesh_path)
# 	mesh = o3d.io.read_triangle_mesh(mesh_path)
# 	vertices = np.asarray(mesh.vertices)
# 	vertices[:,1:] *= -1
# 	vertices*=0.33
# 	vertices += np.array([0.5, 0.5, 0.5])
# 	# vertices = vertices[:,[1,2,0]]
#
# 	# mat[:, 1] *= -1  # flip axis
# 	# mat[:, 2] *= -1
# 	# mat[:, 3] *= 0.33  # scale
# 	# mat[:, 3] += [0.5, 0.5, 0.5]  # offset
# 	#
# 	# mat = mat[[1, 2, 0], :]  # swap axis
#
# 	Rotation, Translate, fovs_x, fovs_y = load_cam_params(camera_path)
# 	for i in range(len(Rotation)):
# 		Rotation[i] = mvs_to_ngp(Rotation[i])
# 		Translate[i] += [1, 1, 1]
# 		Translate[i] *= 2                   #scale
#
# 	q, t = load_base_cam(root + '/key_frame.json')
# 	q = q[0]
# 	t = t[0]
# 	R = quat2mat(q)
#
# 	R0 = Rotation[0].copy()
# 	T0 = Translate[0].copy()
# 	Rinv = np.linalg.inv(R)
# 	Tinv = -np.dot(Rinv,t)   ### w2c ngp
# 	vertices = vertices.transpose(1,0)
# 	vertices = Rinv @ vertices + Tinv[:,None]
#
#
# 	vertices = R0 @ vertices + T0[:,None]   ### c2w
# 	vertices = vertices.transpose(1,0)
#
# 	vertices/=2                           #scale
# 	vertices-= np.array([1, 1, 1])
# 	# vertices[:,1:]*= -1
#
# 	vertices-= np.array([0.006, -1.644, 0.010])
# 	# mesh = trimesh.Trimesh(vertices=vertices,faces=mesh.faces)
# 	# mesh = trimesh.Trimesh(vertices=vertices,faces=mesh.faces)
# 	# trimesh.exchange.export.export_mesh(mesh,save_path)
# 	new_mesh = o3d.geometry.TriangleMesh()
# 	new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
# 	new_mesh.triangles = mesh.triangles
# 	o3d.io.write_triangle_mesh(save_path,new_mesh)






if __name__ == '__main__':

	case_name = 'wig1'

	####1.
	# xforms,fov = load_transofrm_json('E:\wukeyu\hair\data\mvshair\wky07-27\Real_data\wig2\colmap1/transforms.json')
	# # save_base_cam_json1(xforms, fov)
	# quat = []
	# trans =[]
	# fovs =[]
	# for file,xf in xforms.items():
	# 	if '3955' in file:
	# 		q,t = nerf_to_ngp(np.array(xf),True)
	# 		quat.append(q)
	# 		trans.append(t)
	# 		fovs.append(fov)
	#
	# save_path = 'E:\wukeyu\hair\data\mvshair\wky07-27\Real_data\wig2\colmap1/test.json'
	# save_base_cam_json(quat[:1],trans[:1],fovs[:1],save_path)

	####2. generate pose for each capture images
	# select_files = []
	# files = os.listdir(r'E:\wukeyu\hair\data\mvshair\wky07-27\Real_data\{}\capture_images'.format(case_name))
	# for i, file in enumerate(files):
	# 	if i % 1 == 0:
	# 		select_files.append(file[:-4])
	# # select_files = ['IMG_3097']
	# data_folder = r'E:\wukeyu\hair\data\mvshair\wky07-27\Real_data\{}/ours'.format(case_name)
	# generate_mvs_pose_from_base_cam(data_folder, select_files, image_size=[1120, 1992])

	#### 3. generate 16 fixed camera pose
	# generate_ngp_posefrom_cam_params()


	#### 4. convert nerf mesh to mvs
	# root = 'E:\wukeyu\hair\data\mvshair\wky07-27\Real_data\{}\ours'.format(case_name)
	# convert_mesh_to_mvs(root)

	### 5 cut frame from video
	# video_path = 'E:\wukeyu\hair\data\mvshair\wky07-27\Real_data\wig1\colmap1/base_video.mp4'
	# cut_video(video_path)

	### 6 render depth
	colmap_points_path = 'E:\wukeyu\hair\data\mvshair\wky07-27\Real_data\wig2\colmap1/colmap_points.obj'
	camera_path = 'camera/calib_data/wky07-22/cam_params.json'
	save_root = r'E:\wukeyu\hair\data\mvshair\wky07-27\Real_data\wig2\imgs'
	render_bust_hair_depth(colmap_points_path,camera_path,save_root)




# pose_path = r'E:\wukeyu\Instant-NGP\data1\wig1/camera_pose.json'
# transforms_path = r'E:\wukeyu\Instant-NGP\data1\wig1/transforms.json'
# filename = 'IMG_2652'
# generate_pose(pose_path,transforms_path,filename)
