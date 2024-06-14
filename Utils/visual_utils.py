import numpy as np
import open3d as o3d


def vis_point_colud(points,colors=None):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(colors)
    return pc

def visual_volum(ori):
    z,y,x = np.nonzero(np.linalg.norm(ori,2,-1))
    # volum_xyz = np.concatenate((x[:,None],y[:,None],z[:,None]),-1) ### up is +y in opengl, up-to-down is +y in grid
    volum_xyz = np.concatenate((x[:,None],-y[:,None],-z[:,None]),-1) ### consistent with the grid, top left is (0,0,0)

    # color = ori[z[:, 0], y[:, 0], x[:, 0]]
    color = ori[z[:], y[:], x[:]]

    pc = vis_point_colud(volum_xyz, color[:, [1, 0, 2]])
    return pc

def visual_strands_with_tangent(strands,ori,bbox_min,vsize):
    strands_ori = strands.copy()
    strands[:, 1:] *= -1    ###  consistent with the grid, top left is (0,0,0)
    indexs = (strands - bbox_min) / vsize
    indexs = np.round(indexs)
    indexs = indexs.astype(np.int32)
    x, y, z = np.split(indexs, 3, -1)

    color = ori[z[:, 0], y[:, 0], x[:, 0]]

    # color = color[:, [1, 0, 2]]  ### if y-axis correspond R-channel
    # color[:,:2]*=-1   ### to consistent with 2D Ori
    # # color = (color+1)*0.5
    # color[:, 2] = 0
    color = color/np.maximum(np.linalg.norm(color,axis=-1,keepdims=True),1e-6)
    pc = vis_point_colud(strands_ori,color)
    return pc

def visual_strands(strands):
    color = []
    points = []
    for i in range(len(strands)):
        strand = strands[i]
        points.append(strand)
        c = np.concatenate([strand[1:] - strand[:-1], strand[-1:] - strand[-2:-1]],0)
        # c = np.concatenate([strand[1:11] - strand[:10], strand[-1:] - strand[-2:-1]],0)
        # c = strand[1:11] - strand[:10]
        c = c/np.linalg.norm(c,2,-1,keepdims=True)
        color.append(c)
    color = np.concatenate(color,0)
    points = np.concatenate(points,0)
    pc = vis_point_colud(points, (color+1)*0.5)
    vis_norm = vis_normals(points,color*0.004)
    return [pc,vis_norm]

def vis_mesh(vertices,faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices= o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))
    mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()
    mesh.vertex_normals =o3d.utility.Vector3dVector(-1*np.asarray(mesh.vertex_normals))
    # mesh.triangle_normals =o3d.utility.Vector3dVector(np.ones_like(faces)*np.array([0,0,1]))
    mesh.triangle_normals =o3d.utility.Vector3dVector(-np.array(mesh.triangle_normals))
    return mesh

def vis_normals(points,normals,color=None):
    points1 = points + normals
    line1 = np.arange(0, points.shape[0])
    line2 = np.arange(points.shape[0], 2 * points.shape[0])
    line = np.concatenate([line1[:, None], line2[:, None]], 1)
    print(line.shape[:])
    print(points.shape[:])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.concatenate([points, points1], 0))
    line_set.lines = o3d.utility.Vector2iVector(line)
    if color is not None:
        color = color/np.linalg.norm(color,2,-1,keepdims=True)
    else:
        color = normals/np.linalg.norm(normals,2,-1,keepdims=True)
        color = (color+1)*0.5
    line_set.colors = o3d.utility.Vector3dVector(color)
    return line_set

def draw_scene(scene):
    o3d.visualization.draw_geometries(scene)