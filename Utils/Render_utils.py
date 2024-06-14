import numpy as np
import moderngl
import cv2
import os
import open3d as o3d
from Utils.Camera_utils import load_cam,parsing_camera

class StrandsObj():
    def __init__(self, strands, ctx):
        self.ctx = ctx
        num_strands = len(strands)
        print('num of strands:', num_strands)
        self.Lines = []
        self.tangent = []
        for strand in strands:
            tangent = strand[1:] - strand[:-1]
            tangent = np.concatenate([tangent, strand[-1:] - strand[-2:-1]], 0)
            num_v = strand.shape[0]
            index1 = np.arange(0, num_v - 1, dtype=np.int32)
            index2 = np.arange(1, num_v, dtype=np.int32)
            index = np.concatenate([index1, index2], 0)
            index = np.sort(index)
            line = strand[index]
            self.tangent.append(tangent[index])
            self.Lines.append(line)
        self.Lines = np.concatenate(self.Lines, 0)
        self.tangent = np.concatenate(self.tangent, 0)
        self.ctx.line_width = 3.0
        self.colorOption = 1

    def loadObject(self):
        line_buffer = self.ctx.buffer(self.Lines.astype('f4'))
        tangent_buffer = self.ctx.buffer(self.tangent.astype('f4'))

        vao_content = [
            (line_buffer, '3f', 'LinePosition'),
            (tangent_buffer, '3f', 'Tangent')
        ]
        self.vao = self.ctx.vertex_array(self.prog, vao_content)

    def loadShader(self):
        self.prog = self.ctx.program(
            vertex_shader='''
            #version 330

            uniform mat4 projection;
            uniform mat4 transform;

            layout(location = 0) in vec3 LinePosition;
            layout(location = 1) in vec3 Tangent;

            out float depth;
            out vec2 Tangent_2d;


            void main(){
            vec4 camera_v = transform * vec4(LinePosition,1);
            gl_Position = projection * camera_v;
            vec2 pos_2d = gl_Position.xy / gl_Position.w;

            float step_length = 0.01;
            vec3 forward_step = normalize(Tangent) * step_length;
            vec3 nxt_pos = LinePosition + forward_step;
            vec4 nxt_pos_ndc = projection * transform * vec4(nxt_pos, 1.0);
            vec2 nxt_pos_2d = nxt_pos_ndc.xy / nxt_pos_ndc.w;
            Tangent_2d = nxt_pos_2d - pos_2d;

            depth = -camera_v.z;

            }
            '''
            ,
            fragment_shader='''
            #version 330 core
            uniform int colorOption;
            layout(location = 0) out vec4 FragData;
            in float depth;
            in vec2 Tangent_2d;

            void main(){
            float pi = 3.14159265;
            float theta_2d = atan(Tangent_2d.y, Tangent_2d.x);
            switch (colorOption) {
            case 0:
                // depth
                float depth_range = 2.0f;
                float depth_norm = depth / depth_range;
                FragData = vec4(depth_norm,depth_norm,depth_norm,1.0f);
                break;
            case 1:
                //color
                vec3 theta_2d_color_dir = vec3(cos(theta_2d), sin(theta_2d), 0.0);
                FragData = vec4((theta_2d_color_dir + vec3(1.0, 1.0, 0.0)) * 0.5, 1.0);
                break;
            case 2:

                vec3 theta_2d_color = vec3(cos(2 * theta_2d), sin(2 * theta_2d), 0.0);
                FragData = vec4((theta_2d_color + vec3(1.0, 1.0, 0.0)) * 0.5, 1.0);
                break;
                    
            case 3:
                 FragData = vec4(1.0,1.0,1.0, 1.0);
                 break;

            }

            }

            '''
        )

    def rendering(self, projection, pose):
        projection = projection.T
        pose = pose.T
        self.prog['projection'].value = tuple(projection.flatten())
        self.prog['transform'].value = tuple(pose.flatten())
        # self.prog['colorOption'].value = tuple(np.array([1,1,1.],dtype='f4'))
        self.prog['colorOption'].value = self.colorOption

        self.vao.render(moderngl.LINES)

    def makeContext(self):
        self.loadShader()
        self.loadObject()

    def set_colorOption(self, value):
        self.colorOption = value


class BustObj():
    def __init__(self, bust_vertices, bust_faces, ctx):
        self.ctx = ctx
        self.vertices = bust_vertices
        self.faces = bust_faces
        self.depthOption = 0

    def loadObject(self):
        pos_buffer = self.ctx.buffer(self.vertices.astype('f4'))

        index_buffer = self.ctx.buffer(np.array(self.faces, dtype='u4'))
        vao_content = [
            (pos_buffer, '3f', 'vertexPosition')
        ]
        self.vao = self.ctx.vertex_array(self.prog, vao_content, index_buffer)

    def loadShader(self):
        self.prog = self.ctx.program(
            vertex_shader='''
            #version 330

            uniform mat4 projection;
            uniform mat4 transform;
            layout(location = 0) in vec3 vertexPosition;
            out float depth;

            void main(){
            vec4 camera_v = transform * vec4(vertexPosition,1);
            gl_Position = projection * camera_v;

            depth = -camera_v.z;
            }
            '''
            ,
            fragment_shader='''
            #version 330 core
            layout(location = 0) out vec4 FragData;
            in float depth;
            uniform int depthOption;
            uniform sampler2D myTextureSampler;
            void main(){
            switch (depthOption) {
            case 0:
                float depth_range = 2.0f;
                float depth_norm = depth / depth_range;
                FragData = vec4(depth_norm,depth_norm,depth_norm,1.0f);
                break;

            case 1:
                FragData = vec4(0.,0.,0.,1.0f);
                break;
            case 2:
                FragData = vec4(1.,1.,1.,1.0f);
                break;
            }

            }

            '''
        )

    def rendering(self, projection, pose):
        projection = projection.T
        pose = pose.T
        self.prog['projection'].value = tuple(projection.flatten())
        self.prog['transform'].value = tuple(pose.flatten())
        self.prog['depthOption'].value = self.depthOption
        self.vao.render(moderngl.TRIANGLES)

    def makeContext(self):
        self.loadShader()
        self.loadObject()

    def set_depthOption(self, value):
        self.depthOption = value


class Renderer():

    def __init__(self, camera, Width=1120, Height=1992, Headless=False, **kwargs):
        super().__init__(**kwargs)
        self.Width = Width
        self.Height = Height
        self.camera = camera
        self.components = 3
        if Headless:
            # self.ctx = moderngl.create_context(standalone=True, libgl='libGL.so.1', libx11='libX11.so.6')
            self.ctx = moderngl.create_context(standalone=True, backend='egl', libgl='libGL.so.1',
                                               libegl='libEGL.so.1', )
        else:
            self.ctx = moderngl.create_context(standalone=True)
        self.init_buffer()
        self.init_context()
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def add_mesh(self, meshWithRender):
        meshWithRender.makeContext()
        self.meshes.append(meshWithRender)

    def init_context(self):
        self.meshes = []

    def init_buffer(self):
        render_buffer = self.ctx.renderbuffer(size=(self.Width, self.Height), components=self.components, dtype='f4', )
        dbo = self.ctx.depth_texture(size=(self.Width, self.Height), alignment=1)
        self.fbo = self.ctx.framebuffer(render_buffer, depth_attachment=dbo)
        self.fbo.use()

    def draw(self, camera_view, clear_color=[1.0, 1.0, 1.0]):
        self.ctx.clear(clear_color[0], clear_color[1], clear_color[2])
        # P = Matrix44.perspective_projection(38.058, 1920. / 1080, 0.001, 1000)
        # P = P.transpose()
        # print('P',P)
        # print( camera_view.proj.cpu().numpy())
        # projection = np.asarray(P).astype('f4')
        projection = camera_view.proj.cpu().numpy().astype('f4')

        pose = camera_view.pose.cpu().numpy().astype('f4')

        for mesh in self.meshes:
            mesh.rendering(projection, pose)

    def ReadBuffer(self):
        data = self.fbo.read(components=3, dtype='f4')
        image = np.frombuffer(data, dtype='f4')
        image = image.reshape((self.Height, self.Width, 3))
        image = np.flip(image, 0)
        # image = Image.frombytes('F', self.fbo.size, data)
        # image = image.transpose(Image.FLIP_TOP_BOTTOM)
        # image.save('output.png')

        # data = np.frombuffer(data, dtype='f4', )
        # print(data.shape[:])
        # data = np.reshape(data, (self.Height, self.Width, 1))
        # cv2.imwrite('test.png', data * 255)
        return image


def render_data(camera, strands, vertices, faces, image_size=[1280, 720], save_root=None):
    Render = Renderer(camera, Width=image_size[1], Height=image_size[0], Headless=True)
    ### create Strands and BustObj with render
    renderStrands = StrandsObj(strands, Render.ctx)
    renderBust = BustObj(vertices, faces, Render.ctx)
    Render.add_mesh(renderBust)
    os.makedirs(save_root, exist_ok=True)
    for view, c in camera.items():
        Render.draw(c, clear_color=[1., 1., 1.])
        depth = Render.ReadBuffer()
        cv2.imwrite(os.path.join(save_root, view, 'bust_depth.png'), depth * 255,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    Render.add_mesh(renderStrands)

    renderStrands.set_colorOption(2)
    renderBust.set_depthOption(1)
    for view, c in camera.items():
        Render.draw(c, clear_color=[0., 0., 0.])
        color = Render.ReadBuffer()
        cv2.imwrite(os.path.join(save_root, view, 'undirectional_map.png'), color[..., [2, 1, 0]] * 255,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])



    renderStrands.set_colorOption(3)
    for view, c in camera.items():
        Render.draw(c, clear_color=[0., 0., 0.])
        color = Render.ReadBuffer()
        cv2.imwrite(os.path.join(save_root, view, 'mask.png'), color[..., [2, 1, 0]] * 255,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    renderBust.set_depthOption(2)
    renderStrands.set_colorOption(0)
    for view, c in camera.items():
        Render.draw(c, clear_color=[1., 1., 1.])
        depth = Render.ReadBuffer()
        cv2.imwrite(os.path.join(save_root, view, 'hair_depth.png'), depth * 255,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def render_bust_hair_depth(colmap_points_path, camera_path, save_root, image_size=[1280, 720], capture_imgs=False,
                           bust_path=None, Headless=True):
    bust_to_origin = np.array([0.006, -1.644, 0.010])
    # mesh = trimesh.load(colmap_points_path)
    mesh = o3d.io.read_triangle_mesh(colmap_points_path)
    colmap_points = np.array(mesh.vertices)
    colmap_faces = np.array(mesh.triangles)
    colmap_points += bust_to_origin
    camera = load_cam(camera_path)
    camera = parsing_camera(camera, )

    Render = Renderer(camera, Width=image_size[1], Height=image_size[0], Headless=Headless)
    render_colmap = BustObj(colmap_points, colmap_faces, Render.ctx)
    Render.add_mesh(render_colmap)
    if bust_path is not None:
        bust = o3d.io.read_triangle_mesh(bust_path)
        bust_points = np.array(bust.vertices)
        bust_faces = np.array(bust.triangles)
        bust_points += bust_to_origin
        render_bust = BustObj(bust_points, bust_faces, Render.ctx)
        Render.add_mesh(render_bust)

    for view, c in camera.items():
        Render.draw(c)
        depth = Render.ReadBuffer()

        if capture_imgs:
            # depth = np.asarray(depth)
            depth_save = depth.copy() * 255.

            np.save(os.path.join(save_root, view + '.npy'), depth_save)
            # depth_save.tofile(os.path.join(save_root, view+'.dat'))
            cv2.imwrite(os.path.join(save_root, view + '.JPG'), depth_save)
        # depth.save(os.path.join(save_root, view+'.JPG'))
        else:
            # depth = depth.convert('RGB')
            # depth.save(os.path.join(save_root, view,'bust_hair_depth.png'))
            cv2.imwrite(os.path.join(save_root, view, 'bust_hair_depth.png'), depth * 255)
