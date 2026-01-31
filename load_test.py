from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    loadPrcFileData, DirectionalLight, NodePath, ShaderBuffer, GeomEnums,
    Shader, Vec2, Vec4
)
import numpy as np

config_vars = """
gl-version 4 3
gl-debug true
win-size 1280 720
//show-frame-rate-meter 1
hardware-animated-vertices true
framebuffer-srgb true
basic-shaders-only false
model-cache-dir
"""
loadPrcFileData("", config_vars)

ShowBase()

base.set_background_color(.8,.8,.8,1)

light = DirectionalLight('dir_light')
light.setShadowCaster(True, 512, 512)
light_np = render.attachNewNode(light)
light_np.setHpr(20, -80, 0)
light_np.set_color(.5,.45,.39)
render.setLight(light_np)
test_model = loader.loadModel("blob_default.bam")
#test_model.setP(90)
test_model.reparent_to(render)
test_model.set_pos(0.,0.,0.)

test_model.node().modify_geom(0).make_nonindexed(False)

test_model.set_depth_offset(1)

load_data = test_model.node().get_geom(0).get_vertex_data()
#, dtype=np.float32
#print(load_data)

# make an SSBO from the model data for vertex pulling
p3d_array = load_data.get_array_handle(0).get_data()
custom_array = load_data.get_array_handle(1).get_data()
p3d_array = np.append(p3d_array, custom_array)
byte_data = bytearray(p3d_array)
#print(f"byte data for SSBO: {byte_data}")
#byte_data.extend(custom_array)
buffer = ShaderBuffer("ssbo", bytes(byte_data), GeomEnums.UHDynamic)

test_model.set_shader(Shader.load(Shader.SL_GLSL, vertex="blob_jiggle.vert", fragment="default_shader.frag"))
test_model.set_shader_input("ssbo", buffer)
test_model.set_shader_input("model_velocity", Vec2(0.,0.))
test_model.set_shader_input("radius", 1.)
test_model.set_shader_input("col", Vec4(0.,1.,0.,1.))


base.cam.setPos(0,-3,1)
base.cam.setHpr(0,-22,0)

base.run()
