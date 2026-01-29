from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, DirectionalLight, NodePath

config_vars = """
gl-version 4 3
gl-debug true
win-size 1280 720
show-frame-rate-meter 1
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
test_model_np = test_model.reparent_to(render)
test_model_np.set_pos(0.,0.,0.)

base.camera.setPos(0,-3,4)
base.camera.setHpr(0,-22,0)

base.run()
