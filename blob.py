from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    loadPrcFileData, Vec2, Vec3, Vec4, Geom, GeomNode, GeomEnums, NodePath, BoundingBox,
    GeomTrifans, GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData, InternalName,
    Shader, ShaderBuffer
)
import struct
import numpy as np
#import array

EPSILON = 0.0001
radius = 1.
BASIS_VECS = [0.,     0.,
              -.866, -.5,
              -.5,   -.866,
              0.,    -1.,
              .5,    -.866,
              .866,  -.5,
              1.,     0.,
              .866,   .5,
              .5,     .866,
              0.,     1.,
              -.5,    .866,
              -.866,  .5,
              -1.,    0.]

config_vars: str = """
win-size 1200 800
show-frame-rate-meter 1
hardware-animated-vertices true
framebuffer-srgb true
model-cache-dir
gl-debug 1
gl-version 4 3
basic-shaders-only false
//threading-model Cull/Draw
"""
loadPrcFileData("", config_vars)

def update(task):
    global pos
    global vel
    # processor-killing debug
    #print(f"nodepath position: {pos}")
    #vtx_view = memoryview(nodepath.node().get_vertex_data().modify_array(0)).cast('f')
    #print(f"some verts: 0: {vtx_view[0]}, 1: {vtx_view[1]}, 2: {vtx_view[2]}")

    nodepath.set_pos(pos + Vec3(vel, 0.))
    nodepath.set_shader_input("model_velocity", vel)
    vel = vel/10. if vel > EPSILON else Vec2(0.,0.) # friction slows us

    return task.cont

def move(direction):
    global vel
    match direction:
        case "left":             # go left
            vel -=  Vec2(.05,0.)
        case "right":                       # go right
            vel +=  Vec2(.05,0.)
        case "fwd":                         # go forwards
            vel +=  Vec2(0.,.05)
        case "back":                        # ...you guessed it
            vel -=  Vec2(0.,.05)
        case _: 
            print("Move direction not recognised!")


ShowBase() # Showbase initialised

vtx_format  = GeomVertexFormat()
arrayFormat = GeomVertexArrayFormat()
arrayFormat.set_divisor(0)
arrayFormat.set_stride(64)
arrayFormat.add_column(InternalName.get_vertex(),4,GeomEnums.NT_float32, GeomEnums.C_point)
arrayFormat.add_column(InternalName.get_normal(),4,GeomEnums.NT_float32, GeomEnums.C_normal)
arrayFormat.add_column(InternalName.get_color(),4,GeomEnums.NT_float32, GeomEnums.C_color)
arrayFormat.add_column(InternalName.get_vertex().get_parent().append("basis"),2,GeomEnums.NT_float32, GeomEnums.C_point)
arrayFormat.add_column(InternalName.get_vertex().get_parent().append("velocity"),2,GeomEnums.NT_float32, GeomEnums.C_point)
arrayFormat.pack_columns()
vtx_format.add_array(arrayFormat)
vtx_format  = GeomVertexFormat.register_format(vtx_format)
vtx_data    = GeomVertexData('blob_verts', vtx_format, Geom.UHStatic)
vtx_data.unclean_set_num_rows(13) # 1 row per vertex (12 rim, 1 centre)

print("-- Formats registered. Creating geometry...")

# open memoryviews to write position, normal, colour, basis, and velocity data to VBO
view   = memoryview(vtx_data.modify_array(0)).cast('B')

col = Vec4(0.,0.,1.,1.)
pos = Vec3(0.,-5.,0.)
vel = Vec2(0.,0.)

vals = bytearray()
for i in range(13):
    # populate the bytearray with each row
    vals.extend(struct.pack('4f', BASIS_VECS[i*2], BASIS_VECS[i*2 + 1], 0., 1.))
    vals.extend(struct.pack('4f', 0.,0.,1.,0.))
    vals.extend(struct.pack('4f', col.x, col.y, col.z, col.w))
    vals.extend(struct.pack('2f', BASIS_VECS[i*2], BASIS_VECS[i*2 + 1]))
    vals.extend(struct.pack('2f', vel.x, vel.y))

# write to VBO and SSBO
view[:] = vals
buffer = ShaderBuffer("ssbo", bytes(vals), GeomEnums.UHDynamic)

# finally, create a mesh ('Geom') from the vertices- containing one trifan defined above as blobPrim
geom     = Geom(vtx_data)
blobPrim = GeomTrifans(Geom.UHStatic)
blobPrim.add_consecutive_vertices(0,12) # add all the verts
blobPrim.add_vertex(1) # close the circle
blobPrim.closePrimitive()
geom.addPrimitive(blobPrim)

# set up a bounding volume to prevent culling
geom.set_bounds(BoundingBox((-1, -1, -.5), (1, 1, .5)))
geom.doublesideInPlace()
geom_node = GeomNode('blob-geom_node')
geom_node.addGeom(geom)
nodepath = base.render.attach_new_node(geom_node)
nodepath.set_shader(Shader.load(Shader.SL_GLSL, vertex="blob_jiggle.vert", fragment="default_shader.frag"))
nodepath.set_shader_input("ssbo", buffer)
nodepath.set_shader_input("model_velocity", vel)
nodepath.set_shader_input("radius", radius)
nodepath.set_shader_input("col", col)

base.taskMgr.add(update, "update", taskChain='default')

base.accept("arrow_left", move, ["left"])
base.accept("arrow_left-repeat", move, ["left"])
base.accept("a", move, ["left"])
base.accept("a-repeat", move, ["left"])
base.accept("arrow_right", move, ["right"])
base.accept("arrow_right-repeat", move, ["right"])
base.accept("d", move, ["right"])
base.accept("d-repeat", move, ["right"])
base.accept("arrow_up", move, ["fwd"])
base.accept("arrow_up-repeat", move, ["fwd"])
base.accept("w", move, ["fwd"])
base.accept("w-repeat", move, ["fwd"])
base.accept("arrow_down", move, ["back"])
base.accept("arrow_down-repeat", move, ["back"])
base.accept("s", move, ["back"])
base.accept("s-repeat", move, ["back"])

base.cam.setPos(0,-18,5)
base.cam.setHpr(0,-15,0)

base.run() # run Showbase
