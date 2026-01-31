from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    loadPrcFileData, Vec2, Vec3, Vec4, Geom, GeomNode, GeomEnums, NodePath, BoundingBox,
    GeomTrifans, GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData, InternalName,
    Shader, ShaderBuffer, ComputeNode
)
import struct
import numpy as np

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
//show-frame-rate-meter 1
hardware-animated-vertices true
framebuffer-srgb true
model-cache-dir
gl-debug 1
gl-version 4 3
basic-shaders-only false
//threading-model Cull/Draw
"""
loadPrcFileData("", config_vars)

def GENERATE_VTX_DATA(num_rows: int):
    print("-> Creating GeomVertexData and associated formats...")
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
    vtx_data.unclean_set_num_rows(num_rows)                           # 1 row per vertex (12 rim, 1 centre)
    return vtx_data

def POPULATE_VTX_DATA(vtx_data, col: Vec4, vel: Vec2):
    # open memoryviews to write position, normal, colour, basis, and velocity data to VBO
    print("-> Populating vertex data...")
    view   = memoryview(vtx_data.modify_array(0)).cast('B')

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
    return vtx_data, buffer

# create a GeomTrifans primitive for making a complete circle
def GENERATE_PRIM(num_verts: int):
    print("-> Generating blob model primitives...")
    blobPrim = GeomTrifans(Geom.UHStatic)                   # make a trifan
    blobPrim.add_consecutive_vertices(0,(num_verts-1))      # add all the verts
    blobPrim.add_vertex(1)                                  # close the circle
    blobPrim.closePrimitive()                               # close the primitive
    return blobPrim

def CREATE_COMP():
    # set up the compute node to calculate the vertex positions
    print("-> Setting up a compute shader for a blob...")
    comp_node = ComputeNode("blob_animation")
    comp_node.add_dispatch(1, 1, 1)
    comp_np = base.render.attach_new_node(comp_node)
    comp_np.set_shader(Shader.load_compute(Shader.SL_GLSL, "blob.comp"))
    return comp_np

class CellBlob:
    # initialise a circle with 1 inner and 12 outer vertices, with compute-managed animation
    def __init__(self, pos=Vec3(0.,0.,0.),vel=Vec2(0.,0.),col=Vec4(1.,1.,1.,1.),rad=1.):
        print("-> Initialising blob...")
        self.vel = vel
        self.col = col
        self.radius = rad

        # get a new VBO and SSBO with a blob model
        vtx_data = GENERATE_VTX_DATA(13)
        vtx_data, self.buffer = POPULATE_VTX_DATA(vtx_data, col, vel)

        # create a compute shader and send it blob information
        comp_np = CREATE_COMP()
        comp_np.set_shader_input("ssbo", self.buffer)
        comp_np.set_shader_input("radius", self.radius)
        comp_np.set_shader_input("model_velocity", self.vel)

        # create a mesh from the vertices 
        print("-> Creating geometry...")
        geom = Geom(vtx_data)
        geom.addPrimitive(GENERATE_PRIM(13))
        # set up a bounding volume to prevent culling
        geom.set_bounds(BoundingBox((-1, -1, -.5), (1, 1, .5)))
        #geom.doublesideInPlace()
        geom_node = GeomNode('blob-geom_node')
        geom_node.addGeom(geom)

        # make nodepath and activate custom shaders, send SSBO and colour
        print("-> Composing blob NodePath...")
        self.nodepath = base.render.attach_new_node(geom_node)
        self.nodepath.set_shader(Shader.load(
            Shader.SL_GLSL, 
            vertex="default_shader.vert", 
            fragment="default_shader.frag"
        ))
        self.nodepath.set_shader_input("ssbo", self.buffer)
        self.nodepath.set_shader_input("col", self.col)

        # store position in the nodepath
        self.nodepath.set_pos(pos)

        base.taskMgr.add(self.update, "update", taskChain='default')

    def update(self, task):
        # processor-killing debug
        #print(f"nodepath position: {pos}")
        #vtx_view = memoryview(nodepath.node().get_vertex_data().modify_array(0)).cast('f')
        #print(f"some verts: 0: {vtx_view[0]}, 1: {vtx_view[1]}, 2: {vtx_view[2]}")

        # update nodepath position by velocity
        self.nodepath.set_pos(self.nodepath.get_pos() + Vec3(self.vel, 0.))
        self.nodepath.set_shader_input("model_velocity", self.vel)
        self.vel = self.vel/2. if self.vel > EPSILON else Vec2(0.,0.)     # friction slows us

        return task.cont

    def move(self, direction):
        match direction:
            case "left":                                                    # go left
                self.vel -=  Vec2(.05, 0.)
            case "right":                                                   # go right
                self.vel +=  Vec2(.05, 0.)
            case "fwd":                                                     # go forwards
                self.vel +=  Vec2(0., .05)
            case "back":                                                    # ...you guessed it
                self.vel -=  Vec2(0., .05)
            case _: 
                print("Move direction not recognised!")

def bind_blob(bound_blob):
    print(f"-> Binding {bound_blob} to user input...")
    base.accept("arrow_left", bound_blob.move, ["left"])
    base.accept("arrow_left-repeat", bound_blob.move, ["left"])
    base.accept("a", bound_blob.move, ["left"])
    base.accept("a-repeat", bound_blob.move, ["left"])
    base.accept("arrow_right", bound_blob.move, ["right"])
    base.accept("arrow_right-repeat", bound_blob.move, ["right"])
    base.accept("d", bound_blob.move, ["right"])
    base.accept("d-repeat", bound_blob.move, ["right"])
    base.accept("arrow_up", bound_blob.move, ["fwd"])
    base.accept("arrow_up-repeat", bound_blob.move, ["fwd"])
    base.accept("w", bound_blob.move, ["fwd"])
    base.accept("w-repeat", bound_blob.move, ["fwd"])
    base.accept("arrow_down", bound_blob.move, ["back"])
    base.accept("arrow_down-repeat", bound_blob.move, ["back"])
    base.accept("s", bound_blob.move, ["back"])
    base.accept("s-repeat", bound_blob.move, ["back"])

if __name__ == "__main__":
    print("="*30 + "\n... Loading Blob ...")        # application entry point -----------

    print("-> Initialising ShowBase...")
    ShowBase()                                      # Showbase initialised

    player_1 = CellBlob(col=Vec4(0.,0.,1.,1.))                           # CellBlob for player 1 constructed

    bind_blob(player_1)                             # Player 1 bound to user input

    base.cam.setPos(0,-18,5)                        # Adjust camera position and angle
    base.cam.setHpr(0,-15,0)

    print("-> All ready! Running ShowBase:")
    base.run()                                      # run Showbase
