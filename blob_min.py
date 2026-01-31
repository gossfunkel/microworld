from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    loadPrcFileData, Vec2, Vec3, Vec4, Geom, GeomNode, GeomEnums, NodePath, BoundingBox,
    GeomTrifans, GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData, InternalName,
    Shader, ShaderBuffer
)
import struct

BASIS_VECS = [0., 0., -.866, -.5, -.5, -.866, 0., -1., .5, -.866, .866, -.5, 1., 0.,
              .866, .5, .5, .866, 0., 1., -.5, .866, -.866, .5, -1., 0.]

VTX_SHADE = """
#version 430

uniform mat4 p3d_ModelViewProjectionMatrix;

struct Buffer_Data {
    vec4 pos;                       // 4x 4B
    vec4 normal;                    // 4x 4B
    vec4 colour;                    // 4x 4B
    vec2 basis;                     // 2x 4B
    vec2 vel;                       // 2x 4B
};

// SSBO for vertex pulling
layout (std430, binding = 0) buffer ssbo { 
    Buffer_Data p3d_data[13];    // = 13x 64B
};                               //     = 832B buffer

void main() {
    uint vtx = gl_VertexID;
    vec4 new_pos = vec4(0.,0.,0.,1.);       // default value for centrepoint, 
    if (vtx != 0) {
        vec2 desire_vtx = p3d_data[vtx].basis;
    	new_pos = vec4(desire_vtx,0.,1.);
    }
    gl_Position = p3d_ModelViewProjectionMatrix * new_pos;
}
""".strip()

FRG_SHADE = """
#version 430

// out to screen
out vec4 p3d_FragColor;

void main() {
    p3d_FragColor = vec4(1.,0.,0.,1.);
}
""".strip()

config_vars: str = """
win-size 1200 800
show-frame-rate-meter 1
framebuffer-srgb true
model-cache-dir
gl-debug 1
gl-version 4 3
basic-shaders-only false
"""
loadPrcFileData("", config_vars)

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

vals = bytearray()
for i in range(13):
    # populate the bytearray with each row
    vals.extend(struct.pack('4f', BASIS_VECS[i*2], BASIS_VECS[i*2 + 1], 0., 1.))
    vals.extend(struct.pack('4f', 0.,0.,1.,0.))
    vals.extend(struct.pack('4f', 1., 1., 1., 1.))
    vals.extend(struct.pack('2f', BASIS_VECS[i*2], BASIS_VECS[i*2 + 1]))
    vals.extend(struct.pack('2f', 0., 0.))

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
geom_node = GeomNode('blob-geom_node')
geom_node.addGeom(geom)
nodepath = base.render.attach_new_node(geom_node)
nodepath.set_shader(Shader.make(Shader.SL_GLSL, vertex=VTX_SHADE, fragment=FRG_SHADE))
nodepath.set_shader_input("ssbo", buffer)

base.cam.setPos(0,-8,4)
base.cam.setHpr(0,-22,0)

base.run() # run Showbase
