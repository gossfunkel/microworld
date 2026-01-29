from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    loadPrcFileData, Vec3, Vec4, Geom, GeomNode, GeomEnums, NodePath,
    GeomTrifans, GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData, InternalName
)
import struct

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

print("="*20 + "Welcome to the Blob Bam Generator")

CONFIG: str = """
window-type none
framebuffer-srgb true
"""
loadPrcFileData("", CONFIG)

ShowBase()

print("-- ShowBase initialised. Creating formats...")

blobPrim    = GeomTrifans(Geom.UHStatic)
blobPrim.add_consecutive_vertices(0,13)
blobPrim.add_vertex(1)
blobPrim.closePrimitive()
# vtx_format = GeomVertexFormat.getV3n3c4()
vtx_format  = GeomVertexFormat()
arrayFormat = GeomVertexArrayFormat()
arrayFormat.add_column(InternalName.get_vertex(),4,GeomEnums.NT_float32, GeomEnums.C_point)
vtx_format.add_array(arrayFormat)
arrayFormat = GeomVertexArrayFormat()
arrayFormat.add_column(InternalName.get_normal(),3,GeomEnums.NT_float32, GeomEnums.C_point)
vtx_format.add_array(arrayFormat)
arrayFormat = GeomVertexArrayFormat()
arrayFormat.add_column(InternalName.get_color(),4,GeomEnums.NT_uint8, GeomEnums.C_color)
vtx_format.add_array(arrayFormat)
arrayFormat = GeomVertexArrayFormat()
arrayFormat.add_column("basis",2,GeomEnums.NT_float32, GeomEnums.C_point)
vtx_format.add_array(arrayFormat)
vtx_format  = GeomVertexFormat.register_format(vtx_format)
vtx_data    = GeomVertexData('blob_verts', vtx_format, Geom.UHStatic)
vtx_data.unclean_set_num_rows(13) # 1 row per vertex (12 rim, 1 centre)

print("-- Formats registered. Creating geometry...")

# open memoryviews to write position, normal, and colour data to VBO
pos_view   = memoryview(vtx_data.modify_array(0)).cast('B')
norm_view  = memoryview(vtx_data.modify_array(1)).cast('B')
col_view   = memoryview(vtx_data.modify_array(2)).cast('B')
basis_view = memoryview(vtx_data.modify_array(3)).cast('B')

vtx_vals = bytearray()
basis_vals = bytearray()
for i in range(13):
    # generate circular layout with basis vectors
    vtx_vals.extend(struct.pack('4f', BASIS_VECS[i*2], BASIS_VECS[i*2 + 1], 0., 1.))
    basis_vals.extend(struct.pack('2f', BASIS_VECS[i*2], BASIS_VECS[i*2 + 1]))

# now generate and pack the normals and colours the same way
norm_vals = bytearray()
col_vals  = bytearray()
for _ in range(13):
    norm_vals.extend(struct.pack('3f', 0.,0.,1.))
    col_vals.extend(struct.pack('4B', 255, 255, 255, 255))

# write to VBO
pos_view[:] = vtx_vals
norm_view[:] = norm_vals
col_view[:]  = col_vals
basis_view[:] = basis_vals

# finally, create a mesh ('Geom') from the vertices- containing one trifan defined above as blobPrim
geom = Geom(vtx_data)
geom.addPrimitive(blobPrim)
geom_node = GeomNode('blob-geom_node')
geom_node.addGeom(geom)

print("-- Mesh made. Creating NodePath...")

# create a new node and attach to base.render
nodepath = base.render.attach_new_node(geom_node)
# store position in the node FIXME currently nodepath pos must stay at origin for the world matrix
#nodepath.set_pos(pos.x, pos.y, 0)
# give the blob a depth offset to prevent self-shadowing etc
nodepath.setDepthOffset(1)

print("-- NodePath made. Writing to file 'blob_default.bam'")

if nodepath.write_bam_file("blob_default.bam"):
    print("File written successfully!")
else: 
    print("Writing to bam file failed.")

print("="*20 + " goodbye!")
