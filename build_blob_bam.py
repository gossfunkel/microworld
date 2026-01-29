from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    loadPrcFileData, Vec3, Vec4, Geom, GeomNode, GeomEnums, NodePath, BoundingBox,
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

# vtx_format = GeomVertexFormat.getV3n3c4()
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
#stride = 24

vals = bytearray()
for i in range(13):
    # populate the bytearray with each row
    vals.extend(struct.pack('4f', BASIS_VECS[i*2], BASIS_VECS[i*2 + 1], 0., 1.))
    vals.extend(struct.pack('4f', 0.,0.,1.,0.))
    vals.extend(struct.pack('4f', 1., 1., 1., 1.))
    vals.extend(struct.pack('2f', BASIS_VECS[i*2], BASIS_VECS[i*2 + 1]))
    vals.extend(struct.pack('2f', 0., 0.))

# write to VBO
view[:] = vals

# finally, create a mesh ('Geom') from the vertices- containing one trifan defined above as blobPrim
geom     = Geom(vtx_data)
blobPrim = GeomTrifans(Geom.UHStatic)
blobPrim.add_consecutive_vertices(0,13)
blobPrim.add_vertex(1)
blobPrim.closePrimitive()
geom.addPrimitive(blobPrim)
# set up a bounding volume to prevent culling
geom.set_bounds(BoundingBox((-1, -1, -.5), (1, 1, .5)))
geom.doublesideInPlace()
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
