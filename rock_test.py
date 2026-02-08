from direct.showbase.ShowBase import ShowBase

from panda3d.core import (
    loadPrcFileData, GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    GeomEnums, Geom, GeomNode, NodePath, ModelRoot, BoundingBox, GeomTrifans
)
import numpy as np
import struct

TAU         = np.pi * 2.
CONFIG: str = """
framebuffer-srgb true
"""
loadPrcFileData("", CONFIG)

if __name__ == '__main__':
    ShowBase()
    #base.disableMouse() 

    rock_verts: int = 8 # number of vertices in geom

    # set up vertex format and make VBO
    vtx_format = GeomVertexFormat.getV3c4()
    vtx_data   = GeomVertexData('rock_vbo', vtx_format, Geom.UHStatic)
    vtx_data.set_num_rows(rock_verts)

    # fill VBO with initial data and create geometry primitives
    vtx_writer = GeomVertexWriter(vtx_data, "vertex")
    col_writer = GeomVertexWriter(vtx_data, "color")
    for point in range(rock_verts):
        random_radius = np.random.uniform(.6,1.4)
        vtx_writer.add_data3(np.cos(TAU*point/rock_verts)*random_radius,
                             np.sin(TAU*point/rock_verts)*random_radius, 
                             0.)
        col_writer.add_data4(.07,.04,.06,.6)

    # finally, create a mesh ('Geom') from the vertices
    geom = Geom(vtx_data)
    prim = GeomTrifans(Geom.UHStatic)
    prim.add_consecutive_vertices(0,rock_verts) # add all the verts
    prim.add_vertex(1) # close the circle
    prim.closePrimitive()
    geom.addPrimitive(prim)
    # set up a bounding volume to prevent culling
    geom.set_bounds(BoundingBox((-1, -1, -.5), (1, 1, .5)))
    geom.doublesideInPlace()
    geom_node = GeomNode('rock-geom_node')
    geom_node.addGeom(geom)

    # Ensure mesh has a root
    root = ModelRoot("Rock_model_root")
    root.addChild(geom_node)
    # create a new nodepath for the model
    nodepath = NodePath(root)
    # give the blob a depth offset to prevent self-shadowing etc
    nodepath.setDepthOffset(1)
    nodepath.reparent_to(base.render)
    nodepath.set_pos(0.,0.,0.)

    base.cam.set_pos(0.,-5.,2.)
    base.cam.set_hpr(0.,-25.,0.)

    base.run()