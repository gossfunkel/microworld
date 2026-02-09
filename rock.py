from direct.showbase.ShowBase import ShowBase

from panda3d.core import (
    loadPrcFileData, GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    GeomEnums, Geom, GeomNode, NodePath, ModelRoot, BoundingBox, GeomTrifans
)
import numpy as np
import struct

TAU             = np.pi * 2.
ROCK_VTX_FORMAT = GeomVertexFormat.getV3c4()

class Rock:
    def __init__(self, verts: int = 8, 
                       pos: tuple[float] = (0.,0.,0.),
                       radius: float = 1., 
                       col: tuple[float]=(.045,.025,.03,.6)):
        verts_3d = verts // 4
        self.verts = verts + verts_3d + 1                           # number of vertices in geom
        self.radius = radius

        # TODO define area at initialisation and conform random vertex positions to limited area

        # make VBO
        vtx_data   = GeomVertexData('rock_vbo', ROCK_VTX_FORMAT, Geom.UHStatic)
        vtx_data.set_num_rows(self.verts)

        # create a mesh ('Geom') from the vertices
        geom = Geom(vtx_data)

        # fill VBO with initial data and create geometry primitives
        vtx_writer = GeomVertexWriter(vtx_data, "vertex")
        col_writer = GeomVertexWriter(vtx_data, "color")
        vtx_writer.add_data3(0.,0.,0.)
        col_writer.add_data4(col)
        for point in range(verts):
            random_radius = np.random.uniform(.6,1.4)
            rad = random_radius + self.radius
            vtx_writer.add_data3(np.cos(TAU*point/verts)*random_radius,
                                 np.sin(TAU*point/verts)*random_radius, 
                                 0.)
            col_writer.add_data4(col)                               # TODO add some colour variation

        prim = GeomTrifans(Geom.UHStatic)
        prim.add_consecutive_vertices(0,verts+1)                      # add the flat verts
        prim.add_vertex(1)                                          # close the flat shape
        prim.closePrimitive()
        geom.addPrimitive(prim)

        # build the 3D geometry TODO this only works for certain numbers of input verts
        for point in range(verts_3d):
            updown = 2.*(point%2)                                   # half above, half below
            vtx_writer.add_data3(np.random.uniform(-.3,.3),
                                 np.random.uniform(-.3,.3),
                                 (1. - updown)/2.)
            col_writer.add_data4(col[0],col[1],col[2], col[3]*.5)   # reduce opacity of 3d verts
        
        prim_up   = GeomTrifans(Geom.UHStatic)
        prim_up.add_vertex(verts+1)
        prim_up.add_consecutive_vertices(1,verts)
        prim_up.add_vertex(1)
        #prim_up.add_vertex(1)
        prim_up.closePrimitive()
        geom.addPrimitive(prim_up)
        
        prim_down = GeomTrifans(Geom.UHStatic)
        prim_down.add_vertex(verts+2)
        prim_down.add_consecutive_vertices(1,verts)
        prim_down.add_vertex(1)
        #prim_down.add_vertex(1)
        prim_down.closePrimitive()
        geom.addPrimitive(prim_down)
        
        geom.set_bounds(BoundingBox((-1, -1, -1.5), (1, 1, 1.5)))     # set up a bounding volume to prevent culling
        geom.doublesideInPlace()
        geom_node = GeomNode('rock-geom_node')
        geom_node.addGeom(geom)

        root = ModelRoot("Rock_model_root")                         # Ensure mesh has a root for saving
        root.addChild(geom_node)
        
        self.nodepath = NodePath(root)                              # create a new nodepath for the model
        
        #self.nodepath.setDepthOffset(1)                             # give the node a depth offset to prevent self-shadowing etc
        self.nodepath.set_transparency(1)                           # enable transparency
        self.nodepath.reparent_to(base.render)
        self.nodepath.set_pos(pos)

def update(task):
    base.cam.set_pos(np.sin(task.frame/100)*5,-np.cos(task.frame/100)*5,2.)
    base.cam.look_at(base.test_rock.nodepath)
    return task.cont

if __name__ == '__main__':
    CONFIG: str     = """
    framebuffer-srgb true
    """
    loadPrcFileData("", CONFIG)

    ShowBase()
    base.disableMouse() 

    base.test_rock = Rock()

    base.taskMgr.add(update, "update-cam")

    base.cam.set_pos(0.,-5.,2.)
    base.cam.set_hpr(0.,-25.,0.)

    base.run()