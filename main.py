from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import numpy as np

config_vars: str = """
win-size 1200 800
show-frame-rate-meter 1
hardware-animated-vertices true
//framebuffer-srgb true
model-cache-dir
basic-shaders-only false
threading-model Cull/Draw
"""

loadPrcFileData("", config_vars)
app = ShowBase()

class Cell():
    def __init__(self, num, position, colour) -> None:
        self.num: int = num
        self.position = position
        self.colour: Vec3 = colour
        self.size: float = 1.0
        self.ph: float = 6.0
        self.speed: Vec3 = Vec3(0.,0.,0.)

        # format contains vertex (3), normal (3), and colour (4)
        format = GeomVertexFormat.getV3n3c4()
        vdata  = GeomVertexData('cell-'+str(num), format, Geom.UHDynamic)
        vdata.setNumRows(14)
        vert_vertex = GeomVertexWriter(vdata, 'vertex')
        vert_normal = GeomVertexWriter(vdata, 'normal')
        vert_color  = GeomVertexWriter(vdata, 'color')

        # for i in range(numPoints):
        #     vert_vertex.addData3(np.cos((i/numPoints)*360)*np.sin(i/(numPoints/2)) + position[0], 
        #                          np.sin((i/numPoints)*360)*np.sin(i/(numPoints/2)) + position[1], 
        #                          position[2])# - (i%3))
        #     vert_normal.addData3(0,0,1)#np.cos((i/numPoints)*360)*np.sin(i/(numPoints/2))*2. * position[0], 
        #                          #np.sin((i/numPoints)*360)*np.sin(i/(numPoints/2))*2. * position[1], 
        #                          #2 * (1 + position[2] - (i%3)))
        #v0
        vert_vertex.addData3(position[0],position[1],position[2])
        vert_normal.addData3(0,0,1)
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v1
        vert_vertex.addData3(position[0],position[1],position[2]-1)
        vert_normal.addData3(0,0,-1)
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v2
        vert_vertex.addData3(position[0]+.5,position[1]+1.118,position[2])
        vert_normal.addData3(.5,1.118,1)#TODO
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v3
        vert_vertex.addData3(position[0]+1,position[1],position[2])
        vert_normal.addData3(1,0,1)#TODO
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v4
        vert_vertex.addData3(position[0]+.5,position[1]-1.118,position[2])
        vert_normal.addData3(.5,-1.118,1)#TODO
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v5
        vert_vertex.addData3(position[0]-.5,position[1]-1.118,position[2])
        vert_normal.addData3(-.5,-1.118,1)#TODO
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v6
        vert_vertex.addData3(position[0]-1,position[1],position[2])
        vert_normal.addData3(-1,0,1)#TODO
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v7
        vert_vertex.addData3(position[0]-.5,position[1]+1.118,position[2])
        vert_normal.addData3(-.5,1.118,1)#TODO
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v8
        vert_vertex.addData3(position[0]+.5,position[1]+1.118,position[2]-1)
        vert_normal.addData3(.5,1.118,-1)#TODO
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v9
        vert_vertex.addData3(position[0]+1,position[1],position[2]-1)
        vert_normal.addData3(1,0,-1)#TODO
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v10
        vert_vertex.addData3(position[0]+.5,position[1]-1.118,position[2]-1)
        vert_normal.addData3(.5,-1.118,-1)#TODO
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v11
        vert_vertex.addData3(position[0]-.5,position[1]-1.118,position[2]-1)
        vert_normal.addData3(-.5,-1.118,-1)#TODO
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v12
        vert_vertex.addData3(position[0]-1,position[1],position[2]-1)
        vert_normal.addData3(-1,0,-1)#TODO
        vert_color.addData4(colour[0], colour[1], colour[2], 1)
        #v13
        vert_vertex.addData3(position[0]-.5,position[1]+1.118,position[2]-1)
        vert_normal.addData3(-.5,1.118,-1) #TODO
        vert_color.addData4(colour[0], colour[1], colour[2], 1)

        cellGeom = Geom(vdata)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(0,2,3)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(0,3,4)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(0,4,5)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(0,5,6)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(0,6,7)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(0,7,2)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(1,8,9)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(1,9,10)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(1,10,11)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(1,11,12)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(1,12,13)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(1,13,8)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(2,3,8)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(8,9,3)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(3,4,9)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(4,5,10)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(10,11,5)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(5,6,11)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(11,12,6)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(6,7,12)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(12,13,7)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(7,2,13)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)
        prim = GeomTriangles(Geom.UHStatic)
        prim.add_vertices(13,8,2)
        prim.closePrimitive()
        cellGeom.addPrimitive(prim)

        geomNode = GeomNode('cellGeomNode')
        geomNode.addGeom(cellGeom)
        self.geomNP = base.render.attachNewNode(geomNode)

testCell = Cell(0, Vec3(0,2,0), Vec3(0,0,1))
base.camera.setPos(0,-2,0)

app.run()
