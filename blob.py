from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, NodePath, Vec3, InternalName
from panda3d.core import Geom, GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData, GeomEnums
from panda3d.core import GeomTrifans, GeomNode
import struct
import numpy as np
#import array

epsilon = 0.0001
dampRatio = 0.5 # constant variable that sets springyness of object
distBetweenEdgepoints = 0.517472489875 # hopefully should be compatible with the radius
volumeScaleFactor = 1.0
radius = 1.0

config_vars: str = """
win-size 1200 800
show-frame-rate-meter 1
hardware-animated-vertices true
framebuffer-srgb true
model-cache-dir
basic-shaders-only false
//threading-model Cull/Draw
"""
loadPrcFileData("", config_vars)

blobPrim = GeomTrifans(Geom.UHStatic)
blobPrim.add_consecutive_vertices(0,13)
blobPrim.add_vertex(1)
blobPrim.closePrimitive()
#vertexFormat = GeomVertexFormat.getV3n3c4()
vertexFormat = GeomVertexFormat()
arrayFormat = GeomVertexArrayFormat()
arrayFormat.add_column(InternalName.get_vertex(),3,GeomEnums.NT_float32, GeomEnums.C_point)
vertexFormat.add_array(arrayFormat)
arrayFormat = GeomVertexArrayFormat()
arrayFormat.add_column(InternalName.get_normal(),3,GeomEnums.NT_float32, GeomEnums.C_point)
vertexFormat.add_array(arrayFormat)
arrayFormat = GeomVertexArrayFormat()
arrayFormat.add_column(InternalName.get_color(),4,GeomEnums.NT_uint8, GeomEnums.C_color)
vertexFormat.add_array(arrayFormat)
vertexFormat = GeomVertexFormat.register_format(vertexFormat)

#stride = vertexFormat.arrays[0].stride # size of data row in bytes

def calcDampedSHM(pos,vel,equilibriumPos,deltaTime,angularFreq):
	assert (angularFreq >= 0.), f'SHM angular frequency parameter must be positive!'
	assert (dampRatio >= 0.), f'SHM damping ratio parameter must be positive!'

	if (angularFreq < epsilon):
		print("SHM frequency too low to change motion!")
		pospos = 1.
		posvel = 0.
		velpos = 0.
		velvel = 1.
	else:
		if (dampRatio > 1. + epsilon):
			# overdamped formula
			za = -angularFreq * dampRatio
			zb = angularFreq * np.sqrt(dampRatio*dampRatio - 1.)
			z1 = za - zb
			z2 = za + zb

			e1 = np.exp(z1 * deltaTime)
			e2 = np.exp(z2 * deltaTime)

			invTwoZb = 1. / (2. * zb)

			e1OverTwoZb = e1 * invTwoZb
			e2OverTwoZb = e2 * invTwoZb

			z1e1OverTwoZb = z1 * e1OverTwoZb
			z2e2OverTwoZb = z2 * e2OverTwoZb

			pospos = e1OverTwoZb * z2e2OverTwoZb + e2OverTwoZb
			posvel = -e1OverTwoZb + e2OverTwoZb
			velpos = (z1e1OverTwoZb - z2e2OverTwoZb + e2) * z2 
			velvel = -z1e1OverTwoZb + z2e2OverTwoZb
		elif (dampRatio < 1. - epsilon):
			# underdamped formula
			omegaZeta = angularFreq * dampRatio
			alpha 	  = angularFreq * np.sqrt(1. - dampRatio * dampRatio)

			expTerm = np.exp(-omegaZeta * deltaTime)
			cosTerm = np.cos(alpha * deltaTime)
			sinTerm = np.sin(alpha * deltaTime)

			invAlpha = 1. / alpha 

			expSin = expTerm * sinTerm
			expCos = expTerm * cosTerm
			expOmegaZetaSinOverAlpha = expTerm * omegaZeta * sinTerm * invAlpha

			pospos = expCos + expOmegaZetaSinOverAlpha
			posvel = expSin * invAlpha
			velpos = -expSin * alpha - omegaZeta * expOmegaZetaSinOverAlpha
			velvel = expCos - expOmegaZetaSinOverAlpha
		else:
			# critically damped formula
			expTerm = np.exp(-angularFreq * deltaTime)
			timeExp = deltaTime * expTerm
			timeExpFreq = timeExp * angularFreq

			pospos = timeExpFreq + expTerm
			posvel = timeExp
			velpos = -angularFreq * timeExpFreq
			velvel = -timeExpFreq + expTerm

	pos = pos - equilibriumPos
	oldvel = vel
	vel = pos * velpos + oldvel * velvel
	pos = pos * pospos + oldvel * posvel + equilibriumPos

	return pos, vel

class Player():
	def __init__(self, name, pos, col) -> None:
		self.pos: Vec3 = pos
		self.vertexData = GeomVertexData(name+'-verts', vertexFormat, Geom.UHStatic)
		self.vertexData.unclean_set_num_rows(13) # 1 row per vertex (12 rim, 1 centre)

		# vertices: Numpy.Array = np.array([[pos.x, pos.y, pos.z],
		# 								[pos.x - .866, pos.y - .5, pos.z],
		# 								[pos.x - .5, pos.y - .866, pos.z],
		# 								[pos.x, pos.y - 1., pos.z],
		# 								[pos.x + .5, pos.y - .866, pos.z],
		# 								[pos.x + .866, pos.y - .5, pos.z],
		# 								[pos.x + 1., pos.y, pos.z],
		# 								[pos.x + .866, pos.y + .5, pos.z],
		# 								[pos.x + .5, pos.y + .866, pos.z],
		# 								[pos.x, pos.y + 1., pos.z],
		# 								[pos.x - .5, pos.y + .866, pos.z],
		# 								[pos.x - .866, pos.y + .5, pos.z],
		# 								[pos.x - 1., pos.y, pos.z]], dtype='f')
		# vertexNorms: Numpy.Array = np.array([[0.,0.,1.], ] * 13, dtype='f')
		#vertexCols: Numpy.Array = np.array([[col[0], col[1], col[2], 255], ] * 13)

		# create memoryview for float assignment - shouldn't need to cast('f') due to using structs
		#vertexView = memoryview(self.vertexData.modify_array(0)).cast('B').cast('f')
		positionView = memoryview(self.vertexData.modify_array(0)).cast('B')#.cast('f')
		#positionView[:] = vertices
		normalView = memoryview(self.vertexData.modify_array(1)).cast('B')#.cast('f')
		#normalView[:] = vertexNorms
		#vertexData = vertexData.set_color(col[0], col[1], col[2], 255)
		#positionView[:] = np.hstack((vertices, vertexNorms)).astype(np.float32)
		#vertexData.format(GeomVertexFormat.getV3n3c4())
		#colourView = memoryview(self.vertexData.modify_array(2)).cast('b')

		vertexValues = bytearray()
		# 	colValues.extend(struct.pack('4B', ))
		# positionView[:] = vertexValues
		
		# colourView[:] 	= colValues

		vertexValues.extend(struct.pack(
			'3f',
			pos.x, pos.y, pos.z))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x - .866, pos.y - .5, pos.z))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x - .5, pos.y - .866, pos.z))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x, pos.y - 1, pos.z))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x + .5, pos.y - .866, pos.z))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x + .866, pos.y - .5, pos.z))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x + 1, pos.y, pos.z))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x + .866, pos.y + .5, pos.z))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x + .5, pos.y + .866, pos.z))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x, pos.y + 1, pos.z))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x - .5, pos.y + .866, pos.z))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x - .866, pos.y + .5, pos.z))
		vertexValues.extend(struct.pack(
			#'6f4B',
			'3f',
			pos.x - 1, pos.y, pos.z,))
		#	col[0],col[1],col[2],255))
		# write values to buffer via memoryview
		positionView[:] = vertexValues

		# now pack the normals the same way
		normalValues = bytearray()
		for _ in range(13):
			normalValues.extend(struct.pack('3f', 0.,0.,1.))
		normalView[:] 	= normalValues

		# finally, create a mesh ('Geom') from the vertices- containing one trifan defined above as blobPrim
		geom = Geom(self.vertexData)
		geom.addPrimitive(blobPrim)
		self.geomNode = GeomNode(name+'-geomnode')
		self.geomNode.addGeom(geom)
		self.nodepath = base.render.attach_new_node(self.geomNode)

		self.velocity = Vec3(0.,0.,0.)

		base.taskMgr.add(self.update, str(name)+"-update", taskChain='default')

	def update(self, task):
		#floatView = memoryview(self.vertexData.modify_array(0)).cast('B').cast('f')
		floatView = memoryview(self.vertexData.modify_array(0)).cast('B').cast('f')
		#print(floatView.itemsize)
		# trapezoid integration
		area: float = 0.0
		for vertex in range(12):
			vertex += 1
			vertex *= 3
			# area = d(x) * avg(y), i.e. (x1 - x2) * (y1 + y2) / 2
			area += (floatView[(vertex+3)%36] - floatView[vertex]) * ((floatView[(vertex+4)%36] + floatView[vertex+1] )/2.)
		volumeScale: float = volumeScaleFactor * (3.1415926535 - area) # area of a circle of radius 1 is pi
		# force calculation
		for vertex in range(12):
			vertex += 1
			vertex *= 3
			pos: Vec3 = Vec3(floatView[vertex], floatView[vertex+1], floatView[vertex+2])
			centrepoint: Vec3 = Vec3(floatView[0],floatView[1],floatView[2])
			neighbour1: Vec3 = Vec3(floatView[(vertex-3)%36],floatView[(vertex-2)%36], floatView[(vertex-1)%36])
			neighbour2: Vec3 = Vec3(floatView[(vertex+3)%36],floatView[(vertex+2)%36],floatView[(vertex+1)%36])

			midCentrepoint: Vec3 = Vec3((pos.x+centrepoint.x) / 2,(pos.y+centrepoint.y) / 2,(pos.z+centrepoint.z) / 2)
			diffCentrepoint: Vec3 = Vec3(pos.x - centrepoint.x, pos.y - centrepoint.y, pos.z - centrepoint.z)
			distCentrepoint: float = np.sqrt(diffCentrepoint.x*diffCentrepoint.x + diffCentrepoint.y*diffCentrepoint.y + diffCentrepoint.z*diffCentrepoint.z)
			centrepointForceMag: float = (radius - distCentrepoint) * volumeScale
			centrepointForce: Vec3 = (diffCentrepoint / np.linalg.norm(diffCentrepoint)) * centrepointForceMag

			midNeighbour1: Vec3 = Vec3((pos.x+neighbour1.x) / 2,(pos.y+neighbour1.y) / 2,(pos.z+neighbour1.z) / 2)
			diffNeighbour1: Vec3 = Vec3(pos.x - neighbour1.x, pos.y - neighbour1.y, pos.z - neighbour1.z)
			distNeighbour1: float = np.sqrt(diffNeighbour1.x*diffNeighbour1.x + diffNeighbour1.y*diffNeighbour1.y + diffNeighbour1.z*diffNeighbour1.z)
			neighbour1Force: Vec3 = (diffNeighbour1 / np.linalg.norm(diffNeighbour1)) * (radius - distNeighbour1)

			midNeighbour2: Vec3 = Vec3((pos.x+neighbour2.x) / 2,(pos.y+neighbour2.y) / 2,(pos.z+neighbour2.z) / 2)
			diffNeighbour2: Vec3 = Vec3(pos.x - neighbour2.x, pos.y - neighbour2.y, pos.z - neighbour2.z)
			distNeighbour2: float = np.sqrt(diffNeighbour2.x*diffNeighbour2.x + diffNeighbour2.y*diffNeighbour2.y + diffNeighbour2.z*diffNeighbour2.z)
			neighbour2Force: Vec3 = (diffNeighbour2 / np.linalg.norm(diffNeighbour2)) * (radius - distNeighbour2)

			averageForce: Vec3 = Vec3((centrepointForce.x + neighbour1Force.x + neighbour2Force.x) / 3.,
										(centrepointForce.y + neighbour1Force.y + neighbour2Force.y) / 3.,
										(centrepointForce.z + neighbour1Force.z + neighbour2Force.z) / 3.)
			avgMagnitude: float = np.sqrt(averageForce.x*averageForce.x+averageForce.y*averageForce.y+averageForce.z*averageForce.z)
			self.velocity += averageForce
			pos, self.velocity = calcDampedSHM(pos,self.velocity,pos + averageForce,globalClock.getDt(),1./avgMagnitude)

			floatView[vertex]   = pos.x
			floatView[vertex+1] = pos.y
			floatView[vertex+2] = pos.z
		return task.cont

	def move(self, direction) -> bool:
		#print(self.view[1].to_bytes())
		floatView = memoryview(self.vertexArray).cast('B').cast('f')
		if direction == "left":
			# go left
			xpos = floatView[0]
			print("left: " + str(xpos))
			floatView[0] = xpos - 1.
			return 1
		elif direction == "right":
			# go right
			xpos = floatView[0]
			print("right: " + str(xpos))
			floatView[0] = xpos + 1.
			return 1
		elif direction == "fwd":
			# go forwards
			ypos = floatView[1]
			print("fwd: " + str(ypos))
			floatView[1] = ypos + 1.
			return 1
		elif direction == "back":
			# ...you guessed it
			ypos = floatView[1]
			print("back: " + str(ypos))
			floatView[1] = ypos - 1.
			return 1
		else: return 0

#def move_p1_left():
#	p1.pos -= (1,0,0)
#	#view = memoryview(vertexArray)

ShowBase() # Showbase initialised

p1: Player = Player("p1",Vec3(0,-5,0),[0,0,255])
p2: Player = Player("p2",Vec3(0,5,0),[255,0,0])

base.accept("arrow_left-repeat", p1.move, ["left"])
base.accept("a-repeat", p1.move, ["left"])
base.accept("arrow_right-repeat", p1.move, ["right"])
base.accept("d-repeat", p1.move, ["right"])
base.accept("arrow_up-repeat", p1.move, ["fwd"])
base.accept("w-repeat", p1.move, ["fwd"])
base.accept("arrow_down-repeat", p1.move, ["back"])
base.accept("s-repeat", p1.move, ["back"])

base.cam.setPos(0,-18,5)
base.cam.setHpr(0,-15,0)

base.run() # run Showbase
