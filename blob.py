from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, NodePath, Vec2, Vec3, InternalName
from panda3d.core import Geom, GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData, GeomEnums
from panda3d.core import GeomTrifans, GeomNode
import struct
import numpy as np
#import array

epsilon = 0.0001
dampRatio = 0.8 # constant variable that sets springyness of object
distBetweenEdgepoints = 0.517472489875 # hopefully should be compatible with the radius
volumeScaleFactor = 1.0 - epsilon
radius = 1.0 - epsilon

config_vars: str = """
win-size 1200 800
show-frame-rate-meter 1
hardware-animated-vertices true
framebuffer-srgb true
model-cache-dir
gl-debug 1
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

#def calcForce(pos, point)

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
		self.pos: Vec2 = pos
		self.vertexData = GeomVertexData(name+'-verts', vertexFormat, Geom.UHStatic)
		self.vertexData.unclean_set_num_rows(13) # 1 row per vertex (12 rim, 1 centre)

		# vertices: Numpy.Array = np.array([[pos.x, pos.y, 0.],
		# 								[pos.x - .866, pos.y - .5, 0.],
		# 								[pos.x - .5, pos.y - .866, 0.],
		# 								[pos.x, pos.y - 1., 0.],
		# 								[pos.x + .5, pos.y - .866, 0.],
		# 								[pos.x + .866, pos.y - .5, 0.],
		# 								[pos.x + 1., pos.y, 0.],
		# 								[pos.x + .866, pos.y + .5, 0.],
		# 								[pos.x + .5, pos.y + .866, 0.],
		# 								[pos.x, pos.y + 1., 0.],
		# 								[pos.x - .5, pos.y + .866, 0.],
		# 								[pos.x - .866, pos.y + .5, 0.],
		# 								[pos.x - 1., pos.y, 0.]], dtype='f')
		# vertexNorms: Numpy.Array = np.array([[0.,0.,1.], ] * 13, dtype='f')
		#vertexCols: Numpy.Array = np.array([[col[0], col[1], col[2], 255], ] * 13)

		# create memoryview for float assignment - shouldn't need to cast('f') due to using structs
		#vertexView = memoryview(self.vertexData.modify_array(0)).cast('B').cast('f')
		positionView = memoryview(self.vertexData.modify_array(0)).cast('B')#.cast('f')
		#positionView[:] = vertices
		normalView = memoryview(self.vertexData.modify_array(1)).cast('B')#.cast('f')
		#normalView[:] = vertexNorms
		
		#positionView[:] = np.hstack((vertices, vertexNorms)).astype(np.float32)
		colourView = memoryview(self.vertexData.modify_array(2)).cast('B')

		vertexValues = bytearray()
		# positionView[:] = vertexValues

		vertexValues.extend(struct.pack(
			'3f',
			pos.x, pos.y, 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x - .866, pos.y - .5, 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x - .5, pos.y - .866, 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x, pos.y - 1, 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x + .5, pos.y - .866, 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x + .866, pos.y - .5, 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x + 1, pos.y, 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x + .866, pos.y + .5, 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x + .5, pos.y + .866, 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x, pos.y + 1, 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x - .5, pos.y + .866, 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x - .866, pos.y + .5, 0.))
		vertexValues.extend(struct.pack(
			#'6f4B',
			'3f',
			pos.x - 1, pos.y, 0.,))
		#	col[0],col[1],col[2],255))
		# write values to buffer via memoryview
		positionView[:] = vertexValues

		# now pack the normals the same way
		normalValues = bytearray()
		colValues 	 = bytearray()
		for _ in range(13):
			normalValues.extend(struct.pack('3f', 0.,0.,1.))
			colValues.extend(struct.pack('4B', col[0], col[1], col[2], 255))
		normalView[:] = normalValues
		colourView[:] = colValues

		#self.vertexData = self.vertexData.set_color(col[0], col[1], col[2], 255)

		#vertexData.format(GeomVertexFormat.getV3n3c4())

		# finally, create a mesh ('Geom') from the vertices- containing one trifan defined above as blobPrim
		geom = Geom(self.vertexData)
		geom.addPrimitive(blobPrim)
		self.geomNode = GeomNode(name+'-geomnode')
		self.geomNode.addGeom(geom)
		self.nodepath = base.render.attach_new_node(self.geomNode)

		self.velocity = Vec2(0.,0.)

		base.taskMgr.add(self.update, str(name)+"-update", taskChain='default')

	def update(self, task):
		#floatView = memoryview(self.vertexData.modify_array(0)).cast('B').cast('f')
		floatView = memoryview(self.vertexData.modify_array(0)).cast('B').cast('f')
		#print(floatView.itemsize)
		# trapezoid integration
		area: float = 0.0
		for vertex in range(12):
			vertex *= 3
			vertex += 1
			# area = d(x) * avg(y), i.e. (x1 - x2) * (y1 + y2) / 2
			area += (floatView[(vertex+3)%36] - floatView[vertex]) * ((floatView[(vertex+4)%36] + floatView[vertex+1])/2.)
		#volumeScale: float = volumeScaleFactor * (3.1415926535 - area) # area of a circle of radius 1 is pi
		volumeScale: float = 1.0
		# force calculation
		for vertex in range(12):
			vertex *= 3
			#vertex += 1

			# pos: Vec3 = Vec3(floatView[vertex], floatView[vertex+1], floatView[vertex+2])
			# centrepoint: Vec3 = Vec3(floatView[0],floatView[1],floatView[2])
			# neighbour1: Vec3 = Vec3(floatView[(vertex-3)%36],floatView[(vertex-2)%36], floatView[(vertex-1)%36])
			# neighbour2: Vec3 = Vec3(floatView[(vertex+3)%36],floatView[(vertex+2)%36],floatView[(vertex+1)%36])

			# midCentrepoint: Vec3 = Vec3((pos.x+centrepoint.x) / 2,(pos.y+centrepoint.y) / 2,(0.+centrepoint.z) / 2)
			# diffCentrepoint: Vec3 = Vec3(pos.x - centrepoint.x, pos.y - centrepoint.y, 0. - centrepoint.z)
			# distCentrepoint: float = np.sqrt(diffCentrepoint.x*diffCentrepoint.x + diffCentrepoint.y*diffCentrepoint.y + diffCentrepoint.z*diffCentrepoint.z)
			# centrepointForceMag: float = (radius - distCentrepoint) * volumeScale
			# centrepointForce: Vec3 = (diffCentrepoint / np.linalg.norm(diffCentrepoint)) * centrepointForceMag

			# midNeighbour1: Vec3 = Vec3((pos.x+neighbour1.x) / 2,(pos.y+neighbour1.y) / 2,(0.+neighbour1.z) / 2)
			# diffNeighbour1: Vec3 = Vec3(pos.x - neighbour1.x, pos.y - neighbour1.y, 0. - neighbour1.z)
			# distNeighbour1: float = np.sqrt(diffNeighbour1.x*diffNeighbour1.x + diffNeighbour1.y*diffNeighbour1.y + diffNeighbour1.z*diffNeighbour1.z)
			# neighbour1Force: Vec3 = (diffNeighbour1 / np.linalg.norm(diffNeighbour1)) * (radius - distNeighbour1)

			# midNeighbour2: Vec3 = Vec3((pos.x+neighbour2.x) / 2,(pos.y+neighbour2.y) / 2,(0.+neighbour2.z) / 2)
			# diffNeighbour2: Vec3 = Vec3(pos.x - neighbour2.x, pos.y - neighbour2.y, 0. - neighbour2.z)
			# distNeighbour2: float = np.sqrt(diffNeighbour2.x*diffNeighbour2.x + diffNeighbour2.y*diffNeighbour2.y + diffNeighbour2.z*diffNeighbour2.z)
			# neighbour2Force: Vec3 = (diffNeighbour2 / np.linalg.norm(diffNeighbour2)) * (radius - distNeighbour2)

			# averageForce: Vec3 = Vec3((centrepointForce.x + neighbour1Force.x + neighbour2Force.x) / 3.,
			# 							(centrepointForce.y + neighbour1Force.y + neighbour2Force.y) / 3.,
			# 							(centrepointForce.z + neighbour1Force.z + neighbour2Force.z) / 3.)
			# avgMagnitude: float = np.sqrt(averageForce.x*averageForce.x+averageForce.y*averageForce.y+averageForce.z*averageForce.z)
			pos: Vec2 = Vec2(floatView[vertex], floatView[vertex+1])
			centrepoint: Vec2 = Vec2(floatView[0],floatView[1])
			neighbour1: Vec2 = Vec2(floatView[(vertex-3)%36],floatView[(vertex-2)%36])
			neighbour2: Vec2 = Vec2(floatView[(vertex+3)%36],floatView[(vertex+4)%36])

			midCentrepoint: Vec2 = Vec2((pos.x+centrepoint.x) / 2,(pos.y+centrepoint.y) / 2)
			diffCentrepoint: Vec2 = Vec2(pos.x - centrepoint.x, pos.y - centrepoint.y)
			distCentrepoint: float = np.sqrt(diffCentrepoint.x*diffCentrepoint.x + diffCentrepoint.y*diffCentrepoint.y)
			print("distance to centrepoint: " + str(distCentrepoint))
			centrepointForceMag: float = np.absolute((radius - distCentrepoint) * volumeScale)
			centrepointForce: Vec2 = (diffCentrepoint / np.linalg.norm(diffCentrepoint)) * centrepointForceMag
			print("force from centrepoint: " + str(centrepointForce))

			midNeighbour1: Vec2 = Vec2((pos.x + neighbour1.x) / 2,(pos.y+neighbour1.y) / 2)
			diffNeighbour1: Vec2 = Vec2(pos.x - neighbour1.x, pos.y - neighbour1.y)
			distNeighbour1: float = np.sqrt(diffNeighbour1.x*diffNeighbour1.x + diffNeighbour1.y*diffNeighbour1.y)
			neighbour1Force: Vec2 = (diffNeighbour1 / np.linalg.norm(diffNeighbour1)) * (radius - distNeighbour1)
			print("force from neighbour1: " + str(neighbour1Force))

			midNeighbour2: Vec2 = Vec2((pos.x+neighbour2.x) / 2,(pos.y+neighbour2.y) / 2)
			diffNeighbour2: Vec2 = Vec2(pos.x - neighbour2.x, pos.y - neighbour2.y)
			distNeighbour2: float = np.sqrt(diffNeighbour2.x*diffNeighbour2.x + diffNeighbour2.y*diffNeighbour2.y)
			neighbour2Force: Vec2 = (diffNeighbour2 / np.linalg.norm(diffNeighbour2)) * (radius - distNeighbour2)
			print("force from neighbour2: " + str(neighbour2Force))

			#sumForce: Vec2 = Vec2((centrepointForce.getX() + neighbour1Force.getX() + neighbour2Force.getX()) / 3.,
			#						(centrepointForce.getY() + neighbour1Force.getY() + neighbour2Force.getY()) / 3.)
			sumForce: Vec2 = (centrepointForce + neighbour1Force + neighbour2Force) / 3.
			print(">>> total force vector: " + str(sumForce))
			print(">>> total force components: [x: " + str(sumForce.getX()) + ", y: " + str(sumForce.getY()) + "]")
			if np.isnan(sumForce.x): sumForce.x = 0.
			if np.isnan(sumForce.y): sumForce.y = 0.
			averageForce: float = sumForce.x * sumForce.x + sumForce.y * sumForce.y
			print("average force: " + str(averageForce))
			avgMagnitude: float = np.sqrt(averageForce)
			print("average force magnitude: " + str(avgMagnitude))
			self.velocity += Vec2(sumForce.x,sumForce.y)
			#pos: Vec3 = Vec3(pos.x,pos.y,0.)
			pos, self.velocity = calcDampedSHM(pos,self.velocity,pos + sumForce,globalClock.getDt(),avgMagnitude)

			floatView[vertex]   = pos.x if not np.isnan(pos.x) else 0
			floatView[vertex+1] = pos.y if not np.isnan(pos.y) else 0
			floatView[vertex+2] = 0. # pos.z
		return task.cont

	def move(self, direction) -> bool:
		#print(self.view[1].to_bytes())
		floatView = memoryview(self.vertexData.modify_array(0)).cast('B').cast('f')
		if direction == "left":
			# go left
			xpos = floatView[0]
			print("left: " + str(xpos))
			floatView[0] = xpos - .05
			return 1
		elif direction == "right":
			# go right
			xpos = floatView[0]
			print("right: " + str(xpos))
			floatView[0] = xpos + .05
			return 1
		elif direction == "fwd":
			# go forwards
			ypos = floatView[1]
			print("fwd: " + str(ypos))
			floatView[1] = ypos + .05
			return 1
		elif direction == "back":
			# ...you guessed it
			ypos = floatView[1]
			print("back: " + str(ypos))
			floatView[1] = ypos - .05
			return 1
		else: return 0

#def move_p1_left():
#	p1.pos -= (1,0,0)
#	#view = memoryview(vertexArray)

ShowBase() # Showbase initialised

p1: Player = Player("p1",Vec3(0.,-5.,0.),[0,0,255])
p2: Player = Player("p2",Vec3(0.,5.,0.),[255,0,0])

base.accept("arrow_left", p1.move, ["left"])
base.accept("arrow_left-repeat", p1.move, ["left"])
base.accept("a", p1.move, ["left"])
base.accept("a-repeat", p1.move, ["left"])
base.accept("arrow_right", p1.move, ["right"])
base.accept("arrow_right-repeat", p1.move, ["right"])
base.accept("d", p1.move, ["right"])
base.accept("d-repeat", p1.move, ["right"])
base.accept("arrow_up", p1.move, ["fwd"])
base.accept("arrow_up-repeat", p1.move, ["fwd"])
base.accept("w", p1.move, ["fwd"])
base.accept("w-repeat", p1.move, ["fwd"])
base.accept("arrow_down", p1.move, ["back"])
base.accept("arrow_down-repeat", p1.move, ["back"])
base.accept("s", p1.move, ["back"])
base.accept("s-repeat", p1.move, ["back"])

base.cam.setPos(0,-18,5)
base.cam.setHpr(0,-15,0)

base.run() # run Showbase
