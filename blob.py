from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, NodePath, Vec2, Vec3, InternalName
from panda3d.core import Geom, GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData, GeomEnums
from panda3d.core import GeomTrifans, GeomNode
import struct
import numpy as np
#import array

epsilon = 0.0001
dampRatio = .3 # constant variable that sets springyness of object
distBetweenEdgepoints = .51 # hopefully should be compatible with the radius
volumeScaleFactor = 1.0
radius = 1.

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
	def __init__(self, name: str, pos: Vec2, col: tuple) -> None:
		self.pos: Vec2 = pos
		self.size: float = 1.
		self.vertexData = GeomVertexData(name+'-verts', vertexFormat, Geom.UHStatic)
		self.vertexData.unclean_set_num_rows(13) # 1 row per vertex (12 rim, 1 centre)

		#  save initial positions as basis vectors for the distance to centrepoint calculations
		# 	to ensure positive and negative values scale appropriately per their angle to the centrepoint
		self.basisVecs: Numpy.Array = np.array([0., 0.,
										-.866, -.5,
										-.5, -.866,
										0., -1.,
										.5, -.866,
										.866, -.5,
										1., 0.,
										.866, .5,
										.5, .866,
										0., 1.,
										-.5, .866,
										-.866, .5,
										-1., 0.], dtype='f')
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
			pos.x+self.basisVecs[0], pos.y+self.basisVecs[1], 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x+self.basisVecs[2], pos.y+self.basisVecs[3], 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x+self.basisVecs[4], pos.y+self.basisVecs[5], 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x+self.basisVecs[6], pos.y+self.basisVecs[7], 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x+self.basisVecs[8], pos.y+self.basisVecs[9], 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x+self.basisVecs[10], pos.y+self.basisVecs[11], 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x+self.basisVecs[12], pos.y+self.basisVecs[13], 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x+self.basisVecs[14], pos.y+self.basisVecs[15], 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x+self.basisVecs[16], pos.y+self.basisVecs[17], 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x+self.basisVecs[18], pos.y+self.basisVecs[19], 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x+self.basisVecs[20], pos.y+self.basisVecs[21], 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x+self.basisVecs[22], pos.y+self.basisVecs[23], 0.))
		vertexValues.extend(struct.pack(
			'3f',
			pos.x + self.basisVecs[24], pos.y + self.basisVecs[25], 0.))
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

		self.velocities: list[Vec2] = [Vec2(0.,0.) for _ in range(12)]

		base.taskMgr.add(self.update, str(name)+"-update", taskChain='default')

	def update(self, task):
		#floatView = memoryview(self.vertexData.modify_array(0)).cast('B').cast('f')
		floatView = memoryview(self.vertexData.modify_array(0)).cast('B').cast('f')
		#print(floatView.itemsize)
		# trapezoid integration
		# area: float = 0.0
		# for vertex in range(12):
		# 	vertex *= 3
		# 	vertex += 1
		# 	# area = d(x) * avg(y), i.e. (x1 - x2) * (y1 + y2) / 2
		# 	area += (floatView[(vertex+3)%36] - floatView[vertex]) * ((floatView[(vertex+4)%36] + floatView[vertex+1])/2.)
		# volumeScale: float = volumeScaleFactor * (3.1415926535 - area) # area of a circle of radius 1 is pi
		#volumeScale: float = 1.0
		# force calculation
		for vertex in range(12):
			vel: Vec2 = self.velocities[vertex]
			dt: float = globalClock.getDt()
			vertex += 1
			basis: Vec2 = Vec2(self.basisVecs[vertex*2],self.basisVecs[vertex*2+1])
			vertex *= 3

			pos: Vec2 = Vec2(floatView[vertex], floatView[vertex+1])
			centrepoint: Vec2 = Vec2(floatView[0],floatView[1])
			neighbour1: Vec2 = Vec2(floatView[(vertex-3)%39],floatView[(vertex-2)%39])
			neighbour2: Vec2 = Vec2(floatView[(vertex+3)%39],floatView[(vertex+4)%39])

			#midCentrepoint: Vec2 = np.divide(2., pos+centrepoint)
			# diffCentrepoint: Vec2 = centrepoint - pos
			# distCentrepoint: float = np.sqrt(diffCentrepoint.x*diffCentrepoint.x + diffCentrepoint.y*diffCentrepoint.y)
			# print("distance to centrepoint: " + str(distCentrepoint))
			# centrepointForceMag: float = np.absolute(radius*self.size - np.absolute(distCentrepoint))
			# directionCentrepoint: Vec2 = diffCentrepoint.normalized()
			# print("direction to centrepoint: " + str(directionCentrepoint))
			# centrepointForce: Vec2 = directionCentrepoint.normalized() * centrepointForceMag
			# #centrepointForce: Vec2 = basis * centrepointForceMag
			# print("force from centrepoint: " + str(centrepointForce))

			#midNeighbour1: Vec2 = np.divide(2., pos + neighbour1)
			diffNeighbour1: Vec2 = neighbour1 - pos
			distNeighbour1: float = np.sqrt(diffNeighbour1.x*diffNeighbour1.x + diffNeighbour1.y*diffNeighbour1.y)
			neighbour1Force: Vec2 = diffNeighbour1.normalized() * np.absolute(distBetweenEdgepoints - np.absolute(distNeighbour1))
			if (np.absolute(neighbour1Force.x)+np.absolute(neighbour1Force.y)) < epsilon: neighbour1Force = Vec2(0.,0.)
			print("force from neighbour1: " + str(neighbour1Force))

			#midNeighbour2: Vec2 = np.divide(2., pos + neighbour2)
			diffNeighbour2: Vec2 = neighbour2 - pos
			distNeighbour2: float = np.sqrt(diffNeighbour2.x*diffNeighbour2.x + diffNeighbour2.y*diffNeighbour2.y)
			neighbour2Force: Vec2 = diffNeighbour2.normalized() * np.absolute(distBetweenEdgepoints - np.absolute(distNeighbour2))
			if (np.absolute(neighbour2Force.x)+np.absolute(neighbour2Force.y)) < epsilon: neighbour2Force = Vec2(0.,0.)
			#print("diffNeighbour2: " + str(diffNeighbour2))
			#print("normalised diffNeighbour2: " + str(diffNeighbour2.normalized()))
			print("force from neighbour2: " + str(neighbour2Force))

			# #sumForce: Vec2 = Vec2((centrepointForce.getX() + neighbour1Force.getX() + neighbour2Force.getX()) / 3.,
			# #						(centrepointForce.getY() + neighbour1Force.getY() + neighbour2Force.getY()) / 3.)
			sumForce: Vec2 = neighbour1Force - neighbour2Force
			print(">>> total force vector: " + str(sumForce))
			# print(">>> total force components: [x: " + str(sumForce[0]) + ", y: " + str(sumForce[1]) + "]")
			# if np.isnan(sumForce[0]): sumForce[0] = 0.
			# if np.isnan(sumForce[1]): sumForce[1] = 0.
			# averageForce: float = sumForce[0] * sumForce[0] + sumForce[1] * sumForce[1]
			# print("average force: " + str(averageForce))
			# avgMagnitude: float = np.sqrt(averageForce)
			# print("average force magnitude: " + str(avgMagnitude))

			pos += sumForce
			sprungPos, vel = calcDampedSHM(pos,vel,centrepoint+basis*self.size,dt,10.)
			#sprungPos, vel = calcDampedSHM(pos,vel,(basis),globalClock.getDt(),2.)
			#sprungPos, vel = calcDampedSHM(pos,vel,centrepoint+(directionCentrepoint*radius),1/120,1.)
			#pos: Vec3 = Vec3(pos.x,pos.y,0.)
			pos = sprungPos + vel*dt
			self.velocities[int(vertex/3-1)] = vel
			print(">>>>>NEW POSITION: " + str(pos))
			print(">>>>>NEW VELOCITY: " + str(vel))
			print("=====")

			assert not np.isnan(pos.x), f'X POSITION IS NAN; SEGFAULT MAY OCCUR'
			assert not np.isnan(pos.y), f'Y POSITION IS NAN; SEGFAULT MAY OCCUR'
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

	def pointToBoundary(self, point: Vec2):
		# move a point to the boundary of the shape
		floatView = memoryview(self.vertexData.modify_array(0)).cast('B').cast('f')
		centrepoint: Vec2 = Vec2(floatView[0],floatView[1])
		diffCentrepoint: Vec2 = centrepoint - point
		distCentrepoint: float = np.sqrt(diffCentrepoint.x*diffCentrepoint.x + diffCentrepoint.y*diffCentrepoint.y)
		return radius*self.size - distCentrepoint

	def isPointInside(self, point: Vec2):
		# test if the point is bounded by the vertices		
		return (self.pointToBoundary(point) > 0)

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
