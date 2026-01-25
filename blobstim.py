from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    loadPrcFileData, Vec2, Vec3, Vec4,
    GeomTrifans, GeomVertexFormat, GeomVertexArrayFormat, InternalName, GeomEnums,
    GeomVertexData, Geom, GeomNode, DirectionalLight, UserDataAudio, AntialiasAttrib
)
import numpy as np
import struct

EPSILON: float = .0001              # a very small change
DAMP_RATIO: float = .3              # sets springyness of object
DIST_EDGEPOINTS: float = .51        # hopefully should be compatible with the radius
# VOL_SCALE_FACTOR: float = 1.      # for volume preservation
# RADIUS: float = 1.                  # for maintaining roundness
TAU: float = np.pi * 2              # for calculating circles
BASIS_VECS = np.array([0.,     0.,
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
                       -1.,    0.], dtype='f')

CONFIG: str = """
// gl-version 4 3
// gl-debug 1
win-size 1200 800
show-frame-rate-meter 1
hardware-animated-vertices true
framebuffer-srgb true
basic-shaders-only false
framebuffer-multisample 1
multisamples 2
//threading-model Cull/Draw
"""
loadPrcFileData("", CONFIG)

def GEN_PRIMS_FORMAT():
    blobPrim = GeomTrifans(Geom.UHStatic)
    blobPrim.add_consecutive_vertices(0,13)
    blobPrim.add_vertex(1)
    blobPrim.closePrimitive()
    #vtx_format = GeomVertexFormat.getV3n3c4()
    vtx_format = GeomVertexFormat()
    arrayFormat = GeomVertexArrayFormat()
    arrayFormat.add_column(InternalName.get_vertex(),3,GeomEnums.NT_float32, GeomEnums.C_point)
    vtx_format.add_array(arrayFormat)
    arrayFormat = GeomVertexArrayFormat()
    arrayFormat.add_column(InternalName.get_normal(),3,GeomEnums.NT_float32, GeomEnums.C_point)
    vtx_format.add_array(arrayFormat)
    arrayFormat = GeomVertexArrayFormat()
    arrayFormat.add_column(InternalName.get_color(),4,GeomEnums.NT_uint8, GeomEnums.C_color)
    vtx_format.add_array(arrayFormat)
    vtx_format = GeomVertexFormat.register_format(vtx_format)
    return blobPrim, vtx_format

# construct bong
atk: np.array = np.linspace(0,1,400, dtype=np.float32)                # ascending attack
dec: np.array = np.linspace(1,0,6000 - len(atk), dtype=np.float32)   # descending decay
env: np.array = np.append(atk, dec)                                   # generate a simple AD envelope
# freq is 400hz because sample length is 1/80s
BONG_SAMPLE   = np.array(np.sin(TAU * 50 * np.linspace(0,1,6000,endpoint=False)) * env * 32767, dtype=np.int16)
bong_audio_buff = UserDataAudio(48000,1,False)
bong_audio_buff.append(BONG_SAMPLE.tobytes())
bong_audio_buff.done()

#BONG_SRC = Source(Buffer([AL_FORMAT_MONO16, sample.tolist(), len(sample), 48000]))

def SHO(pos: Vec2, vel: Vec2, equilibriumPos: Vec2, deltaTime: float, angularFreq: float):
    assert (angularFreq >= 0.), f'SHM angular frequency parameter must be positive!'
    assert (DAMP_RATIO >= 0.), f'SHM damping ratio parameter must be positive!'

    if (angularFreq < EPSILON):
        print("SHM frequency too low to change motion!")
        pospos = 1.
        posvel = 0.
        velpos = 0.
        velvel = 1.
    else:
        if (DAMP_RATIO > 1. + EPSILON):
            # overdamped formula
            za = -angularFreq * DAMP_RATIO
            zb = angularFreq * np.sqrt(DAMP_RATIO*DAMP_RATIO - 1.)
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
        elif (DAMP_RATIO < 1. - EPSILON):
            # underdamped formula
            omegaZeta = angularFreq * DAMP_RATIO
            alpha     = angularFreq * np.sqrt(1. - DAMP_RATIO * DAMP_RATIO)

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

class Blob:
    def __init__(self, name: str, pos: Vec2, col: tuple) -> None:
        # TODO move data into C++ obj tags or VBO
        self.name = name
        self.pos: Vec2    = pos
        self.col: tuple   = col
        self.size: float  = 1.
        self.verts: int   = 12
        prims, vtx_format = GEN_PRIMS_FORMAT()
        self.vtx_data     = GeomVertexData(name+'-verts', vtx_format, Geom.UHStatic)
        self.vtx_data.unclean_set_num_rows(self.verts+1) # 1 row per vertex (12 rim, 1 centre)

        # open memoryviews to write position, normal, and colour data to VBO
        pos_view  = memoryview(self.vtx_data.modify_array(0)).cast('B')
        norm_view = memoryview(self.vtx_data.modify_array(1)).cast('B')
        col_view  = memoryview(self.vtx_data.modify_array(2)).cast('B')

        vtx_vals = bytearray()
        for i in range(self.verts+1):
            # generate circular layout with basis vectors
            vtx_vals.extend(struct.pack(
                '3f',
                pos.x+BASIS_VECS[i], pos.y+BASIS_VECS[i+1], 0.))
        # pack values into memoryview
        pos_view[:] = vtx_vals

        # now generate and pack the normals and colours the same way
        norm_vals = bytearray()
        col_vals  = bytearray()
        for _ in range(13):
            norm_vals.extend(struct.pack('3f', 0.,0.,1.))
            col_vals.extend(struct.pack('4B', col[0], col[1], col[2], 255))
        norm_view[:] = norm_vals
        col_view[:]  = col_vals

        # finally, create a mesh ('Geom') from the vertices- containing one trifan defined above as blobPrim
        geom = Geom(self.vtx_data)
        geom.addPrimitive(prims)
        self.geom_node = GeomNode(name+'-geom_node')
        self.geom_node.addGeom(geom)

        # create a new node and attach to base.render
        self.nodepath = base.render.attach_new_node(self.geom_node)
        # give the blob a depth offset to prevent self-shadowing etc
        self.nodepath.setDepthOffset(1)

        # start at rest
        self.velocities: list[Vec2] = [Vec2(0.,0.) for _ in range(12)]

        base.taskMgr.add(self.update, str(name)+"-update")

        print(f"== blob {name} created!")

    def update(self, task):
        vtx_view_f32 = memoryview(self.vtx_data.modify_array(0)).cast('B').cast('f')

        # calculate internal blob forces
        for vtx in range(12):
            vel: Vec2 = self.velocities[vtx]
            dt: float = globalClock.getDt()
            vtx += 1
            basis: Vec2 = Vec2(BASIS_VECS[vtx*2], BASIS_VECS[vtx*2+1])
            vtx *= 3

            pos: Vec2 = Vec2(vtx_view_f32[vtx], vtx_view_f32[vtx+1])
            centrepoint: Vec2 = Vec2(vtx_view_f32[0],vtx_view_f32[1])

            sprungPos, vel = SHO(pos,vel,centrepoint+basis*self.size,dt,10.)
            pos = sprungPos + vel*dt
            self.velocities[int(vtx/3-1)] = vel
            #print(">>>>>NEW POSITION: " + str(pos))
            #print(">>>>>NEW VELOCITY: " + str(vel))
            #print("=====")

            assert not np.isnan(pos.x), f'X POSITION IS NAN; SEGFAULT MAY OCCUR'
            assert not np.isnan(pos.y), f'Y POSITION IS NAN; SEGFAULT MAY OCCUR'
            vtx_view_f32[vtx]   = pos.x if not np.isnan(pos.x) else 0
            vtx_view_f32[vtx+1] = pos.y if not np.isnan(pos.y) else 0
            vtx_view_f32[vtx+2] = 0. # pos.z
        return task.cont

    def move(self, direction) -> bool:
        #print(self.view[1].to_bytes())
        vtx_view_f32 = memoryview(self.vtx_data.modify_array(0)).cast('B').cast('f')
        if direction == "left":
            # go left
            vtx_view_f32[0] -= .05
            self.pos.x -= .05
            return 1
        elif direction == "right":
            # go right
            vtx_view_f32[0] += .05
            self.pos.x += .05
            return 1
        elif direction == "fwd":
            # go forwards
            vtx_view_f32[1] += .05
            self.pos.y += .05
            return 1
        elif direction == "back":
            # ...you guessed it
            vtx_view_f32[1] -= .05
            self.pos.y -= .05
            return 1
        else: return 0


class Ball:
    def __init__(self, blob: Blob, bong):
        self.blob: Blob = blob
        self.bounce: float = 0.
        self.radius: float = .15
        self.velocity: Vec3 = Vec3(0,0,0)
        self.bong = bong

        model = base.loader.load_model("sphere.egg")
        model.set_color(Vec4(self.blob.col, 1))
        model.set_scale(.1)
        self.nodepath = base.render.attach_new_node(f"ball-{self.blob.name}")
        model.reparent_to(self.nodepath)
        self.nodepath.set_pos(self.blob.pos + Vec3(0,0,1))

        base.taskMgr.add(self.update, f"update_ball-{self.blob.name}")

    def update(self, task):
        self.velocity -= Vec3(0,0,.01)                                    # gravity
        pos = self.nodepath.get_pos()                                     # current ball position
        if ((pos + self.velocity).z < (self.blob.pos.z+self.radius)):     # collision check
            self.velocity = -(self.velocity * .95)
            self.bong.play()
        self.nodepath.set_pos(self.blob.pos.x,self.blob.pos.y, pos.z + self.velocity.z)
        return task.cont       


if __name__ == "__main__":
    print("="*20 + " Welcome to blobstim:) " + 20*"=")
    ShowBase()                                      # Showbase initialised
    render.setShaderAuto()                          # auto shaders for shadow and glow mapping
    render.setAntialias(AntialiasAttrib.MAuto)      # set global antialiasing

    base.set_background_color(0,0,0,1)              # dark background

    bong_smp = loader.loadSfx(bong_audio_buff)

    big_light_np = render.attachNewNode(DirectionalLight('the_big_light'))
    big_light_np.node().setShadowCaster(True, 512, 512)
    big_light_np.set_color(1,.9,.78)
    big_light_np.setHpr(20, -80, 0)
    render.setLight(big_light_np)                   # set a warm directional light on the whole scene

    p1 = Blob("p1",Vec3(0.,-5.,0.),(0,0,255))       # create a test blob

    # awsd/keypad movement for p1 blob
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

    ball1 = Ball(p1, bong_smp)                                # give each blob a bouncing ball

    # make each blob bong at a different frequency when the ball bounces

    # make the balls glowy and cool

    # allow blobs to exchange balls (?!)

    base.cam.setPos(0,-18,5)                        # move camera to a better angle for us
    base.cam.setHpr(0,-15,0)

    base.run()                                      # taskMgr blocks
    oalQuit()                                  # close audio stream
