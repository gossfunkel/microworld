from direct.showbase.ShowBase import ShowBase
from direct.filter.CommonFilters import CommonFilters
from direct.interval.IntervalGlobal import *
from panda3d.core import (
    loadPrcFileData, Vec2, Vec3, Vec4,
    GeomTrifans, GeomVertexFormat, GeomVertexArrayFormat, InternalName, GeomEnums,
    GeomVertexData, Geom, GeomNode, DirectionalLight, UserDataAudio, AntialiasAttrib,
    TextureStage, Texture
)
import numpy as np
import struct

# this is a little toy with glowy lights and nice sounds
# i'm thinking about testing ideas for chembattle with this
# the floating blobs can be hp (heal), mana (abilities), protein (grow), fats (defense), salts (metabolism)
# TODO: the blobs don't interact. They should slide past/off each other
# verticality; since we can only move on a plane, the z-dim can be used to make things inaccessible

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
CAM_POS: Vec3 = Vec3(0,-8,4)       # for keeping the camera a constant vector from the player blob

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
def make_bong(freq, length=6000, atk=400):
    freq_scale = length/48000
    atk: np.array = np.linspace(0,1,atk, dtype=np.float32)                # ascending attack
    dec: np.array = np.linspace(1,0,length - len(atk), dtype=np.float32)   # descending decay
    env: np.array = np.append(atk, dec)                                   # generate a simple AD envelope
    # freq is 400hz because sample length is 1/80s
    BONG_SAMPLE   = np.array(np.sin(TAU * freq_scale * freq * np.linspace(0,1,length,endpoint=False)) * env * 32767, dtype=np.int16)
    bong_audio_buff = UserDataAudio(48000,1,False)
    bong_audio_buff.append(BONG_SAMPLE.tobytes())
    bong_audio_buff.done()
    return base.loader.loadSfx(bong_audio_buff)

def ABS_DIST(a: Vec3, b: Vec3) -> float:
    return np.sqrt((a.x-b.x)*(a.x-b.x) + 
                   (a.y-b.y)*(a.y-b.y) +
                   (a.z-b.z)*(a.z-b.z))

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
    def __init__(self, name: str, pos: Vec2, col: tuple, bong_freq: float) -> None:
        # TODO move data into C++ obj tags or VBO
        self.name = name
        self.pos = Vec3(pos,0)
        self.col: tuple   = col
        self.size: float  = 2.
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
        # store position in the node FIXME currently nodepath pos must stay at 0,0,0 for the world matrix
        #self.nodepath.set_pos(pos.x, pos.y, 0)
        # give the blob a depth offset to prevent self-shadowing etc
        self.nodepath.setDepthOffset(1)
        #self.nodepath.setShaderAuto()

        # now set up accessories
        self.bong         = make_bong(bong_freq)
        ball_tex = loader.loadTexture("teal.png")
        self.balls        = [Ball(self, ball_tex)]                   # give each blob a floating ball

        # start at rest
        self.velocities: list[Vec2] = [Vec2(0.,0.) for _ in range(12)]

        base.taskMgr.add(self.update, str(name)+"-update")

        print(f"== blob {name} created!")

    # getter to make this quicker
    # def pos(self) -> Vec3:
    #     return self.nodepath.get_pos()

    def add_ball(self, ball=None, balls: int = 1):
        print(f"adding ball to {self.name}")
        if isinstance(ball,type(None)):
            ball_tex = loader.loadTexture("teal.png")
            self.balls.append(Ball(self, ball_tex))
        else:
            self.balls.append(ball)
            ball.blob = self

    def give_ball(self, blob):
        num_balls = len(self.balls)
        if num_balls > 0:
            given_ball = self.balls[num_balls-1]
            self.bong.play()
            blob.add_ball(given_ball)
            self.balls.remove(given_ball)
            given_ball.fly_to_target(blob)
            print(f"{self.name} gave 1 ball to {blob.name}")
        else: 
            print("No balls to give!")

    def update(self, task):
        # TODO move the procedural animation to a vertex shader (collisions??)
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

            # TODO modify movement based on collisions

            assert not np.isnan(pos.x), f'X POSITION IS NAN; SEGFAULT MAY OCCUR'
            assert not np.isnan(pos.y), f'Y POSITION IS NAN; SEGFAULT MAY OCCUR'
            vtx_view_f32[vtx]   = pos.x if not np.isnan(pos.x) else 0
            vtx_view_f32[vtx+1] = pos.y if not np.isnan(pos.y) else 0
            vtx_view_f32[vtx+2] = 0. # pos.z
        return task.cont

    # move the blob by its centrepoint
    def move(self, direction) -> bool:
        #print(self.view[1].to_bytes())
        vtx_view_f32 = memoryview(self.vtx_data.modify_array(0)).cast('B').cast('f')
        match direction:
            case "left":                        # go left
                vtx_view_f32[0] -= .05
                self.pos.x -= .05
                return 1
            case "right":                       # go right
                vtx_view_f32[0] += .05
                self.pos.x += .05
                return 1
            case "fwd":                         # go forwards
                vtx_view_f32[1] += .05
                self.pos.y += .05
                return 1
            case "back":                        # ...you guessed it
                vtx_view_f32[1] -= .05
                self.pos.y -= .05
                return 1
            case _: 
                return 0


class Ball:
    def __init__(self, blob: Blob, colour_tex: Texture):
        self.blob: Blob = blob
        self.bounce: float = 0.
        self.radius: float = .15
        # self.velocity: Vec3 = Vec3(0,0,np.random.uniform(-.1,.05))
        self.ticker = 0
        self.angle = 0
        self.orbiting = True

        model = base.loader.load_model("sphere.egg")
        ts_col = TextureStage('ts_col')
        white_tex = loader.loadTexture("white65.png")
        model.setTexture(ts_col, white_tex)
        model.setTransparency(1)
        # model.set_color(Vec4(self.blob.col, 1))
        ts_glow = TextureStage('ts_glow')
        ts_glow.setMode(TextureStage.MGlow)
        # black_tex = loader.loadTexture("black.png")
        model.setTexture(ts_glow, colour_tex)
        model.set_scale(.06)
        self.nodepath = base.render.attach_new_node(f"ball-{self.blob.name}")
        model.reparent_to(self.nodepath)
        self.nodepath.set_pos(self.blob.pos + Vec3(.5,0,.22)) # adjustments for oscillations
        #self.nodepath.setShaderAuto()

        self.blob.bong.play()

        base.taskMgr.add(self.update, f"update_ball-{self.blob.name}")

    def set_orbiting_true(self):
        self.orbiting = True

    def fly_to_target(self, target: Blob):
        self.orbiting = False
        # abs_dist = ABS_DIST(self.nodepath.get_pos(),target)
        elevation = self.radius*1.2 + .04 * np.sin(self.ticker + .5)   # this works if dt is in seconds (doubt, hahaha)
        move_int = self.nodepath.posInterval(1., target.pos + Vec3(0,0,elevation), fluid=1)
        Sequence(
            move_int,
            Func(self.set_orbiting_true),
            Func(target.bong.play)
        ).start()

    def update(self, task):
        if (self.orbiting):
            pos = self.nodepath.get_pos()                                       # current ball position
            # self.velocity -= Vec3(0,0,.1) * globalClock.getDt()               # gravity
            # if ((pos + self.velocity).z < (self.blob.pos.z+self.radius)):     # collision check
            #     self.velocity = -self.velocity* .979
            # self.nodepath.set_pos(self.blob.pos.x,self.blob.pos.y, pos.z + self.velocity.z)
            self.ticker += globalClock.getDt()*.5
            self.angle = self.ticker%360
            elevation = self.radius*1.2 + .04 * np.sin(self.ticker)
            blobpos = self.blob.pos
            aimpos = Vec3(blobpos.x + np.cos(self.angle)/2.,
                          blobpos.y + np.sin(self.angle)/2.,
                          blobpos.z + elevation)
            abs_dist: float = ABS_DIST(pos, aimpos)
            damper = min(1, max(0, abs_dist))# 1 if far, 0 if close
            self.nodepath.set_pos(pos + (aimpos-pos)*damper)
        return task.cont       

class GameBase(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.set_background_color(0,0,0,1)              # dark background

        render.setAntialias(AntialiasAttrib.MAuto)      # set global antialiasing
        render.setShaderAuto()

        big_light_np = render.attachNewNode(DirectionalLight('the_big_light'))
        big_light_np.node().setShadowCaster(True, 512, 512)
        big_light_np.set_color(.5,.45,.39)
        big_light_np.setHpr(20, -80, 0)
        render.setLight(big_light_np)                   # set a warm directional light on the whole scene

        self.p1 = Blob("p1",Vec2(0.,-5.),(0,0,255), 200)  # create a test blob
        self.p2 = Blob("p2",Vec2(0., 5.),(0,255,0), 300)  # create a second test blob

        # awsd/keypad movement for p1 blob
        self.accept("arrow_left", self.p1.move, ["left"])
        self.accept("arrow_left-repeat", self.p1.move, ["left"])
        self.accept("a", self.p1.move, ["left"])
        self.accept("a-repeat", self.p1.move, ["left"])
        self.accept("arrow_right", self.p1.move, ["right"])
        self.accept("arrow_right-repeat", self.p1.move, ["right"])
        self.accept("d", self.p1.move, ["right"])
        self.accept("d-repeat", self.p1.move, ["right"])
        self.accept("arrow_up", self.p1.move, ["fwd"])
        self.accept("arrow_up-repeat", self.p1.move, ["fwd"])
        self.accept("w", self.p1.move, ["fwd"])
        self.accept("w-repeat", self.p1.move, ["fwd"])
        self.accept("arrow_down", self.p1.move, ["back"])
        self.accept("arrow_down-repeat", self.p1.move, ["back"])
        self.accept("s", self.p1.move, ["back"])
        self.accept("s-repeat", self.p1.move, ["back"])

        
        self.accept("b", self.p1.add_ball)                  # allow player to spawn balls
        self.accept("space", self.p1.give_ball, [self.p2])  # allow player to gift balls 

        self.cam.setPos(CAM_POS)                            # move camera to a better angle for us
        self.cam.setHpr(0,-22,0)

        filters = CommonFilters(self.win, self.cam)
        filters.setBloom(blend=(0,0,0,1), size="small", desat=0)

        self.taskMgr.add(self.update_cam, "update_cam")

    def update_cam(self, task):
        self.cam.setPos(self.p1.pos + CAM_POS)
        # print(f"blobpos: {self.p1.pos}; cam pos: {self.cam.getPos()}")
        return task.cont

if __name__ == "__main__":
    print("="*20 + " Welcome to blobstim:) " + 20*"=")
    base = GameBase()                               # Showbase initialised

    # simplepbr.init(use_330=True)

    base.run()                                      # taskMgr blocks
