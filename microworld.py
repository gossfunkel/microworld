from direct.showbase.ShowBase import ShowBase
from direct.filter.CommonFilters import CommonFilters
from direct.interval.IntervalGlobal import *
from panda3d.core import (
    loadPrcFileData, Vec2, Vec3, Vec4,
    GeomTrifans, GeomVertexFormat, GeomVertexArrayFormat, InternalName, GeomEnums,
    GeomVertexData, Geom, GeomNode, DirectionalLight, UserDataAudio, AntialiasAttrib,
    TextureStage, Texture, TextNode, Thread, Shader, ShaderBuffer
)
import numpy as np
from enum import Enum
from scipy.signal import chirp
import struct

# this is a little toy with glowy lights and nice sounds for testing ideas for chembattle or something
# the two blobs are supposed to be little microbial cell guys
# the floating balls can be hp (heal), mana (abilities), protein (upgrade), fats (defense), salts (constitution)
# considering swapping mana for water and making it a plentiful but constant resource
# TODO: the blobs don't interact. They should slide past/off each other
# TODO: spatial partitioning. This ^ and the ball-collection would benefit a lot
# verticality; since we can only move on a plane, the z-dim could be used to make things inaccessible?

EPSILON: float = .0001              # a very small change
DAMP_RATIO: float = .3              # sets springyness of object
DIST_EDGEPOINTS: float = .51        # hopefully should be compatible with the radius
# VOL_SCALE_FACTOR: float = 1.      # for volume preservation
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

class BallType(Enum):
    MANA = "teal.png"       # energy sources - electron transfer agents
    FOOD = "gold.png"       # energy sources - catabolic substrate
    HEAL = "green.png"      # nutrition - phospholipids
    FRNA = "purple.png"     # nutrition - nucleotides
    SALT = "white65.png"    # salts

CONFIG: str = """
gl-version 4 3
gl-debug true
gl-debug-buffers true
premunge-data false
win-size 1200 800
//show-frame-rate-meter true
hardware-animated-vertices true
framebuffer-srgb true
basic-shaders-only false
//gl-interleaved-arrays true
"""
loadPrcFileData("", CONFIG)

def ABS_DIST(a: Vec3, b: Vec3) -> float:
    return np.sqrt((a.x-b.x)*(a.x-b.x) + 
                   (a.y-b.y)*(a.y-b.y) +
                   (a.z-b.z)*(a.z-b.z))


# container for sound effects
class SoundFX:
    def __init__(self):
        self.bongs = []
        self.brrps = []

    def add_bong(self, freq: int, length: int = 6000, atk: int = 400) -> int:
        freq_scale = length / 48000                                             # translate length into seconds
        atk: np.array = np.linspace(0,1,atk, dtype=np.float32)                  # ascending attack
        dec: np.array = np.linspace(1,0,length - len(atk), dtype=np.float32)    # descending decay
        env: np.array = np.append(atk, dec)                                     # generate a simple AD envelope
        # freq is 400hz because sample length is 1/80s
        BONG_SAMPLE   = np.array(np.sin(TAU * freq_scale * freq * np.linspace(0,1,length,endpoint=False)) * env * 32767, dtype=np.int16)
        bong_audio_buff = UserDataAudio(48000,1,False)                          # create audio buffer view
        bong_audio_buff.append(BONG_SAMPLE.tobytes())                           # add sample
        bong_audio_buff.done()
        self.bongs.append(base.loader.loadSfx(bong_audio_buff))                 # load buffer to bongs list
        return len(self.bongs)-1                                                # return index of bong

    def add_brrp(self, length=3200) -> int:
        freq_scale: float = length / 48000                                      # translate length into seconds
        atk: np.array = np.linspace(0,1,2500, dtype=np.float32)                 # ascending attack
        dec: np.array = np.linspace(1,0,length - len(atk), dtype=np.float32)    # descending decay
        env: np.array = np.append(atk, dec)                                     # generate a simple AD envelope
        amp: float    = .2
        BRRP_SAMPLE = np.array(chirp(np.linspace(0,1,length,endpoint=False),    # return value from scipy.signal.chirp with our parameters
                                     f0=450,
                                     f1=1000, 
                                     t1=freq_scale, 
                                     method='hyperbolic') * env  * 32767 * amp, dtype=np.int16)
        BRRP_SAMPLE_2 = np.array(chirp(np.linspace(0,1,length,endpoint=False),  # return value from scipy.signal.chirp with our parameters
                                     f0=300,
                                     f1=700, 
                                     t1=freq_scale, 
                                     method='hyperbolic') * env  * 32767 * (amp*.5), dtype=np.int16)
        BRRP_SAMPLE_3 = np.array(chirp(np.linspace(0,1,length,endpoint=False),  # return value from scipy.signal.chirp with our parameters
                                     f0=150,
                                     f1=400, 
                                     t1=freq_scale, 
                                     method='hyperbolic') * env  * 32767 * (amp*.25), dtype=np.int16)
        BRRP_SAMPLE = np.append(BRRP_SAMPLE, BRRP_SAMPLE_2)
        BRRP_SAMPLE = np.append(BRRP_SAMPLE, BRRP_SAMPLE_3)
        brrp_audio_buff = UserDataAudio(48000,1,False)                          # create audio buffer view
        brrp_audio_buff.append(BRRP_SAMPLE.tobytes())                           # add sample
        brrp_audio_buff.done()
        self.brrps.append(base.loader.loadSfx(brrp_audio_buff))                 # load buffer to brrps list
        return len(self.brrps)-1                                                # return index of brrp

# a PC. cell-like blobby guy that blobs about
class Blob:
    def __init__(self, name: str, pos: Vec2, col: Vec4, bong_freq: float) -> None:
        # TODO move data into C++ obj tags or VBO
        self.type           = type
        self.name           = name
        self.col: tuple     = col
        self.radius: float  = 2.
        self.verts: int     = 12        # number of OUTER vertices (blob requires centre)
        self.velocity       = Vec2(0.,0.)
        self.colliding      = False

        # load the default blob from file
        self.nodepath = loader.loadModel("blob_default.bam")
        self.nodepath.set_color(col)

        print("Loading VBO data...")

        # modify geom vertex data from file
        print(self.nodepath.node().get_geom(0))
        vtx_data = self.nodepath.node().get_geom(0).get_vertex_data()
        print(f"VBO data from file: {vtx_data}")

        # make an SSBO from the model data for vertex pulling
        p3d_array = vtx_data.get_array_handle(0).get_data()
        #custom_array = vtx_data.get_array_handle(1).get_data()
        byte_data = bytearray(p3d_array)
        print(f"byte data for SSBO: {byte_data}")
        #byte_data.extend(custom_array)
        self.buffer = ShaderBuffer("ssbo", bytes(byte_data), GeomEnums.UHDynamic)

        # node = GeomNode(f"{self.name}_geom")
        # node.add_geom(geom)
        self.nodepath.reparent_to(base.render)
        # store position in the node 
        self.nodepath.set_pos(pos.x, pos.y, 0)
        # give the blob a depth offset to prevent self-shadowing etc
        self.nodepath.setDepthOffset(1)

        # activate the jiggle shader on the blob
        self.nodepath.set_shader(Shader.load(Shader.SL_GLSL, vertex="blob_jiggle.vert", fragment="default_shader.frag"))
        self.nodepath.set_shader_input("ssbo", self.buffer)
        self.nodepath.set_shader_input("model_velocity", self.velocity)
        self.nodepath.set_shader_input("radius", self.radius)
        self.nodepath.set_shader_input("col", self.col)

        # now set up accessories
        self.bong           = base.sfx.add_bong(bong_freq)  # generate sound effect at given freq
        self.balls          = []                            # array for balls on blob
        self.spinner: float = 0                             # this tells balls on this blob how to rotate neatly
        self.num_balls: int = len(self.balls)               # to help with angle calculations

        base.taskMgr.add(self.update, str(name)+"-update")

        print(f"== blob {name} created!")

    # alias to make this quicker
    def pos(self) -> Vec3:
        return self.nodepath.get_pos()

    def grow(self):
        self.radius *= 1.1                                    # make the blob bigger
        # TODO update the basis vecs
        self.nodepath.set_shader_input("radius", self.radius) # update the shader

    def add_ball(self, ball=None, balls: int = 1):
        # print(f"adding ball to {self.name}")
        if isinstance(ball,type(None)): # make a new mana ball
            # FIXME this will break when a ball is generated after giving one away due to names
            self.balls.append(Ball(BallType.MANA, self.name+"-ball-"+str(self.num_balls), blob=self, index=self.num_balls))
            self.num_balls += 1
        else:
            if ball.type is BallType.FOOD:
                ball.consume()
                self.grow()
                # TODO add vertex
                # TODO update VBO on adding/removing verts
            else:
                self.balls.append(ball)     # add ball to balls
                ball.blob = self            # change ball references to self
                ball.index = self.num_balls # n.b. this is only incremented after this, so the index is correct
                ball.set_orbiting_true()
                base.sfx.bongs[self.bong].play()
                self.num_balls += 1

    def give_ball(self, blob):
        if self.num_balls > 0:
            given_ball = self.balls[self.num_balls-1]
            base.sfx.bongs[self.bong].play()
            blob.add_ball(given_ball)
            self.balls.remove(given_ball)
            given_ball.fly_to_target(blob)
            # print(f"{self.name} gave 1 ball to {blob.name}")
            self.num_balls -= 1
        else: 
            print("No balls to give!")

    def update(self, task):
        # processor-killing debug
        #print(f"blob {self.name} nodepath position: {self.pos()}")
        #vtx_view = memoryview(self.nodepath.node().get_vertex_data().modify_array(0)).cast('B').cast('f')
        #print(f"some verts from {self.name}: 0: {vtx_view[0]}, 1: {vtx_view[1]}, 2: {vtx_view[2]}")

        # naive collision check with items - TODO spacial hashing
        for item in base.floating_items:
            if ABS_DIST(self.pos(), Vec3(item.nodepath.get_pos().xy, 0)) < (self.radius + item.radius):
                self.add_ball(item)
                base.floating_items.remove(item)

        self.nodepath.set_pos(self.pos() + Vec3(self.velocity, 0.))
        self.nodepath.set_shader_input("model_velocity", self.velocity)
        self.velocity = self.velocity/10. if self.velocity > EPSILON else Vec2(0.,0.) # friction slows us

        self.spinner += (globalClock.getDt())%360
        return task.cont

    # move the blob by its nodepath.pos
    def move(self, direction):
        pos = self.pos()
        #gsg = base.win.get_gsg()
        match direction:
            case "left":             # go left              SSBO         GSG  NEW_VEL     OFFSET (array 1 row 0 col 1: 48+8= 56B)
                self.velocity -=  Vec2(.05,0.)
                #base.graphics_engine.update_shader_buffer_data(self.buffer, gsg, bytes(-.05), 56)
            case "right":                       # go right
                self.velocity +=  Vec2(.05,0.)
                #base.graphics_engine.update_shader_buffer_data(self.buffer, gsg, bytes(.05),  56)
            case "fwd":                         # go forwards                             (array 1 row 0 col 1 comp 1: 48+8+4= 60B)
                self.velocity +=  Vec2(0.,.05)
                #base.graphics_engine.update_shader_buffer_data(self.buffer, gsg, bytes(.05), 60)
            case "back":                        # ...you guessed it
                self.velocity -=  Vec2(0.,.05)
                #base.graphics_engine.update_shader_buffer_data(self.buffer, gsg, bytes(-.05), 60)
            case _: 
                print("Move direction not recognised!")

# floating resource orbs. can bind to blobs
class Ball:
    def __init__(self, type: BallType, name: str, pos: Vec3 | None = None, 
                 blob: Blob | None = None, index: int | None = 0, sfx = None):
        self.type  = type
        self.name  = name
        self.blob  = blob           # blob that ball is attached to, if any
        self.index = index          # helps balls rotate on the blobs neatly
        self.sfx   = sfx            # associated noise
        self.radius: float = .15    # personal space
        # self.velocity: Vec3 = Vec3(0,0,np.random.uniform(-.1,.05))
        self.angle = 0
        self.orbiting = True if blob is not None else False

        model = base.loader.load_model("sphere.egg")
        model.setTransparency(1)
        ts_col = TextureStage('ts_col')
        model.setTexture(ts_col, loader.loadTexture(self.type.value))
        ts_glow = TextureStage('ts_glow')
        ts_glow.setMode(TextureStage.MGlow)
        black_tex = loader.loadTexture("black.png")
        model.setTexture(ts_glow, black_tex)
        model.set_scale(.06)
        self.nodepath = base.render.attach_new_node(f"ball-{self.name}")
        model.reparent_to(self.nodepath)
        if self.blob is None:
            assert pos is not None, f"A free ball must have a position!"
            self.nodepath.set_pos(pos)                            # use given position
        else:
            self.nodepath.set_pos(self.blob.pos() + Vec3(.5,0,.22)) # adjustments for oscillations
        
            base.sfx.bongs[self.blob.bong].play()
        self.task_name = f"update_ball-{self.name}"
        base.taskMgr.add(self.update, self.task_name)

    def set_orbiting_true(self):
        self.orbiting = True

    def fly_to_target(self, target: Blob):
        self.orbiting = False
        # abs_dist = ABS_DIST(self.nodepath.get_pos(),target)
        ratio = (self.index+1) / self.blob.num_balls
        elevation = self.radius*1.2 + .04 * np.sin((target.spinner + .5) * ratio)   # this works if dt is in seconds (doubt, hahaha)
        move_int = self.nodepath.posInterval(1., target.pos() + Vec3(0,0,elevation), fluid=1)
        Sequence(
            move_int,
            Func(self.set_orbiting_true),
            Func(base.sfx.bongs[target.bong].play)
        ).start()

    def update(self, task):
        pos = self.nodepath.get_pos()                               # current ball position
        if (self.orbiting):
            # each ball gets a root of unity of num_balls (arrange them in an even circle)
            ratio = (self.index + 1)/ self.blob.num_balls
            self.angle = TAU * ratio + self.blob.spinner
            # bob up and down
            elevation = self.radius*1.2 + .04 * np.sin(self.angle)
            # go round in a little circle over the blob
            aimpos = self.blob.pos() + Vec3(np.cos(self.angle)/2.,
                                            np.sin(self.angle)/2.,
                                            elevation)
            abs_dist: float = ABS_DIST(pos, aimpos)                 # absolute distance of that lad
            damper = min(1, max(0, abs_dist/2))                     # 1 if far, 0 if close
            self.nodepath.set_pos(pos + (aimpos-pos)*damper)
        else:
            self.nodepath.set_pos(pos.x,pos.y,0+np.sin(globalClock.getDt())*.1)
        return task.cont

    def consume(self):
        self.nodepath.remove_node(Thread.current_thread)
        base.sfx.brrps[self.sfx].play()
        taskMgr.remove(self.task_name)
        #if self.blob is not None:
        #    self.blob.remove_blob(self)
        # play a little consume brrrrp

class GameBase(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.set_background_color(0.12,0.05,0.22,1.)                # dark background

        render.setAntialias(AntialiasAttrib.MAuto)        # set global antialiasing
        #render.setShaderAuto()

        self.sfx = SoundFX()                              # initialise sound effect library

        big_light_np = render.attachNewNode(DirectionalLight('the_big_light'))
        big_light_np.node().setShadowCaster(True, 512, 512)
        big_light_np.set_color(.5,.45,.49)
        big_light_np.setHpr(20, -80, 0)
        render.setLight(big_light_np)                     # set a warm directional light on the whole scene

        self.p1 = Blob("p1",Vec2(0.,-5.),Vec4(0.,0.,1.,1.), 200)  # create a test blob
        self.p2 = Blob("p2",Vec2(0., 5.),Vec4(0.,1.,0.,1.), 300)  # create a second test blob

        self.p1_label = TextNode("p1 balls: ")
        self.p1_label.setTextColor(1,1,1,1)
        self.p1_label.setTextScale(0.1)
        p1_label_np = aspect2d.attach_new_node(self.p1_label)
        p1_label_np.set_pos((1.,0.,.85))
        self.p1_label_v = TextNode("0")
        self.p1_label_v.setTextColor(1,1,1,1)
        self.p1_label_v.setTextScale(0.1)
        p1_label_v_np = aspect2d.attach_new_node(self.p1_label_v)
        p1_label_v_np.set_pos((1.3,0.,.85))
        # self.p1_label.set_transparency(1)
        self.p2_label = TextNode("p2 balls: ")
        self.p2_label.setTextColor(1,1,1,1)
        self.p2_label.setTextScale(0.1)
        p2_label_np = aspect2d.attach_new_node(self.p2_label)
        p2_label_np.set_pos((1.,0.,.7))
        self.p2_label_v = TextNode("0")
        self.p2_label_v.setTextColor(1,1,1,1)
        self.p2_label_v.setTextScale(0.1)
        p2_label_v_np = aspect2d.attach_new_node(self.p2_label_v)
        p2_label_v_np.set_pos((1.3,0.,.7))
        # self.p2_label.set_transparency(1)

        # TODO: Nodepath or spacial partitioning
        self.floating_items = []                          # big list of all nearby collectable items

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

        self.accept("escape", self.userExit)                # quickly quit the game

        self.cam.setPos(CAM_POS)                            # spawn camera distance from origin
        self.cam.setHpr(0,-22,0)                            # look down at your blob! 

        #filters = CommonFilters(self.win, self.cam)         # bloom/glow
        #filters.setBloom(blend=(0,0,0,1), size="small", desat=0)

        self.taskMgr.add(self.update, "update")
        # TODO spacial partitioning 

    # camera follows p1
    def update(self, task):
        self.cam.setPos(self.p1.pos() + CAM_POS)
        # print(f"blobpos: {self.p1.pos}; cam pos: {self.cam.getPos()}")

        # update UI
        self.p1_label_v.setText(str(self.p1.num_balls))
        self.p2_label_v.setText(str(self.p2.num_balls))
        return task.cont

    def __del__(self):
        print("="*20 + " See you soon!:) " + 20*"=")

if __name__ == "__main__":
    print("="*20 + " Welcome to microworld! v0.0.1 " + 20*"=")
    base = GameBase()                                       # Showbase initialised

    hp_ball_pos_1 = Vec3(-4,0,0)
    hp_ball_pos_2 = Vec3(3,-2,0)
    base.floating_items.append(Ball(BallType.HEAL, f"ball-{len(base.floating_items)}", pos=hp_ball_pos_1))
    base.floating_items.append(Ball(BallType.HEAL, f"ball-{len(base.floating_items)}", pos=hp_ball_pos_2))
    atk_ball_pos_1 = Vec3(-2,3,0)
    base.floating_items.append(Ball(BallType.FRNA, f"ball-{len(base.floating_items)}", pos=atk_ball_pos_1))
    food_ball_pos_1 = Vec3(4,4,0)
    food_ball_pos_2 = Vec3(0,-2,0)
    food_ball_sfx = base.sfx.add_brrp()
    base.floating_items.append(Ball(BallType.FOOD, f"ball-{len(base.floating_items)}", 
                                    pos=food_ball_pos_1, sfx=food_ball_sfx))
    base.floating_items.append(Ball(BallType.FOOD, f"ball-{len(base.floating_items)}", 
                                    pos=food_ball_pos_2, sfx=food_ball_sfx))

    base.run()                                              # taskMgr blocks
