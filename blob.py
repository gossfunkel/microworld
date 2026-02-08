from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    loadPrcFileData, Vec2, Vec3, Vec4, Geom, GeomNode, GeomEnums, NodePath, BoundingBox,
    GeomTrifans, GeomVertexFormat, GeomVertexArrayFormat, GeomVertexData, InternalName,
    Shader, ShaderBuffer, ComputeNode, TextureStage, Thread, UserDataAudio
)
import struct
from enum import Enum
from scipy.signal import chirp
import numpy as np

# constants 
EPSILON    = 0.0001
TAU        = np.pi * 2.
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

# Panda3d configuration file
config_vars: str = """
win-size 1200 800
//show-frame-rate-meter 1
hardware-animated-vertices true
framebuffer-srgb true
model-cache-dir
gl-debug 1
gl-version 4 3
basic-shaders-only false
//threading-model Cull/Draw
"""
loadPrcFileData("", config_vars)

# initialise a new GeomVertexData and associated formats
def GENERATE_VTX_DATA(num_rows: int):
    print("-> Creating GeomVertexData and associated formats...")
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
    vtx_data.unclean_set_num_rows(num_rows)                           # 1 row per vertex (12 rim, 1 centre)
    return vtx_data

# fill a GeomVertexData with basis vectors, colour, and velocity information
def POPULATE_VTX_DATA(vtx_data: GeomVertexData, col: Vec4, vel: Vec2):
    # open memoryviews to write position, normal, colour, basis, and velocity data to VBO
    print("-> Populating vertex data...")
    view   = memoryview(vtx_data.modify_array(0)).cast('B')

    vals = bytearray()
    for i in range(13):
        # populate the bytearray with each row
        vals.extend(struct.pack('4f', BASIS_VECS[i*2], BASIS_VECS[i*2 + 1], 0., 1.))
        vals.extend(struct.pack('4f', 0.,0.,1.,0.))
        vals.extend(struct.pack('4f', col.x, col.y, col.z, col.w))
        vals.extend(struct.pack('2f', BASIS_VECS[i*2], BASIS_VECS[i*2 + 1]))
        vals.extend(struct.pack('2f', vel.x, vel.y))

    # write to VBO and SSBO
    view[:] = vals
    buffer = ShaderBuffer("ssbo", bytes(vals), GeomEnums.UHDynamic)
    return vtx_data, buffer

# create a GeomTrifans primitive for making a complete circle
def GENERATE_PRIM(num_verts: int):
    print("-> Generating blob model primitives...")
    blobPrim = GeomTrifans(Geom.UHStatic)                   # make a trifan
    blobPrim.add_consecutive_vertices(0,(num_verts-1))      # add all the verts
    blobPrim.add_vertex(1)                                  # close the circle
    blobPrim.closePrimitive()                               # close the primitive
    return blobPrim

# make a compute shader to procedurally animate vertices
def CREATE_COMP():
    # set up the compute node to calculate the vertex positions
    print("-> Setting up a compute shader for a blob...")
    comp_node = ComputeNode("blob_animation")
    comp_node.add_dispatch(1, 1, 1)
    comp_np = base.render.attach_new_node(comp_node)
    comp_np.set_shader(Shader.load_compute(Shader.SL_GLSL, "blob.comp"))
    return comp_np

# couple a blob to the user input
def BIND_BLOB(bound_blob):
    print(f"-> Binding {bound_blob} to user input...")
    base.accept("arrow_left", bound_blob.move, ["left"])
    base.accept("arrow_left-repeat", bound_blob.move, ["left"])
    base.accept("a", bound_blob.move, ["left"])
    base.accept("a-repeat", bound_blob.move, ["left"])
    base.accept("arrow_right", bound_blob.move, ["right"])
    base.accept("arrow_right-repeat", bound_blob.move, ["right"])
    base.accept("d", bound_blob.move, ["right"])
    base.accept("d-repeat", bound_blob.move, ["right"])
    base.accept("arrow_up", bound_blob.move, ["fwd"])
    base.accept("arrow_up-repeat", bound_blob.move, ["fwd"])
    base.accept("w", bound_blob.move, ["fwd"])
    base.accept("w-repeat", bound_blob.move, ["fwd"])
    base.accept("arrow_down", bound_blob.move, ["back"])
    base.accept("arrow_down-repeat", bound_blob.move, ["back"])
    base.accept("s", bound_blob.move, ["back"])
    base.accept("s-repeat", bound_blob.move, ["back"])

# calculate the absolute distance between two 3D points
def ABS_DIST(a: Vec3, b: Vec3) -> float:
    return np.sqrt((a.x-b.x)*(a.x-b.x) + 
                   (a.y-b.y)*(a.y-b.y) +
                   (a.z-b.z)*(a.z-b.z))

# Enum relating Ball types to image files
class BallType(Enum):
    MANA = "teal.png"       # energy sources - electron transfer agents
    FOOD = "gold.png"       # energy sources - catabolic substrate
    HEAL = "green.png"      # nutrition - phospholipids
    FRNA = "purple.png"     # nutrition - nucleotides
    SALT = "white65.png"    # salts

# container for sound effects
class SoundFX:
    def __init__(self):
        self.bongs = []
        self.bong_300 = self.add_bong(300)
        self.brrp_consume = self._gen_brrp()

    def _gen_bong(self, freq: int, length: int = 6000, atk: int = 400) -> int:
        freq_scale = length / 48000                                             # translate length into seconds
        atk: np.array = np.linspace(0,1,atk, dtype=np.float32)                  # ascending attack
        dec: np.array = np.linspace(1,0,length - len(atk), dtype=np.float32)    # descending decay
        env: np.array = np.append(atk, dec)                                     # generate a simple AD envelope
        # freq is 400hz because sample length is 1/80s
        BONG_SAMPLE   = np.array(np.sin(TAU * freq_scale * freq * np.linspace(0,1,length,endpoint=False)) * env * 32767, dtype=np.int16)
        bong_audio_buff = UserDataAudio(48000,1,False)                          # create audio buffer view
        bong_audio_buff.append(BONG_SAMPLE.tobytes())                           # add sample
        bong_audio_buff.done()
        return base.loader.loadSfx(bong_audio_buff)                             # load sample and return

    def _gen_brrp(self, length=3200) -> int:
        freq_scale: float = length / 48000                                      # translate length into seconds
        atk: np.array = np.linspace(0,1,2500, dtype=np.float32)                 # ascending attack
        dec: np.array = np.linspace(1,0,length - len(atk), dtype=np.float32)    # descending decay
        env: np.array = np.append(atk, dec)                                     # generate a simple AD envelope
        amp: float    = .2
        BRRP_SAMPLE = np.array(chirp(np.linspace(0,1,length,endpoint=False),    # get value from scipy.signal.chirp with our parameters
                                     f0=450,
                                     f1=1000, 
                                     t1=freq_scale, 
                                     method='hyperbolic') * env  * 32767 * amp, dtype=np.int16)
        BRRP_SAMPLE_2 = np.array(chirp(np.linspace(0,1,length,endpoint=False),  # get 'echo' 1
                                     f0=300,
                                     f1=700, 
                                     t1=freq_scale, 
                                     method='hyperbolic') * env  * 32767 * (amp*.5), dtype=np.int16)
        BRRP_SAMPLE_3 = np.array(chirp(np.linspace(0,1,length,endpoint=False),  # get 'echo' 2
                                     f0=150,
                                     f1=400, 
                                     t1=freq_scale, 
                                     method='hyperbolic') * env  * 32767 * (amp*.25), dtype=np.int16)
        BRRP_SAMPLE = np.append(BRRP_SAMPLE, BRRP_SAMPLE_2)
        BRRP_SAMPLE = np.append(BRRP_SAMPLE, BRRP_SAMPLE_3)
        brrp_audio_buff = UserDataAudio(48000,1,False)                          # create audio buffer view
        brrp_audio_buff.append(BRRP_SAMPLE.tobytes())                           # add sample
        brrp_audio_buff.done()
        return base.loader.loadSfx(brrp_audio_buff)                             # load sample and return

    def add_bong(self, freq: int):
        self.bongs.append(self._gen_bong(freq))                                 # load buffer to bongs list
        return len(self.bongs)-1                                                # return index of bong

# mesh, data, and behaviour for a cell ('blob')
class CellBlob:
    # initialise a circle with 1 inner and 12 outer vertices, with compute-managed animation
    def __init__(self, pos=Vec3(0.,0.,0.),vel=Vec2(0.,0.),col=Vec4(1.,1.,1.,1.),rad=1.):
        print("-> Initialising blob...")
        self.vel: Vec2 = vel
        self.col: Vec4 = col
        self.radius: float = rad
        self.balls: list[Ball] = []
        self.spinner: float = 0                                             # this tells balls on this blob how to rotate neatly
        self.num_balls: int = len(self.balls)                               # to help with angle calculations

        # get a new VBO and SSBO with a blob model
        vtx_data = GENERATE_VTX_DATA(13)
        vtx_data, self.buffer = POPULATE_VTX_DATA(vtx_data, col, vel)

        self.comp_np = CREATE_COMP()                                        # create a compute shader
        self.comp_np.set_shader_input("ssbo", self.buffer)                  # send it blob information
        self.comp_np.set_shader_input("radius", self.radius)
        self.comp_np.set_shader_input("model_velocity", self.vel)

        # create a mesh from the vertices 
        print("-> Creating geometry...")
        geom = Geom(vtx_data)
        geom.addPrimitive(GENERATE_PRIM(13))
        # set up a bounding volume to prevent culling
        geom.set_bounds(BoundingBox((-1, -1, -.5), (1, 1, .5)))
        #geom.doublesideInPlace()
        geom_node = GeomNode('blob-geom_node')
        geom_node.addGeom(geom)
        
        #jiggle_vert  = Shader.load("blob_jiggle.vert", Shader.SL_GLSL)
        #default_frag = Shader.load("default_shader.frag", Shader.SL_GLSL)
        #self.nodepath.set_shader(Shader.make(Shader.SL_GLSL, vertex=jiggle_vert, fragment=default_frag))

        print("-> Composing blob NodePath...")
        self.nodepath = base.render.attach_new_node(geom_node)              # make nodepath
        self.nodepath.set_shader(Shader.load(                               # activate custom shaders
            Shader.SL_GLSL, 
            vertex="default_shader.vert", 
            fragment="default_shader.frag"
        ))
        self.nodepath.set_shader_input("ssbo", self.buffer)                 # send shader SSBO and colour
        self.nodepath.set_shader_input("col", self.col)

        self.nodepath.set_pos(pos)                                          # store position in the nodepath

        base.taskMgr.add(self.update, "update", taskChain='default')

    def update(self, task):
        pos = self.nodepath.get_pos()                                       # get own position

        for item in base.floating_balls:                                    # naive collision check with balls
                                                                            # TODO spacial hashing
            if ABS_DIST(pos, Vec3(item.nodepath.get_pos().xy, 0)) < (self.radius + item.radius):
                base.floating_balls.remove(item)                            # remove reference in base
                self.add_ball(item)                                         # collect if collides

        self.nodepath.set_pos(pos + Vec3(self.vel, 0.))                     # update nodepath position by velocity
        self.comp_np.set_shader_input("model_velocity", self.vel)           # update shader
        self.vel = self.vel/2. if self.vel > EPSILON else Vec2(0.,0.)       # friction slows us

        self.spinner += (globalClock.getDt())%360                           # increment spinner for balls

        return task.cont

    def grow(self):
        self.radius *= 1.1                                                  # make the blob bigger
        self.comp_np.set_shader_input("radius", self.radius)                # update the shader

    def add_ball(self, ball=None, balls: int = 1):
        if isinstance(ball,type(None)): # make a new mana ball
            # FIXME this will break when a ball is generated after giving one away due to names
            self.balls.append(Ball(BallType.MANA, self.name+"-ball-"+str(self.num_balls), blob=self, index=self.num_balls))
            self.num_balls += 1
        else:
            if ball.type is BallType.FOOD:
                ball.consume()                                              # consume food to grow!
                self.grow()                                                 # TODO add vertices at size thresholds
            else:
                self.balls.append(ball)                                     # add ball to balls
                ball.blob = self                                            # change ball references to self
                ball.index = self.num_balls                                 # n.b. this is only incremented after this, so the index is correct
                ball.set_orbiting_true()
                base.sfx.bong_300.play()
                self.num_balls += 1

    def move(self, direction):
        match direction:
            case "left":                                                    # go left
                self.vel -=  Vec2(.05, 0.)
            case "right":                                                   # go right
                self.vel +=  Vec2(.05, 0.)
            case "fwd":                                                     # go forwards
                self.vel +=  Vec2(0., .05)
            case "back":                                                    # ...you guessed it
                self.vel -=  Vec2(0., .05)
            case _: 
                print("Move direction not recognised!")

# floating resource orbs. can bind to blobs
class Ball:
    def __init__(self, type: BallType, name: str, pos: Vec3 | None = None, 
                 blob: CellBlob | None = None, index: int | None = 0, sfx = None):
        self.type  = type
        self.name  = name
        self.blob  = blob           # blob that ball is attached to, if any
        self.index = index          # helps balls rotate on the blobs neatly
        self.sfx   = sfx            # associated noise
        self.radius: float = .15    # personal space
        # self.velocity: Vec3 = Vec3(0,0,np.random.uniform(-.1,.05))
        self.angle = 0
        self.orbiting = True if blob is not None else False
        self.removing = False       # flag for removing ball

        model = base.loader.load_model("sphere.egg")
        #model.setTransparency(1)
        ts_col = TextureStage('ts_col')
        model.setTexture(loader.loadTexture(self.type.value))
        # ts_glow = TextureStage('ts_glow')
        # ts_glow.setMode(TextureStage.MGlow)
        # black_tex = loader.loadTexture("black.png")
        # model.setTexture(ts_glow, black_tex)
        model.set_scale(.06)
        self.nodepath = base.render.attach_new_node(f"ball-{self.name}")
        model.reparent_to(self.nodepath)                                    # attach node to render
        if self.blob is None:
            assert pos is not None, f"A free ball must have a position!"
            self.nodepath.set_pos(pos)                                      # use given position
        else:
            self.nodepath.set_pos(self.blob.nodepath.get_pos() + Vec3(.5,0,.22)) # adjustments for oscillations
        
            base.sfx.bong_300.play()                                        # make a lil bong noise
        self.task_name = f"update_ball-{self.name}"                         # save task name so it can be removed later
        base.taskMgr.add(self.update, self.task_name)                       # add update to taskMgr

    def set_orbiting_true(self):
        self.orbiting = True

    def fly_to_target(self, target: CellBlob):
        self.orbiting = False
        ratio = (self.index+1) / self.blob.num_balls
        elevation = self.radius*1.2 + .04 * np.sin((target.spinner + .5) * ratio) 
        move_int = self.nodepath.posInterval(1., target.pos() + Vec3(0,0,elevation), fluid=1)
        Sequence(
            move_int,
            Func(self.set_orbiting_true),
            Func(base.sfx.bongs[target.bong].play)
        ).start()

    def update(self, task):
        if self.removing: return task.done
        pos = self.nodepath.get_pos()                                       # current ball position
        if (self.orbiting):
            # each ball gets a root of unity of num_balls (i.e. arrange them in an even circle)
            ratio = (self.index + 1)/ self.blob.num_balls
            self.angle = TAU * ratio + self.blob.spinner

            elevation = self.radius*1.2 + .04 * np.sin(self.angle)          # bob up and down rhythmically
                                                                            # go round in a little circle over the blob
            aimpos = self.blob.nodepath.get_pos() + Vec3(np.cos(self.angle)/2.,
                                            np.sin(self.angle)/2.,
                                            elevation)
            abs_dist: float = ABS_DIST(pos, aimpos)                         # absolute distance of that lad
            damper = min(1, max(0, abs_dist/2))                             # 1 if far, 0 if close

            self.nodepath.set_pos(pos + (aimpos-pos)*damper)                # load pos into the nodepath
        else:
            self.nodepath.set_pos(pos.x,pos.y,0+np.sin(globalClock.getDt())*.1)
        return task.cont

    def consume(self):
        self.removing = True
        self.nodepath.remove_node(Thread.current_thread)                    # remove nodepath from graph
        base.sound_fx.brrp_consume.play()                                   # play a little consume brrrrp
        taskMgr.remove(self.task_name)                                      # remove update method from taskMgr

#    def __del__(self):


# default entry point: launch game
if __name__ == "__main__":
    print("="*30 + "\n... Loading Blob ...")        # application entry point ---------------------

    print("-> Initialising ShowBase...")
    ShowBase()                                      # Showbase initialised

    base.set_background_color(.2,.1,.2,1.)

    base.sound_fx = SoundFX()                       # initialise sound library and put in base
    base.floating_balls = []                        # construct a list to store balls in 

                                                    # construct CellBlob for player 1
    player_1 = CellBlob(pos=Vec3(0.,-5.,0.),col=Vec4(0.,0.,1.,1.))

    BIND_BLOB(player_1)                             # Player 1 bound to user input
    base.accept("escape", base.userExit)            # hit `Esc` to quickly quit the game

                                                    # make some test balls to collect
    test_ball_1 = Ball(BallType.FOOD, f"ball-{len(base.floating_balls)}", pos=Vec3(-1.,-2.,0.))
    base.floating_balls.append(test_ball_1)
    test_ball_2 = Ball(BallType.HEAL, f"ball-{len(base.floating_balls)}", pos=Vec3(1.,-2.,0.))
    base.floating_balls.append(test_ball_1)

    base.cam.setPos(0,-18,5)                        # Adjust camera position and angle
    base.cam.setHpr(0,-15,0)

    print("-> All ready! Running ShowBase:")
    base.run()                                      # run Showbase
