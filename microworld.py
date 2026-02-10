from direct.showbase.ShowBase import ShowBase
#from direct.filter.CommonFilters import CommonFilters
from direct.interval.IntervalGlobal import *
from panda3d.core import (
    loadPrcFileData, Vec2, Vec3, Vec4, DirectionalLight, UserDataAudio, 
    TextureStage, Texture, TextNode, CardMaker, ColorBlendAttrib, TransparencyAttrib
)
import numpy as np
from scipy.signal import chirp
import struct
import copy
from rock import Rock
import mol

TAU: float = np.pi * 2              # for calculating circles

# the Cells are supposed to be little microbial cell guys
# the floating mols can be hp (heal), mana (survival), protein (upgrade), fats (grow), salts (constitution)
# TODO: the cells and rocks don't interact. They should slide past/off each other
# TODO: spatial partitioning. This ^ and the mol-collection would benefit a lot
# verticality; since we can only move on a plane, the z-dim could be used to make things inaccessible?

# TODO update camera position when player Cell gets significantly bigger
CAM_POS: Vec3 = Vec3(0,-8,3)        # for keeping the camera a constant vector from the player Cell

CONFIG: str = """
gl-version 4 3
gl-debug true
gl-debug-buffers true
//gl-force-glsl-version 430
gl-support-spirv false
//premunge-data false
win-size 1200 800
//show-frame-rate-meter true
//model-cache-compiled-shaders false
hardware-animated-vertices true
framebuffer-srgb true
//basic-shaders-only false
//gl-interleaved-arrays true
"""
loadPrcFileData("", CONFIG)

CHUNK_SIZE = 30.

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
        atk: np.array = np.linspace(0,1,2000, dtype=np.float32)                 # ascending attack
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

class GameBase(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.set_background_color(0.12,0.05,0.22,1.)                # dark background
        self.CHUNK_SIZE = CHUNK_SIZE
        self.loaded_chunks = None

        #render.setAntialias(AntialiasAttrib.MAuto)                 # set global antialiasing
        #render.setShaderAuto()

        self.sfx = SoundFX()                                        # initialise sound effect library
        self.sfx.brrps.append(self.sfx.add_brrp(2800))              # initialise default mol brrp
        self.setup_grid()

        # big_light_np = render.attachNewNode(DirectionalLight('the_big_light'))
        # big_light_np.node().setShadowCaster(True, 512, 512)
        # big_light_np.set_color(.5,.45,.49)
        # big_light_np.setHpr(20, -80, 0)
        # render.setLight(big_light_np)                             # set a warm directional light on the whole scene

        self.p1 = mol.Cell("p1",Vec2(0., 0.),Vec4(0.,0.,1.,1.), 200, (0,0))    # create player 1's Cell
        self.p2 = mol.Cell("p2",Vec2(0., 25.),Vec4(0.,1.,0.,1.), 300, (0,0))   # create a second Cell
        self.grid[0][0].append(self.p1)
        self.grid[0][0].append(self.p2)
        self.load_chunk((0,0))

        # mol counters
        self.p1_label_v = TextNode("0")
        self.p1_label_v.setTextColor(1,1,1,1)
        self.p1_label_v.setTextScale(0.1)
        p1_label_v_np = aspect2d.attach_new_node(self.p1_label_v)
        p1_label_v_np.set_pos((1.3,0.,.85))
        
        # self.p2_label_v = TextNode("0")
        # self.p2_label_v.setTextColor(1,1,1,1)
        # self.p2_label_v.setTextScale(0.1)
        # p2_label_v_np = aspect2d.attach_new_node(self.p2_label_v)
        # p2_label_v_np.set_pos((1.3,0.,.7))

        bar_maker = CardMaker("bars")
        bar_maker.set_frame(0.,1.,0.,.1,)

        #    render   = incoming_col * A - framebuffer_col * B
        bar_text_cba  = ColorBlendAttrib.make(ColorBlendAttrib.M_subtract, 
                                              ColorBlendAttrib.O_one, 
                                              ColorBlendAttrib.O_one)
        bar_text_trat = TransparencyAttrib.make(TransparencyAttrib.M_binary)

        # energy HUD
        self.nrg_bar = aspect2d.attach_new_node(bar_maker.generate())
        self.nrg_bar.set_pos((-1.4,0.,-.9))
        self.nrg_bar.set_texture(loader.loadTexture(mol.MOLTYPE.MANA.value))
        
        self.nrg_label = TextNode("nrg_label")
        self.nrg_label.set_attrib(bar_text_trat, 1000)
        self.nrg_label.setTextColor(.5,1.,1.,.8)
        self.nrg_label.setTextScale(0.078)
        self.nrg_label.setText("ENERGY:")
        self.nrg_label.setAttrib(bar_text_cba)
        nrg_label_np = aspect2d.attach_new_node(self.nrg_label)
        nrg_label_np.set_pos((-1.35,0.,-.872))
        # energy HUD value output
        self.energy_label_v = TextNode("0")
        self.energy_label_v.set_attrib(bar_text_trat, 1000)
        self.energy_label_v.setTextColor(.6,1.,1.,.8)
        self.energy_label_v.setTextScale(0.08)
        self.energy_label_v.setAttrib(bar_text_cba)
        energy_label_v_np = aspect2d.attach_new_node(self.energy_label_v)
        energy_label_v_np.set_pos((-.97,0.,-.87))

        # health HUD
        self.hp_bar = aspect2d.attach_new_node(bar_maker.generate())
        self.hp_bar.set_pos((-1.4,0.,-.75))
        self.hp_bar.set_texture(loader.loadTexture(mol.MOLTYPE.HEAL.value))

        self.hp_label = TextNode("hp_label")
        self.hp_label.set_attrib(bar_text_trat, 1000)
        self.hp_label.setTextColor(.5,1.,.5,.8)
        self.hp_label.setTextScale(0.078)
        self.hp_label.setText("HEALTH:")
        self.hp_label.setAttrib(bar_text_cba)
        hp_label_np = aspect2d.attach_new_node(self.hp_label)
        hp_label_np.set_pos((-1.35,0.,-.722))
        # health HUD value output
        self.health_label_v = TextNode("0")
        self.health_label_v.set_attrib(bar_text_trat, 1000)
        self.health_label_v.setTextColor(.6,1.,.6,.8)
        self.health_label_v.setTextScale(0.08)
        self.health_label_v.setAttrib(bar_text_cba)
        health_label_v_np = aspect2d.attach_new_node(self.health_label_v)
        health_label_v_np.set_pos((-.97,0.,-.72))
        
        # TODO: Nodepath or spacial partitioning
        self.floating_items = []                                    # big list of all nearby collectable items

        # awsd/keypad movement for p1 Cell
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
        
        self.accept("r", self.p1.make_mol, [mol.MOLTYPE.FRNA])        # make an frna mol from energy
        self.accept("f", self.p1.make_mol, [mol.MOLTYPE.FOOD])        # make an frna mol from energy
        self.accept("space", self.p1.consume_mol)                  # consume a mol
        self.accept("escape", self.userExit)                        # quickly quit the game

        self.cam.setPos(CAM_POS)                                    # spawn camera distance from origin
        self.cam.setHpr(0,-18,0)                                    # look down at your Cell! 

        self.taskMgr.add(self.update, "update")                     # global game update

    def setup_grid(self):
        # generate a 3x3 array of collider arrays
        self.grid = np.array([[None] for _ in range(81)]) # u, v, contents
        self.grid.resize((9,9))
        for i in range(81):
            self.grid[i%9,i//9] = self.load_level_random(((i%9)-1,(i//9)-1))
        return self.grid

    def get_chunk(self, uv: tuple[int]):
        # calculate actual array indices for given uvs (i.e. -1,-1 = 0,0 for a 3x3 grid)
        gridsize = self.grid.shape
        return self.grid[uv[0]+gridsize[0]//2, uv[1]+gridsize[1]//2]

    def get_loaded_chunks(self):
        return self.loaded_chunks

    def load_chunk(self, uv: tuple[int]):
        if self.loaded_chunks == None:
            self.loaded_chunks = self.get_chunk((uv[0],uv[1]))
            #self.get_chunk((uv[0],uv[1])) = None
        else:
            for item in base.get_chunk((uv[0]+1,uv[1])):
                self.loaded_chunks.append(item)
            #self.get_chunk((uv[0],uv[1])) = None

    # takes uv coords, not a base.grid index
    def load_level_random(self, chunk_id: tuple[int]):
        chunk = []

        # position of the CENTRE of the chunk
        gridpos = Vec3(chunk_id[0]*CHUNK_SIZE,chunk_id[1]*CHUNK_SIZE,0.)
        for _ in range(np.random.randint(1,4)):
            # spawn health mols
            hp_mol_pos = Vec3(np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              0) + gridpos
            chunk.append(mol.Mol(mol.MOLTYPE.HEAL, f"mol-{chunk_id}-{len(chunk)}", 
                                      chunk_id, pos=hp_mol_pos))
        for _ in range(np.random.randint(1,6)):
            # spawn salt mols
            salt_mol_pos = Vec3(np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              0) + gridpos
            chunk.append(mol.Mol(mol.MOLTYPE.SALT, f"mol-{chunk_id}-{len(chunk)}", 
                                      chunk_id, pos=salt_mol_pos))
        mana_mol_sfx = base.sfx.add_brrp(2200)
        for _ in range(np.random.randint(2,8)):
            # spawn mana mols
            mana_mol_pos = Vec3(np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              0) + gridpos
            chunk.append(mol.Mol(mol.MOLTYPE.MANA, f"mol-{chunk_id}-{len(chunk)}", 
                                      chunk_id, pos=mana_mol_pos, sfx=mana_mol_sfx))
        for _ in range(np.random.randint(0,1)):
            # spawn protein mols
            frna_mol_pos = Vec3(np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              0) + gridpos
            chunk.append(mol.Mol(mol.MOLTYPE.FRNA, f"mol-{chunk_id}-{len(chunk)}", 
                                      chunk_id, pos=frna_mol_pos))
        for _ in range(np.random.randint(0,2)):
            # spawn carb mols
            food_mol_pos = Vec3(np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              0) + gridpos
            chunk.append(mol.Mol(mol.MOLTYPE.FOOD, f"mol-{chunk_id}-{len(chunk)}", 
                                      chunk_id, pos=food_mol_pos))
        for _ in range(np.random.randint(1,4)):
            # spawn some rocks
            rock_pos = Vec3(np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              0) + gridpos
            chunk.append(Rock(8,rock_pos, np.random.uniform(1.,3.)))
        return chunk
    
    def load_level(self, chunk: tuple[int]):
        for item in floating_items:
            chunk.append(item)
        for rock in rocks:
            chunk.append(rock)
        return chunk

    def update(self, task):
        self.cam.setPos(self.p1.pos() + CAM_POS)                    # camera follows p1
        # print(f"Cellpos: {self.p1.pos}; cam pos: {self.cam.getPos()}")
        self.p1_label_v.setText(str(self.p1.num_mols))             # update UI
        #self.p2_label_v.setText(str(self.p2.num_mols))
        self.energy_label_v.setText(str(self.p1.nrg)[:4]+"/"+str(self.p1.max_hp)[:2])
        self.health_label_v.setText(str(self.p1.hp)[:4]+"/"+str(self.p1.max_nrg)[:2])
        self.nrg_bar.set_scale(self.p1.nrg/self.p1.max_nrg,1.,1.)
        self.hp_bar.set_scale(self.p1.hp/self.p1.max_hp,1.,1.)
        return task.cont

    def game_over(self, msg: str = None):
        # TODO show game over screen
        print("\t\t=== GAME OVER ===")
        if msg is not None:
            print(msg)
        self.userExit()                                             # exit the game

    def __del__(self):                                              # ShowBase deconstructor
        print("="*20 + " See you soon!:) " + 20*"=")

if __name__ == "__main__":
    print("="*20 + " Welcome to microworld! v0.0.2 " + 20*"=")
    base = GameBase()                                               # Showbase initialised

    base.run()                                                      # taskMgr blocks
