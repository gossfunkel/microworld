from direct.showbase.ShowBase import ShowBase
#from direct.filter.CommonFilters import CommonFilters
from direct.interval.IntervalGlobal import *
from panda3d.core import (
    loadPrcFileData, Vec2, Vec3, Vec4,
    GeomTrifans, GeomVertexFormat, GeomVertexArrayFormat, InternalName, GeomEnums,
    GeomVertexData, Geom, GeomNode, DirectionalLight, UserDataAudio, AntialiasAttrib,
    TextureStage, Texture, TextNode, Thread, Shader, ShaderBuffer, BamFile
)
import numpy as np
from scipy.signal import chirp
import struct
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

        #render.setAntialias(AntialiasAttrib.MAuto)                 # set global antialiasing
        #render.setShaderAuto()

        self.sfx = SoundFX()                                        # initialise sound effect library
        self.sfx.brrps.append(self.sfx.add_brrp(2800))              # initialise default mol brrp

        # big_light_np = render.attachNewNode(DirectionalLight('the_big_light'))
        # big_light_np.node().setShadowCaster(True, 512, 512)
        # big_light_np.set_color(.5,.45,.49)
        # big_light_np.setHpr(20, -80, 0)
        # render.setLight(big_light_np)                             # set a warm directional light on the whole scene

        self.p1 = mol.Cell("p1",Vec2(0., 0.),Vec4(0.,0.,1.,1.), 200)    # create player 1's Cell
        self.p2 = mol.Cell("p2",Vec2(0., 50.),Vec4(0.,1.,0.,1.), 300)   # create a second Cell

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
        
        # energy HUD
        self.nrg_label = TextNode("nrg_label")
        self.nrg_label.setTextColor(.5,.5,1,1)
        self.nrg_label.setTextScale(0.1)
        self.nrg_label.setText("ENERGY:")
        nrg_label_np = aspect2d.attach_new_node(self.nrg_label)
        nrg_label_np.set_pos((-1.4,0.,-.85))
        # energy HUD value output
        self.energy_label_v = TextNode("0")
        self.energy_label_v.setTextColor(.6,.6,1.,1.)
        self.energy_label_v.setTextScale(0.1)
        energy_label_v_np = aspect2d.attach_new_node(self.energy_label_v)
        energy_label_v_np.set_pos((-.98,0.,-.85))

        # health HUD
        self.hp_label = TextNode("hp_label")
        self.hp_label.setTextColor(.5,1,.5,1)
        self.hp_label.setTextScale(0.1)
        self.hp_label.setText("HEALTH:")
        hp_label_np = aspect2d.attach_new_node(self.hp_label)
        hp_label_np.set_pos((-1.4,0.,-.7))
        # health HUD value output
        self.health_label_v = TextNode("0")
        self.health_label_v.setTextColor(.6,1.,.6,1.)
        self.health_label_v.setTextScale(0.1)
        health_label_v_np = aspect2d.attach_new_node(self.health_label_v)
        health_label_v_np.set_pos((-.98,0.,-.7))
        
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

    def update(self, task):
        self.cam.setPos(self.p1.pos() + CAM_POS)                    # camera follows p1
        # print(f"Cellpos: {self.p1.pos}; cam pos: {self.cam.getPos()}")
        self.p1_label_v.setText(str(self.p1.num_mols))             # update UI
        #self.p2_label_v.setText(str(self.p2.num_mols))
        self.energy_label_v.setText(str(self.p1.nrg)[:4]+"/"+str(self.p1.max_hp)[:2])
        self.health_label_v.setText(str(self.p1.hp)[:4]+"/"+str(self.p1.max_nrg)[:2])
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
    print("="*20 + " Welcome to microworld! v0.0.1 " + 20*"=")
    base = GameBase()                                               # Showbase initialised

    hp_mol_pos_1 = Vec3(-10,5,0)                                    # spawn some items
    hp_mol_pos_2 = Vec3(8,20,0)
    base.floating_items.append(mol.Mol(mol.MOLTYPE.HEAL, f"mol-{len(base.floating_items)}", pos=hp_mol_pos_1))
    base.floating_items.append(mol.Mol(mol.MOLTYPE.HEAL, f"mol-{len(base.floating_items)}", pos=hp_mol_pos_2))
    salt_mol_pos_1 = Vec3(-20,10,0)
    salt_mol_pos_2 = Vec3(-7,34,0)
    salt_mol_pos_3 = Vec3(12,-5,0)
    base.floating_items.append(mol.Mol(mol.MOLTYPE.SALT, f"mol-{len(base.floating_items)}", pos=salt_mol_pos_1))
    base.floating_items.append(mol.Mol(mol.MOLTYPE.SALT, f"mol-{len(base.floating_items)}", pos=salt_mol_pos_2))
    base.floating_items.append(mol.Mol(mol.MOLTYPE.SALT, f"mol-{len(base.floating_items)}", pos=salt_mol_pos_3))
    mana_mol_pos_1 = Vec3(4,4,0)
    mana_mol_pos_2 = Vec3(0,20,0)
    mana_mol_pos_3 = Vec3(-6,42,0)
    mana_mol_pos_4 = Vec3(34,11,0)
    mana_mol_sfx = base.sfx.add_brrp(2200)
    base.floating_items.append(mol.Mol(mol.MOLTYPE.MANA, f"mol-{len(base.floating_items)}", 
                                    pos=mana_mol_pos_1, sfx=mana_mol_sfx))
    base.floating_items.append(mol.Mol(mol.MOLTYPE.MANA, f"mol-{len(base.floating_items)}", 
                                    pos=mana_mol_pos_2, sfx=mana_mol_sfx))
    base.floating_items.append(mol.Mol(mol.MOLTYPE.MANA, f"mol-{len(base.floating_items)}", 
                                    pos=mana_mol_pos_3, sfx=mana_mol_sfx))
    base.floating_items.append(mol.Mol(mol.MOLTYPE.MANA, f"mol-{len(base.floating_items)}", 
                                    pos=mana_mol_pos_4, sfx=mana_mol_sfx))

    test_rock = Rock(8,(0.,5.,0.), 2.)

    base.run()                                                      # taskMgr blocks
