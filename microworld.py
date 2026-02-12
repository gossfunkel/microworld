import struct                                                       #python imports
import copy

from direct.showbase.ShowBase import ShowBase                       # library imports
#from direct.filter.CommonFilters import CommonFilters
from direct.interval.IntervalGlobal import *
from panda3d.core import (
    loadPrcFileData, Vec2, Vec3, Vec4, DirectionalLight, UserDataAudio, 
    TextureStage, Texture, TextNode, CardMaker, ColorBlendAttrib, TransparencyAttrib
)
import numpy as np
from scipy.signal import chirp

from rock import Rock                                               # game imports
import mol
import ui
import user_controls as controls

TAU: float = np.pi * 2                                              # for calculating circles

# the Cells are supposed to be little microbial cell guys
# the floating mols can be hp (heal), mana (survival), protein (upgrade), fats (grow), salts (constitution)
# TODO: the cells and rocks don't interact. They should slide past/off each other
# TODO: spatial partitioning. This ^ and the mol-collection would benefit a lot
# verticality; since we can only move on a plane, the z-dim could be used to make things inaccessible?

CHUNK_SIZE = 30.                                                    # constants
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

# Main game class and loop manager
class GameBase(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.set_background_color(0.12,0.05,0.22,1.)                # dark background
        self.CHUNK_SIZE = CHUNK_SIZE
        self.loaded_chunks = [(0,0)]

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
        self.grid[0,0].append(self.p1)
        self.grid[0,0].append(self.p2)
        self.chunks = []
        self.load_chunk((0,0))

        ui.setup_ui()

        controls.bind_cell(self.p1)

        self.taskMgr.add(ui.update, "update")                       # global game update

    # initialise a 9x9 grid of chunks
    def setup_grid(self):
        # generate a 3x3 array of collider arrays
        self.grid = np.array([[None] for _ in range(81)]) # u, v, contents
        self.grid.resize((9,9))
        for i in range(81):
            self.grid[i%9,i//9] = self.load_level_random(((i%9)-4,(i//9)-4))
        return self.grid

    # find a chunk in the grid by uv coords
    def get_chunk(self, uv: tuple[int]) -> list:
        # calculate actual array indices for given uvs (i.e. -1,-1 = 0,0 for a 3x3 grid)
        gridsize = self.grid.shape
        assert abs(uv[0]) <= gridsize[0]//2, f"REQUESTING CHUNK OUT OF BOUNDS ON U: {uv[0]}"
        assert abs(uv[1]) <= gridsize[1]//2, f"REQUESTING CHUNK OUT OF BOUNDS ON V: {uv[1]}"
        return self.grid[uv[0]+gridsize[0]//2, uv[1]+gridsize[1]//2]

    # get the cache of chunk data
    def get_chunks(self):
        return self.chunks

    # get the list of which chunks have had their data cached for collisions
    def get_loaded_chunks(self):
        return self.loaded_chunks

    # load data from chunk at uv into collision cache 
    def load_chunk(self, uv: tuple[int]):
        #for item in base.get_chunk((uv[0],uv[1])):
        #    self.chunks.append(item)
        chunk = base.get_chunk(uv)
        #print(f"Loading chunk {chunk}")
        self.chunks.append(chunk)
        self.loaded_chunks.append(uv)
        #self.get_chunk((uv[0],uv[1])) = None

    # load data from multiple chunks at uvs in list into collision cache 
    def load_chunks(self, uvs: list[tuple[int]]):
        for uv in uvs:
            #for item in base.get_chunk((uv[0],uv[1])):
            #    self.chunks.append(item)
            chunk = base.get_chunk(uv)
            self.chunks.append(chunk)
            self.loaded_chunks.append(uv)                           # take note of which chunks have been loaded
            #self.get_chunk((uv[0],uv[1])) = None

    # check if any of a list of uvs of chunks is not loaded. returns uvs of chunks not loaded to collision cache 
    def check_loaded_chunks(self, chunks: list[tuple[int]]) -> list[tuple[int]]:
        missing_chunks = []
        for chunk in chunks:
            if chunk not in self.loaded_chunks:
                missing_chunks.append(chunk)
        return missing_chunks

    # TODO something to keep track of which chunks are no longer nearby and can be unloaded from collision cache

    # takes uv coords, not a base.grid index
    def load_level_random(self, chunk_id: tuple[int]):
        chunk = []

        # position of the CENTRE of the chunk
        gridpos = Vec3(chunk_id[0]*CHUNK_SIZE,chunk_id[1]*CHUNK_SIZE,0.)
        for _ in range(np.random.randint(4,8)):
            # spawn water mols
            water_mol_pos = Vec3(np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              0) + gridpos
            chunk.append(mol.Mol(mol.MOLTYPE.WATER, f"mol-{chunk_id}-{len(chunk)}", 
                                      chunk_id, pos=water_mol_pos))
        for _ in range(np.random.randint(1,6)):
            # spawn salt mols
            salt_mol_pos = Vec3(np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              0) + gridpos
            chunk.append(mol.Mol(mol.MOLTYPE.SALT, f"mol-{chunk_id}-{len(chunk)}", 
                                      chunk_id, pos=salt_mol_pos))
        sugar_mol_sfx = base.sfx.add_brrp(2200)
        for _ in range(np.random.randint(2,5)):
            # spawn mana mols
            sugar_mol_pos = Vec3(np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              0) + gridpos
            chunk.append(mol.Mol(mol.MOLTYPE.SUGAR, f"mol-{chunk_id}-{len(chunk)}", 
                                      chunk_id, pos=sugar_mol_pos, sfx=sugar_mol_sfx))
        for _ in range(np.random.randint(0,2)):
            # spawn carb mols
            carb_mol_pos = Vec3(np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              0) + gridpos
            chunk.append(mol.Mol(mol.MOLTYPE.CARB, f"mol-{chunk_id}-{len(chunk)}", 
                                      chunk_id, pos=carb_mol_pos))
        for _ in range(np.random.randint(0,2)):
            # spawn health mols
            oils_mol_pos = Vec3(np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              0) + gridpos
            chunk.append(mol.Mol(mol.MOLTYPE.OILS, f"mol-{chunk_id}-{len(chunk)}", 
                                      chunk_id, pos=oils_mol_pos))
        for _ in range(np.random.randint(0,1)):
            # spawn protein mols
            amino_mol_pos = Vec3(np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              np.random.uniform(-CHUNK_SIZE,CHUNK_SIZE),
                              0) + gridpos
            chunk.append(mol.Mol(mol.MOLTYPE.AMINO, f"mol-{chunk_id}-{len(chunk)}", 
                                      chunk_id, pos=amino_mol_pos))
        for _ in range(np.random.randint(1,3)):
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
