from direct.interval.IntervalGlobal import *
from panda3d.core import (
    Vec2, Vec3, Vec4, BamFile, Shader, ShaderBuffer, GeomEnums, Thread
)
import numpy as np
from enum import Enum
import struct
import copy

import user_controls as controls

EPSILON: float = .0001              # a very small change
TAU: float = np.pi * 2              # for calculating circles

def ABS_DIST(a: Vec3, b: Vec3) -> float:
    return np.sqrt((a.x-b.x)*(a.x-b.x) + 
                   (a.y-b.y)*(a.y-b.y) +
                   (a.z-b.z)*(a.z-b.z))

class MOLTYPE(Enum):
    WATER = "teal.png"              # essential for metabolism                  = balance metabolism
    SUGAR = "rose.png"             # energy sources - electron transfer agents = restore energy
    CARB  = "gold.png"              # energy sources - catabolic substrate      = grow cell
    OILS  = "green.png"              # nutrition - lipids                        = heal damage
    AMINO = "purple.png"            # nutrition - nucleotides                   = power up cell
    SALT  = "white.png"             # salts                                     = slows energy loss

# floating resource orbs. can bind to cells
class Mol:
    def __init__(self, type: MOLTYPE, name: str, uv: tuple[int], pos: Vec3 | None = None, 
                 cell = None, index: int | None = None, sfx = None):
        self.type  = type
        self.name  = name
        self.cell  = cell       # cell that mol is attached to, if any
        self.index = index          # helps mols rotate on the cells neatly
        self.sfx   = sfx            # associated sound effect
        self.radius: float = .15    # personal space
        # self.velocity: Vec3 = Vec3(0,0,np.random.uniform(-.1,.05))
        self.angle = 0
        self.orbiting = True if cell is not None else False
        self.uv    = uv
        self.ismol = True

        model = base.loader.load_model("sphere.egg")
        #model.setTransparency(1)
        #ts_col = TextureStage('ts_col')
        model.setTexture(loader.loadTexture(self.type.value))
        # ts_glow = TextureStage('ts_glow')
        # ts_glow.setMode(TextureStage.MGlow)
        # black_tex = loader.loadTexture("black.png")
        # model.setTexture(ts_glow, black_tex)
        model.set_scale(.06)
        self.nodepath = base.render.attach_new_node(f"mol-{self.name}")
        model.reparent_to(self.nodepath)
        if self.cell is None:
            assert pos is not None, f"A free mol must have a position!"
            self.nodepath.set_pos(pos)                              # use given position
            self.sfx = base.sfx.brrps[0]
        else:
            self.nodepath.set_pos(self.cell.pos() + Vec3(.5,0,.22)) # adjustments for oscillations
            self.sfx = base.sfx.bongs[self.cell.bong]               # set sound effect to cell bong
            self.sfx.play()                                         # play bong
        self.task_name = f"update_mol-{self.name}"
        base.taskMgr.add(self.update, self.task_name)               # add update to taskMgr

    # Function to enable orbiting with Sequences
    def set_orbiting_true(self):
        self.orbiting = True

    # Lerp the mol to a cell
    def fly_to_target(self, target):
        self.orbiting = False
        # abs_dist = ABS_DIST(self.nodepath.get_pos(),target)
        ratio = (self.index+1) / len(self.cell.mols)
        elevation = self.radius*1.2 + .04 * np.sin((target.spinner + .5) * ratio)   # this works if dt is in seconds (doubt, hahaha)
        move_int = self.nodepath.posInterval(1., target.pos() + Vec3(0,0,elevation), fluid=1)
        Sequence(
            move_int,
            Func(self.set_orbiting_true),
            Func(base.sfx.bongs[target.bong].play)
        ).start()

    def update(self, task):
        pos = self.nodepath.get_pos()                               # current mol position
        if (self.orbiting):
            # each mol gets a root of unity of len(mols) (arrange them in an even circle)
            ratio = (self.index + 1)/ len(self.cell.mols)
            self.angle = TAU * ratio + self.cell.spinner
            # bob up and down
            elevation = self.radius*1.2 + .04 * np.sin(self.angle)
            # go round in a little circle over the cell
            aimpos = self.cell.pos() + Vec3(np.cos(self.angle)/2.,
                                            np.sin(self.angle)/2.,
                                            elevation)
            abs_dist: float = ABS_DIST(pos, aimpos)                 # absolute distance of that lad
            damper = min(1, max(0, abs_dist/2))                     # 1 if far, 0 if close
            self.nodepath.set_pos(pos + (aimpos-pos)*damper)
        else:
            self.nodepath.set_pos(pos.x,pos.y,0+np.sin(globalClock.getDt())*.1)
        return task.cont

    def consume(self):
        self.orbiting = False                                       # ensure orbit doesn't run while dying
        self.nodepath.remove_node(Thread.current_thread)
        self.sfx.play()                                             # play a little consume brrrrp 
        taskMgr.remove(self.task_name)
        #if self.cell is not None:
        #    self.cell.remove_cell(self)


# a PC. cell-like blobby guy that blobs about
class Cell:
    def __init__(self, name: str, pos: Vec2, col: Vec4, bong_freq: float, uv: tuple[int]) -> None:
        # TODO move data into C++ obj tags or VBO
        self.name            = name
        self.col: tuple      = col
        self.uv              = uv            # coordinate in the grid

        self.radius: float   = 2.
        self.verts: int      = 12            # number of OUTER vertices (Cell requires centre)
        self.velocity        = Vec2(0.,0.)   # initial speed
        self.colliding       = False         # flag for if Cell is colliding with something

        print(f"Loading {self.name} VBO data...")
        # load the default Cell from file
        loaded_file = BamFile()
        loaded_file.open_read("cell_default.bam")
        node = loaded_file.read_node()
        loaded_file.close()

        # modify geom vertex data from file
        #print(node.get_child(0).get_geom(0))#print(f"VBO data from file: {vtx_data}")
        vtx_data = node.get_child(0).get_geom(0).get_vertex_data()

        # make an SSBO from the model data for vertex pulling
        p3d_array = vtx_data.get_array_handle(0).get_data()
        #custom_array = vtx_data.get_array_handle(1).get_data()
        byte_data = bytearray(p3d_array)
        #byte_data.extend(custom_array)

        self.buffer = ShaderBuffer("ssbo", bytes(byte_data), GeomEnums.UHDynamic)
        # vals = bytearray()
        # for _ in range(13):
        #     vals.extend(struct.pack('4f', 0., 0., 0., 0.))
        # self.coll_buff = ShaderBuffer("coll_buff", bytes(vals), GeomEnums.UHDynamic)

        self.nodepath = base.render.attach_new_node(node)
        # store position in the node 
        self.nodepath.set_pos(pos.x, pos.y, 0)
        # give the Cell a depth offset to prevent self-shadowing etc
        self.nodepath.setDepthOffset(1)

        # activate the jiggle shader on the Cell
        self.nodepath.set_shader(Shader.load(Shader.SL_GLSL, vertex="cell_jiggle.vert", fragment="default_shader.frag"))
        self.nodepath.set_shader_input("ssbo", self.buffer)
        #self.nodepath.set_shader_input("coll_buff", self.coll_buff)
        self.nodepath.set_shader_input("radius", self.radius)
        self.nodepath.set_shader_input("model_velocity", self.velocity)
        self.nodepath.set_shader_input("col", self.col)
        #self.nodepath.set_shader_input("colliding", self.colliding)

        # now set up accessories
        self.bong           = base.sfx.add_bong(bong_freq)          # generate sound effect at given freq
        self.mols          = []                                     # array for mols on Cell
        self.spinner: float = 0                                     # this tells mols on this Cell how to rotate neatly

        # stats BALANCE
        self.max_hp: float   = 10.           # maximum health
        self.hp: float       = self.max_hp   # current health
        self.max_nrg: float  = 10.           # maximum energy
        self.nrg: float      = self.max_nrg  # current energy

        self.salinity: float = 0.            # current SALT level
        self.hydration: float = 5.           # current WATER level
        self.carbs: int      = 2             # quantity of CARBS in cell
        self.oils: int       = 0             # quantity of FATS in cell
        self.aminos: int     = 0             # quantity of AMINOS in cell

        self.speed: float    = 1.            # amount to add to position per frame per dt
        self.nrg_loss_rate   = .01           # rate of energy loss per second
        self.dry_rate: float = .01           # rate of water loss per second
        self.carb_digest_time: float = 5.    # seconds to digest a carb into energy

        self.selected: int   = 0             # currently selected ball

        base.taskMgr.add(self.update, str(name)+"-update")

        print(f"== Cell {name} created!")

    # alias to make this quicker
    def pos(self) -> Vec3:
        return self.nodepath.get_pos()

    def grow(self):
        if (self.carbs >= 1) and (self.oils >= 1):                      # growing costs 1 carb and 1 oil
            self.carbs -= 1
            self.oils  -= 1
            rad_pregrowth = self.radius
            self.max_hp += 1.                                           # increase maximum health
            self.radius *= 1.1                                          # make the Cell bigger
            self.nrg_loss_rate += .001                                  # lose energy faster BALANCE
            self.nodepath.set_shader_input("radius", self.radius)       # update the shader
            if int(rad_pregrowth) < int(self.radius):
                controls.zoom_out()
        else:
            print("Not enough resources (carbs or oils) to grow!")

    def make_mol(self, *mols: MOLTYPE):
        if self.nrg > 1.:
            for mol in mols:
                self.nrg -= 1.
                match mol:
                    case MOLTYPE.AMINO:
                        self.mols.append(Mol(MOLTYPE.AMINO, self.name+"-AMINOmol-"+str( len(self.mols)), 
                                               self.uv, cell=self, index=len(self.mols)))
                    case MOLTYPE.CARB:
                        self.mols.append(Mol(MOLTYPE.CARB, self.name+"-CARBmol-"+str( len(self.mols)), 
                                               self.uv, cell=self, index= len(self.mols)))
        else:
            print("DANGER: Insufficient energy to produce mol!")

    # change currently selected mol TODO show self.selected somehow (ideally highlighting)
    def select_mol(self, direction: str):
        if direction == "left":
            self.selected = (self.selected-1)%max(1, len(self.mols))
        elif direction == "right":
            self.selected = (self.selected+1)%max(1, len(self.mols))
        else:
            print(f"Incorrect input passed to select_mol: {direction}. Pass 'left' or 'right'!")

    def consume_mol(self, mol=None):
        if isinstance(mol,type(None)):                              # consume the first mol in the buffer
            if len(self.mols) > 0:
                mol = self.mols[self.selected]
                mol_type = copy.deepcopy(mol.type)                  # copy mol.type to prevent reference
                mol.consume()
                del self.mols[self.selected]
                if self.selected > len(self.mols)-1:
                    self.selected = 0
            else:
                print("No mols to consume!")                        # consumption failed, return
                return;
        else:                                                       # consume the mol that was passed
            mol_type = copy.deepcopy(mol.type)                      # copy mol.type to prevent reference
            mol.consume()
            del mol

        match mol_type:                                             # activate the appropriate effect
            case MOLTYPE.WATER:
                self.hydration += 1.                                # hydration station! store water
                self.salinity -= .25                                # water reduces salinity
            case MOLTYPE.SALT:
                self.salinity += 1.                                 # TODO salinity moves energy loss to hp loss
                self.nrg_loss_rate *= .75                           # reduce energy loss rate by a quarter BALANCE
                self.dry_rate *= 1.1                                # salt dehydrates you hahaha
            case MOLTYPE.SUGAR:
                self.nrg = min(self.max_nrg, self.nrg + 1.)         # sugar is consumed for immediate energy
            case MOLTYPE.CARB:
                self.carbs += 1                                     # carbs are stored as a resource
            case MOLTYPE.OILS:
                self.oils += 1                                      # oils are stored as a resource
                # no overhealing - just top up health to maximum health at most
                self.hp = min(self.max_hp, self.hp + 1.)
            case MOLTYPE.AMINO:
                print("Power up!")
                self.aminos += 1                                    # aminos are stored as a resource
                # TODO metabolic objectives; growing utilities. Menu or progression? Unlocks?
                self.speed *= 1.5                                   # increase cell speed

    def add_mol(self, mol=None, mols: int = 1):
        #print(f"adding mol to {self.name}")
        self.mols.append(mol)                                       # add mol to mols
        mol.cell = self                                             # change mol references to self
        mol.index = len(self.mols)-1
        mol.set_orbiting_true()                                     # make mol orbit
        base.sfx.bongs[self.bong].play()                            # make a jubilant bong

    def update(self, task):
        # processor-killing debug:
        #print(f"cell {self.name} nodepath position: {self.pos()}")
        #vtx_view = memoryview(self.nodepath.node().get_vertex_data().modify_array(0)).cast('B').cast('f')
        #print(f"some verts from {self.name}: 0: {vtx_view[0]}, 1: {vtx_view[1]}, 2: {vtx_view[2]}")

        dt = globalClock.get_dt()

        if (self.salinity > 4.):
            self.hp -= (self.salinity-4.)*.01*dt                       # damage from salting out BALANCE

        self.nrg -= self.nrg_loss_rate*dt                              # tick energy loss

        if (self.nrg <= 0.) or (self.hp <= 0.): self.die()          # check if cell should die

        # update cell uv as it travels through the chunks
        xneg = -1 if self.pos().x < 0 else 1
        yneg = -1 if self.pos().y < 0 else 1
        self.uv = (int((self.pos().x + xneg*base.CHUNK_SIZE//2)//base.CHUNK_SIZE), 
                   int((self.pos().y + yneg*base.CHUNK_SIZE//2)//base.CHUNK_SIZE))
        #print(f"{self.name} uv is {self.uv}")

        check_chunks = [self.uv]
        # load chunks as cell approaches them
        if (((self.pos().x+base.CHUNK_SIZE/2)%base.CHUNK_SIZE < self.radius) & 
            ((self.pos().x+base.CHUNK_SIZE/2)%base.CHUNK_SIZE > 0.)):
            check_chunks.append((self.uv[0]+1,self.uv[1]))
        elif (((self.pos().x-base.CHUNK_SIZE/2)%base.CHUNK_SIZE > -self.radius) & 
              ((self.pos().x-base.CHUNK_SIZE/2)%base.CHUNK_SIZE < 0.)):
            check_chunks.append((self.uv[0]-1,self.uv[1]))
        if (((self.pos().y+base.CHUNK_SIZE/2)%base.CHUNK_SIZE < self.radius) & 
            ((self.pos().y+base.CHUNK_SIZE/2)%base.CHUNK_SIZE > 0.)):
            check_chunks.append((self.uv[0],self.uv[1]+1))
        elif (((self.pos().y-base.CHUNK_SIZE/2)%base.CHUNK_SIZE > -self.radius) & 
              ((self.pos().y-base.CHUNK_SIZE/2)%base.CHUNK_SIZE < 0.)):
            check_chunks.append((self.uv[0],self.uv[1]-1))

        chunks_to_load = base.check_loaded_chunks(check_chunks)     # check which chunks are loaded
        base.load_chunks(chunks_to_load)                            # load any remaining chunks

        # collision detection - get items loaded from chunks
        check_items = []
        for chunk in base.get_chunks():
            #print(f"Adding chunk {chunk} to check_items from {base.get_chunks()}")
            check_items += chunk
        # check all items in nearby chunks
        for item in check_items:
            #print(f"checking item: {item}")
            #print(f"blob 2D pos: {self.pos().xy}, item 2D pos: {item.nodepath.get_pos().xy}")
            if ABS_DIST(self.pos(), Vec3(item.nodepath.get_pos().xy,0.)) < (self.radius + item.radius):
                if hasattr(item, 'ismol'):
                    #print(f"adding mol {item}")
                    self.add_mol(item)
                    base.get_chunk(item.uv).remove(item)
                else:
                    self.colliding = True
                    print(f"{self.name} colliding with {item}")
                    # item.buffer
                    # self.nodepath.set_shader_input("coll_buff", )
                    # self.nodepath.set_shader_input("colliding", self.colliding)
            else:
                self.colliding = False
                #self.nodepath.set_shader_input("colliding", self.colliding)

        self.nodepath.set_pos(self.pos() + Vec3(self.velocity*dt, 0.))
        self.nodepath.set_shader_input("model_velocity", self.velocity*dt)
        # cell experiences friction, causing velocity to naturally decrease
        self.velocity = self.velocity/10. if self.velocity > EPSILON else Vec2(0.,0.) 

        self.spinner += dt%360
        return task.cont

    # move the cell by its nodepath.pos
    def move(self, direction):
        pos = self.pos()
        match direction:
            case "left":                                            # go left
                self.velocity -=  Vec2(self.speed,0.)
            case "right":                                           # go right
                self.velocity +=  Vec2(self.speed,0.)
            case "fwd":                                             # go forwards
                self.velocity +=  Vec2(0.,self.speed)
            case "back":                                            # ...you guessed it
                self.velocity -=  Vec2(0.,self.speed)
            case _: 
                print("Move direction not recognised!")

    def die(self):
        # drop mols
        for mol in self.mols:
            mol.orbiting = False
            mol.cell = None
            mol.index = None
            mol.uv = self.uv
            base.get_chunk(self.uv).append(mol)
        # die
        taskMgr.remove(self.name+"-update")
        if self.name == "p1":
            if self.nrg <= 0.:
                base.game_over(" You ran out of energy! ")
            elif self.hp <= 0.:
                base.game_over(" Your health ran out! ")
            else:
                base.game_over(" You died! ")