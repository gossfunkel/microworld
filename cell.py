from panda3d.core import (
    Vec2, Vec3, Vec4, BamFile, ShaderBuffer
)
import mol

# a PC. cell-like blobby guy that blobs about
class Cell:
    def __init__(self, name: str, pos: Vec2, col: Vec4, bong_freq: float) -> None:
        # TODO move data into C++ obj tags or VBO
        self.name            = name
        self.col: tuple      = col

        self.radius: float   = 2.
        self.verts: int      = 12            # number of OUTER vertices (Cell requires centre)
        self.velocity        = Vec2(0.,0.)   # initial speed
        self.colliding       = False         # flag for if Cell is colliding with something

        self.max_hp: float   = 10.           # maximum health
        self.hp: float       = self.max_hp   # current health
        self.max_nrg: float  = 10.           # maximum energy
        self.nrg: float      = self.max_nrg  # current energy
        self.nrg_loss_rate   = .001          # rate of energy loss BALANCE
        self.speed: float    = 1.            # amount to add to position per frame per dt
        self.salinity: float = 0.            # current salt level

        print(f"Loading {self.name} VBO data...")
        # load the default Cell from file
        loaded_file = BamFile()
        loaded_file.open_read("Cell_default.bam")
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

        self.nodepath = base.render.attach_new_node(node)
        # store position in the node 
        self.nodepath.set_pos(pos.x, pos.y, 0)
        # give the Cell a depth offset to prevent self-shadowing etc
        self.nodepath.setDepthOffset(1)

        # activate the jiggle shader on the Cell
        self.nodepath.set_shader(Shader.load(Shader.SL_GLSL, vertex="Cell_jiggle.vert", fragment="default_shader.frag"))
        self.nodepath.set_shader_input("ssbo", self.buffer)
        self.nodepath.set_shader_input("model_velocity", self.velocity)
        self.nodepath.set_shader_input("radius", self.radius)
        self.nodepath.set_shader_input("col", self.col)

        # now set up accessories
        self.bong           = base.sfx.add_bong(bong_freq)          # generate sound effect at given freq
        self.mols          = []                                     # array for mols on Cell
        self.spinner: float = 0                                     # this tells mols on this Cell how to rotate neatly
        self.num_mols: int = len(self.mols)                         # to help with angle calculations

        base.taskMgr.add(self.update, str(name)+"-update")

        print(f"== Cell {name} created!")

    # alias to make this quicker
    def pos(self) -> Vec3:
        return self.nodepath.get_pos()

    def grow(self):
        self.max_hp += 1.                                           # increase maximum health
        self.radius *= 1.1                                          # make the Cell bigger
        self.nrg_loss_rate += .001                                  # lose energy faster BALANCE
        self.nodepath.set_shader_input("radius", self.radius)       # update the shader

    def make_mol(self, *mols: MOLTYPE):
        if self.nrg > 1.:
            for mol in mols:
                self.nrg -= 1.
                match mol:
                    case MOLTYPE.FRNA:
                        self.mols.append(Mol(MOLTYPE.FRNA, self.name+"-FRNAmol-"+str(self.num_mols), 
                                               cell=self, index=self.num_mols))
                        self.num_mols += 1
                    case MOLTYPE.FOOD:
                        self.mols.append(Mol(MOLTYPE.FOOD, self.name+"-FOODmol-"+str(self.num_mols), 
                                               cell=self, index=self.num_mols))
                        self.num_mols += 1
        else:
            print("DANGER: Insufficient energy!")

    def consume_mol(self, mol=None):
        if isinstance(mol,type(None)):                              # consume the first mol in the buffer
            if len(self.mols) > 0:
                mol = self.mols[0]
                mol.consume()
                del self.mols[0]
                self.num_mols -= 1
            else:
                print("No mols to consume!")                        # consumption failed, return
                return;
        else:
            mol.consume()
            del mol

        match mol.type:                                             # activate the appropriate effect
            case MOLTYPE.FOOD:
                self.grow()
                # TODO add vertex
                # TODO update VBO on adding/removing verts
            case MOLTYPE.HEAL:
                # no overhealing - just top up health to maximum health at most
                self.hp = min(self.max_hp, self.hp + 1.)
            case MOLTYPE.MANA:
                self.nrg = min(self.max_nrg, self.nrg + 1.)
            case MOLTYPE.SALT:
                self.salinity += 1.                                 # TODO salinity moves energy loss to hp loss
                self.nrg_loss_rate *= .75                           # reduce energy loss rate by a quarter BALANCE
            case MOLTYPE.FRNA:
                print("Power up!")
                # TODO metabolic objectives; growing utilities. Menu or progression? Unlocks?
                self.speed *= 1.5                                   # increase cell speed

    def add_mol(self, mol=None, mols: int = 1):
        print(f"adding mol to {self.name}")
        self.mols.append(mol)                                       # add mol to mols
        mol.cell = self                                             # change mol references to self
        mol.index = self.num_mols                                   # n.b. this is only incremented at the end of method
        mol.set_orbiting_true()                                     # make mol orbit
        base.sfx.bongs[self.bong].play()                            # make a jubilant bong
        self.num_mols += 1                                          # incremement mol count

    def update(self, task):
        # processor-killing debug:
        #print(f"cell {self.name} nodepath position: {self.pos()}")
        #vtx_view = memoryview(self.nodepath.node().get_vertex_data().modify_array(0)).cast('B').cast('f')
        #print(f"some verts from {self.name}: 0: {vtx_view[0]}, 1: {vtx_view[1]}, 2: {vtx_view[2]}")

        # tick energy loss
        self.nrg -= self.nrg_loss_rate

        if (self.nrg <= 0.) or (self.hp <= 0.):
            self.die()

        # naive collision check with items - TODO spacial hashing
        for item in base.floating_items:
            if ABS_DIST(self.pos(), Vec3(item.nodepath.get_pos().xy, 0)) < (self.radius + item.radius):
                self.add_mol(item)
                base.floating_items.remove(item)

        self.nodepath.set_pos(self.pos() + Vec3(self.velocity, 0.))
        self.nodepath.set_shader_input("model_velocity", self.velocity)
        # cell experiences friction, causing velocity to naturally decrease
        self.velocity = self.velocity/10. if self.velocity > EPSILON else Vec2(0.,0.) 

        self.spinner += (globalClock.getDt())%360
        return task.cont

    # move the cell by its nodepath.pos
    def move(self, direction):
        pos = self.pos()
        speed = self.speed * globalClock.getDt()
        match direction:
            case "left":                                            # go left
                self.velocity -=  Vec2(speed,0.)
            case "right":                                           # go right
                self.velocity +=  Vec2(speed,0.)
            case "fwd":                                             # go forwards
                self.velocity +=  Vec2(0.,speed)
            case "back":                                            # ...you guessed it
                self.velocity -=  Vec2(0.,speed)
            case _: 
                print("Move direction not recognised!")

    def die(self):
        # drop mols
        for mol in self.mols:
            mol.orbiting = False
            mol.cell = None
            mol.index = None
            base.floating_items.append(mol)
        # die
        taskMgr.remove(self.name+"-update")
        if self.name == "p1":
            if self.nrg <= 0.:
                base.game_over(" You ran out of energy! ")
            elif self.hp <= 0.:
                base.game_over(" Your health ran out! ")
            else:
                base.game_over(" You died! ")