import ctypes
from ctypes import util, cdll, c_int, c_uint, c_float, c_void_p
import os
from panda3d.core import Vec2, Vec3#, NodePath

print("- Initialising cell_life library bindings...")
panda_path = ctypes.util.find_library("libpanda")
print(f"..Panda library found at {panda_path}")
lpanda = cdll.LoadLibrary(panda_path)

if os.path.exists("build\\RelWithDebInfo\\cell_life.dll"):
    print("..cell_life.dll exists at expected location. Loading...")
else:
    print("... ERROR cell_life.dll not found!")
    exit()
libcell = cdll.LoadLibrary("build\\RelWithDebInfo\\cell_life.dll")

# CELL CONSTRUCTOR
libcell.Cell_new.argtypes = [c_int, c_float, c_int]
libcell.Cell_new.restype  = c_void_p

# CELL DESTRUCTOR
libcell.Cell_delete.argtypes = [c_void_p]
libcell.Cell_delete.restype  = None

# Cell dying getter
libcell.Cell_is_dying.argtypes = [c_void_p]
libcell.Cell_is_dying.restype  = c_int

# RESOURCE METHODS
libcell.Cell_get_resource.argtypes = [c_void_p, c_uint]
libcell.Cell_get_resource.restype  = c_float

libcell.Cell_spend_resource.argtypes = [c_void_p, c_uint, c_float]
libcell.Cell_spend_resource.restype  = c_int

libcell.Cell_add_resource.argtypes = [c_void_p, c_uint, c_float]
libcell.Cell_add_resource.restype  = c_float

# METABOLIC PROCESS METHODS
# returns process index
libcell.Cell_add_process.argtypes = [c_void_p, c_uint, c_uint, c_float, c_float, c_float, c_int]
libcell.Cell_add_process.restype  = c_uint

# returns success/fail
libcell.Cell_pause_process.argtypes = [c_void_p, c_uint]
libcell.Cell_pause_process.restype = c_int

# returns success/fail
libcell.Cell_resume_process.argtypes = [c_void_p, c_uint]
libcell.Cell_resume_process.restype = c_int

# returns pause status as bool
libcell.Cell_toggle_process.argtypes = [c_void_p, c_uint]
libcell.Cell_toggle_process.restype = c_int

# libcell.Process_get_task.argtypes = [c_void_p]
# libcell.Process_get_task.restype  = c_void_p

libcell.num_cells: int = 0

class Cell():
    def __init__(self, name: str, col: tuple, size: float, abilities: int):
        self.name            = name
        self.col: tuple      = col
        self.velocity        = Vec2(0.,0.)   # initial speed
        self.ptr             = libcell.Cell_new(libcell.num_cells, size, abilities)
        self.metabolism      = []
        self.colliding       = False         # flag for if Cell is colliding with something
        #self.bong       = base.sfx.add_bong(bong_freq)          # generate sound effect at given freq

        base.taskMgr.add(self.update, str(name)+"-update")
        libcell.num_cells    += 1
        print(f"== Cell {name} created!")

    def __del__(self):
        libcell.Cell_delete(self.ptr)

    def get_resource(self, res_idx: int):
        return libcell.Cell_get_resource(self.ptr, res_idx)

    def add_resource(self, res_idx: int, qty: float):
        res_value = libcell.Cell_add_resource(self.ptr, res_idx, qty)
        print(f"== Resource added: now contains {res_value}")

    def spend_resource(self, res_idx: int, qty: float):
        return libcell.Cell_spend_resource(self.ptr, res_idx, qty)

    def add_process(self, in_type: int, out_type: int, cost: float, yld: float, time: float, start_paused: bool):
        self.metabolism.append(libcell.Cell_add_process(self.ptr, in_type, out_type, cost, c_float(yld), c_float(time), c_int(start_paused)))
        return len(self.metabolism) - 1

    def pause(self, proc_idx: int):
        if libcell.Cell_pause_process(self.ptr, self.metabolism[proc_idx]):
            print(f"=! Failed to pause process {proc_idx}!")

    def resume(self, proc_idx: int):
        if libcell.Cell_resume_process(self.ptr, self.metabolism[proc_idx]):
            print(f"=! Failed to resume process {proc_idx}!")

    def dying(self): # TODO reference tracking in main to remove references to dead cells
        # check if hp hits 0, and check if c++ process has noticed energy is depleted
        return (libcell.Cell_is_dying(self.ptr) or (self.get_resource(1) <= 0.))

    def update(self, task):
        if self.dying():
            return task.done

        # naive collision check with items - TODO spacial hashing
        # for item in base.floating_items:
        #     if ABS_DIST(self.pos(), Vec3(item.nodepath.get_pos().xy, 0)) < (self.radius + item.radius):
        #         self.add_mol(item)
        #         base.floating_items.remove(item)

        #self.nodepath.set_pos(self.pos() + Vec3(self.velocity, 0.))
        #self.nodepath.set_shader_input("model_velocity", self.velocity)
        # cell experiences friction, causing velocity to naturally decrease
        #self.velocity = self.velocity/10. if self.velocity > EPSILON else Vec2(0.,0.) 

        #self.spinner += (globalClock.getDt())%360
        return task.cont

    # move the cell by its nodepath.pos
    # def move(self, direction):
    #     pos = self.pos()
    #     speed = self.speed * globalClock.getDt()
    #     match direction:
    #         case "left":                                            # go left
    #             self.velocity -=  Vec2(speed,0.)
    #         case "right":                                           # go right
    #             self.velocity +=  Vec2(speed,0.)
    #         case "fwd":                                             # go forwards
    #             self.velocity +=  Vec2(0.,speed)
    #         case "back":                                            # ...you guessed it
    #             self.velocity -=  Vec2(0.,speed)
    #         case _: 
    #             print("Move direction not recognised!")
