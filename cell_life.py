import ctypes
from ctypes import util, cdll, c_int, c_uint, c_float, c_void_p
import os

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
