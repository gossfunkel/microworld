import ctypes
from ctypes import util, cdll, c_int, c_float, c_void_p

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

libcell.Cell_new.argtypes = [c_int, c_float, c_int]
libcell.Cell_new.restype  = c_void_p

libcell.Cell_delete.argtypes = [c_void_p]
libcell.Cell_delete.restype  = None

# water
libcell.Cell_get_water.argtypes = [c_void_p]
libcell.Cell_get_water.restype  = c_float

libcell.Cell_spend_water.argtypes = [c_void_p, c_float]
libcell.Cell_spend_water.restype  = c_int

libcell.Cell_add_water.argtypes = [c_void_p, c_float]
libcell.Cell_add_water.restype  = c_float

# salts
libcell.Cell_get_salts.argtypes = [c_void_p]
libcell.Cell_get_salts.restype  = c_float

libcell.Cell_spend_salts.argtypes = [c_void_p, c_float]
libcell.Cell_spend_salts.restype  = c_int

libcell.Cell_add_salts.argtypes = [c_void_p, c_float]
libcell.Cell_add_salts.restype  = c_float

# oils
libcell.Cell_get_oils.argtypes = [c_void_p]
libcell.Cell_get_oils.restype  = c_float

libcell.Cell_spend_oils.argtypes = [c_void_p, c_float]
libcell.Cell_spend_oils.restype  = c_int

libcell.Cell_add_oils.argtypes = [c_void_p, c_float]
libcell.Cell_add_oils.restype  = c_float

# sugar
libcell.Cell_get_sugar.argtypes = [c_void_p]
libcell.Cell_get_sugar.restype  = c_float

libcell.Cell_spend_sugar.argtypes = [c_void_p, c_float]
libcell.Cell_spend_sugar.restype  = c_int

libcell.Cell_add_sugar.argtypes = [c_void_p, c_float]
libcell.Cell_add_sugar.restype  = c_float

# carbs
libcell.Cell_get_carbs.argtypes = [c_void_p]
libcell.Cell_get_carbs.restype  = c_float

libcell.Cell_spend_carbs.argtypes = [c_void_p, c_float]
libcell.Cell_spend_carbs.restype  = c_int

libcell.Cell_add_carbs.argtypes = [c_void_p, c_float]
libcell.Cell_add_carbs.restype  = c_float

# amino
libcell.Cell_get_amino.argtypes = [c_void_p]
libcell.Cell_get_amino.restype  = c_float

libcell.Cell_spend_amino.argtypes = [c_void_p, c_float]
libcell.Cell_spend_amino.restype  = c_int

libcell.Cell_add_amino.argtypes = [c_void_p, c_float]
libcell.Cell_add_amino.restype  = c_float

# process management
libcell.Cell_add_process.argtypes = [c_void_p, c_int, c_int, c_float, c_float, c_float, c_int]
libcell.Cell_add_process.restype  = c_void_p

libcell.Process_get_task.argtypes = [c_void_p]
libcell.Process_get_task.restype  = c_void_p
