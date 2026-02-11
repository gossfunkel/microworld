from panda3d.core import (
    Vec3
)
import time
import mol

# TODO update camera position when player Cell gets significantly bigger
CAM_POS: Vec3 = Vec3(0,-8,3)        # for keeping the camera a constant vector from the player Cell

def zoom_in():
    current_fov = base.cam.node().getLens().get_fov()
    if (current_fov[0] > .1) & (current_fov[1] > .1):
        base.cam.node().getLens().setFov(current_fov[0]*.9,current_fov[1]*.9)

def zoom_out():
    current_fov = base.cam.node().getLens().get_fov()
    if (current_fov[0] < 180) & (current_fov[1] < 180):
        base.cam.node().getLens().setFov(current_fov[0]*1.1,current_fov[1]*1.1)

def bind_cell(cell):
    base.CAM_POS = CAM_POS
    # awsd/keypad movement for p1 Cell
    base.accept("arrow_left", cell.move, ["left"])
    base.accept("arrow_left-repeat", cell.move, ["left"])
    base.accept("a", cell.move, ["left"])
    base.accept("a-repeat", cell.move, ["left"])
    base.accept("arrow_right", cell.move, ["right"])
    base.accept("arrow_right-repeat", cell.move, ["right"])
    base.accept("d", cell.move, ["right"])
    base.accept("d-repeat", cell.move, ["right"])
    base.accept("arrow_up", cell.move, ["fwd"])
    base.accept("arrow_up-repeat", cell.move, ["fwd"])
    base.accept("w", cell.move, ["fwd"])
    base.accept("w-repeat", cell.move, ["fwd"])
    base.accept("arrow_down", cell.move, ["back"])
    base.accept("arrow_down-repeat", cell.move, ["back"])
    base.accept("s", cell.move, ["back"])
    base.accept("s-repeat", cell.move, ["back"])

    base.accept("wheel_up", zoom_in)
    base.accept("wheel_down", zoom_out)
    
    base.accept("r", cell.make_mol, [mol.MOLTYPE.FRNA])             # make an frna mol from energy
    base.accept("f", cell.make_mol, [mol.MOLTYPE.FOOD])             # make an frna mol from energy
    base.accept("space", cell.consume_mol)                          # consume a mol
    base.accept("escape", base.userExit)                            # quickly quit the game

    base.cam.setPos(base.CAM_POS)                                   # spawn camera distance from origin
    base.cam.setHpr(0,-18,0)                                        # look down at your Cell! 

