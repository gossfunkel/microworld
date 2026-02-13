from panda3d.core import (
    Vec3
)
import time
import mol

CAM_POS: Vec3 = Vec3(0,-6,2)                                        # initial camera position at game load

def zoom_in():
    current_fov = base.cam.node().getLens().get_fov()
    if (current_fov[0] > 1) & (current_fov[1] > 1):
        base.cam.node().getLens().setFov(current_fov[0]*.98,current_fov[1]*.98)

def zoom_out():
    current_fov = base.cam.node().getLens().get_fov()
    if (current_fov[0] < 90) & (current_fov[1] < 90):
        base.cam.node().getLens().setFov(current_fov[0]*1.02,current_fov[1]*1.02)

# click and drag to move the camera
def drag_cam():
    base.taskMgr.add(cam_drag_task, "drag_cam")

def release_cam():
    base.taskMgr.remove("drag_cam")
    del base.last_mouse_pos

def cam_drag_task(task):
    if base.mouseWatcherNode.hasMouse():
        x = base.mouseWatcherNode.getMouseX()                       # get mouse position
        y = base.mouseWatcherNode.getMouseY()
        if hasattr(base, 'last_mouse_pos'):                         # if this has run at least once
            dt = globalClock.get_dt()
            mouse_move = Vec3(float(x - base.last_mouse_pos[0]) * -1.5, float(y - base.last_mouse_pos[1]) * -1.5, 0.)
            base.CAM_POS += mouse_move                              # add change in mouse pos to cam pos
        base.last_mouse_pos = (x,y)
    return task.cont

# bind a cell to the player wasd input and initialise camera controls
def bind_cell(cell):
    base.disableMouse()                                             # switch off default controls
    base.CAM_POS = CAM_POS                                          # initialise base.CAM_POS

    # setup awsd/keypad movement for p1 Cell
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

    # setup dragging to move camera
    base.accept("mouse1", drag_cam)
    base.accept("mouse1-up", release_cam)

    # setup zoom with scrollwheel
    base.accept("wheel_up", zoom_in)
    base.accept("wheel_down", zoom_out)
    
    # game control buttons
    base.accept("r", cell.make_mol, [mol.MOLTYPE.AMINO])            # make an amino mol from energy (r for rna)
    base.accept("f", cell.make_mol, [mol.MOLTYPE.CARB])             # make an carb mol from energy (f for food)
    base.accept("g", cell.grow)                                     # grow cell (if affordable)
    base.accept("h", cell.heal)                                     # heal cell (if affordable)
    #base.accept("b", cell.boost)                                    # give cell speed boost (if affordable)
    base.accept("m", cell.base_metabolism.toggle)                        # switch basic carb breakdown on/off
    base.accept("j", cell.select_mol, ['left'])
    base.accept("k", cell.select_mol, ['right'])
    base.accept("space", cell.consume_mol)                          # consume a mol
    base.accept("escape", base.userExit)                            # quickly quit the game

    # initialise camera position and angle
    base.cam.setPos(base.CAM_POS)                                   # spawn camera distance from origin
    base.cam.setHpr(0,-19,0)                                        # look down at your Cell! 
