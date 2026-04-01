from direct.showbase.ShowBase import ShowBase
from direct.interval.IntervalGlobal import *
from direct.task.Task import Task
from panda3d.core import load_prc_file_data, AsyncTaskManager
import cell_life
from cell_life import libcell

async def print_sugar(task):
    await Wait(1.)
    print(f"Cell sugar: {libcell.Cell_get_resource(base.test_cell, 2)}")
    return task.again

if __name__ == '__main__':
    print(" = = =  TESTING cell_life library python wrapper  = = = ")

    print("Loading Panda3d...")
    config_vars: str = """
    window-type none
    """
    load_prc_file_data("", config_vars)
    ShowBase()

    async_task_mgr = AsyncTaskManager.getGlobalPtr()

    base.test_cell = libcell.Cell_new(0, 1., 0)
    print(f"Cell carbs: {libcell.Cell_get_resource(base.test_cell, 3)}")
    libcell.Cell_add_resource(base.test_cell, 3, 2.)
    print(f"Carbs after adding 2: {libcell.Cell_get_resource(base.test_cell, 3)}")
    print(f"Cell water level: {libcell.Cell_get_resource(base.test_cell, 0)}")
    libcell.Cell_spend_resource(base.test_cell, 0, 3.)
    print(f"Water after spending 3: {libcell.Cell_get_resource(base.test_cell, 0)}")

    # 1 carbs in, 1 sugar out 
    test_proc_idx = libcell.Cell_add_process(base.test_cell, 3, 2, 1., 1., 1., 0)

    base.taskMgr.add(print_sugar, "pr_sgr")

    base.accept("escape", base.userExit)
    base.run()

    libcell.Cell_delete(base.test_cell)
    print(" = = =  testing concluded. goodbye!  = = = ")

