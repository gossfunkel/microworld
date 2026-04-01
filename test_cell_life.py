from direct.showbase.ShowBase import ShowBase
from direct.interval.IntervalGlobal import *
from direct.task.Task import Task
from panda3d.core import load_prc_file_data, AsyncTaskManager
import cell_life
from cell_life import libcell

async def print_cellstats(task):
    await Wait(1.)
    print(f"Py> Cell nrg: {libcell.Cell_get_resource(base.test_cell, 0)}")
    print(f"Py> Cell sugar: {libcell.Cell_get_resource(base.test_cell, 5)}")
    return task.again

if __name__ == '__main__':
    print(" = = =  TESTING cell_life library python wrapper  = = = ")

    print("Py> Loading Panda3d...")
    config_vars: str = """
    window-type none
    """
    load_prc_file_data("", config_vars)
    ShowBase()

    async_task_mgr = AsyncTaskManager.getGlobalPtr()

    base.test_cell = libcell.Cell_new(0, 1., 0)

    base.taskMgr.add(print_cellstats, "pr_cellstats")

    print(f"Py> Cell carbs: {libcell.Cell_get_resource(base.test_cell, 6)}")
    libcell.Cell_add_resource(base.test_cell, 6, 2.)
    print(f"Py> Carbs after adding 2: {libcell.Cell_get_resource(base.test_cell, 6)}")
    print(f"Py> Cell water level: {libcell.Cell_get_resource(base.test_cell, 3)}")
    libcell.Cell_spend_resource(base.test_cell, 3, 3.)
    print(f"Py> Water after spending 3: {libcell.Cell_get_resource(base.test_cell, 3)}")

    # 1 carbs in, 1 sugar out 
    base.test_proc_idx = libcell.Cell_add_process(base.test_cell, 6, 5, 1., 1., 1., 0)

    test_pause_resume = Sequence(
        Wait(3.),
        Func(print, "Py> Pausing carb-sugar metabolic task:"),
        Func(libcell.Cell_pause_process, base.test_cell, base.test_proc_idx),
        Wait(5.),
        Func(print, "Py> Resuming carb-sugar metabolic task:"),
        Func(libcell.Cell_resume_process, base.test_cell, base.test_proc_idx)
    ).start()

    #base.accept("escape", base.userExit)
    base.run()

    libcell.Cell_delete(base.test_cell)
    print(" = = =  python testing concluded. goodbye!  = = = ")

