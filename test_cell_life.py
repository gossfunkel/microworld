from direct.showbase.ShowBase import ShowBase
from direct.interval.IntervalGlobal import *
from direct.task.Task import Task
from panda3d.core import load_prc_file_data, AsyncTaskManager
import cell_life
from cell_life import Cell

async def print_cellstats(task):
    await Wait(1.)
    print(f"Py> Cell nrg: {base.test_cell.get_resource(0)}")
    print(f"Py> Cell sugar: {base.test_cell.get_resource(5)}")
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

    base.test_cell = Cell(1., 0)
    base.taskMgr.add(print_cellstats, "pr_cellstats")

    print(f"Py> Cell carbs: {base.test_cell.get_resource(6)}")
    base.test_cell.add_resource(6, 2.)
    print(f"Py> Carbs after adding 2: {base.test_cell.get_resource(6)}")
    print(f"Py> Cell water level: {base.test_cell.get_resource(3)}")
    base.test_cell.spend_resource(3, 3.)
    print(f"Py> Water after spending 3: {base.test_cell.get_resource(3)}")

    # 1 carbs in, 1 sugar out 
    base.test_proc_idx = base.test_cell.add_process(6, 5, 1., 1., 1., 0)

    test_pause_resume = Sequence(
        Wait(3.),
        Func(print, "Py> Pausing carb-sugar metabolic task:"),
        Func(base.test_cell.pause, base.test_proc_idx),
        Wait(5.),
        Func(print, "Py> Resuming carb-sugar metabolic task:"),
        Func(base.test_cell.resume, base.test_proc_idx)
    ).start()

    #base.accept("escape", base.userExit)
    base.run()

    print(" = = =  python testing concluded. goodbye!  = = = ")
